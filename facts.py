"""Tier 1: deterministic fact extraction from a PatientSubmission.

Everything in this module is pure Python - no LLM calls. It only resolves
facts that are either already structured fields, or closed-vocabulary /
fixed-format values (lab codes, drug brand/generic names, numeric vitals in
standard clinical notation, unit conversions, dates). Anything that requires
reading comprehension over open-ended free text (which document counts as
the real pre-op H&P, whether a consent event is valid, whether a flagged
medication is currently active, whether an anticoagulation plan is adequate)
is explicitly NOT handled here - see judgments.py for those.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from core import Document, Medication, PatientSubmission

# -------------------------
# Shared primitives
# -------------------------


class EntryStatus(str, Enum):
    VALID = "valid"
    ENTERED_IN_ERROR = "entered_in_error"
    UNKNOWN = "unknown"


class Provenance(BaseModel):
    """Where a fact came from, with the exact evidence snippet it was read from."""

    source: Literal["structured_field", "document_text"]
    doc_id: str | None = None
    doc_type: str | None = None  # the document's raw `type` field, when source is a document
    quote: str
    asserted_at: datetime | None = None


def parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO8601 date or datetime string into a naive datetime.

    Timezone info (if any) is dropped: the policy's windows are day-level
    and the dataset does not need cross-timezone precision.
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def days_between(later: date, earlier: date) -> int:
    return (later - earlier).days


# -------------------------
# Labs (Rule 2)
# -------------------------

# Closed map from the submission's `code` field to the canonical test this
# policy cares about. Deliberately NOT fuzzy: "BMP"/"LAB-BMP" (Basic
# Metabolic Panel) must never resolve to CMP, even though both are metabolic
# panels - that distinction is exactly what the policy requires.
LAB_CODE_MAP: dict[str, Literal["CBC", "CMP"]] = {
    "CBC": "CBC",
    "LAB-CBC": "CBC",
    "CMP": "CMP",
    "LAB-CMP": "CMP",
}


class LabFact(BaseModel):
    test_code: Literal["CBC", "CMP"]
    effective_at: datetime
    status: EntryStatus
    provenance: Provenance


class LabsFact(BaseModel):
    results: list[LabFact] = Field(default_factory=list)

    def latest(self, test_code: Literal["CBC", "CMP"]) -> LabFact | None:
        """Most recent VALID result for a test code, or None if there isn't one."""

        candidates = [
            r for r in self.results if r.test_code == test_code and r.status == EntryStatus.VALID
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.effective_at)


def _normalize_lab_status(raw: str | None) -> EntryStatus:
    if raw == "final":
        return EntryStatus.VALID
    if raw == "entered-in-error":
        return EntryStatus.ENTERED_IN_ERROR
    return EntryStatus.UNKNOWN


def extract_labs(submission: PatientSubmission) -> LabsFact:
    results: list[LabFact] = []
    for lab in submission.labs:
        code = (lab.code or "").strip().upper()
        test_code = LAB_CODE_MAP.get(code)
        if test_code is None:
            continue
        effective_at = parse_dt(lab.effective_at)
        if effective_at is None:
            continue
        results.append(
            LabFact(
                test_code=test_code,
                effective_at=effective_at,
                status=_normalize_lab_status(lab.status),
                provenance=Provenance(
                    source="structured_field",
                    doc_id=lab.id,
                    quote=f"{lab.display or lab.code} ({lab.code}) effective {lab.effective_at}, "
                    f"status={lab.status}",
                    asserted_at=effective_at,
                ),
            )
        )
    return LabsFact(results=results)


# -------------------------
# Medications / anticoagulation (Rule 1 trap: anticoagulant vs antiplatelet)
# -------------------------


class DrugClass(str, Enum):
    ANTICOAGULANT = "anticoagulant"
    ANTIPLATELET = "antiplatelet"


# Closed brand -> (generic, class) table. Scoped to what this policy needs to
# distinguish (anticoagulant vs. antiplatelet - Rule 3's true-negative trap),
# not an exhaustive real-world formulary. Covers, by generic:
#   Anticoagulants: warfarin (Coumadin, Jantoven); the DOACs apixaban
#     (Eliquis), rivaroxaban (Xarelto), dabigatran (Pradaxa), edoxaban
#     (Savaysa, Lixiana); the parenteral/bridging agents enoxaparin
#     (Lovenox), dalteparin (Fragmin), heparin, fondaparinux (Arixtra).
#   Antiplatelets: aspirin (ASA); the P2Y12 inhibitors clopidogrel
#     (Plavix), ticagrelor (Brilinta), prasugrel (Effient); dipyridamole
#     (Aggrenox, which is aspirin + extended-release dipyridamole - the
#     brand resolves to the dipyridamole component here).
DRUG_ALIASES: dict[str, tuple[str, DrugClass]] = {
    "warfarin": ("warfarin", DrugClass.ANTICOAGULANT),
    "coumadin": ("warfarin", DrugClass.ANTICOAGULANT),
    "jantoven": ("warfarin", DrugClass.ANTICOAGULANT),
    "apixaban": ("apixaban", DrugClass.ANTICOAGULANT),
    "eliquis": ("apixaban", DrugClass.ANTICOAGULANT),
    "rivaroxaban": ("rivaroxaban", DrugClass.ANTICOAGULANT),
    "xarelto": ("rivaroxaban", DrugClass.ANTICOAGULANT),
    "dabigatran": ("dabigatran", DrugClass.ANTICOAGULANT),
    "pradaxa": ("dabigatran", DrugClass.ANTICOAGULANT),
    "edoxaban": ("edoxaban", DrugClass.ANTICOAGULANT),
    "savaysa": ("edoxaban", DrugClass.ANTICOAGULANT),
    "lixiana": ("edoxaban", DrugClass.ANTICOAGULANT),
    "enoxaparin": ("enoxaparin", DrugClass.ANTICOAGULANT),
    "lovenox": ("enoxaparin", DrugClass.ANTICOAGULANT),
    "dalteparin": ("dalteparin", DrugClass.ANTICOAGULANT),
    "fragmin": ("dalteparin", DrugClass.ANTICOAGULANT),
    "heparin": ("heparin", DrugClass.ANTICOAGULANT),
    "fondaparinux": ("fondaparinux", DrugClass.ANTICOAGULANT),
    "arixtra": ("fondaparinux", DrugClass.ANTICOAGULANT),
    "aspirin": ("aspirin", DrugClass.ANTIPLATELET),
    "asa": ("aspirin", DrugClass.ANTIPLATELET),
    "clopidogrel": ("clopidogrel", DrugClass.ANTIPLATELET),
    "plavix": ("clopidogrel", DrugClass.ANTIPLATELET),
    "ticagrelor": ("ticagrelor", DrugClass.ANTIPLATELET),
    "brilinta": ("ticagrelor", DrugClass.ANTIPLATELET),
    "prasugrel": ("prasugrel", DrugClass.ANTIPLATELET),
    "effient": ("prasugrel", DrugClass.ANTIPLATELET),
    "dipyridamole": ("dipyridamole", DrugClass.ANTIPLATELET),
    "aggrenox": ("dipyridamole", DrugClass.ANTIPLATELET),
}

_DRUG_ALIAS_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(DRUG_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# "ASA" collides with the ASA (American Society of Anesthesiologists) physical
# status classification ("ASA physical status: II", "ASA class III") that
# appears in nearly every anesthesia note - that is a risk score, not a
# mention of the drug aspirin. Verified against the full sample dataset: this
# is the only alias with a real false-positive collision.
_ASA_CLASS_SUFFIX_RE = re.compile(r"\s*(physical\s+status|class)\b", re.IGNORECASE)


class MedicationMention(BaseModel):
    generic_name: str
    drug_class: DrugClass
    active: bool | None = None  # only ever set for structured-field mentions
    provenance: Provenance


class DrugMentionGroup(BaseModel):
    """All mentions of one generic drug, pooled across structured + free text."""

    generic_name: str
    drug_class: DrugClass
    mentions: list[MedicationMention]

    @property
    def structured_mention(self) -> MedicationMention | None:
        return next(
            (m for m in self.mentions if m.provenance.source == "structured_field"), None
        )

    @property
    def document_mentions(self) -> list[MedicationMention]:
        return [m for m in self.mentions if m.provenance.source == "document_text"]

    def needs_activity_judgment(self) -> bool:
        """Whether a Tier 2 call is needed to resolve current activity status.

        Skippable only when the structured field gives an unambiguous
        `active` value AND nothing else in the chart even mentions the drug
        (no conflicting or corroborating free-text signal to weigh).
        """

        s = self.structured_mention
        if s is None or s.active is None:
            return True
        return bool(self.document_mentions)


def _find_drug_matches(text: str) -> list[tuple[str, DrugClass, re.Match[str]]]:
    """First match of each generic drug in text, keeping the match object itself.

    Keeping the match (not just the generic name) matters: the document text
    uses whatever alias was actually written there (e.g. "Xarelto"), which is
    often not the canonical generic name ("rivaroxaban") - so callers that
    need to quote the surrounding text must anchor on the real match
    position, not re-search for the generic name (which may never literally
    appear in the text at all).
    """

    found: dict[str, tuple[DrugClass, re.Match[str]]] = {}
    for match in _DRUG_ALIAS_RE.finditer(text):
        alias = match.group(1).lower()
        if alias == "asa" and _ASA_CLASS_SUFFIX_RE.match(text, match.end()):
            continue  # anesthesiology risk class, not the drug - see comment above the table
        generic, drug_class = DRUG_ALIASES[alias]
        if generic not in found:
            found[generic] = (drug_class, match)
    return [(generic, drug_class, match) for generic, (drug_class, match) in found.items()]


def _mentions_from_medication(med: Medication) -> list[MedicationMention]:
    name = med.name or ""
    mentions = []
    for generic, drug_class, _match in _find_drug_matches(name):
        mentions.append(
            MedicationMention(
                generic_name=generic,
                drug_class=drug_class,
                active=med.active,
                provenance=Provenance(source="structured_field", quote=name, asserted_at=None),
            )
        )
    return mentions


def _mentions_from_document(doc: Document) -> list[MedicationMention]:
    text = doc.text or ""
    doc_dt = parse_dt(doc.date)
    mentions = []
    for generic, drug_class, match in _find_drug_matches(text):
        mentions.append(
            MedicationMention(
                generic_name=generic,
                drug_class=drug_class,
                active=None,
                provenance=Provenance(
                    source="document_text",
                    doc_id=doc.doc_id,
                    doc_type=doc.type,
                    quote=_excerpt_at(text, match.start(), match.end()),
                    asserted_at=doc_dt,
                ),
            )
        )
    return mentions


def _excerpt_at(text: str, match_start: int, match_end: int, radius: int = 160) -> str:
    start = max(0, match_start - radius // 2)
    end = min(len(text), match_end + radius // 2)
    return text[start:end].strip()


def scan_medication_mentions(submission: PatientSubmission) -> list[DrugMentionGroup]:
    """Pool every structured + free-text mention of a known drug, grouped by generic name."""

    all_mentions: list[MedicationMention] = []
    for med in submission.medications:
        all_mentions.extend(_mentions_from_medication(med))
    for doc in submission.documents:
        all_mentions.extend(_mentions_from_document(doc))

    groups: dict[str, DrugMentionGroup] = {}
    for mention in all_mentions:
        group = groups.get(mention.generic_name)
        if group is None:
            groups[mention.generic_name] = DrugMentionGroup(
                generic_name=mention.generic_name,
                drug_class=mention.drug_class,
                mentions=[mention],
            )
        else:
            group.mentions.append(mention)
    return list(groups.values())


# -------------------------
# Vitals (Rule 4) - structured array + regex extraction from free text
# -------------------------


class VitalReading(BaseModel):
    kind: Literal["blood_pressure", "temperature"]
    systolic: float | None = None
    diastolic: float | None = None
    temp_f: float | None = None
    status: EntryStatus
    provenance: Provenance


class VitalsFact(BaseModel):
    blood_pressure: list[VitalReading] = Field(default_factory=list)
    temperature: list[VitalReading] = Field(default_factory=list)

    def latest_blood_pressure(self) -> VitalReading | None:
        candidates = [
            r
            for r in self.blood_pressure
            if r.status == EntryStatus.VALID and r.provenance.asserted_at is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.provenance.asserted_at)

    def latest_temperature(self) -> VitalReading | None:
        candidates = [
            r
            for r in self.temperature
            if r.status == EntryStatus.VALID and r.provenance.asserted_at is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.provenance.asserted_at)


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_BP_RE = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")
_TEMP_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*°?\s*(F|C)\b")

_BP_SYSTOLIC_RANGE = (70, 260)
_BP_DIASTOLIC_RANGE = (40, 150)
_TEMP_F_RANGE = (85.0, 112.0)


def _nearest_preceding_time(pos: int, times: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    candidate: tuple[int, int] | None = None
    for start, hour, minute in times:
        if start <= pos:
            candidate = (hour, minute)
        else:
            break
    return candidate


def _line_containing(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def extract_vitals_from_text(
    text: str, *, doc_id: str | None, fallback_date: datetime, doc_type: str | None = None
) -> list[VitalReading]:
    """Best-effort extraction of BP/temperature readings embedded in clinical note text.

    Associates each numeric reading with the nearest preceding HH:MM time
    token anywhere earlier in the document (not line-scoped, since a single
    note may describe multiple timed readings on one line or across lines),
    falling back to the document's own date when no time token precedes it.
    Placeholder text ("See flowsheet.") naturally yields no matches.
    """

    times = [(m.start(), int(m.group(1)), int(m.group(2))) for m in _TIME_RE.finditer(text)]
    readings: list[VitalReading] = []

    for match in _BP_RE.finditer(text):
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        if not (
            _BP_SYSTOLIC_RANGE[0] <= systolic <= _BP_SYSTOLIC_RANGE[1]
            and _BP_DIASTOLIC_RANGE[0] <= diastolic <= _BP_DIASTOLIC_RANGE[1]
            and systolic > diastolic
        ):
            continue
        hm = _nearest_preceding_time(match.start(), times)
        asserted_at = fallback_date.replace(hour=hm[0], minute=hm[1]) if hm else fallback_date
        readings.append(
            VitalReading(
                kind="blood_pressure",
                systolic=systolic,
                diastolic=diastolic,
                status=EntryStatus.VALID,
                provenance=Provenance(
                    source="document_text",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    quote=_line_containing(text, match.start()),
                    asserted_at=asserted_at,
                ),
            )
        )

    for match in _TEMP_RE.finditer(text):
        value, unit = float(match.group(1)), match.group(2).upper()
        temp_f = value if unit == "F" else value * 9 / 5 + 32
        if not (_TEMP_F_RANGE[0] <= temp_f <= _TEMP_F_RANGE[1]):
            continue
        hm = _nearest_preceding_time(match.start(), times)
        asserted_at = fallback_date.replace(hour=hm[0], minute=hm[1]) if hm else fallback_date
        readings.append(
            VitalReading(
                kind="temperature",
                temp_f=round(temp_f, 1),
                status=EntryStatus.VALID,
                provenance=Provenance(
                    source="document_text",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    quote=_line_containing(text, match.start()),
                    asserted_at=asserted_at,
                ),
            )
        )

    return readings


def extract_vitals(submission: PatientSubmission) -> VitalsFact:
    blood_pressure: list[VitalReading] = []
    temperature: list[VitalReading] = []

    for vital in submission.vitals:
        vtype = getattr(vital, "type", None)
        vdate = parse_dt(getattr(vital, "date", None))
        if vdate is None:
            continue
        if vtype == "blood_pressure":
            systolic = getattr(vital, "systolic", None)
            diastolic = getattr(vital, "diastolic", None)
            if systolic is None or diastolic is None:
                continue
            blood_pressure.append(
                VitalReading(
                    kind="blood_pressure",
                    systolic=float(systolic),
                    diastolic=float(diastolic),
                    status=EntryStatus.VALID,
                    provenance=Provenance(
                        source="structured_field",
                        quote=f"{systolic}/{diastolic} mmHg ({getattr(vital, 'source', None)})",
                        asserted_at=vdate,
                    ),
                )
            )
        elif vtype == "temperature":
            value_f = getattr(vital, "value_f", None)
            if value_f is None:
                continue
            temperature.append(
                VitalReading(
                    kind="temperature",
                    temp_f=float(value_f),
                    status=EntryStatus.VALID,
                    provenance=Provenance(
                        source="structured_field",
                        quote=f"{value_f} F ({getattr(vital, 'source', None)})",
                        asserted_at=vdate,
                    ),
                )
            )

    for doc in submission.documents:
        if not doc.text:
            continue
        doc_dt = parse_dt(doc.date)
        if doc_dt is None:
            continue
        for reading in extract_vitals_from_text(
            doc.text, doc_id=doc.doc_id, fallback_date=doc_dt, doc_type=doc.type
        ):
            if reading.kind == "blood_pressure":
                blood_pressure.append(reading)
            else:
                temperature.append(reading)

    return VitalsFact(blood_pressure=blood_pressure, temperature=temperature)


# -------------------------
# Procedure fields
# -------------------------


def procedure_date(submission: PatientSubmission) -> date | None:
    if submission.procedure is None:
        return None
    dt = parse_dt(submission.procedure.procedure_date)
    return dt.date() if dt else None


def procedure_risk(submission: PatientSubmission) -> str | None:
    if submission.procedure is None:
        return None
    return submission.procedure.procedure_risk
