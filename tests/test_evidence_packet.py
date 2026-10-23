"""Verifies the structured TriageIssue evidence packet: every field is populated,
and quotes are genuinely threaded from the extraction layer's own provenance -
never re-derived or fabricated by a rule function.
"""

from __future__ import annotations

from datetime import date, datetime

from core import PatientSubmission
from facts import EntryStatus, LabFact, LabsFact, Provenance, VitalReading, VitalsFact, extract_vitals
from policy_facts import (
    PolicyFacts,
    ResolvedAnticoagulant,
    ResolvedConsent,
    ResolvedHP,
    confidence_from_candidate_count,
)
from rules import evaluate_all, evaluate_anticoagulation, evaluate_documentation_hp, evaluate_safety, evaluate_testing

PROC_DATE = date(2026, 3, 15)


def _provenance(quote: str, doc_id: str | None = "d1", asserted_at: datetime | None = None) -> Provenance:
    return Provenance(source="document_text", doc_id=doc_id, quote=quote, asserted_at=asserted_at)


def _bp(
    systolic: float, diastolic: float, quote: str, doc_id: str = "d1", asserted_at: datetime | None = None
) -> VitalReading:
    return VitalReading(
        kind="blood_pressure",
        systolic=systolic,
        diastolic=diastolic,
        status=EntryStatus.VALID,
        provenance=_provenance(quote, doc_id, asserted_at or datetime.combine(PROC_DATE, datetime.min.time())),
    )


def make_facts(**overrides: object) -> PolicyFacts:
    defaults: dict[str, object] = dict(
        procedure_date=PROC_DATE,
        procedure_risk="LOW",
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 3, 5),
                    status=EntryStatus.VALID,
                    provenance=_provenance("CBC 03/05"),
                )
            ]
        ),
        vitals=VitalsFact(
            blood_pressure=[_bp(120, 76, "120/76 mmHg")],
            temperature=[
                VitalReading(
                    kind="temperature",
                    temp_f=98.6,
                    status=EntryStatus.VALID,
                    provenance=_provenance("98.6 F", asserted_at=datetime.combine(PROC_DATE, datetime.min.time())),
                )
            ],
        ),
        hp=ResolvedHP(doc_id="hp1", encounter_date=date(2026, 3, 1), evidence_quote="H&P on file"),
        consent=ResolvedConsent(valid=True, doc_id="c1", event_date=date(2026, 3, 10), evidence_quote="signed"),
        anticoagulants=[],
    )
    defaults.update(overrides)
    return PolicyFacts(**defaults)


# -------------------------
# confidence_from_candidate_count
# -------------------------


def test_confidence_high_for_zero_or_one_candidate() -> None:
    assert confidence_from_candidate_count(0) == "high"
    assert confidence_from_candidate_count(1) == "high"


def test_confidence_medium_for_multiple_candidates() -> None:
    assert confidence_from_candidate_count(2) == "medium"
    assert confidence_from_candidate_count(5) == "medium"


# -------------------------
# Every issue has a fully-populated packet
# -------------------------


def test_every_issue_across_all_categories_has_a_complete_evidence_packet() -> None:
    facts = make_facts(
        procedure_date=None,
        hp=ResolvedHP(doc_id=None, encounter_date=None, evidence_quote="no H&P on file"),
        consent=ResolvedConsent(valid=False, doc_id=None, event_date=None, evidence_quote="not signed"),
        vitals=VitalsFact(
            blood_pressure=[_bp(190, 90, "190/90 mmHg")],
            temperature=[],
        ),
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="apixaban",
                active=True,
                activity_evidence="active per consult",
                activity_confidence="medium",
                plan_checked=True,
                plan_adequate=False,
                plan_evidence="no plan found",
            )
        ],
    )
    issues = evaluate_all(facts)
    assert len(issues) >= 4  # missing prerequisites, consent, safety, anticoagulation all fire

    for issue in issues:
        assert issue.policy and isinstance(issue.policy, str)
        assert issue.findings and isinstance(issue.findings, str)
        assert issue.source.kind in ("structured_field", "document_text", "llm_judgment")
        assert isinstance(issue.notes, str)  # may be "" only for prerequisite-type issues
        assert issue.confidence in ("high", "medium", "low")


# -------------------------
# Notes are threaded verbatim from provenance, never re-derived
# -------------------------


def test_safety_issue_notes_is_exactly_the_vital_s_own_provenance_quote() -> None:
    quote = "09:06: BP 196/90 mmHg, Temperature 98.2 F (oral)"  # systolic-only exclusion
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(196, 90, quote)], temperature=[]))
    issues = evaluate_safety(facts)
    assert len(issues) == 1
    assert issues[0].notes == quote


def test_safety_issue_notes_is_a_real_substring_of_the_source_document_text() -> None:
    # End-to-end through the real extractor (not hand-built), proving the quote
    # genuinely comes from the document, not invented by a rule function.
    doc_text = (
        "PRE-ADMISSION NURSING INTAKE\nVisit date: 03/15/2026\n\n"
        "VITAL SIGNS: On arrival at 09:06: BP 196/90 mmHg, Temperature 98.2 F (oral), "
        "HR 93, SpO2 99% RA."
    )
    submission = PatientSubmission.model_validate(
        {
            "patient": {"id": "p1"},
            "procedure": {"case_id": "c1"},
            "vitals": [],
            "labs": [],
            "medications": [],
            "conditions": [],
            "documents": [
                {"doc_id": "d1", "type": "Pre-Admission Testing Visit", "date": "2026-03-15", "text": doc_text}
            ],
        }
    )
    vitals = extract_vitals(submission)
    facts = make_facts(vitals=vitals)
    issues = evaluate_safety(facts)
    assert len(issues) == 1
    assert issues[0].notes in doc_text  # exact substring of the real document, not paraphrased
    assert issues[0].source.doc_id == "d1"
    assert issues[0].source.doc_type == "Pre-Admission Testing Visit"


def test_testing_issue_notes_is_exactly_the_lab_s_own_provenance_quote() -> None:
    quote = "Hemogram/CBC (CBC) effective 2026-01-01T00:00:00, status=final"
    facts = make_facts(
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 1, 1),
                    status=EntryStatus.VALID,
                    provenance=_provenance(quote, doc_id="lab1"),
                )
            ]
        )
    )
    issues = evaluate_testing(facts)
    assert len(issues) == 1
    assert issues[0].notes == quote
    assert issues[0].source.doc_id == "lab1"


def test_hp_issue_notes_and_sample_are_threaded_from_resolved_hp_verbatim() -> None:
    facts = make_facts(
        hp=ResolvedHP(
            doc_id=None,
            encounter_date=None,
            evidence_quote="No document in the chart qualifies as a pre-op H&P.",
            rejected_candidates=["d2: wellness visit, too old", "d3: retained for context only"],
            confidence="medium",
        )
    )
    issue = evaluate_documentation_hp(facts)
    assert issue is not None
    assert issue.notes == facts.hp.evidence_quote
    assert issue.sample == facts.hp.rejected_candidates
    assert issue.confidence == "medium"


def test_anticoagulation_issue_notes_is_exactly_the_plan_evidence_not_reworded() -> None:
    plan_quote = '"Hold warfarin 5 days before surgery." No resume instructions given.'
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="warfarin",
                active=True,
                activity_evidence="active per structured field",
                plan_checked=True,
                plan_adequate=False,
                plan_evidence=plan_quote,
            )
        ]
    )
    issues = evaluate_anticoagulation(facts)
    assert len(issues) == 1
    assert issues[0].notes == plan_quote


# -------------------------
# Confidence and sample reflect real candidate counts
# -------------------------


def test_single_cbc_candidate_is_high_confidence_with_no_sample() -> None:
    facts = make_facts(
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 1, 1),
                    status=EntryStatus.VALID,
                    provenance=_provenance("only CBC"),
                )
            ]
        )
    )
    issues = evaluate_testing(facts)
    assert len(issues) == 1
    assert issues[0].confidence == "high"
    assert issues[0].sample is None


def test_multiple_cbc_candidates_is_medium_confidence_with_the_others_sampled() -> None:
    stale = LabFact(
        test_code="CBC",
        effective_at=datetime(2026, 1, 1),
        status=EntryStatus.VALID,
        provenance=_provenance("stale CBC quote"),
    )
    excluded = LabFact(
        test_code="CBC",
        effective_at=datetime(2026, 3, 14),
        status=EntryStatus.ENTERED_IN_ERROR,
        provenance=_provenance("entered-in-error CBC quote"),
    )
    facts = make_facts(labs=LabsFact(results=[stale, excluded]))
    issues = evaluate_testing(facts)
    assert len(issues) == 1
    assert issues[0].confidence == "medium"
    assert issues[0].sample == ["entered-in-error CBC quote"]


def test_multiple_bp_readings_sample_excludes_the_selected_one() -> None:
    older = _bp(119, 78, "older reading: 119/78", asserted_at=datetime(2026, 3, 1, 9, 0))
    latest = _bp(190, 95, "latest reading: 190/95", asserted_at=datetime(2026, 3, 15, 9, 0))
    facts = make_facts(vitals=VitalsFact(blood_pressure=[older, latest], temperature=[]))
    issues = evaluate_safety(facts)
    assert len(issues) == 1
    assert issues[0].notes == "latest reading: 190/95"
    assert issues[0].sample == ["older reading: 119/78"]
    assert issues[0].confidence == "medium"
