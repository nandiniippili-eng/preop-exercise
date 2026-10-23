"""Assembly layer: Tier 1 extraction + conditional Tier 2 judgment calls -> PolicyFacts.

This is the only module that orchestrates LLM calls. It decides, per the
rules spelled out in facts.py/judgments.py, which of the four judgment calls
are actually needed for a given submission, and resolves each fact to either
a concrete value or an explicit "unresolved" state when a call exhausts its
retries - rules.py then maps "unresolved" to MISSING_REQUIRED_DATA, so
triage_submission never has to special-case a failed call.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from core import PatientSubmission
from facts import (
    DrugClass,
    DrugMentionGroup,
    LabsFact,
    VitalsFact,
    extract_labs,
    extract_vitals,
    parse_dt,
    procedure_date,
    procedure_risk,
    scan_medication_mentions,
)
from judgments import (
    ActivityJudgmentRequest,
    ConsentDocCandidate,
    ConsentJudgmentRequest,
    HPDocCandidate,
    HPJudgmentRequest,
    PlanJudgmentDocument,
    PlanJudgmentRequest,
    judge_anticoag_plan,
    judge_consent,
    judge_hp_document,
    judge_medication_activity,
)


Confidence = Literal["high", "medium", "low"]


def confidence_from_candidate_count(count: int) -> Confidence:
    """high: 0 or 1 candidate existed (no ambiguity). medium: 2+ competed.

    "low" is never returned here - it's reserved for when a resolution
    genuinely failed/was exhausted, which is a different axis from how many
    candidates existed, and is set directly at those call sites.
    """

    return "high" if count <= 1 else "medium"


class ResolvedHP(BaseModel):
    doc_id: str | None
    doc_type: str | None = None
    encounter_date: date | None
    evidence_quote: str
    rejected_candidates: list[str] = Field(default_factory=list)
    confidence: Confidence = "high"
    unresolved: bool = False


class ResolvedConsent(BaseModel):
    valid: bool
    doc_id: str | None
    doc_type: str | None = None
    event_date: date | None
    evidence_quote: str
    rejected_candidates: list[str] = Field(default_factory=list)
    confidence: Confidence = "high"
    unresolved: bool = False


class ResolvedAnticoagulant(BaseModel):
    generic_name: str
    active: bool | None
    activity_evidence: str
    activity_unresolved: bool = False
    activity_confidence: Confidence = "high"
    activity_source_kind: Literal["structured_field", "llm_judgment"] = "llm_judgment"
    bridging_agent_only: bool = False
    plan_checked: bool = False
    plan_adequate: bool | None = None
    plan_evidence: str | None = None
    plan_unresolved: bool = False
    plan_confidence: Confidence | None = None


class PolicyFacts(BaseModel):
    procedure_date: date | None
    procedure_risk: str | None
    labs: LabsFact
    vitals: VitalsFact
    hp: ResolvedHP
    consent: ResolvedConsent
    anticoagulants: list[ResolvedAnticoagulant]


def _resolve_hp(submission: PatientSubmission, *, client: Any, model: str) -> ResolvedHP:
    docs = submission.documents
    if not docs:
        return ResolvedHP(
            doc_id=None, encounter_date=None, evidence_quote="No documents on file.",
            confidence="high",
        )

    request = HPJudgmentRequest(
        procedure_type=submission.procedure.procedure_type if submission.procedure else None,
        procedure_date=submission.procedure.procedure_date if submission.procedure else None,
        documents=[
            HPDocCandidate(doc_id=d.doc_id or "", raw_type=d.type, date=d.date, text=d.text or "")
            for d in docs
        ],
    )
    result = judge_hp_document(client, model=model, request=request)
    if result is None:
        return ResolvedHP(
            doc_id=None,
            encounter_date=None,
            evidence_quote="H&P identity judgment call failed after retries.",
            unresolved=True,
            confidence="low",
        )

    enc_dt = parse_dt(result.encounter_date)
    doc_type = next((d.type for d in docs if d.doc_id == result.hp_doc_id), None)
    candidate_count = (1 if result.hp_doc_id else 0) + len(result.rejected_candidates)
    return ResolvedHP(
        doc_id=result.hp_doc_id,
        doc_type=doc_type,
        encounter_date=enc_dt.date() if enc_dt else None,
        evidence_quote=result.evidence_quote,
        rejected_candidates=result.rejected_candidates,
        confidence=confidence_from_candidate_count(candidate_count),
    )


def _resolve_consent(submission: PatientSubmission, *, client: Any, model: str) -> ResolvedConsent:
    docs = submission.documents
    if not docs:
        return ResolvedConsent(
            valid=False, doc_id=None, event_date=None, evidence_quote="No documents on file.",
            confidence="high",
        )

    request = ConsentJudgmentRequest(
        scheduled_procedure=submission.procedure.procedure_type if submission.procedure else None,
        documents=[
            ConsentDocCandidate(doc_id=d.doc_id or "", date=d.date, text=d.text or "") for d in docs
        ],
    )
    result = judge_consent(client, model=model, request=request)
    if result is None:
        return ResolvedConsent(
            valid=False,
            doc_id=None,
            event_date=None,
            evidence_quote="Consent validity judgment call failed after retries.",
            unresolved=True,
            confidence="low",
        )

    event_dt = parse_dt(result.event_date)
    doc_type = next((d.type for d in docs if d.doc_id == result.doc_id), None)
    candidate_count = (1 if result.doc_id else 0) + len(result.rejected_candidates)
    return ResolvedConsent(
        valid=result.valid,
        doc_id=result.doc_id,
        doc_type=doc_type,
        event_date=event_dt.date() if event_dt else None,
        evidence_quote=result.evidence_quote,
        rejected_candidates=result.rejected_candidates,
        confidence=confidence_from_candidate_count(candidate_count),
    )


def _resolve_anticoagulant(
    group: DrugMentionGroup, submission: PatientSubmission, *, client: Any, model: str
) -> ResolvedAnticoagulant:
    if not group.needs_activity_judgment():
        # Exactly one clean signal by construction (needs_activity_judgment()
        # only returns False when there's a structured entry and no document
        # mentions at all) - always a single candidate, so always "high".
        structured = group.structured_mention
        active = structured.active if structured else None
        evidence = structured.provenance.quote if structured else ""
        resolved = ResolvedAnticoagulant(
            generic_name=group.generic_name,
            active=active,
            activity_evidence=evidence,
            activity_confidence="high",
            activity_source_kind="structured_field",
        )
    else:
        structured = group.structured_mention
        request = ActivityJudgmentRequest(
            generic_name=group.generic_name,
            structured_raw_name=structured.provenance.quote if structured else None,
            structured_active=structured.active if structured else None,
            structured_active_present=structured is not None,
            document_mentions=[m.provenance.quote for m in group.document_mentions],
        )
        result = judge_medication_activity(client, model=model, request=request)
        if result is None:
            return ResolvedAnticoagulant(
                generic_name=group.generic_name,
                active=None,
                activity_evidence="Activity judgment call failed after retries.",
                activity_unresolved=True,
                activity_confidence="low",
            )
        candidate_count = (1 if structured else 0) + len(group.document_mentions)
        resolved = ResolvedAnticoagulant(
            generic_name=group.generic_name,
            active=result.active,
            activity_evidence=result.evidence_quote,
            bridging_agent_only=result.bridging_agent_only,
            activity_confidence=confidence_from_candidate_count(candidate_count),
        )

    if resolved.active is True and not resolved.bridging_agent_only:
        resolved.plan_checked = True
        plan_request = PlanJudgmentRequest(
            generic_name=group.generic_name,
            documents=[
                PlanJudgmentDocument(doc_id=d.doc_id or "", date=d.date, text=d.text or "")
                for d in submission.documents
            ],
        )
        plan_result = judge_anticoag_plan(client, model=model, request=plan_request)
        if plan_result is None:
            resolved.plan_unresolved = True
            resolved.plan_confidence = "low"
        else:
            resolved.plan_adequate = plan_result.is_adequate
            resolved.plan_evidence = plan_result.evidence_quote
            # No separate plan-candidate count is tracked (plan judgment reads
            # all documents, not a discrete candidate list), so this reuses
            # the drug's own activity-resolution confidence as the best
            # available measure of how much ambiguity surrounded this drug.
            resolved.plan_confidence = resolved.activity_confidence

    return resolved


def extract_policy_facts(
    submission: PatientSubmission, *, client: Any, model: str
) -> PolicyFacts:
    labs = extract_labs(submission)
    vitals = extract_vitals(submission)

    hp = _resolve_hp(submission, client=client, model=model)
    consent = _resolve_consent(submission, client=client, model=model)

    anticoagulants = [
        _resolve_anticoagulant(group, submission, client=client, model=model)
        for group in scan_medication_mentions(submission)
        if group.drug_class == DrugClass.ANTICOAGULANT
    ]

    return PolicyFacts(
        procedure_date=procedure_date(submission),
        procedure_risk=procedure_risk(submission),
        labs=labs,
        vitals=vitals,
        hp=hp,
        consent=consent,
        anticoagulants=anticoagulants,
    )
