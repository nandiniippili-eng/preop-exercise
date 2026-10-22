"""Shared schemas for the pre-op triage scripts, and the triage_submission entrypoint.

triage_submission is a deterministic rule engine (see rules.py) over facts
extracted from the submission (see facts.py for the closed-vocabulary /
structured-field extraction, and judgments.py + policy_facts.py for the
handful of scoped, schema-constrained LLM calls used only where free-text
reading comprehension is unavoidable - e.g. which document is the real
pre-op H&P). No single large LLM call makes the clearance decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

class TriageIssue(BaseModel):

    category: IssueCategory
    evidence: str

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str


def _fallback_output(evidence: str) -> TriageOutput:
    """The policy's own default: unknown/unevaluable -> NEEDS_FOLLOW_UP, never a crash."""

    return TriageOutput(
        decision="NEEDS_FOLLOW_UP",
        issues=[TriageIssue(category="MISSING_REQUIRED_DATA", evidence=evidence)],
        explanation=(
            "Automated triage could not fully evaluate this submission; defaulting to "
            "NEEDS_FOLLOW_UP per policy for missing/unknown required data. "
            f"Detail: {evidence}"
        ),
    )


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Deterministic rule-engine triage, with scoped LLM judgment calls for the handful of
    facts that require reading free text (see policy_facts.py / judgments.py). Always
    returns a valid TriageOutput: any unexpected failure degrades to the policy's own
    "missing/unknown required data" default rather than raising.
    """

    # Imported lazily to avoid a core <-> policy_facts import cycle (policy_facts imports
    # the schemas defined in this module) and to keep core.py importable without OpenAI
    # installed when only the schemas are needed.
    from policy_facts import extract_policy_facts
    from rules import aggregate_decision, build_explanation, evaluate_all

    try:
        if isinstance(submission, PatientSubmission):
            parsed = submission
        else:
            parsed = PatientSubmission.model_validate(submission)
    except Exception as exc:  # pragma: no cover - defensive, submission is pre-validated upstream
        return _fallback_output(f"Submission failed schema validation: {exc}")

    try:
        from openai import OpenAI

        client = OpenAI()
        facts = extract_policy_facts(parsed, client=client, model=model)
        issues = evaluate_all(facts)
        decision = aggregate_decision(issues)
        explanation = build_explanation(decision, issues)
        return TriageOutput(decision=decision, issues=issues, explanation=explanation)
    except Exception as exc:  # pragma: no cover - last-resort safety net
        return _fallback_output(f"Internal error while evaluating submission: {exc}")
