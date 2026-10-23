"""Tier 2: scoped, schema-constrained LLM judgment calls.

Each function here answers exactly one narrow semantic question that Tier 1
(facts.py) cannot resolve deterministically, because it requires reading
comprehension over open-ended free text rather than closed-vocabulary
matching: which document is the real pre-op H&P, whether a consent event is
valid, whether a flagged medication is currently active, and whether an
anticoagulation management plan is adequate.

Every call is schema-constrained (structured JSON output) and made with
temperature=0. Each call gets at most two attempts total (one retry) on
*any* failure - a network/timeout error or a schema-validation failure of
the model's output. If both attempts fail, the function returns None rather
than raising, so the caller can resolve the corresponding fact to "unknown"
and let the policy default (missing required data -> NEEDS_FOLLOW_UP) apply.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

MAX_ATTEMPTS = 2

# Shared quote-discipline instruction injected into every judgment prompt below.
# A pre-op nurse reviews every non-READY case against evidence_quote directly, so
# it has to be literally findable in the record, not a model-composed summary -
# this is what keeps the evidence packet human-verifiable rather than just
# plausible-sounding.
QUOTE_DISCIPLINE = """
evidence_quote must be copied character-for-character from ONE contiguous span
of a single source document - never combine or stitch together text from
different parts of a document or from different documents, and never insert
an ellipsis ("...") to splice separate passages together. Keep it to the
shortest single passage (typically one sentence) that actually supports your
answer. If the supporting fact is spread across more than one place in the
record, pick the single most decisive sentence for evidence_quote and put any
additional context in `reasoning` instead - do not merge multiple passages
into the quote.
If there is nothing in the record to quote (e.g. no plan exists at all, no
document qualifies), say so plainly in evidence_quote (e.g. "No such plan
found in the chart.") rather than inventing or stitching together an
unrelated quote.
""".strip()


# -------------------------
# 1. H&P document identity
# -------------------------

HP_JUDGMENT_SYSTEM_PROMPT = f"""
You resolve which single document, if any, is the patient's valid pre-operative
History & Physical (H&P) for the scheduled procedure.

A valid pre-op H&P is a pre-operative evaluation written in connection with this
upcoming procedure. It is NOT:
- a stale or unrelated encounter (e.g. an annual wellness visit, an old
  establish-care visit) even if its title contains "History and Physical"
- a document that itself says it is retained only for context, superseded by a
  newer evaluation, or otherwise not the operative pre-op workup

Document `type` fields may be generic, misspelled, or otherwise unreliable -
judge by what the document's content actually is, not by its title alone.
When more than one document could qualify, pick the one that is actually the
pre-operative evaluation for this procedure.

If no document in the provided list qualifies, return hp_doc_id as null.
{QUOTE_DISCIPLINE}
Give brief reasoning as well.

Also list every OTHER document you genuinely weighed as a plausible candidate
before ruling it out (not every document in the list - only ones a reasonable
reviewer might have mistaken for the real pre-op H&P), each as a short
"<doc_id>: <why it was rejected>" string. Leave this empty if there was no
real ambiguity (e.g. only one document could plausibly be the H&P at all).

Respond with JSON matching the given schema only.
""".strip()


class HPDocCandidate(BaseModel):
    doc_id: str
    raw_type: str | None = None
    date: str | None = None
    text: str


class HPJudgmentRequest(BaseModel):
    procedure_type: str | None = None
    procedure_date: str | None = None
    documents: list[HPDocCandidate]


class HPJudgmentResult(BaseModel):
    hp_doc_id: str | None
    encounter_date: str | None
    evidence_quote: str
    reasoning: str
    rejected_candidates: list[str] = Field(default_factory=list)


# -------------------------
# 2. Consent validity
# -------------------------

CONSENT_JUDGMENT_SYSTEM_PROMPT = f"""
You determine whether there is a currently valid, patient-signed surgical
consent for the scheduled procedure.

A valid consent requires BOTH:
1. The patient's signature is actually completed - not pending, blank, printed
   for review, or shown as "not completed" in a portal.
2. The consent document is written for the SAME procedure that is scheduled
   (a signed consent for a different procedure does not count).

If multiple consent-related documents exist, the most recent VALID signed
event governs, even if an earlier attempt was declined, left pending, or taken
home for review and only later superseded by a signed one.

If no document qualifies, return valid=false and doc_id=null.
{QUOTE_DISCIPLINE}
Give brief reasoning as well.

Also list every OTHER consent-related document you genuinely weighed before
ruling it out (an earlier pending/declined attempt, a consent for the wrong
procedure, etc.) - not every document in the list, only real candidates -
each as a short "<doc_id>: <why it was rejected>" string. Leave this empty if
there was no real ambiguity (e.g. only one consent event exists at all).

Respond with JSON matching the given schema only.
""".strip()


class ConsentDocCandidate(BaseModel):
    doc_id: str
    date: str | None = None
    text: str


class ConsentJudgmentRequest(BaseModel):
    scheduled_procedure: str | None = None
    documents: list[ConsentDocCandidate]


class ConsentJudgmentResult(BaseModel):
    valid: bool
    doc_id: str | None
    event_date: str | None
    evidence_quote: str
    reasoning: str
    rejected_candidates: list[str] = Field(default_factory=list)


# -------------------------
# 3a. Medication activity resolution
# -------------------------

ACTIVITY_JUDGMENT_SYSTEM_PROMPT = f"""
You determine whether a specific medication is CURRENTLY being taken by the
patient, using a structured medication-list entry (if any) and any mentions of
it found in the clinical documents.

Resolve conflicting or stale signals using the most clinically specific and
recent information available. For example: a note stating the drug was
discontinued or that the patient was switched to a different medication
supersedes an older or more generic "active" flag; a patient self-report of
currently taking a drug counts even if it is absent from the reconciled
medication list.

If the chart genuinely does not allow you to determine current status (for
example, conflicting information with no way to tell which is current, or an
explicit statement that status is unconfirmed), return active as null - do NOT
guess.

Separately, decide whether this drug is only a perioperative BRIDGING AGENT
named inside another anticoagulant's management plan, rather than the
patient's own independently-prescribed home medication. For example, "stop
warfarin 5 days before surgery with enoxaparin bridging, resume warfarin the
evening of surgery" mentions enoxaparin only as the bridge therapy used while
warfarin is held - the patient is not independently prescribed enoxaparin as
a standing home medication, and it does not need its own separate
hold/resume plan the way warfarin does. Set bridging_agent_only=true in that
case. If the drug is the patient's own standing anticoagulant (the one being
held and resumed, not the bridge used while holding something else), set
bridging_agent_only=false, even if a bridging agent is also mentioned
alongside it.

{QUOTE_DISCIPLINE}
Give brief reasoning as well.
Respond with JSON matching the given schema only.
""".strip()


class ActivityJudgmentRequest(BaseModel):
    generic_name: str
    structured_raw_name: str | None = None
    structured_active: bool | None = None
    structured_active_present: bool
    document_mentions: list[str]


class ActivityJudgmentResult(BaseModel):
    active: bool | None
    bridging_agent_only: bool
    evidence_quote: str
    reasoning: str


# -------------------------
# 3b. Anticoagulation plan adequacy
# -------------------------

PLAN_JUDGMENT_SYSTEM_PROMPT = f"""
You evaluate whether the chart contains an adequate documented perioperative
management plan for one specific anticoagulant the patient is currently
taking.

An adequate plan must satisfy ALL of the following:
(a) It is written for THIS specific drug. A plan written for a different
    anticoagulant (e.g. a plan for warfarin after the patient was switched to
    apixaban) does NOT count, even if it is the only plan on file.
(b) It gives clear instructions for holding/stopping the drug before surgery.
(c) It gives clear instructions for resuming the drug after surgery.

A plan that defers the decision to the surgical team WITHOUT giving any
concrete timeframe or criterion (e.g. "will be addressed after surgery", "up
to the surgical team to decide", "recommendations will follow", "to be
determined") is NOT adequate for whichever half it defers.

Do NOT treat a plan as deferred merely because it attributes or credits an
instruction to the surgical team or a protocol, as long as a concrete
timeframe or criterion is actually stated. For example: "resume 24-48 hours
after surgery once hemostasis is secure, per surgical team" IS a complete,
adequate resume instruction - "per surgical team" here only names who
carries it out, it does not erase the concrete 24-48 hour timeframe already
given. Only set deferred_to_surgical_team=true when the instruction itself
is genuinely open-ended, with no timeframe or criterion stated anywhere.

If there is no plan for this drug at all, set plan_exists_for_this_drug to
false and the other boolean fields to false.

{QUOTE_DISCIPLINE}
Give brief reasoning as well.
Respond with JSON matching the given schema only.
""".strip()


class PlanJudgmentDocument(BaseModel):
    doc_id: str
    date: str | None = None
    text: str


class PlanJudgmentRequest(BaseModel):
    generic_name: str
    documents: list[PlanJudgmentDocument]


class PlanJudgmentResult(BaseModel):
    plan_exists_for_this_drug: bool
    has_hold_instructions: bool
    has_resume_instructions: bool
    deferred_to_surgical_team: bool
    evidence_quote: str
    reasoning: str

    @property
    def is_adequate(self) -> bool:
        return (
            self.plan_exists_for_this_drug
            and self.has_hold_instructions
            and self.has_resume_instructions
            and not self.deferred_to_surgical_team
        )


# -------------------------
# Generic schema-constrained call wrapper with bounded retry
# -------------------------


def _network_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import openai
    except ImportError:
        return tuple(types)
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "APIError",
    ):
        cls = getattr(openai, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    return tuple(types)


def _call_judgment(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, object],
    result_model: type[BaseModel],
    schema_name: str,
) -> BaseModel | None:
    """Make a schema-constrained, temperature=0 call with bounded retry.

    Returns the validated result, or None if every attempt (up to
    MAX_ATTEMPTS) failed, whether from a network/timeout error or the
    model's output failing schema validation.
    """

    network_errors = _network_error_types()

    for _attempt in range(MAX_ATTEMPTS):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(user_payload, sort_keys=True, default=str),
                            }
                        ],
                    }
                ],
                temperature=0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": result_model.model_json_schema(),
                        "strict": False,
                    }
                },
            )
            return result_model.model_validate_json(response.output_text)
        except network_errors:
            continue
        except ValidationError:
            continue

    return None


def judge_hp_document(
    client: Any, *, model: str, request: HPJudgmentRequest
) -> HPJudgmentResult | None:
    result = _call_judgment(
        client,
        model=model,
        system_prompt=HP_JUDGMENT_SYSTEM_PROMPT,
        user_payload=request.model_dump(),
        result_model=HPJudgmentResult,
        schema_name="hp_judgment",
    )
    assert result is None or isinstance(result, HPJudgmentResult)
    return result


def judge_consent(
    client: Any, *, model: str, request: ConsentJudgmentRequest
) -> ConsentJudgmentResult | None:
    result = _call_judgment(
        client,
        model=model,
        system_prompt=CONSENT_JUDGMENT_SYSTEM_PROMPT,
        user_payload=request.model_dump(),
        result_model=ConsentJudgmentResult,
        schema_name="consent_judgment",
    )
    assert result is None or isinstance(result, ConsentJudgmentResult)
    return result


def judge_medication_activity(
    client: Any, *, model: str, request: ActivityJudgmentRequest
) -> ActivityJudgmentResult | None:
    result = _call_judgment(
        client,
        model=model,
        system_prompt=ACTIVITY_JUDGMENT_SYSTEM_PROMPT,
        user_payload=request.model_dump(),
        result_model=ActivityJudgmentResult,
        schema_name="activity_judgment",
    )
    assert result is None or isinstance(result, ActivityJudgmentResult)
    return result


def judge_anticoag_plan(
    client: Any, *, model: str, request: PlanJudgmentRequest
) -> PlanJudgmentResult | None:
    result = _call_judgment(
        client,
        model=model,
        system_prompt=PLAN_JUDGMENT_SYSTEM_PROMPT,
        user_payload=request.model_dump(),
        result_model=PlanJudgmentResult,
        schema_name="plan_judgment",
    )
    assert result is None or isinstance(result, PlanJudgmentResult)
    return result
