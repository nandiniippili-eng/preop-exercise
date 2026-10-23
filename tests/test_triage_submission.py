from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import policy_facts
import pytest
from core import PatientSubmission, triage_submission
from judgments import ActivityJudgmentResult, ConsentJudgmentResult, HPJudgmentResult, PlanJudgmentResult

READY_SUBMISSION: dict[str, object] = {
    "patient": {"id": "p1"},
    "procedure": {
        "case_id": "c1",
        "procedure_type": "total knee arthroplasty",
        "procedure_risk": "MODERATE",
        "procedure_date": "2026-03-01",
    },
    "vitals": [
        {"type": "blood_pressure", "systolic": 118, "diastolic": 74, "date": "2026-02-24T09:00:00Z"},
        {"type": "temperature", "value_f": 98.2, "date": "2026-02-24T09:00:00Z"},
    ],
    "labs": [
        {"id": "l1", "code": "CBC", "display": "CBC", "effective_at": "2026-02-20T08:00:00Z", "status": "final"},
    ],
    "medications": [],
    "conditions": [],
    "documents": [
        {
            "doc_id": "hp1",
            "type": "History and Physical",
            "date": "2026-02-19",
            "author": "Dr. X",
            "text": "Pre-op evaluation complete for planned procedure.",
        },
        {
            "doc_id": "consent1",
            "type": "Surgical Consent",
            "date": "2026-02-23",
            "author": "Dr. Y",
            "text": "Electronic consent obtained and signed by patient for procedure.",
        },
    ],
}


def ready_submission() -> dict[str, object]:
    return copy.deepcopy(READY_SUBMISSION)


@pytest.fixture(autouse=True)
def openai_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """triage_submission does `from openai import OpenAI` lazily; only the client's
    existence matters here since every judgment call is mocked at the policy_facts
    wiring level, never reaching client.responses.create.
    """

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: object()))


@pytest.fixture(autouse=True)
def default_judgment_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy H&P/consent for READY_SUBMISSION by default; individual tests override
    whichever mock they care about.
    """

    def fake_hp(client, *, model, request):
        return HPJudgmentResult(
            hp_doc_id="hp1",
            encounter_date="2026-02-19",
            evidence_quote="Pre-op evaluation complete for planned procedure.",
            reasoning="This is the pre-op H&P for the scheduled procedure.",
        )

    def fake_consent(client, *, model, request):
        return ConsentJudgmentResult(
            valid=True,
            doc_id="consent1",
            event_date="2026-02-23",
            evidence_quote="Electronic consent obtained and signed by patient for procedure.",
            reasoning="Signed consent for the correct procedure.",
        )

    def fail_if_called(client, *, model, request):
        raise AssertionError(
            "anticoagulation judgment call should not be made when no anticoagulant is present"
        )

    monkeypatch.setattr(policy_facts, "judge_hp_document", fake_hp)
    monkeypatch.setattr(policy_facts, "judge_consent", fake_consent)
    monkeypatch.setattr(policy_facts, "judge_medication_activity", fail_if_called)
    monkeypatch.setattr(policy_facts, "judge_anticoag_plan", fail_if_called)


def test_ready_submission_resolves_to_ready() -> None:
    output = triage_submission(ready_submission(), model="test-model")
    assert output.decision == "READY"
    assert output.issues == []


def test_accepts_already_validated_patient_submission() -> None:
    parsed = PatientSubmission.model_validate(ready_submission())
    output = triage_submission(parsed, model="test-model")
    assert output.decision == "READY"


def test_fever_vital_overrides_otherwise_ready_chart_to_not_cleared() -> None:
    submission = ready_submission()
    submission["vitals"] = [
        {"type": "blood_pressure", "systolic": 118, "diastolic": 74, "date": "2026-02-24T09:00:00Z"},
        {"type": "temperature", "value_f": 101.2, "date": "2026-02-24T09:00:00Z"},
    ]
    output = triage_submission(submission, model="test-model")
    assert output.decision == "NOT_CLEARED"
    assert any(i.category == "ACUTE_SAFETY_EXCLUSION" for i in output.issues)


def test_invalid_consent_judgment_yields_needs_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_consent(client, *, model, request):
        return ConsentJudgmentResult(
            valid=False,
            doc_id=None,
            event_date=None,
            evidence_quote="Signature line blank; printed for review.",
            reasoning="Not yet signed.",
        )

    monkeypatch.setattr(policy_facts, "judge_consent", fake_consent)

    output = triage_submission(ready_submission(), model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(i.category == "REQUIRED_DOCUMENTATION" for i in output.issues)


def test_antiplatelet_only_never_triggers_anticoagulation_judgment_calls() -> None:
    # The default_judgment_mocks fixture raises if judge_medication_activity or
    # judge_anticoag_plan is ever called; aspirin must never reach Tier 2 at all.
    submission = ready_submission()
    submission["medications"] = [{"name": "ASA 81 mg PO daily", "active": True}]

    output = triage_submission(submission, model="test-model")

    assert output.decision == "READY"
    assert output.issues == []


def test_active_anticoagulant_without_adequate_plan_needs_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = ready_submission()
    submission["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": True}]

    def fake_plan(client, *, model, request):
        assert request.generic_name == "rivaroxaban"
        return PlanJudgmentResult(
            plan_exists_for_this_drug=False,
            has_hold_instructions=False,
            has_resume_instructions=False,
            deferred_to_surgical_team=False,
            evidence_quote="No management plan found in the chart.",
            reasoning="No document discusses stopping or resuming rivaroxaban.",
        )

    monkeypatch.setattr(policy_facts, "judge_anticoag_plan", fake_plan)
    # active=True with no document mentions is a clean structured signal -> no
    # activity judgment call needed; judge_medication_activity should not be hit.

    output = triage_submission(submission, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(i.category == "ANTICOAGULATION_MANAGEMENT" for i in output.issues)


def test_active_anticoagulant_with_adequate_plan_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    submission = ready_submission()
    submission["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": True}]

    def fake_plan(client, *, model, request):
        return PlanJudgmentResult(
            plan_exists_for_this_drug=True,
            has_hold_instructions=True,
            has_resume_instructions=True,
            deferred_to_surgical_team=False,
            evidence_quote="Hold 2 days before, resume 24-48h after.",
            reasoning="Clear hold and resume instructions for rivaroxaban.",
        )

    monkeypatch.setattr(policy_facts, "judge_anticoag_plan", fake_plan)

    output = triage_submission(submission, model="test-model")

    assert output.decision == "READY"


def test_bridging_agent_only_skips_plan_judgment_call_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (case_00004): enoxaparin mentioned only as a perioperative
    # bridge for warfarin must never reach its own plan-adequacy call - there
    # is never a standalone "hold/resume enoxaparin" plan to find, so asking
    # would always (wrongly) come back inadequate. The activity judgment call
    # is what flags this via bridging_agent_only=True.
    submission = ready_submission()
    submission["documents"].append(
        {
            "doc_id": "consult1",
            "type": "Hematology Consultation",
            "date": "2026-02-24",
            "author": "Dr. Z",
            "text": "Perioperative enoxaparin bridging planned per protocol.",
        }
    )

    def fake_activity(client, *, model, request):
        assert request.generic_name == "enoxaparin"
        return ActivityJudgmentResult(
            active=True,
            bridging_agent_only=True,
            evidence_quote="enoxaparin bridging",
            reasoning="Only mentioned as the bridge therapy within the warfarin plan.",
        )

    def fail_if_plan_called(client, *, model, request):
        raise AssertionError(
            "plan-adequacy call should never be made for a bridging-only agent"
        )

    monkeypatch.setattr(policy_facts, "judge_medication_activity", fake_activity)
    monkeypatch.setattr(policy_facts, "judge_anticoag_plan", fail_if_plan_called)

    output = triage_submission(submission, model="test-model")

    assert output.decision == "READY"
    assert output.issues == []


def test_unconfirmed_medication_activity_yields_missing_required_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = ready_submission()
    # Blank `active` forces a Tier 2 activity judgment call.
    submission["medications"] = [{"name": "Coumadin 5 mg PO daily", "active": None}]

    def fake_activity(client, *, model, request):
        assert request.generic_name == "warfarin"
        return ActivityJudgmentResult(
            active=None,
            bridging_agent_only=False,
            evidence_quote="Patient could not confirm whether still taking it.",
            reasoning="Genuinely unconfirmable from the chart.",
        )

    monkeypatch.setattr(policy_facts, "judge_medication_activity", fake_activity)

    output = triage_submission(submission, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(i.category == "MISSING_REQUIRED_DATA" for i in output.issues)


def test_all_judgment_calls_exhausted_still_returns_valid_needs_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_facts, "judge_hp_document", lambda client, *, model, request: None)
    monkeypatch.setattr(policy_facts, "judge_consent", lambda client, *, model, request: None)

    output = triage_submission(ready_submission(), model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    categories = {i.category for i in output.issues}
    assert "MISSING_REQUIRED_DATA" in categories


def test_internal_error_never_raises_and_defaults_to_needs_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(submission, *, client, model):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr("policy_facts.extract_policy_facts", boom)

    output = triage_submission(ready_submission(), model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert output.issues[0].category == "MISSING_REQUIRED_DATA"
    assert "simulated internal failure" in output.issues[0].evidence


def test_malformed_submission_never_raises() -> None:
    output = triage_submission({"not": "a valid submission shape at all", "procedure": "oops"}, model="test-model")
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert output.issues[0].category == "MISSING_REQUIRED_DATA"
