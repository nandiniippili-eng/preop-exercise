from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import BaseModel

from judgments import (
    MAX_ATTEMPTS,
    ActivityJudgmentRequest,
    ConsentJudgmentRequest,
    HPDocCandidate,
    HPJudgmentRequest,
    HPJudgmentResult,
    PlanJudgmentDocument,
    PlanJudgmentRequest,
    _call_judgment,
    judge_anticoag_plan,
    judge_consent,
    judge_hp_document,
    judge_medication_activity,
)


class _DummyResult(BaseModel):
    value: str


def _response(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(payload))


# -------------------------
# Generic retry wrapper
# -------------------------


def test_call_judgment_succeeds_on_first_attempt() -> None:
    client = Mock()
    client.responses.create.return_value = _response({"value": "ok"})

    result = _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={"a": 1},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    assert result == _DummyResult(value="ok")
    assert client.responses.create.call_count == 1


def test_call_judgment_uses_temperature_zero_and_schema_constrained_format() -> None:
    client = Mock()
    client.responses.create.return_value = _response({"value": "ok"})

    _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={"a": 1},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "test-model"
    assert kwargs["instructions"] == "sys"
    fmt = kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "dummy"
    assert fmt["schema"] == _DummyResult.model_json_schema()


def test_call_judgment_retries_once_after_network_error_then_succeeds() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        ConnectionError("network blip"),
        _response({"value": "ok"}),
    ]

    result = _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    assert result == _DummyResult(value="ok")
    assert client.responses.create.call_count == 2


def test_call_judgment_returns_none_after_two_network_errors() -> None:
    client = Mock()
    client.responses.create.side_effect = [TimeoutError("slow"), ConnectionError("still down")]

    result = _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    assert result is None
    assert client.responses.create.call_count == MAX_ATTEMPTS == 2


def test_call_judgment_returns_none_after_two_schema_validation_failures() -> None:
    client = Mock()
    # Missing the required "value" field both times -> ValidationError both times.
    client.responses.create.return_value = _response({"unexpected": "shape"})

    result = _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    assert result is None
    assert client.responses.create.call_count == 2


def test_call_judgment_recovers_after_one_schema_validation_failure() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        _response({"unexpected": "shape"}),
        _response({"value": "ok"}),
    ]

    result = _call_judgment(
        client,
        model="test-model",
        system_prompt="sys",
        user_payload={},
        result_model=_DummyResult,
        schema_name="dummy",
    )

    assert result == _DummyResult(value="ok")
    assert client.responses.create.call_count == 2


# -------------------------
# Public judgment call wiring
# -------------------------


def test_judge_hp_document_returns_typed_result() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        {
            "hp_doc_id": "doc-1",
            "encounter_date": "2026-02-21",
            "evidence_quote": "PREOPERATIVE HISTORY AND PHYSICAL ...",
            "reasoning": "This is the pre-op workup for the scheduled procedure.",
        }
    )
    request = HPJudgmentRequest(
        procedure_type="inguinal hernia repair",
        procedure_date="2026-03-17",
        documents=[HPDocCandidate(doc_id="doc-1", raw_type="H&P [PDF]", date="2026-02-21", text="...")],
    )

    result = judge_hp_document(client, model="test-model", request=request)

    assert isinstance(result, HPJudgmentResult)
    assert result.hp_doc_id == "doc-1"
    assert result.encounter_date == "2026-02-21"


def test_judge_hp_document_returns_none_when_calls_exhausted() -> None:
    client = Mock()
    client.responses.create.side_effect = ConnectionError("down")
    request = HPJudgmentRequest(documents=[])

    result = judge_hp_document(client, model="test-model", request=request)

    assert result is None
    assert client.responses.create.call_count == 2


def test_judge_consent_returns_typed_result() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        {
            "valid": False,
            "doc_id": "doc-2",
            "event_date": None,
            "evidence_quote": "Patient signature: ______ [printed for review; not yet signed]",
            "reasoning": "No completed signature on file.",
        }
    )
    request = ConsentJudgmentRequest(scheduled_procedure="thyroidectomy", documents=[])

    result = judge_consent(client, model="test-model", request=request)

    assert result is not None
    assert result.valid is False
    assert result.doc_id == "doc-2"


def test_judge_medication_activity_returns_typed_result() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        {
            "active": None,
            "evidence_quote": "Pharmacy fill history was requested; patient unsure.",
            "reasoning": "Active status is explicitly unconfirmed in the chart.",
        }
    )
    request = ActivityJudgmentRequest(
        generic_name="warfarin",
        structured_raw_name="Coumadin 5 mg PO daily",
        structured_active=None,
        structured_active_present=True,
        document_mentions=["couldn't confirm whether she's still taking it"],
    )

    result = judge_medication_activity(client, model="test-model", request=request)

    assert result is not None
    assert result.active is None


def test_judge_anticoag_plan_returns_typed_result_and_computes_adequacy() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        {
            "plan_exists_for_this_drug": True,
            "has_hold_instructions": True,
            "has_resume_instructions": False,
            "deferred_to_surgical_team": False,
            "evidence_quote": "stop warfarin 5 days out ... will be addressed after surgery",
            "reasoning": "Hold instructions present; resume side left unaddressed.",
        }
    )
    request = PlanJudgmentRequest(
        generic_name="warfarin",
        documents=[PlanJudgmentDocument(doc_id="d1", date="2026-03-15", text="...")],
    )

    result = judge_anticoag_plan(client, model="test-model", request=request)

    assert result is not None
    assert result.is_adequate is False  # missing resume instructions
