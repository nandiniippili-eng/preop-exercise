from __future__ import annotations

from datetime import date, datetime

from facts import EntryStatus, LabFact, LabsFact, Provenance, VitalReading, VitalsFact
from policy_facts import PolicyFacts, ResolvedAnticoagulant, ResolvedConsent, ResolvedHP
from rules import aggregate_decision, evaluate_all

PROC_DATE = date(2026, 3, 15)


def _provenance(quote: str = "evidence", asserted_at: datetime | None = None) -> Provenance:
    return Provenance(source="document_text", doc_id="d1", quote=quote, asserted_at=asserted_at)


def _bp(systolic: float, diastolic: float, *, days_before: int = 5) -> VitalReading:
    return VitalReading(
        kind="blood_pressure",
        systolic=systolic,
        diastolic=diastolic,
        status=EntryStatus.VALID,
        provenance=_provenance(asserted_at=datetime.combine(PROC_DATE, datetime.min.time())),
    )


def _temp(value_f: float) -> VitalReading:
    return VitalReading(
        kind="temperature",
        temp_f=value_f,
        status=EntryStatus.VALID,
        provenance=_provenance(asserted_at=datetime.combine(PROC_DATE, datetime.min.time())),
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
        vitals=VitalsFact(blood_pressure=[_bp(120, 76)], temperature=[_temp(98.6)]),
        hp=ResolvedHP(doc_id="hp1", encounter_date=date(2026, 3, 1), evidence_quote="H&P on file"),
        consent=ResolvedConsent(valid=True, doc_id="c1", event_date=date(2026, 3, 10), evidence_quote="signed"),
        anticoagulants=[],
    )
    defaults.update(overrides)
    return PolicyFacts(**defaults)


def categories(facts: PolicyFacts) -> set[str]:
    return {issue.category for issue in evaluate_all(facts)}


# -------------------------
# Baseline / decision aggregation
# -------------------------


def test_all_criteria_satisfied_is_ready() -> None:
    facts = make_facts()
    issues = evaluate_all(facts)
    assert issues == []
    assert aggregate_decision(issues) == "READY"


def test_aggregate_decision_ready_when_no_issues() -> None:
    assert aggregate_decision([]) == "READY"


def test_aggregate_decision_needs_follow_up_for_non_exclusion_issues() -> None:
    facts = make_facts(consent=ResolvedConsent(valid=False, doc_id=None, event_date=None, evidence_quote="pending"))
    issues = evaluate_all(facts)
    assert aggregate_decision(issues) == "NEEDS_FOLLOW_UP"


def test_aggregate_decision_not_cleared_wins_over_other_issues() -> None:
    facts = make_facts(
        consent=ResolvedConsent(valid=False, doc_id=None, event_date=None, evidence_quote="pending"),
        vitals=VitalsFact(blood_pressure=[_bp(190, 90)], temperature=[_temp(98.6)]),
    )
    issues = evaluate_all(facts)
    cats = {i.category for i in issues}
    assert "REQUIRED_DOCUMENTATION" in cats
    assert "ACUTE_SAFETY_EXCLUSION" in cats
    assert aggregate_decision(issues) == "NOT_CLEARED"


# -------------------------
# Rule 1: documentation
# -------------------------


def test_missing_hp_is_required_documentation() -> None:
    facts = make_facts(hp=ResolvedHP(doc_id=None, encounter_date=None, evidence_quote="no H&P"))
    assert categories(facts) == {"REQUIRED_DOCUMENTATION"}


def test_hp_outside_30_day_window_is_required_documentation() -> None:
    facts = make_facts(hp=ResolvedHP(doc_id="hp1", encounter_date=date(2026, 2, 1), evidence_quote="31 days out"))
    assert categories(facts) == {"REQUIRED_DOCUMENTATION"}


def test_hp_exactly_30_days_out_passes() -> None:
    facts = make_facts(hp=ResolvedHP(doc_id="hp1", encounter_date=date(2026, 2, 13), evidence_quote="30 days out"))
    assert categories(facts) == set()


def test_hp_judgment_unresolved_is_missing_required_data_not_documentation() -> None:
    facts = make_facts(hp=ResolvedHP(doc_id=None, encounter_date=None, evidence_quote="call failed", unresolved=True))
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


def test_consent_invalid_is_required_documentation() -> None:
    facts = make_facts(consent=ResolvedConsent(valid=False, doc_id=None, event_date=None, evidence_quote="not signed"))
    assert categories(facts) == {"REQUIRED_DOCUMENTATION"}


def test_consent_unresolved_is_missing_required_data() -> None:
    facts = make_facts(
        consent=ResolvedConsent(valid=False, doc_id=None, event_date=None, evidence_quote="call failed", unresolved=True)
    )
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


# -------------------------
# Missing prerequisites (procedure_date / procedure_risk / vitals)
# -------------------------


def test_missing_procedure_date_skips_date_window_rules_and_reports_once() -> None:
    facts = make_facts(procedure_date=None)
    issues = evaluate_all(facts)
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}
    assert len([i for i in issues if i.category == "MISSING_REQUIRED_DATA"]) == 1


def test_missing_procedure_risk_skips_testing_rule_and_reports_once() -> None:
    facts = make_facts(procedure_risk=None)
    issues = evaluate_all(facts)
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}
    assert len([i for i in issues if i.category == "MISSING_REQUIRED_DATA"]) == 1


def test_missing_blood_pressure_is_missing_required_data() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[], temperature=[_temp(98.6)]))
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


def test_missing_temperature_is_missing_required_data() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(120, 76)], temperature=[]))
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


# -------------------------
# Rule 2: testing
# -------------------------


def test_missing_cbc_for_low_risk_is_required_testing() -> None:
    facts = make_facts(labs=LabsFact(results=[]))
    assert categories(facts) == {"REQUIRED_TESTING"}


def test_cbc_outside_30_day_window_for_low_risk_is_required_testing() -> None:
    facts = make_facts(
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 1, 1),
                    status=EntryStatus.VALID,
                    provenance=_provenance("stale CBC"),
                )
            ]
        )
    )
    assert categories(facts) == {"REQUIRED_TESTING"}


def test_high_risk_requires_both_cbc_and_cmp_independently() -> None:
    facts = make_facts(
        procedure_risk="HIGH",
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 3, 10),
                    status=EntryStatus.VALID,
                    provenance=_provenance("CBC ok"),
                )
            ]
        ),
    )
    issues = evaluate_all(facts)
    testing_issues = [i for i in issues if i.category == "REQUIRED_TESTING"]
    assert len(testing_issues) == 1
    assert "CMP" in testing_issues[0].evidence


def test_entered_in_error_is_excluded_from_testing_evaluation() -> None:
    facts = make_facts(
        labs=LabsFact(
            results=[
                LabFact(
                    test_code="CBC",
                    effective_at=datetime(2026, 3, 14),
                    status=EntryStatus.ENTERED_IN_ERROR,
                    provenance=_provenance("bad entry"),
                )
            ]
        )
    )
    assert categories(facts) == {"REQUIRED_TESTING"}


# -------------------------
# Rule 4: acute safety exclusions
# -------------------------


def test_systolic_at_threshold_excludes() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(180, 80)], temperature=[_temp(98.6)]))
    assert categories(facts) == {"ACUTE_SAFETY_EXCLUSION"}


def test_diastolic_at_threshold_excludes() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(140, 110)], temperature=[_temp(98.6)]))
    assert categories(facts) == {"ACUTE_SAFETY_EXCLUSION"}


def test_temperature_above_threshold_excludes() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(120, 76)], temperature=[_temp(100.5)]))
    assert categories(facts) == {"ACUTE_SAFETY_EXCLUSION"}


def test_temperature_at_threshold_does_not_exclude() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(120, 76)], temperature=[_temp(100.4)]))
    assert categories(facts) == set()


def test_normal_vitals_do_not_exclude() -> None:
    facts = make_facts(vitals=VitalsFact(blood_pressure=[_bp(179, 109)], temperature=[_temp(100.3)]))
    assert categories(facts) == set()


# -------------------------
# Rule 3: anticoagulation
# -------------------------


def test_inactive_anticoagulant_needs_no_plan() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(generic_name="warfarin", active=False, activity_evidence="stopped in 2020")
        ]
    )
    assert categories(facts) == set()


def test_active_anticoagulant_with_adequate_plan_has_no_issue() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="rivaroxaban",
                active=True,
                activity_evidence="active per consult",
                plan_checked=True,
                plan_adequate=True,
                plan_evidence="hold 2 days, resume 24-48h post-op",
            )
        ]
    )
    assert categories(facts) == set()


def test_active_anticoagulant_without_adequate_plan_is_anticoagulation_management() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="apixaban",
                active=True,
                activity_evidence="switched last month",
                plan_checked=True,
                plan_adequate=False,
                plan_evidence="only a warfarin plan exists",
            )
        ]
    )
    assert categories(facts) == {"ANTICOAGULATION_MANAGEMENT"}


def test_unconfirmed_activity_status_is_missing_required_data() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(generic_name="warfarin", active=None, activity_evidence="patient unsure")
        ]
    )
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


def test_activity_judgment_unresolved_is_missing_required_data() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="warfarin",
                active=None,
                activity_evidence="call failed",
                activity_unresolved=True,
            )
        ]
    )
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


def test_plan_judgment_unresolved_is_missing_required_data() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(
                generic_name="warfarin",
                active=True,
                activity_evidence="active",
                plan_checked=True,
                plan_unresolved=True,
            )
        ]
    )
    assert categories(facts) == {"MISSING_REQUIRED_DATA"}


def test_multiple_anticoagulants_evaluated_independently() -> None:
    facts = make_facts(
        anticoagulants=[
            ResolvedAnticoagulant(generic_name="aspirin_like_but_coag", active=False, activity_evidence="n/a"),
            ResolvedAnticoagulant(
                generic_name="apixaban",
                active=True,
                activity_evidence="active",
                plan_checked=True,
                plan_adequate=False,
                plan_evidence="no plan",
            ),
        ]
    )
    issues = evaluate_all(facts)
    management_issues = [i for i in issues if i.category == "ANTICOAGULATION_MANAGEMENT"]
    assert len(management_issues) == 1
    assert "apixaban" in management_issues[0].evidence
