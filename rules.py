"""Pure rule functions over PolicyFacts -> TriageIssue list -> Decision.

No I/O, no LLM calls, nothing async - every function here is a plain
function of PolicyFacts, which makes each rule independently unit-testable
against hand-built facts instead of full submissions. This is the layer
that encodes the Cadence Surgical Center policy itself (rules.py mirrors the
appendix 1:1 - see the module docstring on each function for the rule it
implements).
"""

from __future__ import annotations

from core import Decision, TriageIssue
from facts import days_between
from policy_facts import PolicyFacts

LAB_WINDOWS_BY_RISK: dict[str, list[tuple[str, int]]] = {
    "LOW": [("CBC", 30)],
    "MODERATE": [("CBC", 30)],
    "HIGH": [("CBC", 14), ("CMP", 14)],
}


def evaluate_missing_prerequisites(facts: PolicyFacts) -> list[TriageIssue]:
    """Fields a rule needs but that are entirely absent from the chart.

    Each of these independently blocks only the rule(s) that need it - the
    caller skips those specific window checks rather than guessing - but is
    itself always reported.
    """

    issues: list[TriageIssue] = []
    if facts.procedure_date is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                evidence="procedure.procedure_date is missing; cannot evaluate the H&P or lab "
                "recency windows.",
            )
        )
    if facts.procedure_risk is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                evidence="procedure.procedure_risk is missing; cannot determine which testing "
                "tier (Rule 2.1 vs 2.2) applies.",
            )
        )
    if facts.vitals.latest_blood_pressure() is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                evidence="No blood pressure reading found in structured vitals or any document; "
                "cannot evaluate the acute safety exclusion for BP.",
            )
        )
    if facts.vitals.latest_temperature() is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                evidence="No temperature reading found in structured vitals or any document; "
                "cannot evaluate the acute safety exclusion for fever.",
            )
        )
    return issues


def evaluate_documentation_hp(facts: PolicyFacts) -> TriageIssue | None:
    """Rule 1.1: History & Physical, completed within 30 days of the procedure date."""

    if facts.hp.unresolved:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            evidence=f"Could not determine which document is the valid pre-op H&P: "
            f"{facts.hp.evidence_quote}",
        )
    if facts.procedure_date is None:
        return None  # window unverifiable; already reported once via prerequisites

    if facts.hp.doc_id is None:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            evidence=f"No valid pre-op H&P on file. {facts.hp.evidence_quote}",
        )
    if facts.hp.encounter_date is None:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            evidence="A pre-op H&P document was identified but its encounter date could not be "
            "determined.",
        )

    age_days = days_between(facts.procedure_date, facts.hp.encounter_date)
    if age_days < 0 or age_days > 30:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            evidence=f"Pre-op H&P dated {facts.hp.encounter_date.isoformat()} is {age_days} "
            f"day(s) before the procedure date {facts.procedure_date.isoformat()}, outside the "
            f"30-day window. {facts.hp.evidence_quote}",
        )
    return None


def evaluate_documentation_consent(facts: PolicyFacts) -> TriageIssue | None:
    """Rule 1.2: a signed surgical consent for the scheduled procedure."""

    if facts.consent.unresolved:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            evidence=f"Could not determine whether a valid signed consent exists: "
            f"{facts.consent.evidence_quote}",
        )
    if not facts.consent.valid:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            evidence=f"No valid, patient-signed surgical consent on file for this procedure. "
            f"{facts.consent.evidence_quote}",
        )
    return None


def evaluate_testing(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 2: CBC (and CMP for HIGH risk) within the risk-tiered window."""

    if facts.procedure_risk is None or facts.procedure_date is None:
        return []  # tier/window unknown; already reported once via prerequisites

    required = LAB_WINDOWS_BY_RISK.get(facts.procedure_risk, [])
    issues: list[TriageIssue] = []
    for code, window_days in required:
        latest = facts.labs.latest(code)  # type: ignore[arg-type]
        if latest is None:
            issues.append(
                TriageIssue(
                    category="REQUIRED_TESTING",
                    evidence=f"No valid {code} result on file; a {facts.procedure_risk} risk "
                    f"procedure requires one within {window_days} days of the procedure date.",
                )
            )
            continue
        age_days = days_between(facts.procedure_date, latest.effective_at.date())
        if age_days < 0 or age_days > window_days:
            issues.append(
                TriageIssue(
                    category="REQUIRED_TESTING",
                    evidence=f"Most recent {code} is {age_days} day(s) before the procedure "
                    f"date, outside the {window_days}-day window for {facts.procedure_risk} "
                    f"risk. {latest.provenance.quote}",
                )
            )
    return issues


def evaluate_safety(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 4: acute safety exclusions from the most recent vitals."""

    issues: list[TriageIssue] = []
    bp = facts.vitals.latest_blood_pressure()
    if bp is not None:
        if bp.systolic is not None and bp.systolic >= 180:
            issues.append(
                TriageIssue(
                    category="ACUTE_SAFETY_EXCLUSION",
                    evidence=f"Most recent systolic BP {bp.systolic:g} mmHg >= 180. "
                    f"{bp.provenance.quote}",
                )
            )
        if bp.diastolic is not None and bp.diastolic >= 110:
            issues.append(
                TriageIssue(
                    category="ACUTE_SAFETY_EXCLUSION",
                    evidence=f"Most recent diastolic BP {bp.diastolic:g} mmHg >= 110. "
                    f"{bp.provenance.quote}",
                )
            )

    temp = facts.vitals.latest_temperature()
    if temp is not None and temp.temp_f is not None and temp.temp_f > 100.4:
        issues.append(
            TriageIssue(
                category="ACUTE_SAFETY_EXCLUSION",
                evidence=f"Most recent temperature {temp.temp_f:g} F > 100.4 F. "
                f"{temp.provenance.quote}",
            )
        )
    return issues


def evaluate_anticoagulation(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 3: a documented, adequate perioperative plan for each active anticoagulant."""

    issues: list[TriageIssue] = []
    for ac in facts.anticoagulants:
        if ac.activity_unresolved:
            issues.append(
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    evidence=f"Could not determine whether {ac.generic_name} is currently "
                    f"active: {ac.activity_evidence}",
                )
            )
            continue
        if ac.active is None:
            issues.append(
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    evidence=f"Current status of {ac.generic_name} is unconfirmed: "
                    f"{ac.activity_evidence}",
                )
            )
            continue
        if ac.active is False:
            continue  # not currently active -> Rule 3 does not apply
        if ac.bridging_agent_only:
            continue  # a bridging therapy within another drug's plan, not its own home med

        if ac.plan_unresolved:
            issues.append(
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    evidence=f"Could not determine whether an adequate perioperative "
                    f"management plan exists for {ac.generic_name}.",
                )
            )
            continue
        if not ac.plan_adequate:
            evidence = ac.plan_evidence or "No documented perioperative management plan found."
            issues.append(
                TriageIssue(
                    category="ANTICOAGULATION_MANAGEMENT",
                    evidence=f"No adequate perioperative management plan on file for "
                    f"{ac.generic_name} (needs clear hold and resume instructions written for "
                    f"this specific drug). {evidence}",
                )
            )
    return issues


def evaluate_all(facts: PolicyFacts) -> list[TriageIssue]:
    issues: list[TriageIssue] = []
    issues.extend(evaluate_missing_prerequisites(facts))

    hp_issue = evaluate_documentation_hp(facts)
    if hp_issue is not None:
        issues.append(hp_issue)

    consent_issue = evaluate_documentation_consent(facts)
    if consent_issue is not None:
        issues.append(consent_issue)

    issues.extend(evaluate_testing(facts))
    issues.extend(evaluate_safety(facts))
    issues.extend(evaluate_anticoagulation(facts))
    return issues


def aggregate_decision(issues: list[TriageIssue]) -> Decision:
    """Final Clearance Determination: exclusions win, then any other issue, else READY."""

    categories = {issue.category for issue in issues}
    if "ACUTE_SAFETY_EXCLUSION" in categories:
        return "NOT_CLEARED"
    if categories:
        return "NEEDS_FOLLOW_UP"
    return "READY"


def build_explanation(decision: Decision, issues: list[TriageIssue]) -> str:
    if not issues:
        return (
            "All required documentation, testing, anticoagulation management, and safety "
            "criteria are satisfied within their required timeframes."
        )
    lines = [f"{decision}: {len(issues)} issue(s) found."]
    lines.extend(f"- [{issue.category}] {issue.evidence}" for issue in issues)
    return "\n".join(lines)
