"""Pure rule functions over PolicyFacts -> TriageIssue list -> Decision.

No I/O, no LLM calls, nothing async - every function here is a plain
function of PolicyFacts, which makes each rule independently unit-testable
against hand-built facts instead of full submissions. This is the layer
that encodes the Cadence Surgical Center policy itself (rules.py mirrors the
appendix 1:1 - see the module docstring on each function for the rule it
implements).

Every TriageIssue built here is a structured evidence packet assembled
entirely from fields the extraction layer already computed (Provenance on
labs/vitals, evidence_quote/rejected_candidates on the Resolved* objects) -
no rule function re-reads raw submission or document text.
"""

from __future__ import annotations

from core import Decision, EvidenceSource, TriageIssue
from facts import EntryStatus, days_between
from policy_facts import PolicyFacts, confidence_from_candidate_count

LAB_WINDOWS_BY_RISK: dict[str, list[tuple[str, int]]] = {
    "LOW": [("CBC", 30)],
    "MODERATE": [("CBC", 30)],
    "HIGH": [("CBC", 14), ("CMP", 14)],
}


def evaluate_missing_prerequisites(facts: PolicyFacts) -> list[TriageIssue]:
    """Fields a rule needs but that are entirely absent from the chart.

    Each of these independently blocks only the rule(s) that need it - the
    caller skips those specific window checks rather than guessing - but is
    itself always reported. Nothing exists to compare, so these are always
    "high" confidence (there is no candidate ambiguity in "the field is
    null").
    """

    issues: list[TriageIssue] = []
    if facts.procedure_date is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                policy="Final Clearance Determination - a required field must be known "
                "to evaluate a rule.",
                findings="procedure.procedure_date is null; the H&P and lab recency "
                "windows (Rules 1.1, 2) cannot be evaluated.",
                source=EvidenceSource(kind="structured_field"),
                notes="",
                confidence="high",
            )
        )
    if facts.procedure_risk is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                policy="Final Clearance Determination - a required field must be known "
                "to evaluate a rule.",
                findings="procedure.procedure_risk is null; Rule 2's testing tier "
                "(2.1 LOW/MODERATE vs 2.2 HIGH) cannot be determined.",
                source=EvidenceSource(kind="structured_field"),
                notes="",
                confidence="high",
            )
        )
    if facts.vitals.latest_blood_pressure() is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                policy="Final Clearance Determination - a required field must be known "
                "to evaluate a rule.",
                findings="No blood pressure reading found in structured vitals or any "
                "document; Rule 4's BP exclusion cannot be evaluated.",
                source=EvidenceSource(kind="structured_field"),
                notes="",
                confidence="high",
            )
        )
    if facts.vitals.latest_temperature() is None:
        issues.append(
            TriageIssue(
                category="MISSING_REQUIRED_DATA",
                policy="Final Clearance Determination - a required field must be known "
                "to evaluate a rule.",
                findings="No temperature reading found in structured vitals or any "
                "document; Rule 4's fever exclusion cannot be evaluated.",
                source=EvidenceSource(kind="structured_field"),
                notes="",
                confidence="high",
            )
        )
    return issues


def evaluate_documentation_hp(facts: PolicyFacts) -> TriageIssue | None:
    """Rule 1.1: History & Physical, completed within 30 days of the procedure date."""

    hp = facts.hp
    if hp.unresolved:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            policy="Rule 1.1 - a valid pre-op H&P document must be identifiable.",
            findings="Could not determine which document, if any, is the valid pre-op H&P.",
            source=EvidenceSource(kind="llm_judgment"),
            notes=hp.evidence_quote,
            confidence="low",
        )
    if facts.procedure_date is None:
        return None  # window unverifiable; already reported once via prerequisites

    sample = hp.rejected_candidates or None

    if hp.doc_id is None:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            policy="Rule 1.1 - History and Physical (H&P) must be completed within "
            "30 days of the scheduled procedure date.",
            findings="No valid pre-op H&P on file.",
            source=EvidenceSource(kind="llm_judgment"),
            notes=hp.evidence_quote,
            sample=sample,
            confidence=hp.confidence,
        )
    if hp.encounter_date is None:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            policy="Rule 1.1 - H&P must be completed within 30 days of the procedure date.",
            findings="A pre-op H&P document was identified but its encounter date could "
            "not be determined.",
            source=EvidenceSource(kind="llm_judgment", doc_id=hp.doc_id, doc_type=hp.doc_type),
            notes=hp.evidence_quote,
            sample=sample,
            confidence="low",
        )

    age_days = days_between(facts.procedure_date, hp.encounter_date)
    if age_days < 0 or age_days > 30:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            policy="Rule 1.1 - H&P must be completed within 30 days of the procedure date.",
            findings=f"Pre-op H&P dated {hp.encounter_date.isoformat()}, procedure date "
            f"{facts.procedure_date.isoformat()}: {age_days} day(s) apart, outside the "
            f"30-day window.",
            source=EvidenceSource(kind="llm_judgment", doc_id=hp.doc_id, doc_type=hp.doc_type),
            notes=hp.evidence_quote,
            sample=sample,
            confidence=hp.confidence,
        )
    return None


def evaluate_documentation_consent(facts: PolicyFacts) -> TriageIssue | None:
    """Rule 1.2: a signed surgical consent for the scheduled procedure."""

    consent = facts.consent
    if consent.unresolved:
        return TriageIssue(
            category="MISSING_REQUIRED_DATA",
            policy="Rule 1.2 - whether a valid signed consent exists must be determinable.",
            findings="Could not determine whether a valid signed consent exists.",
            source=EvidenceSource(kind="llm_judgment"),
            notes=consent.evidence_quote,
            confidence="low",
        )
    if not consent.valid:
        return TriageIssue(
            category="REQUIRED_DOCUMENTATION",
            policy="Rule 1.2 - a signed surgical consent for the scheduled procedure "
            "is required.",
            findings="No valid, patient-signed surgical consent on file for this procedure.",
            source=EvidenceSource(
                kind="llm_judgment", doc_id=consent.doc_id, doc_type=consent.doc_type
            ),
            notes=consent.evidence_quote,
            sample=consent.rejected_candidates or None,
            confidence=consent.confidence,
        )
    return None


def evaluate_testing(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 2: CBC (and CMP for HIGH risk) within the risk-tiered window."""

    if facts.procedure_risk is None or facts.procedure_date is None:
        return []  # tier/window unknown; already reported once via prerequisites

    required = LAB_WINDOWS_BY_RISK.get(facts.procedure_risk, [])
    issues: list[TriageIssue] = []
    for code, window_days in required:
        matching = [r for r in facts.labs.results if r.test_code == code]
        confidence = confidence_from_candidate_count(len(matching))
        latest = facts.labs.latest(code)  # type: ignore[arg-type]

        if latest is None:
            issues.append(
                TriageIssue(
                    category="REQUIRED_TESTING",
                    policy=f"Rule 2 - {code} required within {window_days} days of the "
                    f"procedure date for {facts.procedure_risk} risk procedures.",
                    findings=f"No valid {code} result on file."
                    + (f" {len(matching)} {code}-coded result(s) existed but none were "
                       "usable (entered-in-error or otherwise excluded)." if matching else ""),
                    source=EvidenceSource(kind="structured_field"),
                    notes="; ".join(m.provenance.quote for m in matching) if matching else "",
                    sample=[m.provenance.quote for m in matching] if len(matching) > 1 else None,
                    confidence=confidence,
                )
            )
            continue

        age_days = days_between(facts.procedure_date, latest.effective_at.date())
        if age_days < 0 or age_days > window_days:
            issues.append(
                TriageIssue(
                    category="REQUIRED_TESTING",
                    policy=f"Rule 2 - {code} required within {window_days} days of the "
                    f"procedure date for {facts.procedure_risk} risk procedures.",
                    findings=f"Most recent {code} dated {latest.effective_at.date().isoformat()}, "
                    f"procedure date {facts.procedure_date.isoformat()}: {age_days} day(s) "
                    f"apart, outside the {window_days}-day window.",
                    source=EvidenceSource(kind="structured_field", doc_id=latest.provenance.doc_id),
                    notes=latest.provenance.quote,
                    sample=(
                        [m.provenance.quote for m in matching if m is not latest]
                        if len(matching) > 1
                        else None
                    ),
                    confidence=confidence,
                )
            )
    return issues


def evaluate_safety(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 4: acute safety exclusions from the most recent vitals."""

    issues: list[TriageIssue] = []

    bp = facts.vitals.latest_blood_pressure()
    bp_candidates = [r for r in facts.vitals.blood_pressure if r.status == EntryStatus.VALID]
    bp_confidence = confidence_from_candidate_count(len(bp_candidates))
    bp_sample = (
        [r.provenance.quote for r in bp_candidates if r is not bp]
        if bp is not None and len(bp_candidates) > 1
        else None
    )
    if bp is not None:
        if bp.systolic is not None and bp.systolic >= 180:
            issues.append(
                TriageIssue(
                    category="ACUTE_SAFETY_EXCLUSION",
                    policy="Rule 4 - systolic BP >= 180 mmHg is an acute safety exclusion.",
                    findings=f"Most recent systolic BP {bp.systolic:g} mmHg (threshold 180).",
                    source=EvidenceSource(
                        kind=bp.provenance.source,
                        doc_id=bp.provenance.doc_id,
                        doc_type=bp.provenance.doc_type,
                    ),
                    notes=bp.provenance.quote,
                    sample=bp_sample,
                    confidence=bp_confidence,
                )
            )
        if bp.diastolic is not None and bp.diastolic >= 110:
            issues.append(
                TriageIssue(
                    category="ACUTE_SAFETY_EXCLUSION",
                    policy="Rule 4 - diastolic BP >= 110 mmHg is an acute safety exclusion.",
                    findings=f"Most recent diastolic BP {bp.diastolic:g} mmHg (threshold 110).",
                    source=EvidenceSource(
                        kind=bp.provenance.source,
                        doc_id=bp.provenance.doc_id,
                        doc_type=bp.provenance.doc_type,
                    ),
                    notes=bp.provenance.quote,
                    sample=bp_sample,
                    confidence=bp_confidence,
                )
            )

    temp = facts.vitals.latest_temperature()
    temp_candidates = [r for r in facts.vitals.temperature if r.status == EntryStatus.VALID]
    if temp is not None and temp.temp_f is not None and temp.temp_f > 100.4:
        temp_confidence = confidence_from_candidate_count(len(temp_candidates))
        temp_sample = (
            [r.provenance.quote for r in temp_candidates if r is not temp]
            if len(temp_candidates) > 1
            else None
        )
        issues.append(
            TriageIssue(
                category="ACUTE_SAFETY_EXCLUSION",
                policy="Rule 4 - temperature > 100.4 F is an acute safety exclusion.",
                findings=f"Most recent temperature {temp.temp_f:g} F (threshold 100.4 F).",
                source=EvidenceSource(
                    kind=temp.provenance.source,
                    doc_id=temp.provenance.doc_id,
                    doc_type=temp.provenance.doc_type,
                ),
                notes=temp.provenance.quote,
                sample=temp_sample,
                confidence=temp_confidence,
            )
        )
    return issues


def evaluate_anticoagulation(facts: PolicyFacts) -> list[TriageIssue]:
    """Rule 3: a documented, adequate perioperative plan for each active anticoagulant."""

    issues: list[TriageIssue] = []
    for ac in facts.anticoagulants:
        activity_source = EvidenceSource(kind=ac.activity_source_kind)

        if ac.activity_unresolved:
            issues.append(
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    policy="Rule 3 - current anticoagulant status must be known to evaluate "
                    "the management-plan requirement.",
                    findings=f"Could not determine whether {ac.generic_name} is currently active.",
                    source=activity_source,
                    notes=ac.activity_evidence,
                    confidence="low",
                )
            )
            continue
        if ac.active is None:
            issues.append(
                TriageIssue(
                    category="MISSING_REQUIRED_DATA",
                    policy="Rule 3 - current anticoagulant status must be known to evaluate "
                    "the management-plan requirement.",
                    findings=f"Current status of {ac.generic_name} is unconfirmed.",
                    source=activity_source,
                    notes=ac.activity_evidence,
                    confidence=ac.activity_confidence,
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
                    policy="Rule 3 - whether an adequate perioperative management plan "
                    "exists must be determinable.",
                    findings=f"Could not determine whether an adequate perioperative "
                    f"management plan exists for {ac.generic_name}.",
                    source=EvidenceSource(kind="llm_judgment"),
                    notes="",
                    confidence="low",
                )
            )
            continue
        if not ac.plan_adequate:
            issues.append(
                TriageIssue(
                    category="ANTICOAGULATION_MANAGEMENT",
                    policy="Rule 3 - a documented perioperative anticoagulation management "
                    "plan (clear hold and resume instructions for this specific drug) is "
                    "required for active anticoagulants.",
                    findings=f"No adequate perioperative management plan on file for "
                    f"{ac.generic_name}.",
                    source=EvidenceSource(kind="llm_judgment"),
                    notes=ac.plan_evidence or "No documented perioperative management plan found.",
                    confidence=ac.plan_confidence or ac.activity_confidence,
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
    lines.extend(f"- [{issue.category}] {issue.findings} ({issue.policy})" for issue in issues)
    return "\n".join(lines)
