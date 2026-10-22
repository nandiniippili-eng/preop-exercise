from __future__ import annotations

from datetime import datetime

from core import PatientSubmission
from facts import (
    DrugClass,
    EntryStatus,
    extract_labs,
    extract_vitals,
    extract_vitals_from_text,
    parse_dt,
    procedure_date,
    procedure_risk,
    scan_medication_mentions,
)


def submission(**overrides: object) -> PatientSubmission:
    base: dict[str, object] = {
        "patient": {"id": "p1"},
        "procedure": {"case_id": "c1"},
        "vitals": [],
        "labs": [],
        "medications": [],
        "conditions": [],
        "documents": [],
    }
    base.update(overrides)
    return PatientSubmission.model_validate(base)


# -------------------------
# Labs (Rule 2)
# -------------------------


def test_lab_code_normalization_accepts_known_aliases() -> None:
    sub = submission(
        labs=[
            {"code": "CBC", "display": "Hemogram/CBC", "effective_at": "2026-01-01T00:00:00Z", "status": "final"},
            {"code": "LAB-CBC", "display": "CBC w/ Diff", "effective_at": "2026-01-02T00:00:00Z", "status": "final"},
            {"code": "CMP", "display": "Comprehensive Metabolic Panel", "effective_at": "2026-01-03T00:00:00Z", "status": "final"},
            {"code": "LAB-CMP", "display": "CMP w/ Mg", "effective_at": "2026-01-04T00:00:00Z", "status": "final"},
        ]
    )
    labs = extract_labs(sub)
    assert labs.latest("CBC") is not None
    assert labs.latest("CBC").effective_at == datetime(2026, 1, 2)
    assert labs.latest("CMP") is not None
    assert labs.latest("CMP").effective_at == datetime(2026, 1, 4)


def test_basic_metabolic_panel_never_resolves_to_cmp() -> None:
    sub = submission(
        labs=[
            {"code": "BMP", "display": "Basic Metabolic Panel", "effective_at": "2026-01-01T00:00:00Z", "status": "final"},
            {"code": "LAB-BMP", "display": "Basic Metabolic Panel w/ eGFR", "effective_at": "2026-01-02T00:00:00Z", "status": "final"},
        ]
    )
    labs = extract_labs(sub)
    assert labs.latest("CMP") is None


def test_unrelated_lab_codes_are_ignored() -> None:
    sub = submission(
        labs=[
            {"code": "TSH", "display": "Thyroid Stimulating Hormone", "effective_at": "2026-01-01T00:00:00Z", "status": "final"},
            {"code": "UA", "display": "Urinalysis", "effective_at": "2026-01-01T00:00:00Z", "status": "final"},
        ]
    )
    labs = extract_labs(sub)
    assert labs.latest("CBC") is None
    assert labs.latest("CMP") is None


def test_entered_in_error_lab_is_excluded_from_latest() -> None:
    sub = submission(
        labs=[
            {"code": "CBC", "display": "CBC", "effective_at": "2026-02-10T00:00:00Z", "status": "entered-in-error"},
            {"code": "CBC", "display": "CBC", "effective_at": "2026-02-09T00:00:00Z", "status": "final"},
            {"code": "CBC", "display": "CBC", "effective_at": "2026-01-06T00:00:00Z", "status": "final"},
        ]
    )
    latest = extract_labs(sub).latest("CBC")
    assert latest is not None
    assert latest.effective_at == datetime(2026, 2, 9)
    assert latest.status == EntryStatus.VALID


# -------------------------
# Medications / anticoagulation (Rule 1 trap)
# -------------------------


def test_brand_name_resolves_to_generic_and_class() -> None:
    sub = submission(medications=[{"name": "Xarelto 20 mg PO daily", "active": True}])
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert "rivaroxaban" in groups
    assert groups["rivaroxaban"].drug_class == DrugClass.ANTICOAGULANT


def test_antiplatelet_is_never_classified_as_anticoagulant() -> None:
    sub = submission(
        medications=[
            {"name": "ASA 81 mg PO daily", "active": True},
            {"name": "clopidogrel 75 mg PO daily", "active": True},
        ]
    )
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert groups["aspirin"].drug_class == DrugClass.ANTIPLATELET
    assert groups["clopidogrel"].drug_class == DrugClass.ANTIPLATELET
    assert all(g.drug_class != DrugClass.ANTICOAGULANT for g in groups.values())


def test_structured_entry_with_both_generic_and_brand_dedupes_to_one_group() -> None:
    sub = submission(medications=[{"name": "warfarin (Coumadin) 5 mg PO daily", "active": False}])
    groups = scan_medication_mentions(sub)
    assert len(groups) == 1
    assert groups[0].generic_name == "warfarin"


def test_document_only_mention_is_captured_with_no_structured_entry() -> None:
    sub = submission(
        medications=[],
        documents=[
            {
                "doc_id": "d1",
                "type": "Pre-Admission Testing Visit",
                "date": "2026-03-25",
                "text": "Patient reports taking Eliquis 5 mg BID from an outside cardiologist.",
            }
        ],
    )
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert "apixaban" in groups
    group = groups["apixaban"]
    assert group.structured_mention is None
    assert len(group.document_mentions) == 1
    assert group.needs_activity_judgment() is True


def test_clean_active_structured_entry_with_no_document_mentions_skips_judgment() -> None:
    sub = submission(medications=[{"name": "apixaban 5 mg PO BID", "active": True}])
    group = scan_medication_mentions(sub)[0]
    assert group.needs_activity_judgment() is False


def test_structured_entry_with_blank_active_needs_judgment() -> None:
    sub = submission(medications=[{"name": "Coumadin 5 mg PO daily", "active": None}])
    group = scan_medication_mentions(sub)[0]
    assert group.needs_activity_judgment() is True


def test_structured_entry_corroborated_in_free_text_still_needs_judgment() -> None:
    # Even a clean active=True signal still gets verified if the drug is also
    # discussed elsewhere in the chart (e.g. the chart could describe a switch).
    sub = submission(
        medications=[{"name": "warfarin 5 mg PO daily", "active": True}],
        documents=[
            {
                "doc_id": "d1",
                "type": "Cardiology Consult",
                "date": "2026-03-01",
                "text": "Continue warfarin per prior management.",
            }
        ],
    )
    group = scan_medication_mentions(sub)[0]
    assert group.needs_activity_judgment() is True


def test_asa_physical_status_is_not_mistaken_for_aspirin() -> None:
    # Regression: "ASA physical status: II" is the anesthesiology risk class,
    # not a mention of the drug aspirin. A document containing only this
    # phrase must not produce an aspirin DrugMentionGroup at all.
    sub = submission(
        documents=[
            {
                "doc_id": "d1",
                "type": "Anesthesia Pre-Assessment",
                "date": "2026-03-01",
                "text": "Planned procedure: knee arthroscopy  ASA physical status: II. Airway: Mallampati I.",
            }
        ]
    )
    assert scan_medication_mentions(sub) == []


def test_asa_class_phrasing_is_also_excluded() -> None:
    sub = submission(
        documents=[{"doc_id": "d1", "type": "Anesthesia Note", "date": "2026-03-01", "text": "ASA class III."}]
    )
    assert scan_medication_mentions(sub) == []


def test_real_asa_dosing_mention_still_counts_alongside_physical_status_text() -> None:
    sub = submission(
        medications=[{"name": "ASA 81 mg PO daily", "active": True}],
        documents=[
            {
                "doc_id": "d1",
                "type": "Anesthesia Pre-Assessment",
                "date": "2026-03-01",
                "text": "ASA physical status: II. Airway: Mallampati I.",
            }
        ],
    )
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert "aspirin" in groups
    assert groups["aspirin"].drug_class == DrugClass.ANTIPLATELET
    # Only the real dosing mention counted - the physical-status text didn't
    # also register as a (redundant, but still incorrect) document mention.
    assert len(groups["aspirin"].document_mentions) == 0


def test_dalteparin_brand_and_generic_resolve_to_anticoagulant() -> None:
    sub = submission(
        documents=[
            {
                "doc_id": "d1",
                "type": "Hematology Consult",
                "date": "2026-03-01",
                "text": "Bridge with Fragmin (dalteparin) per protocol.",
            }
        ]
    )
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert "dalteparin" in groups
    assert groups["dalteparin"].drug_class == DrugClass.ANTICOAGULANT


def test_dipyridamole_and_aggrenox_resolve_to_antiplatelet() -> None:
    sub = submission(medications=[{"name": "Aggrenox 25-200 mg PO BID", "active": True}])
    groups = {g.generic_name: g for g in scan_medication_mentions(sub)}
    assert "dipyridamole" in groups
    assert groups["dipyridamole"].drug_class == DrugClass.ANTIPLATELET


# -------------------------
# Vitals (Rule 4) - structured + free text
# -------------------------


def test_extract_vitals_from_structured_array() -> None:
    sub = submission(
        vitals=[
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-01-01T09:00:00Z"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-01-01T09:00:00Z"},
        ]
    )
    vitals = extract_vitals(sub)
    bp = vitals.latest_blood_pressure()
    temp = vitals.latest_temperature()
    assert (bp.systolic, bp.diastolic) == (120, 80)
    assert temp.temp_f == 98.6


def test_extract_vitals_from_text_associates_nearest_preceding_time() -> None:
    text = (
        "VITAL SIGNS:\n"
        "  08:06  NIBP 189/112, T 37.1 C, HR 74, SpO2 96% RA. See addendum.\n\n"
        "ADDENDUM 08:30: Re-measured 137/78 mmHg following 24 minutes in a quiet room."
    )
    readings = extract_vitals_from_text(text, doc_id="d1", fallback_date=datetime(2026, 3, 15))
    bp_readings = [r for r in readings if r.kind == "blood_pressure"]
    temp_readings = [r for r in readings if r.kind == "temperature"]

    assert len(bp_readings) == 2
    first, second = sorted(bp_readings, key=lambda r: r.provenance.asserted_at)
    assert (first.systolic, first.diastolic) == (189, 112)
    assert first.provenance.asserted_at == datetime(2026, 3, 15, 8, 6)
    assert (second.systolic, second.diastolic) == (137, 78)
    assert second.provenance.asserted_at == datetime(2026, 3, 15, 8, 30)

    assert len(temp_readings) == 1
    assert temp_readings[0].temp_f == 98.8  # 37.1 C -> F, associated with the 08:06 reading
    assert temp_readings[0].provenance.asserted_at == datetime(2026, 3, 15, 8, 6)


def test_extract_vitals_from_text_handles_celsius_and_fahrenheit() -> None:
    text = "09:05  BP: 161/117, 37.2 °C tympanic, HR 84, SpO2 95% RA"
    readings = extract_vitals_from_text(text, doc_id="d1", fallback_date=datetime(2026, 2, 28))
    temp = next(r for r in readings if r.kind == "temperature")
    assert temp.temp_f == 99.0  # 37.2 C


def test_extract_vitals_from_text_ignores_placeholder() -> None:
    text = "VITAL SIGNS: See flowsheet."
    readings = extract_vitals_from_text(text, doc_id="d1", fallback_date=datetime(2026, 3, 30))
    assert readings == []


def test_extract_vitals_from_text_does_not_misparse_dates_as_blood_pressure() -> None:
    text = "Date of service: 03/17/2026\nPlanned procedure: knee arthroscopy (scheduled for 04/02/2026)"
    readings = extract_vitals_from_text(text, doc_id="d1", fallback_date=datetime(2026, 3, 17))
    assert readings == []


def test_latest_blood_pressure_prefers_most_recent_across_structured_and_document_sources() -> None:
    sub = submission(
        vitals=[{"type": "blood_pressure", "systolic": 119, "diastolic": 78, "date": "2026-03-01T09:00:00Z"}],
        documents=[
            {
                "doc_id": "d1",
                "type": "Nursing pre-procedure note",
                "date": "2026-03-03",
                "text": "09:05 BP 157/117, HR 88",
            }
        ],
    )
    bp = extract_vitals(sub).latest_blood_pressure()
    assert (bp.systolic, bp.diastolic) == (157, 117)
    assert bp.provenance.source == "document_text"


def test_missing_vitals_resolve_to_none_not_a_default() -> None:
    sub = submission(vitals=[], documents=[])
    vitals = extract_vitals(sub)
    assert vitals.latest_blood_pressure() is None
    assert vitals.latest_temperature() is None


# -------------------------
# Procedure fields + date parsing
# -------------------------


def test_procedure_date_and_risk_pass_through() -> None:
    sub = submission(procedure={"case_id": "c1", "procedure_risk": "HIGH", "procedure_date": "2026-03-10"})
    assert procedure_date(sub) is not None
    assert procedure_date(sub).isoformat() == "2026-03-10"
    assert procedure_risk(sub) == "HIGH"


def test_procedure_date_and_risk_are_none_when_blank() -> None:
    sub = submission(procedure={"case_id": "c1", "procedure_risk": None, "procedure_date": None})
    assert procedure_date(sub) is None
    assert procedure_risk(sub) is None


def test_parse_dt_handles_zulu_and_date_only() -> None:
    assert parse_dt("2026-02-17T09:05:00Z") == datetime(2026, 2, 17, 9, 5)
    assert parse_dt("2026-02-17") == datetime(2026, 2, 17)
    assert parse_dt(None) is None
    assert parse_dt("not-a-date") is None
