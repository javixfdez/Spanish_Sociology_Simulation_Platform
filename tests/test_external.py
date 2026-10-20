from datetime import date

import numpy as np
import pandas as pd
import pyreadstat

from mavericks.etl.external import (
    CUTOFF,
    eb_education,
    ess_education,
    parse_codebook,
    parse_field_dates,
    process_ess,
    process_eurobarometer,
)


def test_education_equivalences_are_documented_approximations():
    ages = pd.Series([97, 13, 15, 18, 22, 98, 0, 99])
    assert list(eb_education(ages)[:5]) == ["sin_estudios", "primaria", "secundaria_1", "secundaria_2", "superiores"]
    assert eb_education(ages)[5:].isna().all()  # sigue estudiando / rechazo: sin categoría
    assert list(ess_education(pd.Series([1, 2, 4, 5, 7, 0]))[:5]) == [
        "primaria", "secundaria_1", "secundaria_2", "fp", "superiores",
    ]


def test_parse_codebook_extracts_wording_between_label_and_value():
    html = (
        "<p>trstprl</p><p>Trust in country's parliament</p><p>On a scale of 0-10 how much do you trust "
        "[Country]'s parliament?</p><p>Value</p><p>Category</p><p>0</p><p>No trust</p>"
        "<p>trstplc</p><p>Trust in the police</p><p>Value</p>"
    )
    wording = parse_codebook(html, ["trstprl", "trstplc"])
    assert wording["trstprl"].startswith("On a scale of 0-10")
    assert "trstplc" not in wording  # sin enunciado propio


def _write_eb(path, n_es=400, version="1.0.0 (2025-02-26)"):
    rng = np.random.default_rng(0)
    n = n_es + 300
    frame = pd.DataFrame(
        {
            "isocntry": ["ES"] * n_es + ["FR"] * 300,
            "version": [version] * n,
            "w1": rng.uniform(0.5, 1.5, n),
            "d10": rng.integers(1, 3, n).astype(float),
            "d11": rng.integers(18, 80, n).astype(float),
            "d8": rng.integers(14, 26, n).astype(float),
            "p1": rng.integers(1, 4, n).astype(float),
            "qa1": rng.choice([1.0, 2.0, 3.0], n, p=[0.6, 0.3, 0.1]),
            "q1.1": rng.integers(0, 2, n).astype(float),  # nacionalidad de otro país: se descarta
            "qa7_1": rng.integers(0, 2, n).astype(float),
            "qa7_2": rng.integers(0, 2, n).astype(float),
        }
    )
    pyreadstat.write_sav(
        frame,
        str(path),
        column_labels={
            "qa1": "SITUATION: COUNTRY GENERAL",
            "q1.1": "NATIONALITY: BELGIUM",
            "qa7_1": "PROBLEMS: ECONOMY",
            "qa7_2": "PROBLEMS: HEALTH",
        },
        variable_value_labels={
            "p1": {1.0: "17-Jan-2022", 2.0: "18-Jan-2022", 3.0: "19-Jan-2022"},
            "qa1": {1.0: "Good", 2.0: "Bad", 3.0: "Don't know (SPONTANEOUS)"},
            "q1.1": {0.0: "Not mentioned", 1.0: "Mentioned"},
            "qa7_1": {0.0: "Not mentioned", 1.0: "Mentioned"},
            "qa7_2": {0.0: "Not mentioned", 1.0: "Mentioned"},
        },
    )


def test_eurobarometer_keeps_only_spain_and_drops_technical_and_foreign_items(tmp_path):
    path = tmp_path / "ZA9999_v1-0-0.sav"
    _write_eb(path)
    records, report = process_eurobarometer(path, "vivienda")
    assert report["n_spain"] == 400 and tuple(report["fieldwork"]) == ("2022-01-17", "2022-01-19")
    assert report["fieldwork_source"] == "entrevistas"
    ids = {r["evidence_id"] for r in records}
    assert ids == {"EBZA9999_qa1", "EBZA9999_qa7"}  # sin d*, p*, w1 ni «NATIONALITY: BELGIUM»
    single = next(r for r in records if r["evidence_id"].endswith("qa1"))
    assert single["options"] == ["Good", "Bad"] and single["idioma"] == "en" and single["source"] == "eurobarometro"
    assert abs(sum(single["metadata"]["empirical_statistics"]["national"]["distribution"].values()) - 1) < 1e-9
    assert "n≈1.000" in single["metadata"]["survey_note"] and single["reservado_examen"] is False
    multiple = next(r for r in records if r["evidence_id"].endswith("qa7"))
    assert multiple["question_type"] == "multiple" and len(multiple["options"]) == 2
    assert date.fromisoformat(report["fieldwork"][1]) <= CUTOFF


def test_field_dates_come_from_the_catalog_title_not_from_the_publication_date(tmp_path):
    path = tmp_path / "ZA1_v1-0-0.sav"
    _write_eb(path, version="1.0.0 (2026-03-18)")
    row = {"title": "Standard Eurobarometer 102.2 (campo 10/10-05/11/2024)", "published": date(2026, 3, 18)}
    records, report = process_eurobarometer(path, "general", row)
    assert report["fieldwork_source"] == "catalogo" and tuple(report["fieldwork"]) == ("2024-10-10", "2024-11-05")
    assert "date_mismatch" in report  # las entrevistas del fichero dicen otra cosa: se avisa, manda el catálogo
    assert records[0]["fieldwork_end"] == "2024-11-05" and records[0]["fieldwork_date"] == "2024-11-05"
    assert records[0]["fecha_publicacion"] == "2026-03-18" and "fieldwork_estimated" not in records[0]
    # la fecha de campo manda: esta evidencia entra en un corte de 2025 aunque se publicara en 2026
    assert records[0]["fieldwork_end"] < "2025-01-01"


def test_publication_date_only_applies_the_june_2026_rule(tmp_path):
    for published, skipped in ((date(2026, 6, 9), False), (date(2026, 9, 1), True)):
        path = tmp_path / f"ZA2_{published}.sav"
        _write_eb(path)
        row = {"title": "EB (campo 26/03-08/04/2025)", "published": published}
        records, report = process_eurobarometer(path, "vivienda", row)
        assert bool(report.get("skipped")) is skipped and bool(records) is not skipped
        if skipped:
            assert "posterior al 30/06/2026" in report["skipped"]


def test_without_catalog_or_interview_dates_evidence_gets_no_date(tmp_path):
    path = tmp_path / "ZA3_v1-0-0.sav"
    frame = pd.DataFrame(
        {"isocntry": ["ES"] * 400, "version": ["1.0.0 (2025-02-26)"] * 400, "w1": [1.0] * 400,
         "qa1": np.random.default_rng(1).choice([1.0, 2.0], 400)}
    )
    pyreadstat.write_sav(frame, str(path), variable_value_labels={"qa1": {1.0: "Good", 2.0: "Bad"}})
    records, report = process_eurobarometer(path, "vivienda")
    assert records and records[0]["fieldwork_end"] is None and "sin fecha" in report["warning"]


def test_parse_field_dates_formats():
    assert parse_field_dates("EB 97.4 (campo 30/05-28/06/2022)") == ("2022-05-30", "2022-06-28")
    assert parse_field_dates("Flash EB 514 (campo 23-30/11/2022)") == ("2022-11-23", "2022-11-30")
    assert parse_field_dates("EB 99.1 (campo 02-26/03/2023)") == ("2023-03-02", "2023-03-26")
    assert parse_field_dates("EB x (campo 28/12-10/01/2025)") == ("2024-12-28", "2025-01-10")
    assert parse_field_dates("ESS ronda 10 (incluye Espana; campo 2022) ed. 3.2") is None
    assert parse_field_dates("sin campo") is None


def _ess_frame():
    return pd.DataFrame(
        {
            "cntry": ["ES"] * 5, "essround": [11.0] * 5, "gndr": [1.0, 2.0, 1.0, 2.0, 1.0],
            "agea": [30.0, 40.0, 50.0, 60.0, 70.0], "anweight": [1.0] * 5, "proddate": ["02.07.2026"] * 5,
            "trstprl": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_ess_edition_produced_after_cutoff_is_excluded_without_a_catalog(tmp_path):
    path = tmp_path / "ESS11_test.sav"
    pyreadstat.write_sav(_ess_frame(), str(path))
    records, report = process_ess(path, "", "general")
    assert records == [] and "posterior al 30/06/2026" in report["skipped"]


def test_ess_uses_catalog_publication_date_and_keeps_a_note(tmp_path):
    path = tmp_path / "ESS11_test.sav"
    pyreadstat.write_sav(_ess_frame(), str(path))
    _, late = process_ess(path, "", "general", {"title": "ESS11", "published": date(2026, 7, 2)})
    assert "posterior al 30/06/2026" in late["skipped"]
    _, earlier = process_ess(path, "", "general", {"title": "ESS11 ed. 4.2 (campo 2023)", "published": date(2026, 1, 13)})
    assert "skipped" not in earlier and "edition_note" in earlier and earlier["published"] == "2026-01-13"
