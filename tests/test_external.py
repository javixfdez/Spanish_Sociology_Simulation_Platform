import numpy as np
import pandas as pd
import pyreadstat

from mavericks.etl.external import (
    CUTOFF,
    eb_education,
    ess_education,
    parse_codebook,
    process_ess,
    process_eurobarometer,
)


def test_education_equivalences_are_documented_approximations():
    ages = pd.Series([97, 13, 15, 18, 22, 98, 0, 99])
    assert list(eb_education(ages)[:5]) == ["sin_estudios", "primaria", "secundaria_1", "secundaria_2", "superiores"]
    assert eb_education(ages)[5:].isna().all()  # sigue estudiando / rechazo: sin categoría
    assert list(ess_education(pd.Series([1, 2, 4, 5, 7, 0]))[:5]) == ["primaria", "secundaria_1", "secundaria_2", "fp", "superiores"]


def test_parse_codebook_extracts_wording_between_label_and_value():
    html = (
        "<p>trstprl</p><p>Trust in country's parliament</p><p>On a scale of 0-10 how much do you trust "
        "[Country]'s parliament?</p><p>Value</p><p>Category</p><p>0</p><p>No trust</p>"
        "<p>trstplc</p><p>Trust in the police</p><p>Value</p>"
    )
    wording = parse_codebook(html, ["trstprl", "trstplc"])
    assert wording["trstprl"].startswith("On a scale of 0-10")
    assert "trstplc" not in wording  # sin enunciado propio


def _write_eb(path, n_es=400):
    rng = np.random.default_rng(0)
    n = n_es + 300
    frame = pd.DataFrame(
        {
            "isocntry": ["ES"] * n_es + ["FR"] * 300,
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
        column_labels={"qa1": "SITUATION: COUNTRY GENERAL", "q1.1": "NATIONALITY: BELGIUM", "qa7_1": "PROBLEMS: ECONOMY", "qa7_2": "PROBLEMS: HEALTH"},
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
    ids = {r["evidence_id"] for r in records}
    assert ids == {"EBZA9999_qa1", "EBZA9999_qa7"}  # sin d*, p*, w1 ni «NATIONALITY: BELGIUM»
    single = next(r for r in records if r["evidence_id"].endswith("qa1"))
    assert single["options"] == ["Good", "Bad"] and single["idioma"] == "en" and single["source"] == "eurobarometro"
    assert abs(sum(single["metadata"]["empirical_statistics"]["national"]["distribution"].values()) - 1) < 1e-9
    assert "n≈1.000" in single["metadata"]["survey_note"] and single["reservado_examen"] is False
    multiple = next(r for r in records if r["evidence_id"].endswith("qa7"))
    assert multiple["question_type"] == "multiple" and len(multiple["options"]) == 2
    assert date_is_before_cutoff(report["fieldwork"][1])


def date_is_before_cutoff(text):
    from datetime import date

    return date.fromisoformat(text) <= CUTOFF


def test_ess_edition_produced_after_cutoff_is_excluded(tmp_path):
    frame = pd.DataFrame(
        {
            "cntry": ["ES"] * 5,
            "essround": [11.0] * 5,
            "gndr": [1.0, 2.0, 1.0, 2.0, 1.0],
            "agea": [30.0, 40.0, 50.0, 60.0, 70.0],
            "anweight": [1.0] * 5,
            "proddate": ["02.07.2026"] * 5,
            "trstprl": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    path = tmp_path / "ESS11_test.sav"
    pyreadstat.write_sav(frame, str(path))
    records, report = process_ess(path, "", "general")
    assert records == [] and "posterior al 30/06/2026" in report["skipped"]
