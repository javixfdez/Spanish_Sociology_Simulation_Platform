import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pyreadstat

from mavericks import categories as cat
from mavericks.etl.cis import parse_fieldwork, run_etl
from mavericks.evidence import survey_distribution
from mavericks.topics import TOPICS


def test_parse_fieldwork_variants():
    assert parse_fieldwork("Fecha de realización: \nDel 2 al 6 de marzo de 2026. \n") == (
        date(2026, 3, 2),
        date(2026, 3, 6),
    )
    assert parse_fieldwork("Fecha de realización:\nDel 27 marzo al 1 de abril de 2025") == (
        date(2025, 3, 27),
        date(2025, 4, 1),
    )
    assert parse_fieldwork("Fecha de realización: Del 30 de diciembre al 5 de enero de 2026") == (
        date(2025, 12, 30),
        date(2026, 1, 5),
    )
    assert parse_fieldwork("sin fecha") is None


def test_common_categories():
    assert cat.map_estudios("Secundaria 1ª etapa") == "secundaria_1"
    assert cat.map_estudios("Secundaria inferior") == "secundaria_1"
    assert cat.map_estudios("Universitarios") == "superiores"
    assert cat.map_estudios("N.P.") is None
    assert cat.map_ccaa("Castilla y León") == "Castilla y León"
    assert cat.map_ccaa("Madrid (Comunidad de)") == "Madrid (Comunidad de)"
    assert cat.map_ingresos("De 3.901 a 5.000 &euro;") == "De 3.901 a 5.000 €"
    assert cat.map_ingresos("Más de 5.000 €") == "Más de 5.000 €"
    assert cat.map_ingresos("Menos o igual a 300 €") is None
    assert list(cat.age_group([18, 24, 25, 80])) == ["18-24", "18-24", "25-34", "75+"]


def _write_study(root, n=120):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "ESTUDIO": float(9999),
            "REGISTRO": np.arange(1, n + 1, dtype=float),
            "CCAA": rng.integers(1, 20, n).astype(float),
            "TAMUNI": rng.integers(1, 8, n).astype(float),
            "SEXO": rng.integers(1, 3, n).astype(float),
            "EDAD": rng.integers(18, 90, n).astype(float),
            "P1": rng.integers(1, 4, n).astype(float),
            "P2_1": rng.integers(1, 3, n).astype(float),
            "P2_2": rng.integers(1, 3, n).astype(float),
            "P3": np.where(np.arange(n) < 60, 0.0, 1.0),
            "ESTUDIOS": rng.integers(1, 7, n).astype(float),
            "IA_E1_DIA": 3.0,
            "IA_E1_MES": 3.0,
            "IA_E1_ANYO": 26.0,
            "IA_OBS1": 1.0,
            "PESO": rng.uniform(0.5, 2, n),
        }
    )
    labels = {
        "CCAA": {float(i): "Andalucía" if i == 1 else "Aragón" for i in range(1, 20)},
        "TAMUNI": {1.0: "Menos o igual a 2.000 habitantes"},
        "SEXO": {1.0: "Hombre", 2.0: "Mujer"},
        "P1": {0.0: "No procede", 1.0: "Sí", 2.0: "No", 3.0: "N.S."},
        "P2_1": {1.0: "Menciona", 2.0: "No menciona"},
        "P2_2": {1.0: "Menciona", 2.0: "No menciona"},
        "P3": {0.0: "N.P.", 1.0: "Sí"},
        "ESTUDIOS": {1.0: "Sin estudios", 2.0: "Primaria", 6.0: "Superiores"},
    }
    sav = root / "9999.sav"
    pyreadstat.write_sav(
        frame,
        str(sav),
        column_labels={"P1": "Pregunta uno con una redacción suficientemente larga", "P2_1": "Opción A", "P2_2": "Opción B", "P3": "Filtrada"},
        variable_value_labels=labels,
    )
    raw = root / "raw" / "vivienda" / "cis"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "CIS_9999_prueba.zip", "w") as z:
        z.write(sav, "9999.sav")
    catalog = root / "catalogo.csv"
    catalog.write_text(
        '"archivo","tema","tipo","titulo","fecha_publicacion","reservado_examen","bytes"\n'
        '"raw/vivienda/cis/CIS_9999_prueba.zip","vivienda","encuesta_cis","","2026-03-02","si","1"\n',
        encoding="utf-8",
    )
    return catalog


def test_etl_cis_end_to_end(tmp_path):
    catalog = _write_study(tmp_path)
    out = tmp_path / "out"
    report = run_etl(tmp_path / "raw", catalog, out)
    assert report["n_studies"] == 1 and report["n_reserved_studies"] == 1
    questions = pd.read_parquet(out / "cis_questions.parquet")
    respondents = pd.read_parquet(out / "cis_respondents.parquet")
    responses = pd.read_parquet(out / "cis_responses.parquet")
    studies = pd.read_parquet(out / "cis_studies.parquet")
    assert set(questions.question_uid) == {"CIS9999_P1", "CIS9999_P2"}  # P3 sin válidos suficientes, IA_* fuera
    assert questions.set_index("question_uid").loc["CIS9999_P2", "question_type"] == "multiple"
    assert bool(questions.reservado_examen.all()) and set(questions.topic) == {"vivienda"} <= set(TOPICS)
    assert studies.fieldwork_end.iloc[0] == "2026-03-03" and studies.fieldwork_source.iloc[0] == "microdatos"
    assert set(respondents.sexo.dropna()) <= {"hombre", "mujer"}
    assert set(respondents.nivel_estudios.dropna()) <= set(cat.NIVEL_ESTUDIOS) | {"otros", "nc"}
    assert respondents.respondent_uid.is_unique
    # «No procede» es estructural: no es opción de la pregunta; NS explícito conservado
    assert "No procede" not in questions.set_index("question_uid").loc["CIS9999_P1", "options"]
    assert "No procede" not in set(responses.answer)
    assert "N.S." in set(responses[responses.question_uid == "CIS9999_P1"].answer)
    # El resultado es consumible por survey_distribution tal cual
    single = survey_distribution(respondents, responses, "CIS9999_P1", group_by="sexo")
    assert set(single) == {"hombre", "mujer"}
    assert abs(sum(single["mujer"]["distribution"].values()) - 1) < 1e-9
    multiple = survey_distribution(respondents, responses, "CIS9999_P2", question_type="multiple")
    assert set(multiple["national"]["distribution"]) == {"Opción A", "Opción B"}


def test_duplicate_registro_falls_back_to_stable_cuestionario(tmp_path):
    from mavericks.etl.cis import build_respondents

    class Meta:
        variable_value_labels = {}
        column_names_to_labels = {}

    frame = pd.DataFrame({"REGISTRO": [1.0, 1.0, 2.0], "CUES": [10.0, 11.0, 12.0], "PESO": [1.0, 1.0, 1.0]})
    warnings = []
    out, _ = build_respondents(frame, Meta(), "1", warnings)
    assert out.respondent_uid.tolist() == ["C10", "C11", "C12"] and warnings
