"""Comparador de formulaciones: reutiliza ``service.predict`` y solo ordena y presenta sus resultados."""

import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mavericks import compare, service
from mavericks.service import OutOfScopeError, ServiceError
from tests.test_ft_adjusted import OPTIONS, demo  # noqa: F401  (demo es una fixture)

OPTS = ["A favor", "En contra"]
PAGE = str(Path(__file__).resolve().parents[1] / "views" / "comparar.py")


def fake_result(favor, core=(0.4, 0.6), wide=(0.2, 0.8), level="media", mode="direct", groups=None):
    groups = groups or {
        "grupo_edad": {"label": "Edad", "groups": {"18-24": {"distribution": {"A favor": favor + 0.10, "En contra": 0.9 - favor}}, "65+": {"distribution": {"A favor": favor - 0.20, "En contra": 1.1 - favor}}}},
        "ccaa": {"label": "Comunidad autónoma", "groups": {"Murcia": {"distribution": {"A favor": favor - 0.05, "En contra": 1.05 - favor}}}},
        "sexo": {"label": "Sexo", "groups": {"mujer": {"distribution": {"A favor": 0.0, "En contra": 1.0}}}},  # el sexo no se usa para «menos apoyo»
    }
    return {
        "national": {"distribution": {"A favor": favor, "En contra": 1 - favor}, "interval_core": {"A favor": list(core), "En contra": [0.1, 0.3]}, "interval": {"A favor": list(wide), "En contra": [0.0, 0.5]}},
        "confidence": {"level": level}, "mode_label": "directo" if mode == "direct" else "análogas", "subgroups": groups,
    }


def test_parse_lines_strips_blanks_and_repeats_in_order():
    assert compare.parse_lines("  A favor \n\nEn  contra\na favor\n") == ["A favor", "En contra"]
    assert compare.parse_lines(None) == []


def test_least_support_looks_at_age_region_and_studies_only():
    result = fake_result(0.6)
    assert compare.least_support(result, "A favor") == ("Edad: 65+", pytest.approx(0.4))
    assert compare.least_support({"subgroups": {}}, "A favor") is None


def test_rows_are_sorted_by_the_chosen_option_and_failures_go_last():
    shares = {"¿Primera?": 0.30, "¿Segunda?": 0.55, "¿Tercera?": 0.42}

    def fake_predict(text, options, qtype, preset, resources=None):
        assert preset == "rapido_sin_llm" and options == OPTS  # siempre modo rápido y las mismas opciones
        if text == "¿Fuera?":
            raise OutOfScopeError({"explanation": "No es una opinión.", "reformulation": "Reformúlala."})
        if text == "¿Rota?":
            raise ServiceError("Revisa la pregunta")
        return fake_result(shares[text])

    seen = []
    rows = compare.compare(["¿Primera?", "¿Fuera?", "¿Segunda?", "¿Rota?", "¿Tercera?"], OPTS, "A favor", predict=fake_predict, progress=lambda n, t: seen.append((n, t)))
    assert seen[-1] == (5, 5)
    frame = compare.to_frame(rows)
    assert list(frame["Formulación"]) == ["¿Segunda?", "¿Tercera?", "¿Primera?", "¿Fuera?", "¿Rota?"]
    top = frame.iloc[0]
    assert top["% estimado"] == 55.0 and top["Rango habitual mín. (%)"] == 40.0 and top["Rango 90 % máx. (%)"] == 80.0
    assert top["Confianza"] == "media" and top["Modo"] == "directo" and top["Grupo con menos apoyo"] == "Edad: 65+" and top["% en ese grupo"] == 35.0
    assert list(frame["Estado"][:3]) == ["ok"] * 3 and frame["Estado"][3].startswith("Fuera de alcance: No es una opinión.") and frame["Estado"][4] == "Revisa la pregunta"
    assert list(frame.columns) == compare.COLUMNS


def test_csv_has_the_same_columns_and_rows_and_opens_in_excel():
    frame = compare.to_frame(compare.compare(["¿Uno?", "¿Dos?"], OPTS, "A favor", predict=lambda *a, **k: fake_result(0.5)))
    data = compare.to_csv(frame)
    assert data.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    assert list(back.columns) == compare.COLUMNS and len(back) == 2


@pytest.mark.parametrize("count", [0, 1, 11])
def test_between_two_and_ten_formulations_are_required(count):
    with pytest.raises(ValueError, match="entre 2 y 10"):
        compare.compare([f"¿Pregunta {i}?" for i in range(count)], OPTS, "A favor", predict=lambda *a, **k: fake_result(0.5))


def test_the_option_must_be_one_of_the_options():
    with pytest.raises(ValueError, match="no está entre las opciones"):
        compare.compare(["¿Uno?", "¿Dos?"], OPTS, "Otra", predict=lambda *a, **k: fake_result(0.5))


def test_numbers_are_exactly_those_of_service_predict(demo):  # noqa: F811
    questions = ["¿Le preocupa que la inteligencia artificial le quite su empleo?", "¿Le inquieta perder su empleo por culpa de la inteligencia artificial?"]
    rows = compare.compare(questions, OPTIONS, "Mucho", resources=demo)
    for row, question in zip(rows, questions, strict=True):
        direct = service.predict(question, OPTIONS, "single", "rapido_sin_llm", resources=demo)
        assert row["% estimado"] == round(100 * direct["national"]["distribution"]["Mucho"], 1) and row["Modo"] == direct["mode_label"]
        assert row["Confianza"] == direct["confidence"]["level"]


def test_page_shows_table_chart_and_download(monkeypatch):
    import mavericks.ui_common as ui_common

    monkeypatch.setattr(ui_common, "get_resources", lambda: None)
    monkeypatch.setattr(ui_common, "setup", lambda *a, **k: None)
    monkeypatch.setattr(service, "predict", lambda text, options, qtype, preset, resources=None: fake_result(0.7 if "apoyaría" in text.lower() else 0.4))
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    at.text_area(key="cmp_formulations").set_value("¿Está de acuerdo con limitar el alquiler?\n¿Apoyaría una ley que limite el alquiler?").run()
    at.text_area(key="cmp_options").set_value("A favor\nEn contra").run()
    at.button[0].click().run()
    assert not at.exception
    frame = at.dataframe[0].value
    assert list(frame["Formulación"]) == ["¿Apoyaría una ley que limite el alquiler?", "¿Está de acuerdo con limitar el alquiler?"]
    assert list(frame["% estimado"]) == [70.0, 40.0]
    assert any("Apoyo estimado a «A favor»" in s.value for s in at.subheader)
    assert len(at.get("plotly_chart")) == 1 and len(at.get("download_button")) == 1


def test_page_asks_for_two_to_ten_formulations(monkeypatch):
    import mavericks.ui_common as ui_common

    monkeypatch.setattr(ui_common, "get_resources", lambda: None)
    monkeypatch.setattr(ui_common, "setup", lambda *a, **k: None)
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.text_area(key="cmp_formulations").set_value("¿Solo una?").run()
    at.text_area(key="cmp_options").set_value("A favor\nEn contra").run()
    at.button[0].click().run()
    assert any("entre 2 y 10" in e.value for e in at.error)
