import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mavericks import assist, explain, measured
from mavericks import service as svc
from mavericks.anchor import option_mapping

REAL = Path("data/processed/cis_responses.parquet").exists() and Path("data/processed_v2/evidence_v2.sqlite").exists() and Path("data/examples/ejemplos_demo.json").exists()
OPTIONS = ["Sí", "No", "Depende"]


def frame():
    return pd.DataFrame({
        "respondent_uid": [f"r{i}" for i in range(8)], "peso": [1.0, 2.0, 0.5, 1.5, 1.0, 1.0, 3.0, 1.0],
        "answer": ["Sí", "Sí", "No", "No", "Depende", "NS/NC", "Sí", "No"], "sexo": ["mujer"] * 4 + ["hombre"] * 4,
    })


def test_tabulate_is_weighted_excludes_non_substantive_and_uses_kish_n_eff():
    p, margin, n, n_eff = measured.tabulate(frame(), OPTIONS)
    f = frame()[frame()["answer"].isin(OPTIONS)]
    w = f["peso"].to_numpy()
    assert n == 7 and n_eff == pytest.approx(w.sum() ** 2 / (w**2).sum())
    expected = f.groupby("answer")["peso"].sum().reindex(OPTIONS) / w.sum()
    assert p == pytest.approx(expected.to_numpy()) and p.sum() == pytest.approx(1)
    assert margin == pytest.approx(1.96 * np.sqrt(p * (1 - p) / n_eff))


def test_small_sample_flag_and_empty_group():
    class Anchors:
        def _frame(self, study, evidence):
            return frame()

    res = type("R", (), {"anchors": Anchors()})()
    spec = {"study_id": "1", "evidence_id": "CIS1_X", "mapping": {o: o for o in [*OPTIONS, "NS/NC"]}}
    out = measured.block(res, spec, OPTIONS, "2025-01-01", [(("sexo", "mujer"),), (("sexo", "alien"),)])
    assert out["national"]["small"] is True and out["crosses"][0]["n"] == 4 and out["crosses"][0]["small"]
    assert out["crosses"][1]["distribution"] is None and out["crosses"][1]["n"] == 0
    assert out["study_id"] == "1" and out["fieldwork_end"] == "2025-01-01"


def test_intent_rules_by_keywords():
    for q in ("¿Qué respondieron los españoles sobre el alquiler?", "Según el CIS, ¿cuánta gente confía en el Gobierno?", "¿Qué porcentaje contestó que sí en el último barómetro?"):
        assert assist.detect_intent(q) == "consulta_dato"
    for q in ("¿Le preocupa la inteligencia artificial?", "¿Está a favor de limitar el alquiler?", ""):
        assert assist.detect_intent(q) == "prediccion"


def result(measured_block=True, intent="prediccion"):
    dist = {"Sí": 0.6, "No": 0.4}
    out = {
        "question": "¿Está a favor?", "options": ["Sí", "No"], "mode": "direct" if measured_block else "none", "mode_label": "directo" if measured_block else "sin ancla", "intent": intent,
        "national": {"distribution": dist, "interval_core": {"Sí": [0.5, 0.7], "No": [0.3, 0.5]}},
        "confidence": {"typical_error": 0.05, "backtest_cases": 10}, "prior_usado": {"metodo": "ancla_directa"}, "subgroups": {}, "evidence": [],
        "measured": None, "measured_note": None,
    }
    if measured_block:
        out["measured"] = {
            "study_id": "3546", "evidence_id": "CIS3546_P1", "fieldwork_end": "2026-03-06",
            "national": {"distribution": {"Sí": 0.52, "No": 0.48}, "margin": {"Sí": 0.02, "No": 0.02}, "interval": {}, "n": 3958, "n_eff": 2000.0, "small": False},
            "crosses": [{"label": "Sexo: mujer", "distribution": {"Sí": 0.58, "No": 0.42}, "margin": {"Sí": 0.09, "No": 0.09}, "n": 30, "n_eff": 20.0, "small": True}],
        }
    else:
        out["measured_note"] = measured.NO_MEASURED
    return out


def test_template_keeps_measured_and_estimated_apart_and_orders_by_intent():
    text = explain.template_text(result())
    assert "Dato medido por el CIS (estudio 3546" in text and "52 %" in text and "muestra pequeña" in text
    assert text.index("La opción más probable") < text.index("Dato medido")  # predicción: primero la estimación
    first = explain.template_text(result(intent="consulta_dato"))
    assert first.startswith("Dato medido por el CIS") and "Estimación del modelo para hoy: la opción más probable" in first
    assert "No hay dato medido" in explain.template_text(result(measured_block=False))


def test_number_check_refuses_to_mix_measured_and_estimated_figures():
    payload = explain.llm_payload(result())
    payload["_all_categories"] = []
    ok = "Según el CIS se midió un 52 % de Sí. La estimación del modelo para hoy es del 60 % de Sí."
    assert explain.check_text(ok, payload) is None
    mixed = "El CIS midió un 60 % de Sí."  # 60 es la estimación del modelo, no lo medido
    assert explain.check_text(mixed, payload).startswith("mezcla lo medido y lo estimado")
    mixed2 = "La estimación del modelo es del 52 % de Sí."  # 52 es lo medido
    assert explain.check_text(mixed2, payload).startswith("mezcla lo medido y lo estimado")


def test_ui_renders_both_blocks_in_the_right_order_and_the_small_sample_warning():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st

        from mavericks.ui_result import render_measured

        render_measured(st.session_state["r"])

    at = AppTest.from_function(app)
    at.session_state["r"] = result()
    at.run()
    assert at.subheader[0].value == "Lo que midió el CIS" and "Muestra pequeña: el dato medido es poco fiable" in at.warning[0].value
    assert "3546" in at.caption[0].value and "3958 encuestados" in at.caption[0].value


# --- igualdad EXACTA con un cálculo independiente sobre los microdatos (preguntas reales) ---------------------------------------------------------

def independent(study, evidence_id, options, cross=None):
    """Cálculo aparte con pandas sobre los parquet originales (sin usar el código del proyecto, salvo ``option_mapping`` para casar etiquetas)."""
    answers = pq.read_table("data/processed/cis_responses.parquet", filters=[("estudio_id", "=", str(study)), ("question_uid", "=", evidence_id)]).to_pandas()
    people = pd.read_parquet("data/processed/cis_respondents.parquet")
    people = people[people["estudio_id"] == str(study)]
    mapping = option_mapping(options, sorted(answers["answer"].unique()))
    back = {v: k for k, v in mapping.items()}
    merged = answers.merge(people, on=["estudio_id", "respondent_uid"], how="left")
    merged["opt"] = merged["answer"].map(back)
    merged = merged[merged["opt"].notna()].drop_duplicates("respondent_uid")
    if cross:
        for var, val in cross:
            merged = merged[merged[var].astype(str) == val]
    w = merged["peso"].to_numpy(float)
    sums = merged.groupby("opt")["peso"].sum().reindex(options).fillna(0.0).to_numpy()
    p = sums / sums.sum()
    n_eff = w.sum() ** 2 / (w**2).sum()
    return p, 1.96 * np.sqrt(p * (1 - p) / n_eff), len(merged), n_eff


@pytest.mark.skipif(not REAL, reason="necesita los datos reales")
def test_measured_block_equals_independent_pandas_on_real_questions_and_a_cross():
    resources = svc.load_resources()
    items = [e for e in json.loads(Path("data/examples/ejemplos_demo.json").read_text(encoding="utf-8"))["items"] if e["mode"] == "direct"][:3]
    assert len(items) == 3
    cross = [[{"var": "sexo", "val": "mujer"}, {"var": "edad", "val": "25-34"}]]
    for k, item in enumerate(items):
        out = svc.predict(item["question"], item["options"], preset="rapido_sin_llm", resources=resources, crosses=cross if k == 0 else None)
        m = out["measured"]
        assert m is not None and m["study_id"] and m["fieldwork_end"] and out["intent"] == "prediccion"
        options = out["options"]
        p, margin, n, n_eff = independent(m["study_id"], m["evidence_id"], options)
        nat = m["national"]
        assert [nat["distribution"][o] for o in options] == pytest.approx(list(p), abs=1e-12, rel=0)
        assert [nat["margin"][o] for o in options] == pytest.approx(list(margin), abs=1e-12, rel=0)
        assert nat["n"] == n and nat["n_eff"] == pytest.approx(n_eff, abs=1e-9, rel=0)
        if k == 0:
            c = m["crosses"][0]
            p, margin, n, n_eff = independent(m["study_id"], m["evidence_id"], options, cross=[("sexo", "mujer"), ("grupo_edad", "25-34")])
            assert c["n"] == n and [c["distribution"][o] for o in options] == pytest.approx(list(p), abs=1e-12, rel=0)
            assert [c["margin"][o] for o in options] == pytest.approx(list(margin), abs=1e-12, rel=0) and c["n_eff"] == pytest.approx(n_eff, abs=1e-9, rel=0)


@pytest.mark.skipif(not REAL, reason="necesita los datos reales")
def test_new_question_has_no_measured_block_and_says_so():
    out = svc.predict("¿Le preocupa que la inteligencia artificial le quite su empleo?", ["Mucho", "Bastante", "Poco", "Nada"], preset="rapido_sin_llm", resources=svc.load_resources())
    assert out["measured"] is None and "No hay dato medido" in out["measured_note"] and out["mode"] != "direct"
