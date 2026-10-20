import asyncio
import types

import numpy as np
import pandas as pd
import pytest

from mavericks import assist, explain, mrp_service
from mavericks import service as svc
from mavericks.mrp import MRP, tilt
from mavericks.schemas import PlanItem, QueryPlan

SEX, AGE, CCAA = ["hombre", "mujer"], ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"], ["Andalucía", "Cataluña", "Galicia"]
EDU, INC = ["primaria", "secundaria_2", "superiores"], ["Menos de 1.100 €", "De 1.100 a 1.800 €", "Más de 5.000 €"]
TAMUNI = ["2.001 a 10.000 habitantes", "Más de 1.000.000 habitantes"]
OPTIONS = ["Sí", "No", "Depende"]


def people(n, seed=0, income_valid=True):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "respondent_uid": [f"r{i}" for i in range(n)], "peso": rng.uniform(0.5, 1.5, n), "sexo": rng.choice(SEX, n), "grupo_edad": rng.choice(AGE, n),
        "ccaa": rng.choice(CCAA, n), "nivel_estudios": rng.choice(EDU, n), "tamuni": rng.choice(TAMUNI, n),
        "ingresos_hogar_tramo": rng.choice(INC, n) if income_valid else "N.S.", "edad": rng.uniform(18, 90, n),
    })


def answers(frame, seed=1):
    """Los mayores de 55 dicen «Sí» con mucha más frecuencia; el resto, «No»."""
    rng = np.random.default_rng(seed)
    old = frame["grupo_edad"].isin(["55-64", "65-74", "75+"]).to_numpy()
    p_yes = np.where(old, 0.8, 0.2)
    draw = rng.random(len(frame))
    return frame.assign(answer=np.where(draw < p_yes, "Sí", np.where(draw < p_yes + 0.1, "Depende", "No")))


def test_tilt_makes_the_weighted_mean_equal_the_goal():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), 500)
    w = rng.uniform(0.5, 1.5, 500)
    w /= w.sum()
    goal = np.array([0.5, 0.3, 0.2])
    q = tilt(p, w, goal)
    assert np.allclose((q * w[:, None]).sum(axis=0), goal, atol=1e-8) and np.allclose(q.sum(axis=1), 1)


def test_mrp_recovers_age_effect_and_matches_national():
    population, train = people(3000, 1), answers(people(3000, 2))
    model = MRP(population).fit(train, OPTIONS)
    goal = np.array([0.45, 0.4, 0.15])
    post = model.postratify(goal)
    assert post[(("grupo_edad", "75+"),)][0] > post[(("grupo_edad", "18-24"),)][0] + 0.3
    everyone = model.postratify(goal, groups=[()])
    assert np.allclose(everyone[()], goal, atol=1e-6)  # el nacional postestratificado es el de entrada
    cross = model.postratify(goal, groups=[(("sexo", "mujer"), ("grupo_edad", "75+"))])
    assert cross[(("sexo", "mujer"), ("grupo_edad", "75+"))].sum() == pytest.approx(1)
    assert model.C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0) and model.omitted == []


def test_dimension_without_valid_values_is_omitted_not_imputed():
    population, train = people(1500, 1), answers(people(1500, 2, income_valid=False))
    model = MRP(population).fit(train, OPTIONS)
    assert model.omitted == ["ingresos_hogar_tramo"] and "ingresos_hogar_tramo" not in model.used
    assert not any(c.startswith("ingresos_hogar_tramo") for c in model.design.columns)


class FakeAnchors:
    def __init__(self, frame):
        self.frame = frame

    def _frame(self, study, evidence):
        return self.frame


@pytest.fixture
def resources(tmp_path, monkeypatch):
    monkeypatch.setattr(mrp_service, "CACHE_DIR", tmp_path / "mrp")
    population, train = people(2000, 1), answers(people(1500, 2))
    spec = {"study_id": "1", "evidence_id": "CIS1_X", "mapping": {o: o for o in OPTIONS}, "question_type": "single"}
    res = types.SimpleNamespace(anchors=FakeAnchors(train), population=types.SimpleNamespace(frame=population))
    return res, spec, train


def test_normalize_cross_accepts_aliases_and_rejects_values_outside_the_schema():
    schema = {"sexo": SEX, "grupo_edad": AGE, "ccaa": CCAA}
    items = [{"var": "Sexo", "val": "Mujer"}, {"var": "edad", "val": "18-24"}, {"var": "ccaa", "val": "andalucía"}]
    assert mrp_service.normalize_cross(items, schema) == (("sexo", "mujer"), ("grupo_edad", "18-24"), ("ccaa", "Andalucía"))
    for bad in ([{"var": "edad", "val": "18-34"}], [{"var": "religion", "val": "x"}], [{"var": "sexo", "val": "mujer"}, {"var": "sexo", "val": "hombre"}], []):
        with pytest.raises(mrp_service.PlanError):
            mrp_service.normalize_cross(bad, schema)


def test_cross_estimates_have_sample_size_small_warning_and_cache(resources):
    res, spec, train = resources
    national = {"Sí": 0.5, "No": 0.4, "Depende": 0.1}
    big = (("sexo", "mujer"),)
    small = (("sexo", "mujer"), ("grupo_edad", "75+"), ("ccaa", "Galicia"), ("nivel_estudios", "superiores"))
    items, info = mrp_service.cross_estimates(res, spec, OPTIONS, national, [big, small])
    assert info["cache"] is False and [i["small"] for i in items] == [False, True]
    expected = int(train[train["sexo"] == "mujer"]["respondent_uid"].nunique())
    assert items[0]["n_cis"] == expected and items[1]["n_cis"] < mrp_service.SMALL_CROSS
    assert sum(items[0]["distribution"].values()) == pytest.approx(1)
    again, info2 = mrp_service.cross_estimates(res, spec, OPTIONS, national, [big, small])
    assert info2["cache"] is True and again[0]["distribution"] == pytest.approx(items[0]["distribution"])  # la caché no cambia las cifras


def test_service_crosses_only_in_direct_mode_and_with_clear_reasons(resources):
    res, spec, _ = resources
    national = {"Sí": 0.5, "No": 0.4, "Depende": 0.1}
    cross = [[{"var": "sexo", "val": "mujer"}, {"var": "edad", "val": "75+"}]]
    items, note, _ = svc._crosses(res, OPTIONS, "direct", spec, national, lambda p: 0.1, cross, False)
    assert len(items) == 1 and items[0]["method"] == "MRP" and items[0]["interval"] and note is None
    none, note, _ = svc._crosses(res, OPTIONS, "none", None, national, lambda p: 0.1, cross, False)
    assert none == [] and "modo directo" in note
    bad, _, warnings = svc._crosses(res, OPTIONS, "direct", spec, national, lambda p: 0.1, [[{"var": "edad", "val": "18-34"}]], False)
    assert bad == [] and "fuera del esquema" in warnings[0]


class PlanLLM:
    def __init__(self, plan=None, delay=0.0, error=None):
        self.plan, self.delay, self.error = plan, delay, error

    async def call(self, schema, payload, **kwargs):
        assert schema is QueryPlan and payload["esquema"] and payload["instrucciones"]
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.plan


SCHEMA = {"sexo": SEX, "grupo_edad": AGE, "ccaa": CCAA}


def plan(**kw):
    base = {"opciones": ["Sí", "No"], "tipo": "si_no", "tema": "vivienda", "subgrupo_o_cruce": [], "respondible": True, "motivo": ""}
    return QueryPlan(**{**base, **kw})


def test_plan_question_valid_and_fallbacks():
    cross = [PlanItem(var="sexo", val="Mujer"), PlanItem(var="edad", val="18-24"), PlanItem(var="ccaa", val="Andalucía")]
    ok = assist.plan_question("¿q?", PlanLLM(plan(subgrupo_o_cruce=cross)), SCHEMA)
    assert ok["ok"] and ok["plan"]["cruce"] == [{"var": "sexo", "val": "mujer"}, {"var": "grupo_edad", "val": "18-24"}, {"var": "ccaa", "val": "Andalucía"}]
    assert not assist.plan_question("¿q?", PlanLLM(plan(subgrupo_o_cruce=[PlanItem(var="edad", val="18-34")])), SCHEMA)["ok"]  # valor fuera del esquema -> reglas
    assert "tema" in assist.plan_question("¿q?", PlanLLM(plan(tema="astrologia")), SCHEMA)["reason"]
    assert not assist.plan_question("¿q?", PlanLLM(plan(opciones=["Sí"])), SCHEMA)["ok"]
    assert "falló" in assist.plan_question("¿q?", PlanLLM(error=RuntimeError("x")), SCHEMA)["reason"]
    assert "tardó" in assist.plan_question("¿q?", PlanLLM(plan(), delay=1.0), SCHEMA, timeout=0.1)["reason"]
    assert assist.plan_question("¿q?", None, SCHEMA)["reason"] == "sin modelo de lenguaje"


def test_plan_with_the_mock_provider(tmp_path):
    from mavericks.llm import LLMService, MockLLMProvider
    from mavericks.schemas import Settings

    service = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "cache")
    out = assist.plan_question("¿Le preocupa la IA?", service, SCHEMA)
    assert out["ok"] and out["plan"]["tipo"] == "escala" and out["plan"]["cruce"] == []


def result_with_cross(small=False):
    dist = {"Sí": 0.6, "No": 0.4}
    return {
        "question": "¿Está a favor?", "options": ["Sí", "No"], "mode": "direct", "mode_label": "directo",
        "national": {"distribution": dist, "interval_core": {"Sí": [0.5, 0.7], "No": [0.3, 0.5]}},
        "confidence": {"typical_error": 0.05, "backtest_cases": 10}, "prior_usado": {"metodo": "ancla_directa"},
        "subgroups": {"sexo": {"label": "Sexo", "groups": {"mujer": {"distribution": {"Sí": 0.7, "No": 0.3}}, "hombre": {"distribution": {"Sí": 0.5, "No": 0.5}}}}},
        "subgroups_method": {"clave": "mrp", "etiqueta": "MRP (modelo multinivel con los microdatos de la oleada de la pregunta)"},
        "crosses": [{"label": "Sexo: mujer · Edad: 18-24", "distribution": {"Sí": 0.75, "No": 0.25}, "n_cis": 30 if small else 400, "small": small, "method": "MRP", "interval": None}],
        "cross_experimental": svc.CROSS_EXPERIMENTAL, "evidence": [],
    }


def test_explanation_names_the_cross_and_method_and_the_number_check_allows_it():
    result = result_with_cross(small=True)
    text = explain.template_text(result)
    assert "Sexo: mujer · Edad: 18-24" in text and "30 encuestados" in text and "75 %" in text and "MRP" in text and "cruce experimental" in text
    payload = explain.llm_payload(result)
    payload["_all_categories"] = explain.all_categories(result)
    good = "En el cruce de mujer de 18-24 años (30 encuestados), la opción más probable es Sí con un 75 %, según el MRP."
    assert explain.check_text(good, payload) is None
    assert explain.check_text("En el cruce el apoyo es del 83 %.", payload).startswith("cifra que no cuadra")


def test_ui_shows_experimental_label_and_yellow_warning_for_small_crosses():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st

        from mavericks.ui_result import render_crosses

        render_crosses(st.session_state["r"])

    at = AppTest.from_function(app)
    at.session_state["r"] = result_with_cross(small=True)
    at.run()
    assert "no están validados" in at.info[0].value and "Solo 30 encuestados" in at.warning[0].value
    at.session_state["r"] = result_with_cross(small=False)
    at.run()
    assert not at.warning
    at.session_state["r"] = {"crosses": [], "cross_note": "sin ancla: los cruces no se ofrecen"}
    at.run()
    assert "no se ofrecen" in at.info[0].value
