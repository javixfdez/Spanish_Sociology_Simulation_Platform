import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest

from mavericks import measured, mrp_service, synthetic
from mavericks import service as svc
from tests.test_mrp import OPTIONS, answers, people

REAL = Path("data/processed/cis_responses.parquet").exists() and Path("data/processed_v2/evidence_v2.sqlite").exists() and Path("data/examples/ejemplos_demo.json").exists()


class FakeAnchors:
    def __init__(self, frame):
        self.frame = frame

    def _frame(self, study, evidence):
        return self.frame


@pytest.fixture
def resources(tmp_path, monkeypatch):
    monkeypatch.setattr(mrp_service, "CACHE_DIR", tmp_path / "mrp")
    population = people(30000, 1)  # con 50.000 personas el error de muestreo es ≈ 0,2 pp; con 30.000, ≈ 0,3 pp
    return types.SimpleNamespace(anchors=FakeAnchors(answers(people(2000, 2))), population=types.SimpleNamespace(frame=population))


def result_for(options, national, method, subgroups=None, anchor=None):
    return {"options": options, "national": {"distribution": dict(zip(options, national, strict=True))}, "subgroups_method": {"clave": method}, "subgroups": subgroups or {}, "anchor": anchor}


def test_mrp_probabilities_aggregate_to_the_national_and_simulation_is_reproducible(resources):
    national = [0.5, 0.35, 0.15]
    anchor = {"study_id": "1", "evidence_id": "CIS1_X", "mapping": {o: o for o in OPTIONS}}
    result = result_for(OPTIONS, national, "mrp", anchor=anchor)
    snapshot = copy.deepcopy(result)
    frame, method = synthetic.population_frame(resources, result)
    assert method == "MRP" and len(frame) == 30000 and result == snapshot  # no cambia ninguna cifra de la predicción
    q = frame[[f"p({o})" for o in OPTIONS]].to_numpy()
    assert np.allclose(q.sum(axis=1), 1, atol=1e-5)
    w = frame["peso"].to_numpy()
    assert np.allclose((q * w[:, None]).sum(axis=0) / w.sum(), national, atol=2e-4)
    shares = synthetic.aggregate(frame, OPTIONS)
    assert all(abs(shares[o] - p) <= 0.01 for o, p in zip(OPTIONS, national, strict=True))  # ± 1 pp
    again, _ = synthetic.population_frame(resources, result)
    assert (again["respuesta_simulada"] == frame["respuesta_simulada"]).all()  # semilla fija


def test_product_of_group_distributions_is_recalibrated_and_does_not_touch_the_result(resources):
    national = [0.5, 0.3, 0.2]
    groups = {"sexo": {"label": "Sexo", "groups": {"hombre": {"distribution": {"Sí": 0.6, "No": 0.25, "Depende": 0.15}}, "mujer": {"distribution": {"Sí": 0.4, "No": 0.35, "Depende": 0.25}}}},
              "grupo_edad": {"label": "Edad", "groups": {"75+": {"distribution": {"Sí": 0.8, "No": 0.1, "Depende": 0.1}}, "18-24": {"distribution": {"Sí": 0.2, "No": 0.5, "Depende": 0.3}}}}}
    result = result_for(OPTIONS, national, "analogas", subgroups=groups)
    snapshot = copy.deepcopy(result)
    frame, method = synthetic.population_frame(resources, result)
    assert method.startswith("producto") and result == snapshot
    shares = synthetic.aggregate(frame, OPTIONS)
    assert all(abs(shares[o] - p) <= 0.01 for o, p in zip(OPTIONS, national, strict=True))
    old = frame[frame["grupo_edad"] == "75+"]
    young = frame[frame["grupo_edad"] == "18-24"]
    assert old["p(Sí)"].mean() > young["p(Sí)"].mean() + 0.2  # las diferencias entre grupos se conservan


def test_sample_has_200_people_and_csv_has_all(resources):
    result = result_for(OPTIONS, [0.5, 0.3, 0.2], "ninguno")
    frame, _ = synthetic.population_frame(resources, result)
    assert len(synthetic.sample(frame)) == 200 and synthetic.to_csv(frame).count(b"\n") == len(frame) + 1


def test_age_warning_after_12_months_from_the_data_cutoff():
    assert measured.age_warning("2025-06-30") is None  # 12 meses exactos: sin aviso
    assert measured.age_warning("2026-03-06") is None
    text = measured.age_warning("2025-05-15")
    assert text == "Dato de mayo de 2025: la opinión puede haber cambiado desde entonces."
    assert measured.age_warning("2024-11-14").startswith("Dato de noviembre de 2024")
    assert measured.age_warning(None) is None


def test_ui_shows_the_age_warning_in_the_measured_block():
    from streamlit.testing.v1 import AppTest

    from tests.test_measured import result

    def app():
        import streamlit as st

        from mavericks.ui_result import render_measured

        render_measured(st.session_state["r"])

    r = result()
    r["measured"]["age_warning"] = measured.age_warning("2024-11-14")
    at = AppTest.from_function(app)
    at.session_state["r"] = r
    at.run()
    assert any("Dato de noviembre de 2024" in w.value for w in at.warning)


@pytest.mark.skipif(not REAL, reason="necesita los datos reales")
def test_real_questions_recover_the_national_with_every_method_and_predictions_do_not_change():
    resources = svc.load_resources()
    items = json.loads(Path("data/examples/ejemplos_demo.json").read_text(encoding="utf-8"))["items"]
    chosen = [e for e in items if e["mode"] == "direct"][:2] + [e for e in items if e["mode"] == "prior"][:1] + [e for e in items if e["mode"] == "none"][:1]
    assert len(chosen) == 4
    for item in chosen:
        out = svc.predict(item["question"], item["options"], preset="rapido_sin_llm", resources=resources)
        snapshot = copy.deepcopy(out)
        frame, method = synthetic.population_frame(resources, out)
        assert len(frame) == 50_000 and out == snapshot
        shares = synthetic.aggregate(frame, out["options"])
        assert max(abs(shares[o] - out["national"]["distribution"][o]) for o in out["options"]) <= 0.01, (item["question"][:50], method)
        assert (method == "MRP") == (out["mode"] == "direct")
