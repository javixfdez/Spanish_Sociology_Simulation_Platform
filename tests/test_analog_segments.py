import numpy as np
import pytest

from mavericks.analog import AnalogConfig, combine
from mavericks.analog_segments import predict, rake, segment_priors
from mavericks.population import PopulationLoader, PopulationRepository, create_demo


def test_rake_reproduces_the_target_weighted_mean_and_keeps_rows_normalised():
    dists = {"a": {"Sí": 0.8, "No": 0.2}, "b": {"Sí": 0.4, "No": 0.6}, "c": {"Sí": 0.5, "No": 0.5}}
    weights = {"a": 1.0, "b": 3.0, "c": 2.0}
    target = {"Sí": 0.7, "No": 0.3}
    out = rake(dists, weights, target)
    total = sum(weights.values())
    mean = sum(weights[i] * out[i]["Sí"] for i in out) / total
    assert mean == pytest.approx(0.7, abs=1e-6)
    assert all(sum(v.values()) == pytest.approx(1) for v in out.values())
    assert out["a"]["Sí"] > out["b"]["Sí"]  # conserva el orden relativo entre segmentos


class FakeAnchors:
    def segment_distribution(self, spec, constraints):
        young = "18-34" in constraints.get("grupo_edad", [])
        p = 0.9 if young else 0.2
        return {"probabilities": {"Sí": p, "No": 1 - p}}


def _result():
    cand = {"evidence_id": "Q", "source": "CIS", "date": None, "lex": 0.9, "emb": None, "kind": "label",
            "dist": {"Sí": 0.5, "No": 0.5}, "study_id": "1", "mapping": {"Sí": "Sí", "No": "No"}, "microdata": True}
    return combine([cand], ["Sí", "No"], AnalogConfig(shrink=0.0, min_sim=0.0))


def test_segment_priors_use_analog_microdata_and_fall_back_to_national(tmp_path):
    population = PopulationLoader().load(create_demo(tmp_path / "fx"))
    repo = PopulationRepository(*population)
    from mavericks.population import build_segments

    segments = build_segments(repo, 3, dimensions=["grupo_edad"])
    result = _result()
    priors = segment_priors(result, ["Sí", "No"], segments, FakeAnchors())
    young = next(s for s in segments if s["demographic_profile"].get("grupo_edad") == "18-34")
    assert priors[young["segment_id"]]["Sí"] == pytest.approx(0.9)
    assert min(p["Sí"] for p in priors.values()) == pytest.approx(0.2)
    national = segment_priors(result, ["Sí", "No"], segments, None)  # sin microdatos: la distribución nacional
    assert all(p["Sí"] == pytest.approx(0.5) for p in national.values())


def test_predict_reproduces_the_total_and_flat_mode_has_no_group_differences(tmp_path):
    repo = PopulationRepository(*PopulationLoader().load(create_demo(tmp_path / "fx")))
    result = _result()
    out = predict(result, ["Sí", "No"], repo, FakeAnchors(), n_segments=3, dimensions=["grupo_edad"])
    assert out["national"]["Sí"] == pytest.approx(result["prior"]["Sí"], abs=1e-6)  # raking
    groups = out["subgroups"]["grupo_edad"]
    assert max(g["Sí"] for g in groups.values()) - min(g["Sí"] for g in groups.values()) > 0.3
    flat = predict(result, ["Sí", "No"], repo, None, n_segments=3, dimensions=["grupo_edad"], deviations=False)
    assert {round(g["Sí"], 6) for g in flat["subgroups"]["grupo_edad"].values()} == {0.5}
    assert np.isclose(sum(flat["national"].values()), 1)
