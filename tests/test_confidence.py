import asyncio

import pytest

from mavericks.evaluation import backtest
from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Prediction, Question, Settings
from mavericks.statistics import (
    audit,
    build_calibration,
    load_calibration,
    pairwise_divergence,
    select_stratum,
)

QUESTION = Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"])


@pytest.fixture
def world(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, meta = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "e.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    return PopulationRepository(frame, meta), evidence


def _service(tmp_path, provider, runs=1):
    settings = Settings(llm_provider="mock", llm_retries=0, stability_runs=runs)
    return LLMService(provider, settings, tmp_path / "cache")


class Jittery(MockLLMProvider):
    """Respuesta distinta en cada repetición (payload['sample'])."""

    async def generate(self, schema, payload, temperature=None):
        if schema is Prediction:
            p = 0.3 + 0.1 * payload.get("sample", 0)
            return Prediction(
                segment_id=payload["segment"]["segment_id"], probabilities={"Sí": p, "No": 1 - p}, explanation="x"
            )
        return await super().generate(schema, payload)


def test_stability_runs_measure_divergence(tmp_path, world):
    population, evidence = world
    result = asyncio.run(simulate(QUESTION, population, evidence, _service(tmp_path, Jittery(), runs=3), 4))
    assert result["execution_metadata"]["status"] == "completed"
    first = next(iter(result["segment_predictions"].values()))
    assert first["stability"]["runs"] == 3 and len(first["repetitions"]) == 3
    assert first["probabilities"]["Sí"] == pytest.approx(0.4)  # media de 0.3, 0.4, 0.5
    stability = result["trust_metrics"]["prediction_stability"]
    assert stability["measured"] and stability["runs"] == 3
    assert stability["js_divergence"] > 0 and stability["national_divergence"] > 0


def test_single_run_leaves_stability_unmeasured(tmp_path, world):
    population, evidence = world
    result = asyncio.run(simulate(QUESTION, population, evidence, _service(tmp_path, MockLLMProvider()), 4))
    assert result["trust_metrics"]["prediction_stability"]["measured"] is False
    assert result["trust_metrics"]["reliability"] == "Fiabilidad predictiva todavía no calibrada"


def test_pairwise_divergence():
    same = {"A": 0.5, "B": 0.5}
    assert pairwise_divergence([same, same]) == pytest.approx(0)
    assert pairwise_divergence([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}]) == pytest.approx(1)
    assert pairwise_divergence([same]) is None


def _result(mode, errors, mae_value=0.05, topic="general"):
    return {
        "status": "completed",
        "anchor_mode": mode,
        "topic": topic,
        "mae": mae_value,
        "mae_uniform_baseline": 0.2,
        "option_errors": errors,
        "subgroup_option_errors": [],
        "synthetic": False,
    }


def test_calibration_strata_and_intervals():
    results = [_result("direct", [0.02, 0.03]) for _ in range(25)] + [_result("none", [0.1, 0.12]) for _ in range(5)]
    calibration = build_calibration(results, model="m", population_version="v1")
    assert calibration["overall"]["n_cases"] == 30
    name, block = select_stratum(calibration, "direct")
    assert "direct" in name and block["n_cases"] == 25
    name, block = select_stratum(calibration, "none")  # solo 5 casos: cae al global
    assert name == "global"
    assert select_stratum(None, "direct") == (None, None)


class _Meta:
    population_version = "v1"
    variable_provenance = {}


def test_audit_calibrated_vs_uncalibrated():
    results = [_result("direct", [0.02, 0.03]) for _ in range(25)]
    calibration = build_calibration(results, model="m", population_version="v1")
    national = {"Sí": 0.4, "No": 0.6}
    trust = audit([], _Meta(), {}, calibration=calibration, national=national, anchor_mode="direct", model="m")
    assert trust["predictive_intervals"]["radius"] == pytest.approx(0.03)
    assert trust["predictive_intervals"]["intervals"]["Sí"] == pytest.approx([0.37, 0.43])
    assert trust["historical_accuracy"]["calibrated"] and trust["confidence"]["level"] == "alta"
    assert "Calibrada con 25 casos" in trust["reliability"]
    other = audit([], _Meta(), {}, calibration=calibration, national=national, anchor_mode="direct", model="otro")
    assert any("otro modelo" in x for x in other["limitations"])
    few = build_calibration(results[:5], model="m")
    weak = audit([], _Meta(), {}, calibration=few, national=national, anchor_mode="direct")
    assert weak["predictive_intervals"] is None and any("insuficiente" in x for x in weak["limitations"])
    assert audit([], _Meta(), {})["predictive_intervals"] is None


def test_backtest_results_feed_calibration(tmp_path, world):
    population, evidence = world
    case = {
        "evidence_id": "X",
        "question": QUESTION.question,
        "options": ["Sí", "No"],
        "fieldwork_date": "2026-01-01",
        "observed_distribution": {"Sí": 0.4, "No": 0.6},
        "allow_previous_waves": True,
    }
    results = asyncio.run(
        backtest([case], population, evidence, lambda: _service(tmp_path, MockLLMProvider()), 4)
    )
    assert len(results[0]["option_errors"]) == 2
    calibration = build_calibration(results, model="mock")
    assert calibration["overall"]["n_cases"] == 1 and calibration["synthetic"]
    path = tmp_path / "cal.json"
    from mavericks.evaluation import save_json

    save_json(path, calibration)
    assert load_calibration(path)["overall"]["n_cases"] == 1
    assert load_calibration(tmp_path / "missing.json") is None
