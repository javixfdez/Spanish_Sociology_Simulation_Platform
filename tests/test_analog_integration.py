import asyncio
import json

import pytest

from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Prediction, Question, Settings

QUESTION = "¿Cree usted que la inteligencia artificial mejorará su vida cotidiana?"


def _world(tmp_path):
    population = PopulationRepository(*PopulationLoader().load(create_demo(tmp_path / "fx")))
    records = [
        dict(evidence_id=f"A{i}", source="CIS 1", study_id=str(i), question_type="single", topic="ia_tecnologia",
             question_text=f"¿Cree usted que la inteligencia artificial mejorará su vida laboral número {i}?", options=["Sí", "No"],
             fieldwork_date="2025-01-01",
             metadata={"empirical_statistics": {"national": {"distribution": {"Sí": 0.8, "No": 0.2}, "n_respondents": 1000}}})
        for i in range(3)
    ]
    path = tmp_path / "a.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    evidence = EvidenceRepository(tmp_path / "e.sqlite")
    evidence.ingest(path)
    return population, evidence


def _run(tmp_path, **settings):
    population, evidence = _world(tmp_path)
    seen = []

    class Spy(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                seen.append(payload)
            return await super().generate(schema, payload)

    calibration = {"blend": {"edges": [0.0, 0.5, 1.01], "params": [[0.5, 1.0], [0.5, 1.0]]}}
    llm = LLMService(Spy(), Settings(llm_provider="mock", llm_retries=0, analog_min_sim=0.2, **settings), tmp_path / "c")
    result = asyncio.run(simulate(Question(question=QUESTION, options=["Sí", "No"]), population, evidence, llm, 3, calibration=calibration))
    return result, seen


def test_analog_off_by_default_changes_nothing(tmp_path):
    result, seen = _run(tmp_path)
    assert seen and all("analog" not in p for p in seen) and all("blend" not in p for p in result["segment_predictions"].values())


def test_analog_prior_and_examples_reach_the_prompt(tmp_path):
    result, seen = _run(tmp_path, analog=True)
    assert seen
    analog = seen[0]["analog"]
    assert analog["prior"]["Sí"] == pytest.approx(0.8) and analog["best_similarity"] > 0.2
    assert 1 <= len(analog["examples"]) <= 3 and analog["examples"][0]["distribution"]["Sí"] == 0.8
    assert result["execution_metadata"]["status"] == "completed"
    assert all("blend" not in p for p in result["segment_predictions"].values())  # sin --blend no se mezcla


def test_blend_mixes_the_llm_with_the_prior_and_records_w_and_alpha(tmp_path):
    result, _ = _run(tmp_path, analog=True, blend=True)
    for prediction in result["segment_predictions"].values():
        info = prediction["blend"]
        assert info["w"] == 0.5 and info["alpha"] == 1.0
        expected = 0.5 * info["llm_mean"]["Sí"] + 0.5 * 0.8
        assert prediction["probabilities"]["Sí"] == pytest.approx(expected)
