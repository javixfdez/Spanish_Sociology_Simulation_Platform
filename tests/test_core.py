import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from mavericks.evaluation import backtest
from mavericks.evidence import EvidenceRepository, survey_distribution
from mavericks.graph import merge_unique, simulate
from mavericks.llm import LLMService, MockLLMProvider, check_ollama
from mavericks.population import (
    PopulationLoader,
    PopulationRepository,
    PopulationValidator,
    build_segments,
    create_demo,
)
from mavericks.schemas import Prediction, Question, Settings
from mavericks.statistics import aggregate, calibrated_intervals, js_divergence, mae


@pytest.fixture
def resources(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    population = PopulationRepository(frame, metadata)
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    return population, evidence


def service(tmp_path, concurrency=1, provider=None):
    return LLMService(
        provider or MockLLMProvider(),
        Settings(llm_provider="mock", llm_retries=0, llm_max_concurrency=concurrency),
        tmp_path / "cache",
    )


def test_population_load_and_optional_columns(resources):
    population, _ = resources
    assert len(population.frame) == 120
    assert "ccaa" not in population.frame
    assert population.metadata.synthetic


@pytest.mark.parametrize(
    "column,value",
    [("edad", 17), ("edad", np.inf), ("peso", 0), ("peso", -1), ("peso", np.nan), ("peso", np.inf)],
)
def test_population_invalid(resources, column, value):
    population, _ = resources
    frame = population.frame.copy()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        PopulationValidator().validate(frame, population.metadata)


def test_population_duplicate_version_categories(resources):
    population, _ = resources
    frame = population.frame.copy()
    frame.loc[0, "person_id"] = frame.loc[1, "person_id"]
    with pytest.raises(ValueError):
        PopulationRepository(frame, population.metadata)
    metadata = population.metadata.model_copy(update={"population_version": "wrong"})
    with pytest.raises(ValueError):
        PopulationRepository(population.frame, metadata)
    metadata = population.metadata.model_copy(update={"categories": {"sexo": ["invalid"]}})
    with pytest.raises(ValueError):
        PopulationRepository(population.frame, metadata)


def test_segments_preserve_weights(resources):
    population, _ = resources
    for limit in [1, 3, 5, 24]:
        segments = build_segments(population, limit)
        assert len(segments) <= limit
        assert sum(s["population_weight"] for s in segments) == pytest.approx(population.frame.peso.sum())
        assert "socioeconomic_profile" in segments[0]


def test_aggregation_weighted():
    segments = [{"segment_id": "a", "population_weight": 1}, {"segment_id": "b", "population_weight": 3}]
    predictions = {
        "a": Prediction(segment_id="a", probabilities={"Y": 1, "N": 0}, explanation="x"),
        "b": Prediction(segment_id="b", probabilities={"Y": 0, "N": 1}, explanation="x"),
    }
    assert aggregate(segments, predictions, ["Y", "N"]) == {"Y": 0.25, "N": 0.75}
    with pytest.raises(ValueError):
        aggregate(segments, {"a": predictions["a"]}, ["Y", "N"])


@pytest.mark.parametrize("probabilities", [{"Y": -0.1, "N": 1.1}, {"Y": np.nan}, {"Y": 0.2, "N": 0.2}])
def test_invalid_predictions(probabilities):
    with pytest.raises(ValueError):
        Prediction(segment_id="a", probabilities=probabilities, explanation="x").validate_context(
            "a", ["Y", "N"], []
        )


def test_unknown_evidence():
    value = Prediction(segment_id="a", probabilities={"Y": 1, "N": 0}, explanation="x", evidence_ids=["fake"])
    with pytest.raises(ValueError):
        value.validate_context("a", ["Y", "N"], [])


@pytest.mark.parametrize("concurrency", [1, 3])
def test_graph_all_stages(resources, tmp_path, concurrency):
    population, evidence = resources
    result = asyncio.run(
        simulate(
            Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"]),
            population,
            evidence,
            service(tmp_path, concurrency),
            max_segments=5,
        )
    )
    assert result["execution_metadata"]["status"] == "completed"
    assert result["execution_metadata"]["coverage"] == 1
    assert result["execution_metadata"]["synthetic"]
    assert sum(result["national_distribution"].values()) == pytest.approx(1)
    for stage in [
        "validate_input",
        "methodologist_agent",
        "retrieve_sociological_evidence",
        "interpret_evidence",
        "segment_population",
        "dispatch_segment_agents",
        "aggregate_predictions",
        "audit_confidence",
        "analyst_agent",
    ]:
        assert stage in result["stages"]
    assert len(result["segment_predictions"]) == len(result["segments"])
    assert result["trust_metrics"]["predictive_intervals"] is None
    assert "ccaa" not in result["subgroup_distributions"]


def test_failed_agent_aborts(resources, tmp_path):
    class Broken(MockLLMProvider):
        async def generate(self, schema, payload):
            if schema is Prediction and payload["segment"]["segment_id"] == "S001":
                raise RuntimeError("Intentional test failure")
            return await super().generate(schema, payload)

    population, evidence = resources
    result = asyncio.run(
        simulate(
            Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"]),
            population,
            evidence,
            service(tmp_path, provider=Broken()),
        )
    )
    assert result["errors"][0]["segment_id"] == "S001"
    assert result["national_distribution"] == {}
    assert result["execution_metadata"]["coverage"] < 1
    assert "analyst_agent" not in result["stages"]


def test_reducer_no_duplicates():
    assert merge_unique({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        merge_unique({"a": 1}, {"a": 2})


def test_rag_idempotent(resources, tmp_path):
    _, evidence = resources
    evidence.ingest(tmp_path / "fixtures/evidence.json")
    assert len(evidence.all()) == 1
    assert evidence.search("inteligencia artificial")[0]["evidence"]["evidence_id"] == "DEMO_AI"
    assert evidence.search("inteligencia", excluded_ids=["DEMO_AI"]) == []
    assert evidence.search("inteligencia", cutoff="2020-01-01") == []


def test_survey_weight_and_study_isolation():
    respondents = pd.DataFrame(
        {
            "estudio_id": ["a", "a", "b"],
            "respondent_uid": ["1", "2", "1"],
            "peso": [1.0, 3.0, 900.0],
            "peso_ccaa": [3.0, 1.0, 900.0],
        }
    )
    responses = pd.DataFrame(
        {
            "estudio_id": ["a", "a"],
            "respondent_uid": ["1", "2"],
            "question_uid": ["Q", "Q"],
            "answer": ["Sí", "NS/NC"],
        }
    )
    national = survey_distribution(respondents, responses, "Q")["national"]
    assert national["distribution"] == {"NS/NC": 0.75, "Sí": 0.25}
    assert national["n_eff"] == pytest.approx(1.6)
    regional = survey_distribution(respondents, responses, "Q", weight="peso_ccaa")["national"]
    assert regional["distribution"]["Sí"] == 0.75
    responses.loc[0, "estudio_id"] = "missing"
    with pytest.raises(ValueError):
        survey_distribution(respondents, responses, "Q")


def test_multiple_response():
    respondents = pd.DataFrame({"estudio_id": ["a", "a"], "respondent_uid": ["1", "2"], "peso": [1, 3]})
    responses = pd.DataFrame(
        {
            "estudio_id": ["a"] * 4,
            "respondent_uid": ["1", "1", "2", "2"],
            "question_uid": ["Q"] * 4,
            "answer": ["A", "B", "A", "B"],
            "selected": [1, 1, 1, 1],
        }
    )
    result = survey_distribution(respondents, responses, "Q", question_type="multiple")["national"]
    assert sum(result["distribution"].values()) == 2
    assert result["n_respondents"] == 2


def test_backtest_holds_out(resources, tmp_path):
    population, evidence = resources
    cases = [
        {
            "evidence_id": "DEMO_AI",
            "question": "¿Utiliza aplicaciones de inteligencia artificial?",
            "options": ["Sí", "No"],
            "observed_distribution": {"Sí": 0.5, "No": 0.5},
            "fieldwork_date": "2020-01-01",
        }
    ]
    result = asyncio.run(backtest(cases, population, evidence, lambda: service(tmp_path)))
    assert result[0]["retrieved_ids"] == []
    assert result[0]["synthetic"]
    assert result[0]["mae"] is not None


def test_ollama_unavailable(monkeypatch):
    import httpx

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", unavailable)
    with pytest.raises(RuntimeError, match="Ollama no accesible"):
        check_ollama(Settings())


def test_metrics_and_uncertainty():
    assert mae({"A": 0.2, "B": 0.8}, {"A": 0.4, "B": 0.6}) == pytest.approx(0.2)
    assert js_divergence({"A": 1, "B": 0}, {"A": 0, "B": 1}) == pytest.approx(1)
    assert calibrated_intervals({"A": 0.5}, []) is None
    assert calibrated_intervals({"A": 0.5}, [0.1] * 20) == {"A": [0.4, 0.6]}


def test_cache(resources, tmp_path):
    population, evidence = resources
    llm = service(tmp_path)
    q = Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"])
    asyncio.run(simulate(q, population, evidence, llm))
    llm2 = service(tmp_path)
    asyncio.run(simulate(q, population, evidence, llm2))
    assert all(call["cache_hit"] for call in llm2.calls)


def test_microdata_attached_to_rag(resources, tmp_path):
    _, evidence = resources
    respondents = pd.DataFrame(
        {
            "estudio_id": ["DEMO", "DEMO"],
            "respondent_uid": ["1", "2"],
            "peso": [1, 3],
            "sexo": ["mujer", "hombre"],
        }
    )
    responses = pd.DataFrame(
        {
            "estudio_id": ["DEMO", "DEMO"],
            "respondent_uid": ["1", "2"],
            "question_uid": ["DEMO_AI", "DEMO_AI"],
            "answer": ["Sí", "No"],
        }
    )
    r, a = tmp_path / "respondents.parquet", tmp_path / "responses.csv"
    respondents.to_parquet(r)
    responses.to_csv(a, index=False)
    assert evidence.attach_microdata(r, a) == 1
    hit = evidence.search("inteligencia artificial")[0]["evidence"]
    assert hit["metadata"]["empirical_statistics"]["national"]["distribution"]["Sí"] == 0.25
    assert hit["sample_size"] == 2


def test_retry_timeout_and_context_validation(tmp_path):
    class InvalidOnce(MockLLMProvider):
        attempts = 0

        async def generate(self, schema, payload):
            self.attempts += 1
            if self.attempts == 1:
                return Prediction(segment_id="wrong", probabilities={"A": 1, "B": 0}, explanation="x")
            return Prediction(segment_id="correct", probabilities={"A": 1, "B": 0}, explanation="x")

    provider = InvalidOnce()
    llm = LLMService(provider, Settings(llm_provider="mock", llm_retries=1), tmp_path / "retry")
    result = asyncio.run(llm.call(Prediction, {}, lambda p: p.validate_context("correct", ["A", "B"], [])))
    assert result.segment_id == "correct"
    assert provider.attempts == 2

    class Slow(MockLLMProvider):
        async def generate(self, schema, payload):
            await asyncio.sleep(0.1)

    llm = LLMService(
        Slow(), Settings(llm_provider="mock", llm_retries=0, llm_timeout=0.001), tmp_path / "slow"
    )
    with pytest.raises(TimeoutError):
        asyncio.run(llm.call(Prediction, {}))


def test_ollama_structured_output_without_server(monkeypatch):
    import langchain_ollama

    import mavericks.llm as module
    from mavericks.llm import StructuredProvider
    from mavericks.schemas import Narrative

    monkeypatch.setattr(module, "check_ollama", lambda settings: {"available": True})

    class FakeChat:
        def __init__(self, **kwargs):
            assert kwargs["reasoning"] is False

        def with_structured_output(self, schema, method, include_raw):
            assert method == "json_schema" and include_raw
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            assert messages[0][0] == "system"
            return {"parsed": self.schema(summary="Test"), "raw": None}

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat)
    provider = StructuredProvider(Settings())
    assert asyncio.run(provider.generate(Narrative, {})).summary == "Test"


def test_ingest_formats_and_pdf_ocr(tmp_path):
    from pypdf import PdfWriter

    evidence = EvidenceRepository(tmp_path / "db.sqlite")
    record = pd.DataFrame(
        [{"evidence_id": "TEST", "source": "SYNTHETIC", "question_text": "Prueba tecnología"}]
    )
    for suffix in ["csv", "parquet", "xlsx"]:
        path = tmp_path / f"input.{suffix}"
        if suffix == "csv":
            record.to_csv(path, index=False)
        elif suffix == "parquet":
            record.to_parquet(path)
        else:
            record.to_excel(path, index=False)
        assert evidence.ingest(path) == 1
    assert len(evidence.all()) == 1
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / "scan.pdf"
    with path.open("wb") as stream:
        writer.write(stream)
    evidence.ingest(path)
    assert any(e.metadata.get("requires_ocr") for e in evidence.all())


def test_temporal_and_equivalent_exclusions(tmp_path):
    import json

    evidence = EvidenceRepository(tmp_path / "db.sqlite")
    records = [
        {
            "evidence_id": "old",
            "source": "SYNTHETIC",
            "question_text": "inteligencia artificial antigua",
            "fieldwork_date": "2019-01-01",
        },
        {
            "evidence_id": "future",
            "source": "SYNTHETIC",
            "question_text": "inteligencia artificial futura",
            "fieldwork_date": "2021-01-01",
        },
        {
            "evidence_id": "equiv",
            "source": "SYNTHETIC",
            "question_text": "inteligencia artificial parecida",
            "fieldwork_date": "2019-01-01",
            "metadata": {"equivalent_ids": ["held"]},
        },
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    evidence.ingest(path)
    hits = evidence.search("inteligencia artificial", cutoff="2020-01-01", excluded_ids=["held"])
    assert [hit["evidence"]["evidence_id"] for hit in hits] == ["old"]


def test_streamlit_app_smoke():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value.startswith("GenAI Mavericks")


def test_search_cache_refreshes_after_ingest(tmp_path):
    repo = EvidenceRepository(tmp_path / "c.sqlite")
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"evidence_id": "A", "source": "t", "question_text": "vivienda alquiler precio"}]), encoding="utf-8")
    repo.ingest(path)
    assert [h["evidence"]["evidence_id"] for h in repo.search("alquiler")] == ["A"]
    path.write_text(json.dumps([{"evidence_id": "B", "source": "t", "question_text": "alquiler turistico"}]), encoding="utf-8")
    repo.ingest(path)
    assert {h["evidence"]["evidence_id"] for h in repo.search("alquiler")} == {"A", "B"}


def test_sav_is_not_ingested_as_evidence(tmp_path):
    from mavericks.evidence import read_records

    with pytest.raises(ValueError, match="etl-cis"):
        read_records(tmp_path / "estudio.sav")


def test_aggregate_rejects_unknown_evidence_ids():
    segments = [{"segment_id": "a", "population_weight": 1}]
    prediction = Prediction(segment_id="a", probabilities={"Y": 1, "N": 0}, explanation="x", evidence_ids=["fake"])
    with pytest.raises(ValueError):
        aggregate(segments, {"a": prediction}, ["Y", "N"], evidence_ids=["real"])
    assert aggregate(segments, {"a": prediction}, ["Y", "N"], evidence_ids=["fake"]) == {"Y": 1.0, "N": 0.0}
