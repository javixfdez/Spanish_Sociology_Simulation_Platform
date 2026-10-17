import asyncio

from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Assessment, Methodology, Narrative, Prediction, Question, Settings


def _resources(tmp_path, with_evidence=True):
    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    if with_evidence:
        evidence.ingest(path.parent / "evidence.json")
    return PopulationRepository(frame, metadata), evidence


def _service(tmp_path, provider, retries=0):
    settings = Settings(llm_provider="mock", llm_retries=retries)
    return LLMService(provider, settings, tmp_path / "cache")


def _run(tmp_path, provider, repo, retries=0):
    population, evidence = repo
    return asyncio.run(
        simulate(
            Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"]),
            population,
            evidence,
            _service(tmp_path, provider, retries),
            max_segments=5,
        )
    )


def test_no_evidence_skips_assessment_llm(tmp_path):
    class Spy(MockLLMProvider):
        schemas = []

        async def generate(self, schema, payload, temperature=None):
            self.schemas.append(schema)
            return await super().generate(schema, payload)

    result = _run(tmp_path, Spy(), _resources(tmp_path, with_evidence=False))
    assert result["execution_metadata"]["status"] == "completed"
    assert Assessment not in Spy.schemas
    assert result["evidence_assessment"]["items"] == []
    assert any("Sin evidencia" in w for w in result["warnings"])


def test_failing_assessment_does_not_abort(tmp_path):
    class BadAssessment(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Assessment:
                return Assessment(items=[{"evidence_id": "inventada", "relevance": "direct"}])
            return await super().generate(schema, payload)

    result = _run(tmp_path, BadAssessment(), _resources(tmp_path))
    assert result["execution_metadata"]["status"] == "completed"
    assert "interpret_evidence:fallback" in result["stages"]
    assert {i["relevance"] for i in result["evidence_assessment"]["items"]} == {"indirect"}
    assert any("Evaluación automática no disponible" in w for w in result["warnings"])


def test_methodologist_and_analyst_failures_degrade(tmp_path):
    class Broken(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema in (Methodology, Narrative):
                raise RuntimeError("caído")
            return await super().generate(schema, payload)

    result = _run(tmp_path, Broken(), _resources(tmp_path))
    assert result["execution_metadata"]["status"] == "completed"
    assert "methodologist_agent:fallback" in result["stages"]
    assert "analyst_agent:fallback" in result["stages"]
    assert result["methodology"]["topic"] == "ia_tecnologia"
    assert sum(result["national_distribution"].values()) == 1.0 or abs(sum(result["national_distribution"].values()) - 1) < 1e-9


def test_retry_carries_error_and_raises_temperature(tmp_path):
    seen = []

    class Flaky(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            seen.append((payload.get("retry"), temperature))
            if len(seen) == 1:
                return Prediction(segment_id="mal", probabilities={"A": 1, "B": 0}, explanation="x")
            return Prediction(segment_id="bien", probabilities={"A": 1, "B": 0}, explanation="x")

    llm = _service(tmp_path, Flaky(), retries=2)
    value = asyncio.run(llm.call(Prediction, {"x": 1}, lambda p: p.validate_context("bien", ["A", "B"], [])))
    assert value.segment_id == "bien"
    assert seen[0] == (None, None)
    retry, temperature = seen[1]
    assert retry["attempt"] == 1 and "Segmento u opciones incorrectos" in retry["previous_error"]
    assert temperature is not None and temperature > 0
