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


def test_invented_assessment_ids_are_ignored_and_missing_ones_neutral(tmp_path):
    class BadAssessment(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Assessment:
                return Assessment(items=[{"evidence_id": "inventada", "relevance": "direct"}])
            return await super().generate(schema, payload)

    result = _run(tmp_path, BadAssessment(), _resources(tmp_path))
    assert result["execution_metadata"]["status"] == "completed"
    assert "interpret_evidence:repaired" in result["stages"]
    assert {i["relevance"] for i in result["evidence_assessment"]["items"]} == {"indirect"}
    assert "inventada" not in {i["evidence_id"] for i in result["evidence_assessment"]["items"]}
    assert any("Evaluador LLM incompleto" in w for w in result["warnings"])


def test_assessment_exception_still_falls_back(tmp_path):
    class Broken(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Assessment:
                raise RuntimeError("caído")
            return await super().generate(schema, payload)

    result = _run(tmp_path, Broken(), _resources(tmp_path))
    assert result["execution_metadata"]["status"] == "completed"
    assert "interpret_evidence:fallback" in result["stages"]


def test_deterministic_assessment_never_calls_the_llm(tmp_path):
    class Spy(MockLLMProvider):
        schemas = []

        async def generate(self, schema, payload, temperature=None):
            self.schemas.append(schema)
            return await super().generate(schema, payload)

    population, evidence = _resources(tmp_path)
    settings = Settings(llm_provider="mock", llm_retries=0, llm_assessment=False)
    result = asyncio.run(
        simulate(
            Question(question="¿Usas inteligencia artificial?", options=["Sí", "No"]),
            population,
            evidence,
            LLMService(Spy(), settings, tmp_path / "c"),
            max_segments=3,
        )
    )
    assert Assessment not in Spy.schemas and "interpret_evidence:deterministic" in result["stages"]
    assert result["execution_metadata"]["status"] == "completed"


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


def test_strip_think_blocks():
    from mavericks.llm import strip_think

    assert strip_think('<think>razono\n\nmucho</think>{"a": 1}') == '{"a": 1}'
    assert strip_think('{"a": 1}<think>sin cerrar') == '{"a": 1}'
    assert strip_think('{"a": 1}') == '{"a": 1}'


def test_provider_recovers_json_after_think_block(monkeypatch):
    import langchain_ollama

    import mavericks.llm as module
    from mavericks.llm import StructuredProvider

    monkeypatch.setattr(module, "check_ollama", lambda settings: {"available": True})

    class Raw:
        content = '<think>pensando…</think>{"summary": "Limpio"}'

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, method, include_raw):
            return self

        async def ainvoke(self, messages):
            return {"parsed": None, "parsing_error": ValueError("json"), "raw": Raw()}

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat)
    value = asyncio.run(StructuredProvider(Settings()).generate(Narrative, {}))
    assert value.summary == "Limpio"
