from pathlib import Path

import pytest

from mavericks.evidence import EvidenceRepository
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Settings
from mavericks.service import Resources, ServiceError, load_resources, predict

REAL = (
    Path("data/processed/synthetic_population.parquet").exists()
    and Path("data/processed_v2/evidence_v2.sqlite").exists()
)
OPTIONS = ["Mucho", "Bastante", "Poco", "Nada"]
QUESTION = "¿Le preocupa que la inteligencia artificial le quite su empleo?"


@pytest.fixture
def demo(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    return Resources(PopulationRepository(frame, metadata), evidence, None, {}, None)


def test_mock_llm_path(demo, tmp_path):
    llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "cache")
    out = predict(
        "¿Usas inteligencia artificial?",
        ["Sí", "No"],
        preset="llm_completo",
        resources=demo,
        service=llm,
        n_segments=5,
    )
    assert out["mode"] == "none"
    assert sum(out["national"]["distribution"].values()) == pytest.approx(1)
    assert out["national"]["interval"] is None
    assert any("calibración" in w for w in out["warnings"])
    assert out["confidence"]["level"] == "baja"


def test_preset_and_options_errors(demo):
    with pytest.raises(ServiceError, match="Preset"):
        predict(QUESTION, OPTIONS, preset="x", resources=demo)
    with pytest.raises(ServiceError, match="opciones"):
        predict(QUESTION, ["Solo una"], resources=demo)
    with pytest.raises(ServiceError, match="modelo"):
        predict(QUESTION, OPTIONS, preset="llm_completo", resources=demo)


def test_missing_data(tmp_path):
    from mavericks.service import Paths

    with pytest.raises(ServiceError, match="Faltan"):
        load_resources(Paths(population=str(tmp_path / "no.parquet")))


@pytest.mark.skipif(not REAL, reason="requiere los datos procesados")
def test_fast_real():
    resources = load_resources()
    out = predict(QUESTION, OPTIONS, resources=resources)
    assert out["mode"] in {"direct", "prior", "none"}
    assert sum(out["national"]["distribution"].values()) == pytest.approx(1)
    assert set(out["subgroups"]) >= {"sexo", "grupo_edad", "ccaa"}
    for dimension in out["subgroups"].values():
        for group in dimension["groups"].values():
            assert sum(group["distribution"].values()) == pytest.approx(1, abs=1e-6)
    assert out["evidence"]
    assert out["seconds"] >= 0


def test_confidence_uses_the_error_of_the_mode_used():
    from mavericks.service import _confidence

    rows = [{"mode": "direct", "mae": 0.02}, {"mode": "direct", "mae": 0.03}, {"mode": "none", "mae": 0.2}]
    direct = _confidence(None, "direct", rows)
    none = _confidence(None, "none", rows)
    assert direct["typical_error"] == pytest.approx(0.025) and direct["backtest_cases"] == 2
    assert none["typical_error"] == pytest.approx(0.2) and none["backtest_cases"] == 1
    assert "sin ancla" in none["basis"]


def test_spontaneous_options_are_flagged():
    from mavericks.service import _spontaneous

    spec = {"mapping": {"Buena": "Buena", "Regular": "(NO LEER) Regular"}}
    assert _spontaneous(spec, ["Buena", "Regular"]) == ["Regular"]
    assert _spontaneous(None, ["Buena"]) == []
