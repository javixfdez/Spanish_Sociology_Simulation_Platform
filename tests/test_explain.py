import asyncio

import pytest

from mavericks import explain
from mavericks import service as svc
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import ResultNarrative, Settings


def make_result(mode="none"):
    dist = {"Sí": 0.63, "No": 0.37}

    def group(p):
        return {"distribution": {"Sí": p, "No": 1 - p}, "interval": None, "population_share": 0.1, "n_cis": None}

    return {
        "question": "¿Está a favor de limitar el alquiler?",
        "options": ["Sí", "No"],
        "mode": mode,
        "mode_label": {"direct": "directo", "prior": "prior", "none": "sin ancla"}[mode],
        "national": {"distribution": dist, "interval_core": {"Sí": [0.4, 0.85], "No": [0.15, 0.6]}},
        "confidence": {"typical_error": 0.05, "backtest_cases": 100},
        "prior_usado": {"metodo": "media_knn_mlp"},
        "subgroups": {
            "grupo_edad": {"label": "Edad", "groups": {"18-24": group(0.75), "75+": group(0.64), "45-54": group(0.60)}},
            "sexo": {"label": "Sexo", "groups": {"mujer": group(0.69), "hombre": group(0.60)}},
        },
        "evidence": [{"study_id": "3546", "fieldwork_end": "2026-03-06", "question": "Limitar el alquiler", "similarity": 0.55}],
    }


class FakeLLM:
    def __init__(self, text=None, delay=0.0, error=None):
        self.text, self.delay, self.error = text, delay, error

    async def call(self, schema, payload, validator=None, **kwargs):
        assert schema is ResultNarrative and "national_percent" in payload
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return ResultNarrative(summary=self.text)


def test_template_has_majority_range_and_only_large_subgroup_differences():
    text = explain.template_text(make_result())
    assert "«Sí»" in text and "63 %" in text and "40–85 %" in text
    assert "Edad 18-24".lower() in text.lower() and "+12" in text  # 75 % − 63 % = 12 > error típico (5)
    assert "mujer" in text  # segunda mayor diferencia (+6)
    assert "75+" not in text and "hombre" not in text  # +1 y −3 puntos: no superan el error típico
    assert "pregunta nueva" in text  # de qué se apoya
    quiet = make_result()
    quiet["confidence"]["typical_error"] = 0.30
    assert "diferencias entre grupos" not in explain.template_text(quiet)  # nada supera el error típico


def test_llm_text_with_correct_numbers_is_accepted():
    result = explain.add_template(make_result())
    text = "La opción más probable es «Sí», con un 63 % (rango habitual entre 40 y 85 %). Entre 18-24 sube al 75 %, 12 puntos más."
    out = explain.add_llm_text(result, FakeLLM(text))["explanation_text"]
    assert out["origen"] == "llm" and out["texto"] == text


def test_invented_figure_falls_back_to_template(caplog):
    result = explain.add_template(make_result())
    out = explain.add_llm_text(result, FakeLLM("La opción más probable es «Sí», con un 80 % de apoyo entre la población."))["explanation_text"]
    assert out["origen"] == "plantilla" and "cifra" in out["motivo"] and "63 %" in out["texto"]


def test_unknown_group_mention_falls_back():
    result = explain.add_template(make_result())
    out = explain.add_llm_text(result, FakeLLM("La opción más probable es «Sí», con un 63 %. Los hombres apoyan menos, y 75+ también."))["explanation_text"]
    assert out["origen"] == "plantilla" and "grupo" in out["motivo"]


def test_timeout_and_failure_fall_back():
    result = explain.add_template(make_result())
    slow = explain.add_llm_text(result, FakeLLM("La opción más probable es «Sí», con un 63 %.", delay=2.0), timeout=0.2)["explanation_text"]
    assert slow["origen"] == "plantilla" and "tardó" in slow["motivo"]
    broken = explain.add_llm_text(make_result(), FakeLLM(error=RuntimeError("sin modelo")))["explanation_text"]
    assert broken["origen"] == "plantilla" and "falló" in broken["motivo"]


def test_mock_provider_text_passes_the_check(tmp_path):
    llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "cache")
    out = explain.add_llm_text(make_result(), llm)["explanation_text"]
    assert out["origen"] == "llm" and "63 %" in out["texto"]


@pytest.fixture
def demo(tmp_path):
    from mavericks.evidence import EvidenceRepository

    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    return svc.Resources(PopulationRepository(frame, metadata), evidence, None, {}, None)


def test_llm_mode_mixes_numbers_with_weight_and_falls_back(demo, tmp_path, monkeypatch):
    llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "cache")
    fast = svc.predict("¿Usas inteligencia artificial?", ["Sí", "No"], preset="rapido_sin_llm", resources=demo)
    mixed = svc.predict("¿Usas inteligencia artificial?", ["Sí", "No"], preset="llm_completo", resources=demo, service=llm, n_segments=5)
    assert mixed["prior_usado"]["metodo"] == "mezcla_llm" and mixed["prior_usado"]["w_llm"] == 0.3
    assert sum(mixed["national"]["distribution"].values()) == pytest.approx(1)
    assert mixed["national"]["distribution"] != fast["national"]["distribution"]
    assert "explanation_text" in mixed and mixed["explanation_text"]["origen"] == "plantilla"

    def boom(*args, **kwargs):
        raise RuntimeError("sin GPU")

    monkeypatch.setattr(svc, "_llm_path", boom)
    back = svc.predict("¿Usas inteligencia artificial?", ["Sí", "No"], preset="llm_completo", resources=demo, service=llm)
    assert back["national"]["distribution"] == pytest.approx(fast["national"]["distribution"])
    assert any("no pudo completar" in w for w in back["warnings"])


def test_gpu_memory_error_falls_back_to_template_with_reason():
    for error in (RuntimeError("model requires more system memory (6.1 GiB) than is available"), MemoryError(), RuntimeError("CUDA out of memory")):
        out = explain.add_llm_text(make_result(), FakeLLM(error=error))["explanation_text"]
        assert out["origen"] == "plantilla" and "memoria de la GPU" in out["motivo"] and out["texto"]
    other = explain.add_llm_text(make_result(), FakeLLM(error=RuntimeError("sin modelo")))["explanation_text"]
    assert "memoria" not in other["motivo"]
