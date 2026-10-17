import asyncio
import json
import sys
from pathlib import Path

import pytest

from mavericks.backtest_cases import build_cases, clean_option
from mavericks.etl.cis import run_etl
from mavericks.evaluation import backtest, subgroup_errors, summarize_backtest
from mavericks.evidence import EvidenceRepository
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Settings

sys.path.insert(0, str(Path(__file__).parent))
from test_etl_cis import _write_study  # noqa: E402

TEXT = "Valoración de la situación económica de España en los próximos meses"


def _records(**extra):
    base = dict(source="CIS", question_text=TEXT, options=["Mejor", "Peor"], topic="consumo", question_type="single")
    return [
        {**base, "evidence_id": "OLD", "study_id": "1", "fieldwork_date": "2026-01-20", **extra},
        {**base, "evidence_id": "SAMEDAY", "study_id": "2", "fieldwork_date": "2026-03-01", **extra},
        {**base, "evidence_id": "HELD", "study_id": "3", "fieldwork_date": "2026-03-06", "reservado_examen": True},
    ]


def test_ingest_skips_reserved_unless_requested(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(_records()), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / "a.sqlite")
    assert repo.ingest(path) == 2 and repo.skipped_reserved == 1
    assert {e.evidence_id for e in repo.all()} == {"OLD", "SAMEDAY"}
    both = EvidenceRepository(tmp_path / "b.sqlite")
    both.ingest(path, include_reserved=True)
    # Aunque estén en la base, el RAG no los devuelve salvo petición explícita
    assert "HELD" not in [h["evidence"]["evidence_id"] for h in both.search(TEXT, limit=10)]
    assert "HELD" in [h["evidence"]["evidence_id"] for h in both.search(TEXT, limit=10, include_reserved=True)]


def test_cutoff_is_strictly_before_target_start(tmp_path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(_records()), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / "a.sqlite")
    repo.ingest(path)
    ids = [h["evidence"]["evidence_id"] for h in repo.search(TEXT, limit=10, cutoff="2026-03-01")]
    assert ids == ["OLD"]  # el estudio que termina el mismo día que empieza el objetivo no vale


@pytest.mark.parametrize("previous_waves,expected", [(True, ["OLD"]), (False, [])])
def test_previous_waves_mode(tmp_path, previous_waves, expected):
    path = tmp_path / "data" / "e.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_records()[:1]), encoding="utf-8")
    demo = create_demo(tmp_path / "fixtures")
    frame, meta = PopulationLoader().load(demo)
    evidence = EvidenceRepository(tmp_path / "db.sqlite")
    evidence.ingest(path)
    case = {
        "evidence_id": "HELD",
        "question": TEXT,
        "options": ["Mejor", "Peor"],
        "fieldwork_date": "2026-03-06",
        "observed_distribution": {"Mejor": 0.4, "Peor": 0.6},
        "observed_subgroups": {"sexo": {"mujer": {"distribution": {"Mejor": 0.3, "Peor": 0.7}, "n": 200}}},
        "allow_previous_waves": previous_waves,
    }
    settings = Settings(llm_provider="mock", llm_retries=0)
    results = asyncio.run(
        backtest([case], PopulationRepository(frame, meta), evidence, lambda: LLMService(MockLLMProvider(), settings, tmp_path / "c"))
    )
    assert results[0]["retrieved_ids"] == expected
    assert results[0]["mae_uniform_baseline"] == pytest.approx(0.1)


def test_build_cases_from_reserved_studies(tmp_path):
    catalog = _write_study(tmp_path, n=1200)  # estudio marcado como reservado en el catálogo
    out = tmp_path / "out"
    run_etl(tmp_path / "raw", catalog, out)
    cases = build_cases(out, max_per_study=5, min_n=100)
    assert cases, "los estudios reservados generan casos"
    case = next(c for c in cases if c["evidence_id"].endswith("P1")) if any(c["evidence_id"].endswith("P1") for c in cases) else cases[0]
    assert case["estudio_id"] == "9999" and case["fieldwork_date"] == case["fieldwork_start"]
    assert sum(case["observed_distribution"].values()) == pytest.approx(1)
    assert all("(NO LEER)" not in o for o in case["options"]) and "N.S." not in case["options"]
    assert case["allow_previous_waves"] is True
    assert build_cases(out, previous_waves=False, min_n=100)[0]["allow_previous_waves"] is False


def test_clean_option_and_metrics():
    assert clean_option("(NO LEER) Regular") == "Regular"
    predicted = {"sexo": {"mujer": {"A": 0.5, "B": 0.5}, "otro": {"A": 1.0, "B": 0.0}}}
    observed = {"sexo": {"mujer": {"distribution": {"A": 0.7, "B": 0.3}, "n": 300}}}
    scores = subgroup_errors(predicted, observed)
    assert len(scores) == 1 and scores[0]["mae"] == pytest.approx(0.2)
    results = [
        {"status": "completed", "topic": "t", "anchor_mode": "direct", "mae": 0.1, "js_divergence": 0.01,
         "subgroup_mae": 0.2, "mae_uniform_baseline": 0.3},
        {"status": "failed", "topic": "t", "anchor_mode": "none", "mae": None, "js_divergence": None,
         "subgroup_mae": None, "mae_uniform_baseline": 0.3},
    ]
    summary = summarize_backtest(results)
    assert summary["overall"]["failed"] == 1 and summary["by_anchor_mode"]["direct"]["mae"] == pytest.approx(0.1)
