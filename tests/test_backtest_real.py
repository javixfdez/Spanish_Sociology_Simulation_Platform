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


def test_national_baseline_and_mode_topic_summary(tmp_path):
    import pandas as pd

    from mavericks.anchor import AnchorStore
    from mavericks.evaluation import national_baseline

    rows = [dict(estudio_id="9001", respondent_uid=str(i), peso=1.0, sexo="mujer", grupo_edad="18-34") for i in range(100)]
    answers = [
        dict(estudio_id="9001", respondent_uid=str(i), question_uid="Q1", answer="Mucho" if i < 70 else "Poco", selected=None)
        for i in range(100)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "cis_respondents.parquet")
    pd.DataFrame(answers).assign(selected=lambda d: d.selected.astype("boolean")).to_parquet(tmp_path / "cis_responses.parquet")
    store = AnchorStore(tmp_path)
    hit = {
        "retrieval_score": 1.0,
        "evidence": {
            "evidence_id": "Q1", "study_id": "9001", "question_text": "Grado de preocupación por el clima",
            "options": ["Mucho", "Poco", "N.S."], "metadata": {"original_fields": {"question_type": "single"}},
        },
    }
    result = {"retrieved_evidence": [hit]}
    by_label = national_baseline(result, {"question": "Grado de preocupación por el clima", "options": ["Mucho", "Poco"]}, store)
    assert by_label["source"] == "etiqueta" and by_label["distribution"]["Mucho"] == pytest.approx(0.7)
    ordinal = national_baseline(result, {"question": "Preocupación por el clima", "options": ["Bastante", "Nada"]}, store)
    assert ordinal["source"] == "ordinal" and ordinal["distribution"]["Bastante"] == pytest.approx(0.7)
    assert national_baseline(result, {"question": "x", "options": ["A", "B", "C"]}, store) is None
    assert national_baseline(result, {"question": "x", "options": ["A", "B"]}, None) is None
    done = {"status": "completed", "topic": "t", "anchor_mode": "none", "mae": 0.1, "js_divergence": 0.01, "subgroup_mae": None,
            "mae_uniform_baseline": 0.3, "mae_national_baseline": 0.2, "assessment_fallback": False, "had_evidence": True}
    summary = summarize_backtest([done])
    block = summary["by_mode_and_topic"]["none | t"]
    assert block["national_baseline"]["wins"] == 1 and block["assessment_fallback_rate"] == 0.0


def test_widened_literal_pool_keeps_leak_protections(tmp_path):
    """El grupo ampliado de 30 candidatos de la consulta literal respeta excluded_ids, corte estricto y reservados."""
    records = []
    for i in range(40):  # mismo enunciado en 40 oleadas, de enero a octubre de 2025
        records.append(
            dict(source="CIS", question_text=TEXT, options=["Mejor", "Peor"], topic="consumo", question_type="single",
                 evidence_id=f"W{i:02d}", study_id=str(100 + i), fieldwork_date=f"2025-{1 + i // 4:02d}-{1 + i % 4 * 7:02d}")
        )
    records.append({**records[0], "evidence_id": "HELD", "study_id": "999", "fieldwork_date": "2026-03-06"})
    records.append({**records[0], "evidence_id": "RESERVED", "study_id": "998", "fieldwork_date": "2025-02-01", "reservado_examen": True})
    path = tmp_path / "w.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    evidence = EvidenceRepository(tmp_path / "w.sqlite")
    evidence.ingest(path, include_reserved=True)  # incluso almacenada, una reservada no puede salir
    demo = create_demo(tmp_path / "fixtures")
    frame, meta = PopulationLoader().load(demo)
    cutoff = "2025-06-01"
    case = {
        "evidence_id": "HELD", "question": TEXT, "options": ["Mejor", "Peor"], "fieldwork_date": cutoff,
        "observed_distribution": {"Mejor": 0.4, "Peor": 0.6}, "allow_previous_waves": True,
    }
    settings = Settings(llm_provider="mock", llm_retries=0, max_evidence_hits=20)
    results = asyncio.run(
        backtest([case], PopulationRepository(frame, meta), evidence, lambda: LLMService(MockLLMProvider(), settings, tmp_path / "c"))
    )
    ids = results[0]["retrieved_ids"]
    assert ids, "debe recuperar oleadas anteriores"
    assert "HELD" not in ids and "RESERVED" not in ids
    dates = {r["evidence_id"]: r["fieldwork_date"] for r in records}
    assert all(dates[i] < cutoff for i in ids), "solo trabajo de campo anterior al inicio del estudio evaluado"
    # Misma garantía directamente sobre la búsqueda con el tamaño de pool que usa el grafo
    pool = [h["evidence"]["evidence_id"] for h in evidence.search(TEXT, limit=30, excluded_ids=["HELD"], cutoff=cutoff)]
    assert "HELD" not in pool and "RESERVED" not in pool and all(dates[i] < cutoff for i in pool)
