import json

import numpy as np
import pandas as pd
import pytest

from mavericks.analog import (
    AnalogConfig,
    align_options,
    build_national_table,
    by_similarity,
    candidates,
    collect_case,
    combine,
    evaluate,
    load_national_table,
    summarize,
    tune,
)
from mavericks.evidence import EvidenceRepository


class FakeEmbedder:
    name = "fake"
    vocab = {"mucho": 0, "bastante": 1, "poco": 2, "nada": 3, "alto": 0, "medio": 1, "bajo": 2, "ninguno": 3}

    def _enc(self, texts):
        out = np.zeros((len(texts), 4))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                if w in self.vocab:
                    out[i, self.vocab[w]] = 1
        return out + 1e-3

    encode_queries = encode_passages = _enc


def test_option_alignment_kinds():
    mapping, kind = align_options(["Sí", "No"], ["Si", "No", "N.S."])
    assert kind == "label" and mapping == {"Sí": "Si", "No": "No"}
    mapping, kind = align_options(["Mucho", "Bastante", "Poco", "Nada"], ["Alto", "Medio", "Bajo", "Ninguno"], FakeEmbedder())
    assert kind == "semantic" and mapping["Mucho"] == "Alto" and mapping["Nada"] == "Ninguno"
    mapping, kind = align_options(["Uno", "Dos"], ["Primero", "Segundo", "N.S."])
    assert kind == "ordinal" and mapping == {"Uno": "Primero", "Dos": "Segundo"}
    assert align_options(["A", "B", "C"], ["X", "Y"]) == (None, None)


def _hit(eid, text, options, dist, source="CIS 1"):
    return {"evidence": {"evidence_id": eid, "question_text": text, "options": options, "source": source, "fieldwork_date": "2025-01-01",
                         "metadata": {"original_fields": {"question_type": "single"}}}, "retrieval_score": 1.0, "embedding_similarity": None}


def test_candidates_use_table_dedupe_by_text_and_drop_unalignable():
    table = {"A": {"n": 9, "dist": {"Sí": 0.7, "No": 0.3}}, "B": {"n": 9, "dist": {"Sí": 0.1, "No": 0.9}}, "C": {"n": 9, "dist": {"X": 1.0}}}
    hits = [
        _hit("A", "¿Está de acuerdo con la medida?", ["Sí", "No"], None),
        _hit("B", "¿Está de acuerdo con la medida?", ["Sí", "No"], None),  # mismo enunciado: se descarta
        _hit("C", "Otra cosa", ["X", "Y", "Z"], None),  # escala no alineable
    ]
    got = candidates(hits, "¿Está de acuerdo con la medida?", ["Sí", "No"], table)
    assert [c["evidence_id"] for c in got] == ["A"] and got[0]["dist"]["Sí"] == pytest.approx(0.7)


def test_combine_weights_by_similarity_and_shrinks_to_uniform():
    cands = [
        {"evidence_id": "a", "source": "CIS", "date": None, "lex": 0.9, "emb": None, "kind": "label", "dist": {"Sí": 0.8, "No": 0.2}},
        {"evidence_id": "b", "source": "CIS", "date": None, "lex": 0.4, "emb": None, "kind": "ordinal", "dist": {"Sí": 0.2, "No": 0.8}},
    ]
    pure = combine(cands, ["Sí", "No"], AnalogConfig(shrink=0.0, power=2.0, min_sim=0.0))
    assert pure["prior"]["Sí"] > 0.7 and pure["best_similarity"] == 0.9 and pure["used"] == 2
    only_close = combine(cands, ["Sí", "No"], AnalogConfig(shrink=0.0, min_sim=0.5))
    assert only_close["prior"]["Sí"] == pytest.approx(0.8) and only_close["used"] == 1
    shrunk = combine(cands, ["Sí", "No"], AnalogConfig(shrink=5.0, min_sim=0.5))
    assert 0.5 < shrunk["prior"]["Sí"] < 0.8 and sum(shrunk["prior"].values()) == pytest.approx(1)
    none = combine(cands, ["Sí", "No"], AnalogConfig(min_sim=0.99))
    assert none["prior"] == {"Sí": 0.5, "No": 0.5} and none["used"] == 0


def _repo(tmp_path):
    base = dict(source="CIS 1", options=["Sí", "No"], question_type="single", topic="general")
    recs = [
        dict(base, evidence_id="OLD", study_id="1", question_text="¿Apoya usted la medida del gobierno?", fieldwork_date="2025-01-10"),
        dict(base, evidence_id="SAME", study_id="2", question_text="¿Apoya usted la medida del gobierno hoy?", fieldwork_date="2025-02-10"),
        dict(base, evidence_id="HELD", study_id="3", question_text="¿Apoya usted la medida del gobierno hoy?", fieldwork_date="2025-04-01"),
        dict(base, evidence_id="LATE", study_id="4", question_text="¿Apoya usted la medida del gobierno?", fieldwork_date="2025-06-01"),
        dict(base, evidence_id="RES", study_id="5", question_text="¿Apoya usted la medida del gobierno?", fieldwork_date="2025-01-01", reservado_examen=True),
    ]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(recs), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / "r.sqlite")
    repo.ingest(path, include_reserved=True)
    table = {k: {"n": 100, "dist": {"Sí": 0.6, "No": 0.4}} for k in ("OLD", "SAME", "HELD", "LATE", "RES")}
    return repo, table


def test_collect_case_keeps_every_leak_protection(tmp_path):
    repo, table = _repo(tmp_path)
    case = {"evidence_id": "HELD", "question": "¿Apoya usted la medida del gobierno hoy?", "options": ["Sí", "No"],
            "fieldwork_date": "2025-04-01", "observed_distribution": {"Sí": 0.6, "No": 0.4}, "topic": "general", "estudio_id": "3"}
    strict = collect_case(case, repo, table, strict=True)
    ids = {c["evidence_id"] for c in strict["candidates"]}
    assert ids == {"OLD"}  # ni ella misma, ni el texto idéntico (SAME), ni posterior (LATE), ni reservada (RES)
    temporal = {c["evidence_id"] for c in collect_case(case, repo, table, strict=False)["candidates"]}
    assert temporal == {"OLD", "SAME"}


def _collected(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        truth = rng.dirichlet([1, 1])
        good = {"Sí": float(truth[0]), "No": float(truth[1])}
        out.append({
            "evidence_id": f"c{i}", "estudio_id": str(i % 8), "topic": "t", "split": "test", "question": "q", "options": ["Sí", "No"],
            "observed": good,
            "candidates": [{"evidence_id": "a", "source": "CIS", "date": None, "lex": 0.9, "emb": None, "kind": "label", "dist": good}],
        })
    return out


def test_evaluate_summarize_bins_and_cross_validated_tuning():
    collected = _collected()
    rows = evaluate(collected, AnalogConfig(shrink=0.0, min_sim=0.0))
    assert summarize(rows)["mae"] < 1e-9 and summarize(rows)["simbench"] == pytest.approx(100.0)
    assert by_similarity(rows)["0.7-1.0"]["n"] == 40 and by_similarity(rows)["0.0-0.3"] is None
    grid = [AnalogConfig(shrink=s, min_sim=m) for s in (0.0, 5.0) for m in (0.0, 0.95)]
    result = tune(collected, grid, folds=4)
    assert result["cv"]["mae"] <= result["default"]["mae"] + 1e-9 and result["final"].min_sim == 0.0 and len(result["held_rows"]) == 40


def test_build_national_table_weights_and_skips_multiple(tmp_path):
    pd.DataFrame([{"question_uid": "Q1", "estudio_id": "1", "question_type": "single"},
                  {"question_uid": "Q2", "estudio_id": "1", "question_type": "multiple"}]).to_parquet(tmp_path / "cis_questions.parquet")
    pd.DataFrame([{"estudio_id": "1", "respondent_uid": str(i), "peso": 3.0 if i == 0 else 1.0} for i in range(4)]).to_parquet(tmp_path / "cis_respondents.parquet")
    answers = [{"estudio_id": "1", "respondent_uid": str(i), "question_uid": "Q1", "answer": "A" if i < 2 else "B", "selected": None} for i in range(4)]
    pd.DataFrame(answers).assign(selected=lambda d: d.selected.astype("boolean")).to_parquet(tmp_path / "cis_responses.parquet")
    n = build_national_table(tmp_path, tmp_path / "t.json")
    table = load_national_table(tmp_path / "t.json")
    assert n == 1 and set(table) == {"Q1"} and table["Q1"]["dist"] == {"A": pytest.approx(4 / 6), "B": pytest.approx(2 / 6)}
