import json

import numpy as np
import pytest

from mavericks.evidence import EvidenceRepository

# «Semántica» mínima: palabras de distintos idiomas con el mismo concepto comparten dimensión.
CONCEPTS = {
    "vivienda": 0, "housing": 0, "alquiler": 1, "rent": 1, "precio": 2, "price": 2,
    "coche": 3, "car": 3, "salud": 4, "health": 4, "clima": 5, "climate": 5,
}


class FakeEmbedder:
    name = "fake-multilingual"

    def _vec(self, text):
        v = np.zeros(len(set(CONCEPTS.values())) + 1, dtype=np.float32)
        for word in text.lower().replace("|", " ").replace(";", " ").split():
            v[CONCEPTS.get(word.strip("¿?.,"), len(v) - 1)] += 1.0
        return v

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    encode_queries = encode_passages


def _repo(tmp_path, retriever, name="db"):
    records = [
        dict(evidence_id="EN_RENT", source="eb", question_text="housing rent price", options=["Up", "Down"],
             topic="vivienda", fieldwork_date="2025-01-10", study_id="1", idioma="en"),
        dict(evidence_id="ES_CAR", source="cis", question_text="coche precio", options=["Sí", "No"],
             topic="consumo", fieldwork_date="2025-01-11", study_id="2"),
        dict(evidence_id="HELD", source="cis", question_text="vivienda alquiler precio", options=["Sí", "No"],
             topic="vivienda", fieldwork_date="2025-03-01", study_id="3"),
        dict(evidence_id="LATE", source="cis", question_text="vivienda alquiler", options=["Sí", "No"],
             topic="vivienda", fieldwork_date="2025-06-01", study_id="4"),
        dict(evidence_id="RESERVED", source="cis", question_text="vivienda alquiler precio", options=["Sí", "No"],
             topic="vivienda", fieldwork_date="2025-01-05", study_id="5", reservado_examen=True),
        dict(evidence_id="EQUAL", source="cis", question_text="vivienda alquiler", options=["Sí", "No"],
             topic="vivienda", fieldwork_date="2025-01-06", study_id="6"),
    ]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / f"{name}.sqlite", embedder=FakeEmbedder(), retriever=retriever)
    repo.ingest(path, include_reserved=True)
    return repo


def test_tfidf_cannot_cross_languages_but_embed_and_hybrid_can(tmp_path):
    query = "vivienda alquiler"
    tfidf = [h["evidence"]["evidence_id"] for h in _repo(tmp_path, "tfidf").search("clima salud", limit=3)]
    assert "EN_RENT" not in tfidf
    english_only = _repo(tmp_path, "tfidf", "t").search("alquiler del piso", limit=6)
    assert "EN_RENT" not in [h["evidence"]["evidence_id"] for h in english_only]
    for mode in ("embed", "hybrid"):
        ids = [h["evidence"]["evidence_id"] for h in _repo(tmp_path, mode, mode).search(query, limit=6, include_reserved=True)]
        assert "EN_RENT" in ids, mode


@pytest.mark.parametrize("mode", ["tfidf", "embed", "hybrid"])
def test_leak_protections_apply_to_every_retriever(tmp_path, mode):
    repo = _repo(tmp_path, mode, mode)
    hits = repo.search(
        "vivienda alquiler precio", limit=10, excluded_ids=["HELD"], excluded_texts=["vivienda alquiler"], cutoff="2025-03-01"
    )
    ids = [h["evidence"]["evidence_id"] for h in hits]
    assert "HELD" not in ids  # nunca la propia pregunta evaluada
    assert "LATE" not in ids  # trabajo de campo posterior al corte
    assert "RESERVED" not in ids  # estudios reservados fuera aunque estén almacenados
    assert "EQUAL" not in ids  # texto excluido
    assert "EN_RENT" in ids if mode != "tfidf" else True
    assert all(h["evidence"]["fieldwork_date"] < "2025-03-01" for h in hits)
    # corte estricto: el estudio que acaba justo el día del corte tampoco entra
    same_day = [h["evidence"]["evidence_id"] for h in repo.search("vivienda", limit=10, cutoff="2025-03-01")]
    assert "HELD" not in same_day


def test_embedding_cache_is_incremental(tmp_path):
    repo = _repo(tmp_path, "embed")
    repo.search("vivienda", limit=2)
    files = list(tmp_path.glob("db.emb-*.npz"))
    assert len(files) == 1
    calls = []

    class Counting(FakeEmbedder):
        def encode_passages(self, texts):
            calls.append(len(texts))
            return super().encode_passages(texts)

    again = EvidenceRepository(tmp_path / "db.sqlite", embedder=Counting(), retriever="embed")
    again.search("vivienda", limit=2)
    assert calls == []  # todo leído de disco
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([dict(evidence_id="NEW", source="cis", question_text="salud health")]), encoding="utf-8")
    again.ingest(extra)
    again.search("salud", limit=2)
    assert calls == [1]  # solo el texto nuevo; los demás vienen de la caché
    assert len(list(tmp_path.glob("db.emb-*.npz"))) == 1


def test_embed_requires_an_embedder(tmp_path):
    with pytest.raises(ValueError):
        EvidenceRepository(tmp_path / "x.sqlite", retriever="hybrid")
    with pytest.raises(ValueError):
        EvidenceRepository(tmp_path / "x.sqlite", retriever="otro")


def test_keep_zero_returns_the_whole_filtered_store_and_default_is_unchanged(tmp_path):
    repo = EvidenceRepository(tmp_path / "z.sqlite")
    path = tmp_path / "z.json"
    records = [
        {"evidence_id": "A", "source": "t", "question_text": "vivienda alquiler precio", "fieldwork_date": "2024-01-01"},
        {"evidence_id": "B", "source": "t", "question_text": "salud sanidad hospital", "fieldwork_date": "2024-02-01"},
        {"evidence_id": "C", "source": "t", "question_text": "clima temperatura", "fieldwork_date": "2026-05-01"},
        {"evidence_id": "R", "source": "t", "question_text": "salud mental", "fieldwork_date": "2024-03-01",
         "reservado_examen": True},
        {"evidence_id": "X", "source": "t", "question_text": "salud propia pregunta", "fieldwork_date": "2024-04-01"},
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    repo.ingest(path, include_reserved=True)
    kwargs = dict(limit=100, cutoff="2025-01-01", excluded_ids=["X"], retriever="tfidf")
    default = {h["evidence"]["evidence_id"] for h in repo.search("alquiler vivienda", **kwargs)}
    everything = {h["evidence"]["evidence_id"] for h in repo.search("alquiler vivienda", keep_zero=True, **kwargs)}
    assert default == {"A"}  # comportamiento por defecto: solo lo que comparte palabras
    assert everything == {"A", "B"}  # sin reservados (R), sin la propia pregunta (X) y sin posteriores al corte (C)
