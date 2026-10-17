import json

import pytest
from pydantic import ValidationError

from mavericks.evidence import EvidenceRepository
from mavericks.schemas import Methodology
from mavericks.topics import TOPICS, Topic, classify_topic, normalize_topic


def test_topics_match_data_folders():
    assert set(TOPICS) == {"general", "vivienda", "salud", "ia_tecnologia", "consumo", "medio_ambiente", "soledad"}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("vivienda", "vivienda"),
        ("Medio ambiente", "medio_ambiente"),
        ("housing", "vivienda"),
        ("Vivienda y acceso al alquiler", "vivienda"),
        ("tecnología inteligencia artificial", "ia_tecnologia"),
        ("Salud mental", "salud"),
    ],
)
def test_normalize_topic_free_text(value, expected):
    assert normalize_topic(value) == expected


def test_normalize_unknown_uses_fallback_text():
    assert normalize_topic("???") is None
    assert normalize_topic("???", "¿Se siente solo o sola con frecuencia?") == "soledad"
    assert classify_topic("¿Qué opina del precio del alquiler?") == "vivienda"
    assert classify_topic("Texto sin pistas") == "general"


def test_methodology_topic_is_closed_enum():
    base = dict(question_type="single", relevant_demographics=[], retrieval_queries=["q"])
    assert Methodology(topic="vivienda", **base).topic is Topic.VIVIENDA
    with pytest.raises(ValidationError):
        Methodology(topic="DEMO / SYNTHETIC", **base)


def _repo(tmp_path):
    records = [
        dict(evidence_id="H1", source="t", question_text="Dificultad para pagar el alquiler de la vivienda",
             topic="vivienda", options=["Sí", "No"]),
        dict(evidence_id="E1", source="t", question_text="Dificultad para pagar el coche eléctrico",
             topic="medio_ambiente", options=["Sí", "No"]),
        dict(evidence_id="G1", source="t", question_text="Dificultad general para llegar a fin de mes",
             topic="Algo raro", options=["Sí", "No"]),
    ]
    path = tmp_path / "e.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / "e.sqlite")
    repo.ingest(path)
    return repo


def test_ingest_normalizes_topic(tmp_path):
    topics = {e.evidence_id: e.topic for e in _repo(tmp_path).all()}
    assert topics == {"H1": "vivienda", "E1": "medio_ambiente", "G1": "general"}


def test_search_filters_by_topic_and_falls_back(tmp_path):
    repo = _repo(tmp_path)
    ids = [h["evidence"]["evidence_id"] for h in repo.search("dificultad para pagar", topic="vivienda")]
    assert "E1" not in ids and ids[0] == "H1" and "G1" in ids
    # Con texto libre del LLM el tema también se reconoce
    assert [h["evidence"]["evidence_id"] for h in repo.search("dificultad pagar", topic="Vivienda y alquiler")][0] == "H1"
    # Sin resultados dentro del tema se amplía y se marca
    fallback = repo.search("coche eléctrico", topic="salud")
    assert fallback and all(h["topic_fallback"] for h in fallback)
    # El filtro puede desactivarse
    assert "E1" in [h["evidence"]["evidence_id"] for h in repo.search("dificultad pagar", topic="vivienda", topic_filter=False)]
