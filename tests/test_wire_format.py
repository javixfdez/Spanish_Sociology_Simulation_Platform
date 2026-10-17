import asyncio
import logging

import pytest

import mavericks.llm as module
from mavericks.llm import StructuredProvider, prediction_answer_model
from mavericks.schemas import Assessment, Prediction, Settings


class Raw:
    def __init__(self, tokens=100, content=""):
        self.usage_metadata = {"input_tokens": tokens}
        self.content = content


def _provider(monkeypatch, answers):
    import langchain_ollama

    monkeypatch.setattr(module, "check_ollama", lambda settings: {"available": True})
    sent = []

    class FakeChat:
        def __init__(self, **kwargs):
            sent.append(("init", kwargs))

        def with_structured_output(self, schema, method, include_raw):
            assert method == "json_schema" and include_raw
            sent.append(("schema", schema))
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            sent.append(("human", messages[1][1]))
            return answers(self.schema)

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat)
    return StructuredProvider(Settings()), sent


PAYLOAD = {
    "segment": {"segment_id": "S007"},
    "question": {"question": "¿Pagarías?", "options": ["Sí", "No", "Quizá"]},
    "options": ["Sí", "No", "Quizá"],
    "evidence": [{"evidence_id": "CIS1_P1"}],
    "anchor": {"mode": "prior", "evidence_id": "CIS1_P1", "prior": {"Sí": 0.5, "No": 0.3, "Quizá": 0.2}},
}


def test_prediction_wire_format_uses_positional_keys_and_code_fills_ids(monkeypatch):
    def answers(schema):
        assert set(schema.model_fields) == {"o1", "o2", "o3", "explanation", "limitations"}
        return {"parsed": schema(o1=0.5, o2=0.3, o3=0.2, explanation="ok"), "raw": Raw()}

    provider, sent = _provider(monkeypatch, answers)
    value = asyncio.run(provider.generate(Prediction, PAYLOAD))
    assert value.segment_id == "S007" and value.probabilities == {"Sí": 0.5, "No": 0.3, "Quizá": 0.2}
    assert value.evidence_ids == ["CIS1_P1"]
    human = next(v for k, v in sent if k == "human")
    assert '"o1": "Sí"' in human and '"segment_id"' in human and "CIS1_P1" in human
    assert '"prior": {"o1": 0.5' in human  # el ancla también va con claves posicionales


def test_assessment_wire_format_refs_are_tolerant(monkeypatch):
    payload = {"question": {}, "evidence": [{"evidence_id": "A"}, {"evidence_id": "B"}, {"evidence_id": "C"}]}

    def answers(schema):
        items = [
            {"ref": "E2", "relevance": "direct"},
            {"ref": "e2", "relevance": "incompatible"},  # duplicado: gana el primero
            {"ref": "9", "relevance": "direct"},  # inexistente: se ignora
            {"ref": "1", "relevance": "indirect"},
        ]
        return {"parsed": schema(items=items), "raw": Raw()}

    provider, sent = _provider(monkeypatch, answers)
    value = asyncio.run(provider.generate(Assessment, payload))
    assert {i.evidence_id: i.relevance for i in value.items} == {"B": "direct", "A": "indirect"}
    human = next(v for k, v in sent if k == "human")
    assert '"evidence_id"' not in human and '"ref": "e1"' in human


def test_context_warning_at_90_percent(monkeypatch, caplog):
    def answers(schema):
        return {"parsed": schema(o1=1.0, o2=0.0, o3=0.0, explanation="x"), "raw": Raw(tokens=7600)}

    provider, _ = _provider(monkeypatch, answers)
    with caplog.at_level(logging.WARNING):
        asyncio.run(provider.generate(Prediction, PAYLOAD))
    assert provider.context_warnings == 1 and "riesgo de truncado" in caplog.text


def test_answer_model_bounds():
    model = prediction_answer_model(2)
    with pytest.raises(ValueError):
        model(o1=1.5, o2=0.0, explanation="x")
