import asyncio

import pytest

from mavericks.llm import LLMService, MockLLMProvider
from mavericks.schemas import Settings
from mavericks.translate import Translation, apply_translation, texts_to_translate, translate_texts


class Upper(MockLLMProvider):
    calls = 0

    async def generate(self, schema, payload, temperature=None):
        if schema is Translation:
            Upper.calls += 1
            return Translation(items=[t.upper() for t in payload["texts"]])
        return await super().generate(schema, payload)


RECORD = {
    "evidence_id": "E1", "question_text": "Do you trust the EU?", "options": ["Yes", "No"], "idioma": "en",
    "metadata": {"empirical_statistics": {
        "national": {"distribution": {"Yes": 0.6, "No": 0.4}, "n_respondents": 1000},
        "sexo": {"mujer": {"distribution": {"Yes": 0.7, "No": 0.3}, "n_respondents": 500}},
    }},
}


def test_translation_is_batched_cached_and_remaps_distribution_keys(tmp_path):
    service = LLMService(Upper(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c")
    texts = texts_to_translate([RECORD, {**RECORD, "evidence_id": "E2"}])
    assert texts == ["Do you trust the EU?", "Yes", "No"]  # sin duplicados
    cache = asyncio.run(translate_texts(texts, service, {}, batch=2))
    assert Upper.calls == 2 and cache["Yes"] == "YES"
    asyncio.run(translate_texts(texts, service, cache, batch=2))
    assert Upper.calls == 2  # todo estaba en la caché: no se vuelve a llamar
    es = apply_translation(RECORD, cache)
    assert es["idioma"] == "es" and es["question_text"] == "DO YOU TRUST THE EU?" and es["options"] == ["YES", "NO"]
    stats = es["metadata"]["empirical_statistics"]
    assert set(stats["national"]["distribution"]) == {"YES", "NO"}
    assert set(stats["sexo"]["mujer"]["distribution"]) == {"YES", "NO"}
    assert es["metadata"]["original_en"]["options"] == ["Yes", "No"] and RECORD["options"] == ["Yes", "No"]


def test_wrong_batch_length_is_rejected(tmp_path):
    class Short(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            return Translation(items=payload["texts"][:-1])

    service = LLMService(Short(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c2")
    with pytest.raises(ValueError, match="traducciones"):
        asyncio.run(translate_texts(["a", "b"], service, {}, batch=2))
