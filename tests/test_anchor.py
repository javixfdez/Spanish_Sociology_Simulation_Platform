import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from mavericks.anchor import (
    AnchorStore,
    option_mapping,
    project_to_band,
    question_similarity,
    select_anchor,
)
from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Prediction, Question, Settings

QUESTION = "¿Utiliza aplicaciones de inteligencia artificial?"


def _hit(question_text, options, qtype="single", score=1.0, eid="CIS9001_P1"):
    return {
        "retrieval_score": score,
        "evidence": {
            "evidence_id": eid,
            "study_id": "9001",
            "question_text": question_text,
            "options": options,
            "fieldwork_date": "2025-01-01",
            "metadata": {"original_fields": {"question_type": qtype}},
        },
    }


def test_option_mapping_accepts_only_nonsubstantive_extras():
    assert option_mapping(["Sí", "No"], ["Si", "NO", "N.S.", "N.C."]) == {"Sí": "Si", "No": "NO"}
    assert option_mapping(["Sí", "No"], ["Sí", "No", "Depende"]) is None
    assert option_mapping(["Sí", "No", "Quizá"], ["Sí", "No"]) is None


def test_select_anchor_modes():
    q = {"question": QUESTION, "options": ["Sí", "No"], "question_type": "single"}
    direct = select_anchor([_hit(QUESTION, ["Sí", "No", "N.S."])], q)
    assert direct["mode"] == "direct" and direct["similarity"] >= 0.9
    prior = select_anchor([_hit("¿Utiliza usted alguna aplicación de inteligencia artificial en su móvil?", ["Sí", "No"])], q)
    assert prior["mode"] == "prior"
    assert select_anchor([_hit("Valoración de la economía de España", ["Sí", "No"])], q) is None
    assert select_anchor([_hit(QUESTION, ["Sí", "No"], qtype="multiple")], q) is None
    assert question_similarity("Hola mundo", "") == 0.0


def test_project_to_band_keeps_sum_and_bounds():
    out = project_to_band([0.9, 0.05, 0.05], [0.4, 0.1, 0.1], [0.6, 0.5, 0.5])
    assert out.sum() == pytest.approx(1.0)
    assert (out >= np.array([0.4, 0.1, 0.1]) - 1e-9).all() and (out <= np.array([0.6, 0.5, 0.5]) + 1e-9).all()


def _write_cis(directory, age_groups):
    """CIS sintético: las mujeres de 18-34 responden Sí con prob. alta; el resto, baja."""
    rng = np.random.default_rng(1)
    rows, answers = [], []
    uid = 0
    for sexo in ["mujer", "hombre"]:
        for grupo in age_groups:
            for _ in range(200):
                uid += 1
                p_yes = 0.8 if (sexo == "mujer" and grupo == "18-34") else 0.3
                rows.append(
                    dict(estudio_id="9001", respondent_uid=str(uid), peso=1.0, sexo=sexo, grupo_edad=grupo)
                )
                answers.append(
                    dict(
                        estudio_id="9001",
                        respondent_uid=str(uid),
                        question_uid="CIS9001_P1",
                        answer="Sí" if rng.random() < p_yes else "No",
                        selected=None,
                    )
                )
    pd.DataFrame(rows).to_parquet(directory / "cis_respondents.parquet")
    pd.DataFrame(answers).assign(selected=lambda d: d.selected.astype("boolean")).to_parquet(
        directory / "cis_responses.parquet"
    )


def test_segment_distribution_is_weighted_and_shrunk(tmp_path):
    _write_cis(tmp_path, ["18-34", "35-54", "55+"])
    store = AnchorStore(tmp_path, shrinkage=25)
    anchor = select_anchor(
        [_hit(QUESTION, ["Sí", "No"])], {"question": QUESTION, "options": ["Sí", "No"], "question_type": "single"}
    )
    young_women = store.segment_distribution(anchor, {"sexo": ["mujer"], "grupo_edad": ["18-34"]})
    national = young_women["national"]["Sí"]
    assert young_women["probabilities"]["Sí"] > national + 0.2
    assert sum(young_women["probabilities"].values()) == pytest.approx(1)
    assert [lv["level"] for lv in young_women["levels"]] == ["nacional", "sexo", "sexo + grupo_edad"]
    # Un segmento sin entrevistados cae al nivel superior en lugar de fallar
    empty = store.segment_distribution(anchor, {"sexo": ["mujer"], "grupo_edad": ["inexistente"]})
    assert empty["probabilities"]["Sí"] == pytest.approx(
        store.segment_distribution(anchor, {"sexo": ["mujer"]})["probabilities"]["Sí"]
    )


@pytest.fixture
def world(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    _write_cis(tmp_path, ["18-34", "35-54", "55+"])
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    record = dict(
        evidence_id="CIS9001_P1",
        source="CIS 9001",
        study_id="9001",
        question_text=QUESTION,
        options=["Sí", "No", "N.S."],
        topic="ia_tecnologia",
        fieldwork_date="2025-01-01",
        question_type="single",
        metadata={"synthetic": True},
    )
    (tmp_path / "e.json").write_text(json.dumps([record]), encoding="utf-8")
    evidence.ingest(tmp_path / "e.json")
    return PopulationRepository(frame, metadata), evidence, AnchorStore(tmp_path)


def _run(tmp_path, world, provider, question=QUESTION):
    population, evidence, store = world
    llm = LLMService(provider, Settings(llm_provider="mock", llm_retries=0), tmp_path / "cache")
    return asyncio.run(
        simulate(
            Question(question=question, options=["Sí", "No"]),
            population,
            evidence,
            llm,
            max_segments=6,
            anchors=store,
        )
    )


def test_same_question_returns_direct_statistics_without_llm(tmp_path, world):
    class Spy(MockLLMProvider):
        predictions = 0

        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                Spy.predictions += 1
            return await super().generate(schema, payload)

    result = _run(tmp_path, world, Spy())
    assert Spy.predictions == 0
    assert all(p["anchor"]["mode"] == "direct" for p in result["segment_predictions"].values())
    assert result["execution_metadata"]["status"] == "completed"
    assert set(result["national_distribution"]) == {"Sí", "No"}
    assert sum(result["national_distribution"].values()) == pytest.approx(1)
    mujeres = [s for s in result["segments"] if s["demographic_profile"].get("sexo") == "mujer"]
    hombres = [s for s in result["segments"] if s["demographic_profile"].get("sexo") == "hombre"]
    assert mujeres and hombres
    assert result["subgroup_distributions"]["sexo"]["mujer"]["Sí"] > result["subgroup_distributions"]["sexo"]["hombre"]["Sí"]


def test_similar_question_uses_prior_and_clamps_llm(tmp_path, world):
    class Extreme(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                assert payload["anchor"]["mode"] == "prior"
                return Prediction(
                    segment_id=payload["segment"]["segment_id"],
                    probabilities={"Sí": 0.99, "No": 0.01},
                    explanation="x",
                    evidence_ids=[],
                )
            return await super().generate(schema, payload)

    result = _run(tmp_path, world, Extreme(), question="¿Utiliza usted alguna aplicación de inteligencia artificial?")
    for p in result["segment_predictions"].values():
        assert p["anchor"]["mode"] == "prior"
        assert p["llm_probabilities"]["Sí"] == 0.99
        assert p["probabilities"]["Sí"] - p["anchor"]["prior"]["Sí"] <= p["anchor"]["max_shift"] + 1e-6
        assert sum(p["probabilities"].values()) == pytest.approx(1)


def test_payload_is_compact(tmp_path, world):
    sizes = []

    class Sizes(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                sizes.append(len(json.dumps(payload, ensure_ascii=False)))
            return await super().generate(schema, payload)

    _run(tmp_path, world, Sizes(), question="¿Utiliza usted alguna aplicación de inteligencia artificial?")
    assert sizes and max(sizes) < 3500


def test_single_answer_sums_are_repaired_within_tolerance(tmp_path):
    values = iter([{"Y": 0.55, "N": 0.45 - 0.03}, {"Y": 0.2, "N": 0.2}])

    class Sloppy(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            return Prediction(segment_id="a", probabilities=next(values), explanation="x")

    llm = LLMService(Sloppy(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c")
    ok = asyncio.run(
        llm.call(Prediction, {"n": 1}, lambda p: p.validate_context("a", ["Y", "N"], []), repair=lambda p: p.renormalized())
    )
    assert sum(ok.probabilities.values()) == pytest.approx(1)
    with pytest.raises(ValueError):  # suma 0.4: fuera de tolerancia, no se inventa una distribución
        asyncio.run(
            llm.call(Prediction, {"n": 2}, lambda p: p.validate_context("a", ["Y", "N"], []), repair=lambda p: p.renormalized())
        )


def test_evaluator_cannot_veto_a_valid_anchor(tmp_path, world):
    from mavericks.schemas import Assessment

    class Strict(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Assessment:
                return Assessment(
                    items=[{"evidence_id": e["evidence_id"], "relevance": "incompatible"} for e in payload["evidence"]]
                )
            return await super().generate(schema, payload)

    result = _run(tmp_path, world, Strict())
    assert result["execution_metadata"]["status"] == "completed"
    assert all(p["anchor"]["mode"] == "direct" for p in result["segment_predictions"].values())
    assert any("marcó como incompatible la evidencia ancla" in w for w in result["warnings"])


def test_direct_anchor_is_independent_of_methodologist_and_assessment(tmp_path, world):
    from mavericks.schemas import Assessment, Methodology

    class Garbage(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Methodology:
                return Methodology(
                    topic="salud", question_type="single", relevant_demographics=["xx"], retrieval_queries=["zzz qqq"]
                )
            if schema is Assessment:
                raise RuntimeError("evaluador caído")
            return await super().generate(schema, payload)

    class Spy(Garbage):
        predictions = 0

        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                Spy.predictions += 1
            return await super().generate(schema, payload)

    result = _run(tmp_path, world, Spy())
    assert Spy.predictions == 0
    assert result["anchor_spec"]["mode"] == "direct"
    assert all(p["anchor"]["mode"] == "direct" for p in result["segment_predictions"].values())


def test_retrieved_evidence_is_capped_and_keeps_the_anchor(tmp_path, world):
    population, evidence, store = world
    extra = [
        dict(evidence_id=f"X{i}", source="t", question_text=f"{QUESTION} variante {i}", options=["Sí", "No"], topic="ia_tecnologia")
        for i in range(12)
    ]
    (tmp_path / "x.json").write_text(json.dumps(extra), encoding="utf-8")
    evidence.ingest(tmp_path / "x.json")
    llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0, max_evidence_hits=4), tmp_path / "c2")
    result = asyncio.run(simulate(Question(question=QUESTION, options=["Sí", "No"]), population, evidence, llm, 4, anchors=store))
    ids = [h["evidence"]["evidence_id"] for h in result["retrieved_evidence"]]
    assert len(ids) <= 4 and "CIS9001_P1" in ids


def test_anchor_is_always_in_the_payload_even_if_outranked(tmp_path, world):
    """Regresión: con muchas preguntas parecidas la ancla quedaba fuera de las 3 evidencias y los segmentos fallaban."""
    population, evidence, store = world
    extra = [
        dict(evidence_id=f"Y{i}", source="t", question_text=f"{QUESTION} variante {i}", options=["Sí", "No"],
             topic="ia_tecnologia", question_type="single", study_id=str(50 + i), fieldwork_date="2025-02-01")
        for i in range(8)
    ]
    (tmp_path / "y.json").write_text(json.dumps(extra), encoding="utf-8")
    evidence.ingest(tmp_path / "y.json")
    seen = []

    class Spy(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                seen.append(payload)
            return await super().generate(schema, payload)

    llm = LLMService(Spy(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c3")
    from mavericks import payloads

    original = payloads.compact_evidence

    def outrank(hits, assessment, anchors=None, limit=3, keep_ids=()):
        # Simula que las variantes superan a la ancla en puntuación
        boosted = [{**h, "retrieval_score": h["retrieval_score"] + (5 if h["evidence"]["evidence_id"].startswith("Y") else 0)} for h in hits]
        return original(boosted, assessment, anchors, limit, keep_ids)

    import mavericks.graph as graph_module

    graph_module.compact_evidence = outrank
    try:
        result = asyncio.run(simulate(Question(question=QUESTION, options=["Sí", "No"]), population, evidence, llm, 4, anchors=store))
    finally:
        graph_module.compact_evidence = original
    assert result["execution_metadata"]["status"] == "completed", result.get("errors")
    assert result["anchor_spec"]["mode"] == "direct"
    assert all(p["anchor"]["mode"] == "direct" for p in result["segment_predictions"].values())


def test_select_anchor_prefers_exact_text_then_most_recent_wave():
    q = {"question": QUESTION, "options": ["Sí", "No"], "question_type": "single"}
    near = _hit(QUESTION + " variante", ["Sí", "No"], score=9.0, eid="NEAR")
    old = _hit(QUESTION, ["Sí", "No"], score=1.0, eid="OLD")
    new = _hit(QUESTION, ["Sí", "No"], score=0.5, eid="NEW")
    old["evidence"]["fieldwork_date"], new["evidence"]["fieldwork_date"] = "2024-01-01", "2025-06-01"
    chosen = select_anchor([near, old, new], q)
    assert chosen["evidence_id"] == "NEW" and chosen["mode"] == "direct"


def _run_with_settings(tmp_path, world, question, **settings):
    population, evidence, store = world
    seen = []

    class Spy(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                seen.append(payload)
            return await super().generate(schema, payload)

    llm = LLMService(Spy(), Settings(llm_provider="mock", llm_retries=0, **settings), tmp_path / "cfs")
    result = asyncio.run(
        simulate(Question(question=question, options=["Sí", "No"]), population, evidence, llm, 4, anchors=store)
    )
    return result, seen


def test_few_shot_adds_real_examples_only_when_enabled_and_not_for_direct(tmp_path, world):
    population, evidence, store = world
    records = [
        dict(evidence_id=f"F{i}", source="CIS", study_id=str(70 + i), question_text=f"Pregunta distinta número {i} sobre inteligencia artificial",
             options=["Sí", "No"], topic="ia_tecnologia", question_type="single", fieldwork_date="2025-02-01",
             metadata={"empirical_statistics": {"national": {"distribution": {"Sí": 0.8, "No": 0.2}, "n_respondents": 1000}}})
        for i in range(4)
    ]
    (tmp_path / "f.json").write_text(json.dumps(records), encoding="utf-8")
    evidence.ingest(tmp_path / "f.json")
    novel = "¿Cree que la inteligencia artificial cambiará su trabajo?"
    _, off = _run_with_settings(tmp_path, world, novel)
    assert off and all("examples" not in p for p in off)
    _, on = _run_with_settings(tmp_path, world, novel, few_shot=3)
    assert on and all(len(p["examples"]) == 3 for p in on)
    example = on[0]["examples"][0]
    assert example["distribution"]["Sí"] == 0.8 and example["n"] == 1000 and example["source"] == "CIS"
    direct_result, direct_payloads = _run_with_settings(tmp_path, world, QUESTION, few_shot=3)
    assert direct_payloads == [] and direct_result["anchor_spec"]["mode"] == "direct"
