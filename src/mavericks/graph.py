import asyncio
import operator
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .anchor import constraints_of, project_to_band, select_anchor
from .payloads import compact_evidence, compact_segment
from .population import build_segments
from .schemas import Assessment, AssessmentItem, Methodology, Narrative, Prediction, Question
from .statistics import aggregate, audit, subgroup_distributions, subgroup_resolution
from .topics import classify_topic


def merge_unique(left, right):
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"Predicciones duplicadas: {overlap}")
    return {**left, **right}


class SimulationState(TypedDict, total=False):
    simulation_id: str
    question: dict
    methodology: dict
    population_version: str
    population_metadata: dict
    retrieved_evidence: list
    anchor_spec: dict
    evidence_assessment: dict
    segments: list
    segment_predictions: Annotated[dict, merge_unique]
    national_distribution: dict
    subgroup_distributions: dict
    subgroup_resolution: dict
    trust_metrics: dict
    warnings: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    stages: Annotated[list, operator.add]
    final_report: dict
    execution_metadata: dict


def build_graph(
    repository,
    evidence_repository,
    service,
    max_segments=5,
    rag=True,
    retrieval_filters=None,
    anchors=None,
):
    filters = retrieval_filters or {}
    settings = service.settings

    def validate_input(state):
        question = Question.model_validate(state["question"])
        warnings = [repository.metadata.coverage]
        if not repository.metadata.categories:
            warnings.append("Sin catálogo de categorías de Fase 1: validación categórica no verificada")
        if repository.metadata.synthetic:
            warnings.append("DEMO / SYNTHETIC: población ficticia, sin representatividad")
        if service.provider.synthetic:
            warnings.append("DEMO / SYNTHETIC: proveedor Mock, sin validez predictiva")
        return {
            "question": question.model_dump(),
            "population_version": repository.metadata.population_version,
            "population_metadata": repository.metadata.model_dump(),
            "warnings": warnings,
            "stages": ["validate_input"],
        }

    async def methodologist(state):
        question = state["question"]
        try:
            value = await service.call(Methodology, question)
            if value.question_type != question["question_type"]:
                raise ValueError("El metodólogo cambió el tipo de respuesta solicitado")
            return {
                "methodology": value.model_dump(),
                "warnings": value.warnings,
                "stages": ["methodologist_agent"],
            }
        except Exception as exc:
            # Sin metodólogo se continúa con una metodología determinista, avisando del fallo.
            fallback = Methodology(
                topic=classify_topic(question["question"]),
                question_type=question["question_type"],
                relevant_demographics=["grupo_edad", "sexo", "nivel_estudios"],
                retrieval_queries=[question["question"]],
                warnings=[f"Metodólogo LLM no disponible ({exc}); metodología determinista"],
            )
            return {
                "methodology": fallback.model_dump(),
                "warnings": fallback.warnings,
                "stages": ["methodologist_agent:fallback"],
            }

    def retrieve(state):
        found = {}
        if rag:
            for query in state["methodology"]["retrieval_queries"]:
                for hit in evidence_repository.search(
                    query,
                    target_population=state["question"]["target_population"],
                    options=state["question"]["options"],
                    topic=state["methodology"]["topic"],
                    **filters,
                ):
                    e = hit["evidence"]
                    if e.get("metadata", {}).get("synthetic") and not repository.metadata.synthetic:
                        continue
                    found[e["evidence_id"]] = hit
        spec = None
        if anchors is not None and found:
            spec = select_anchor(
                list(found.values()),
                state["question"],
                settings.anchor_direct_similarity,
                settings.anchor_prior_similarity,
            )
        return {
            "retrieved_evidence": list(found.values()),
            "anchor_spec": spec,
            "stages": ["retrieve_sociological_evidence"],
            "warnings": []
            if found
            else ["Sin evidencia recuperada: estimación no fundamentada empíricamente"],
        }

    async def interpret(state):
        hits = state["retrieved_evidence"]
        ids = {hit["evidence"]["evidence_id"] for hit in hits}
        if not ids:
            # Sin evidencia no hay nada que evaluar: preguntar al LLM invita a inventar evidencias.
            empty = Assessment(items=[], missing_information=["Sin evidencia recuperada"])
            return {"evidence_assessment": empty.model_dump(), "stages": ["interpret_evidence"]}

        def validate(value):
            returned = [item.evidence_id for item in value.items]
            if set(returned) != ids or len(returned) != len(ids):
                raise ValueError("Assessment debe evaluar exactamente cada evidence_id recuperado")

        try:
            value = await service.call(
                Assessment,
                {"question": state["question"], "evidence": compact_evidence(hits, {}, anchors, limit=None)},
                validate,
            )
        except Exception as exc:
            # Evaluación conservadora: toda la evidencia se considera indirecta, con aviso explícito.
            note = f"Evaluación automática no disponible ({exc}); evidencia tratada como indirecta"
            value = Assessment(
                items=[
                    AssessmentItem(
                        evidence_id=hit["evidence"]["evidence_id"], relevance="indirect", differences=[note]
                    )
                    for hit in hits
                ],
                missing_information=[note],
            )
            return {
                "evidence_assessment": value.model_dump(),
                "warnings": [note],
                "stages": ["interpret_evidence:fallback"],
            }
        return {"evidence_assessment": value.model_dump(), "stages": ["interpret_evidence"]}

    def segment(state):
        wanted = state.get("methodology", {}).get("relevant_demographics", [])
        return {
            "segments": build_segments(repository, max_segments, dimensions=wanted),
            "stages": ["segment_population"],
        }

    def dispatch(state):
        return {"stages": ["dispatch_segment_agents"]}

    def route(state):
        assessment = state["evidence_assessment"]
        evidence = compact_evidence(state["retrieved_evidence"], assessment, anchors)
        allowed = {e["evidence_id"] for e in evidence}
        spec = state.get("anchor_spec")
        if spec and spec["evidence_id"] not in allowed:
            spec = None  # la evidencia ancla fue descartada por incompatible
        total = sum(s["population_weight"] for s in state["segments"])
        sends = []
        for s in state["segments"]:
            anchor = None
            if spec:
                estimate = anchors.segment_distribution(spec, constraints_of(s))
                anchor = {
                    "mode": spec["mode"],
                    "evidence_id": spec["evidence_id"],
                    "study_id": spec["study_id"],
                    "question_similarity": spec["similarity"],
                    "prior": {k: round(v, 4) for k, v in estimate["probabilities"].items()},
                    "national": {k: round(v, 4) for k, v in estimate["national"].items()},
                    "basis": estimate["finest"],
                    "levels": estimate["levels"],
                    "max_shift": settings.anchor_max_shift,
                }
            sends.append(
                Send(
                    "respondent_agent",
                    {
                        "segment": compact_segment(s, total),
                        "question": state["question"],
                        "evidence": evidence,
                        "anchor": anchor,
                    },
                )
            )
        return sends

    async def respondent(state):
        segment_id = state["segment"]["segment_id"]
        question = state["question"]
        ids = [e["evidence_id"] for e in state["evidence"]]
        anchor = state.get("anchor")
        try:
            if anchor and anchor["mode"] == "direct":
                # Misma pregunta en el CIS: estadística directa del segmento, sin intervención del LLM.
                basis = anchor["basis"]
                value = Prediction(
                    segment_id=segment_id,
                    probabilities=anchor["prior"],
                    explanation=f"Estadística directa del CIS (estudio {anchor['study_id']}, pregunta "
                    f"{anchor['evidence_id']}) para el perfil del segmento: nivel {basis['level']}, "
                    f"n={basis['n']}, n efectivo={basis['n_eff']}.",
                    evidence_ids=[anchor["evidence_id"]],
                    limitations=["Sin ajuste por cambios desde el trabajo de campo", "Contracción jerárquica hacia niveles superiores"],
                )
                if question["question_type"] == "single":
                    total = sum(value.probabilities.values())
                    value = value.model_copy(update={"probabilities": {k: v / total for k, v in value.probabilities.items()}})
                value.validate_context(segment_id, question["options"], ids, question["question_type"])
                dump = value.model_dump()
                dump["anchor"] = {"mode": "direct", "basis": basis, "evidence_id": anchor["evidence_id"], "prior": anchor["prior"]}
                return {"segment_predictions": {segment_id: dump}, "stages": [f"respondent_agent:{segment_id}:direct"]}
            payload = {**state, "options": question["options"]}
            value = await service.call(
                Prediction,
                payload,
                lambda p: p.validate_context(segment_id, question["options"], ids, question["question_type"]),
                repair=(lambda p: p.renormalized()) if question["question_type"] == "single" else None,
            )
            dump = value.model_dump()
            if anchor and anchor["mode"] == "prior":
                options = question["options"]
                prior = np.array([anchor["prior"][o] for o in options])
                raw = np.array([value.probabilities[o] for o in options])
                low, high = np.clip(prior - anchor["max_shift"], 0, 1), np.clip(prior + anchor["max_shift"], 0, 1)
                if question["question_type"] == "single":
                    final = project_to_band(raw, low, high)
                else:
                    final = np.clip(raw, low, high)
                dump["llm_probabilities"] = dict(value.probabilities)
                dump["probabilities"] = {o: float(p) for o, p in zip(options, final)}
                dump["anchor"] = {
                    "mode": "prior",
                    "basis": anchor["basis"],
                    "evidence_id": anchor["evidence_id"],
                    "prior": anchor["prior"],
                    "max_shift": anchor["max_shift"],
                }
                Prediction.model_validate(dump).validate_context(
                    segment_id, options, ids, question["question_type"]
                )
            return {
                "segment_predictions": {segment_id: dump},
                "stages": [f"respondent_agent:{segment_id}"],
            }
        except Exception as exc:
            return {
                "errors": [{"segment_id": segment_id, "error": str(exc)}],
                "stages": [f"respondent_agent:{segment_id}:failed"],
            }

    def aggregation(state):
        predictions = {k: Prediction.model_validate(v) for k, v in state["segment_predictions"].items()}
        q = state["question"]
        completed_weight = sum(
            s["population_weight"] for s in state["segments"] if s["segment_id"] in predictions
        )
        coverage = completed_weight / sum(s["population_weight"] for s in state["segments"])
        if state.get("errors") or len(predictions) != len(state["segments"]):
            return {
                "national_distribution": {},
                "subgroup_distributions": {},
                "warnings": [f"Estimación nacional abortada; cobertura={coverage:.3f}"],
                "execution_metadata": {"coverage": coverage, "status": "failed"},
                "stages": ["aggregate_predictions"],
            }
        national = aggregate(state["segments"], predictions, q["options"], q["question_type"])
        return {
            "national_distribution": national,
            "subgroup_distributions": subgroup_distributions(
                state["segments"], predictions, q["options"], q["question_type"]
            ),
            "subgroup_resolution": subgroup_resolution(state["segments"]),
            "execution_metadata": {"coverage": coverage, "status": "completed"},
            "stages": ["aggregate_predictions"],
        }

    def auditing(state):
        return {
            "trust_metrics": audit(
                state["retrieved_evidence"], repository.metadata, state["segment_predictions"]
            ),
            "stages": ["audit_confidence"],
        }

    async def analyst(state):
        try:
            narrative = await service.call(
                Narrative,
                {
                    "national_distribution": state["national_distribution"],
                    "subgroup_distributions": state["subgroup_distributions"],
                    "evidence": compact_evidence(state["retrieved_evidence"], state["evidence_assessment"], anchors),
                    "trust_metrics": state["trust_metrics"],
                    "warnings": state["warnings"],
                },
            )
            stage = "analyst_agent"
        except Exception as exc:
            narrative = Narrative(summary=f"Narrativa no disponible ({exc}). Las cifras son las del modelo.")
            stage = "analyst_agent:fallback"
        # Numerical results remain authoritative, separate from generated interpretation.
        return {
            "final_report": {
                "narrative": narrative.model_dump(),
                "model_estimates": state["national_distribution"],
                "synthetic_explanations": True,
            },
            "stages": [stage],
            "warnings": [] if stage == "analyst_agent" else [narrative.summary],
        }

    graph = StateGraph(SimulationState)
    nodes = {
        "validate_input": validate_input,
        "methodologist_agent": methodologist,
        "retrieve_sociological_evidence": retrieve,
        "interpret_evidence": interpret,
        "segment_population": segment,
        "dispatch_segment_agents": dispatch,
        "respondent_agent": respondent,
        "aggregate_predictions": aggregation,
        "audit_confidence": auditing,
        "analyst_agent": analyst,
    }
    for name, function in nodes.items():
        graph.add_node(name, function)
    chain = [
        START,
        "validate_input",
        "methodologist_agent",
        "retrieve_sociological_evidence",
        "interpret_evidence",
        "segment_population",
        "dispatch_segment_agents",
    ]
    for source, target in zip(chain, chain[1:]):
        graph.add_edge(source, target)
    graph.add_conditional_edges("dispatch_segment_agents", route, ["respondent_agent"])
    graph.add_edge("respondent_agent", "aggregate_predictions")
    graph.add_edge("aggregate_predictions", "audit_confidence")
    graph.add_conditional_edges(
        "audit_confidence",
        lambda s: "failed" if s.get("errors") else "ok",
        {"failed": END, "ok": "analyst_agent"},
    )
    graph.add_edge("analyst_agent", END)
    return graph.compile()


async def simulate(
    question,
    repository,
    evidence_repository,
    service,
    max_segments=5,
    rag=True,
    retrieval_filters=None,
    anchors=None,
):
    started = time.monotonic()
    graph = build_graph(repository, evidence_repository, service, max_segments, rag, retrieval_filters, anchors)
    state = await asyncio.wait_for(
        graph.ainvoke(
            {
                "simulation_id": str(uuid.uuid4()),
                "question": question.model_dump(),
                "segment_predictions": {},
                "warnings": [],
                "errors": [],
                "stages": [],
            },
            config={"max_concurrency": service.settings.llm_max_concurrency},
        ),
        timeout=service.settings.simulation_timeout,
    )
    state["execution_metadata"].update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": time.monotonic() - started,
            "model_name": service.provider.name,
            "synthetic": service.provider.synthetic or repository.metadata.synthetic,
            "llm_calls": service.calls,
            "rag": rag,
            "anchors": anchors is not None,
        }
    )
    return state
