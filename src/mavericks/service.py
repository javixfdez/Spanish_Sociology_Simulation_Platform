"""Capa de servicio: una sola función ``predict`` que usan la interfaz, la CLI y el modo por lotes.

Presets:
  rapido_sin_llm  ancla directa (oleada más reciente) o prior por analogía + segmentos empíricos, sin GPU.
  recomendado     lo anterior y, si hay LLM disponible y calibración, mezcla calibrada para preguntas no repetidas.
  llm_completo    el pipeline completo con LLM (grafo de LangGraph).
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import categories as cat
from .analog import AnalogConfig, analog_for_question, load_national_table
from .analog_segments import group_estimates
from .anchor import AnchorStore, constraints_of, select_anchor
from .evidence import EvidenceRepository
from .population import PopulationLoader, PopulationRepository, build_segments
from .schemas import Prediction, Question, Settings
from .statistics import aggregate, load_calibration, select_stratum
from .topics import TOPICS, classify_topic

PRESETS = ("recomendado", "rapido_sin_llm", "llm_completo")
SUBGROUP_DIMENSIONS = {
    "sexo": "Sexo",
    "grupo_edad": "Edad",
    "ccaa": "Comunidad autónoma",
    "nivel_estudios": "Estudios",
    "ingresos_hogar_tramo": "Ingresos del hogar",
}
_ORDER = {
    "sexo": cat.SEXO,
    "grupo_edad": cat.GRUPOS_EDAD,
    "ccaa": cat.CCAA,
    "nivel_estudios": cat.NIVEL_ESTUDIOS,
    "ingresos_hogar_tramo": cat.INGRESOS,
}
MODE_LABELS = {"direct": "directo", "prior": "prior", "none": "sin ancla"}
CONFIDENCE = {"direct": "alta", "prior": "media", "none": "baja"}
ALPHA = 0.1
CONTRACTION = 0.5  # fracción de la diferencia empírica de las análogas que se aplica a cada grupo
SUBGROUP_NOTE = "Diferencias estimadas a partir de preguntas parecidas del CIS; confianza baja."
# Umbral del modo directo validado con el backtest (153 casos) y con pares de preguntas del CIS: ver README
DIRECT_SIMILARITY = 0.85
# Configuración del prior por analogía elegida por validación cruzada (analog-eval)
ANALOG = AnalogConfig(k=5, power=4.0, min_sim=0.0, shrink=0.0, sim_mode="lex")


class ServiceError(ValueError):
    """Error de uso con un mensaje comprensible para quien utiliza la interfaz."""


@dataclass
class Paths:
    population: str = "data/processed/synthetic_population.parquet"
    evidence: str = "data/processed_v2/evidence_v2.sqlite"
    evidence_fallback: str = "data/processed/evidence.sqlite"
    national_table: str = "data/processed_v2/cis_national.json"
    cis_dir: str = "data/processed"
    calibration: str = "outputs/baseline_v1/calibration.json"
    calibration_fallback: str = "data/processed/calibration.json"


@dataclass
class Resources:
    population: PopulationRepository
    evidence: EvidenceRepository
    anchors: AnchorStore
    table: dict
    calibration: dict | None
    paths: Paths = field(default_factory=Paths)


def load_resources(paths=None):
    paths = paths or Paths()
    for label, path in (
        ("la población sintética", paths.population),
        ("los microdatos del CIS", paths.cis_dir),
    ):
        if not Path(path).exists():
            raise ServiceError(f"Faltan {label} ({path}). Genéralos con etl-population y etl-cis.")
    frame, metadata = PopulationLoader().load(paths.population)
    evidence_path = paths.evidence if Path(paths.evidence).exists() else paths.evidence_fallback
    if not Path(evidence_path).exists():
        raise ServiceError("Falta el almacén de evidencia. Ejecuta ingest-surveys.")
    calibration = load_calibration(paths.calibration) or load_calibration(paths.calibration_fallback)
    return Resources(
        PopulationRepository(frame, metadata),
        EvidenceRepository(evidence_path),
        AnchorStore(paths.cis_dir),
        load_national_table(paths.national_table),
        calibration,
        paths,
    )


def _radius(errors, alpha=ALPHA):
    errors = np.sort(np.asarray(errors, dtype=float))
    if len(errors) < 20:
        return None
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * (1 - alpha))))
    return float(errors[rank - 1])


def _subgroup_radius(calibration, mode):
    """Radio del intervalo de subgrupos: errores de subgrupos del backtest para el modo o, si faltan, los globales."""
    if not calibration:
        return None
    for block in (calibration.get("by_anchor_mode", {}).get(mode), calibration.get("overall")):
        if block and len(block.get("subgroup_option_errors", [])) >= 20:
            return _radius(block["subgroup_option_errors"])
    return None


def _intervals(distribution, radius):
    if radius is None:
        return None
    return {o: [max(0.0, p - radius), min(1.0, p + radius)] for o, p in distribution.items()}


def _confidence(calibration, mode):
    """Nivel de confianza por modo y error típico medido en el backtest para ese modo."""
    name, block = select_stratum(calibration, mode)
    typical = block["mae"] if block else None
    n_cases = block["n_cases"] if block else 0
    return {
        "level": CONFIDENCE[mode],
        "typical_error": typical,
        "backtest_cases": n_cases,
        "basis": name,
    }


def _explain(mode, evidence):
    if mode == "direct" and evidence:
        e = evidence[0]
        return (
            f"Esta pregunta ya existe en el CIS (estudio {e['study_id']}, campo hasta {e['fieldwork_end'] or 's/f'}): "
            "partimos de su resultado real en la oleada más reciente y lo adaptamos al perfil de cada grupo."
        )
    if mode == "prior" and evidence:
        e = evidence[0]
        return (
            f"Hay una pregunta parecida en el CIS (estudio {e['study_id']}, similitud {e['similarity']:.0%}): "
            "usamos su distribución real como punto de partida; la diferencia de redacción añade incertidumbre."
        )
    return (
        "No hay ninguna pregunta equivalente en el CIS. Estimamos la distribución a partir de las preguntas más parecidas "
        "(media ponderada por similitud) y no podemos distinguir entre grupos: es la situación con más incertidumbre."
    )


def _record_lookup(repository, evidence_id):
    index = repository._index()
    position = index["position"].get(evidence_id)
    return index["records"][position] if position is not None else None


def _evidence_entries(repository, table, items):
    """Preguntas del CIS usadas: id, estudio, fecha, enunciado, distribución real y similitud."""
    entries = []
    for item in items:
        record = _record_lookup(repository, item["evidence_id"])
        distribution = item.get("distribution")
        if distribution is None and record is not None:
            entry = table.get(record.evidence_id)
            distribution = entry["dist"] if entry else None
        entries.append(
            {
                "evidence_id": item["evidence_id"],
                "study_id": getattr(record, "study_id", None),
                "fieldwork_end": getattr(record, "fieldwork_date", None),
                "question": item.get("question") or getattr(record, "question_text", ""),
                "distribution": distribution,
                "similarity": item.get("similarity"),
                "source": item.get("source") or getattr(record, "source", ""),
            }
        )
    return entries


def _population_share(frame, dimension, category):
    mask = frame[dimension] == category
    return float(frame.loc[mask, "peso"].sum() / frame["peso"].sum())


def _subgroups(resources, options, mode, spec, group_specs, national, radius):
    """Distribución por categoría de cada dimensión.

    Directo: microdatos de la pregunta repetida. Prior y sin ancla: diferencias empíricas de las preguntas análogas,
    contraídas a la mitad hacia el total y con raking para que cuadren con él (ver ``group_estimates``).
    """
    frame = resources.population.frame
    shares, labels = {}, {}
    for dimension, label in SUBGROUP_DIMENSIONS.items():
        if dimension in frame:
            present = set(frame[dimension].dropna())
            shares[dimension] = {
                c: _population_share(frame, dimension, c) for c in _ORDER[dimension] if c in present
            }
            labels[dimension] = label
    estimated = None
    if mode != "direct" and group_specs:
        estimated = group_estimates(group_specs, options, shares, resources.anchors, national, CONTRACTION)
    out = {}
    for dimension, categories in shares.items():
        groups = {}
        for category, share in categories.items():
            n_cis = None
            if mode == "direct" and spec is not None:
                estimate = resources.anchors.segment_distribution(spec, {dimension: [category]})
                distribution = {o: float(estimate["probabilities"][o]) for o in options}
                n_cis = estimate["finest"]["n"] if estimate["finest"]["level"] != "nacional" else 0
            elif estimated is not None:
                distribution = estimated[dimension][category]
            else:
                distribution = {o: float(national[o]) for o in options}
            groups[category] = {
                "distribution": distribution,
                "interval": _intervals(distribution, radius),
                "population_share": share,
                "n_cis": n_cis,
            }
        out[dimension] = {"label": labels[dimension], "groups": groups}
    return out


def _national_from_segments(resources, options, spec, n_segments):
    segments = build_segments(
        resources.population, n_segments, dimensions=["grupo_edad", "sexo", "nivel_estudios"]
    )
    predictions = {}
    for segment in segments:
        estimate = resources.anchors.segment_distribution(spec, constraints_of(segment))
        probabilities = {o: float(estimate["probabilities"][o]) for o in options}
        predictions[segment["segment_id"]] = Prediction(
            segment_id=segment["segment_id"], probabilities=probabilities, explanation="ancla empírica"
        )
    return aggregate(segments, predictions, options, "single", evidence_ids=[])


def _fast(resources, question, options, topic, n_segments, direct_sim, prior_sim):
    query = f"{question} | " + "; ".join(options[:8])
    found = {}
    for text in (question, query):  # la pregunta literal recupera la misma pregunta en otras oleadas
        for hit in resources.evidence.search(
            text, limit=40, options=options, topic=topic, topic_filter=False, sources={"cis"}
        ):
            found.setdefault(hit["evidence"]["evidence_id"], hit)
    hits = list(found.values())
    spec = select_anchor(
        hits, {"question": question, "options": options, "question_type": "single"}, direct_sim, prior_sim
    )
    warnings = []
    if spec is not None:
        mode = spec["mode"]
        national = _national_from_segments(resources, options, spec, n_segments)
        items = [{"evidence_id": spec["evidence_id"], "similarity": spec["similarity"]}]
        extras = [
            h["evidence"]["evidence_id"] for h in hits if h["evidence"]["evidence_id"] != spec["evidence_id"]
        ][:4]
        items += [{"evidence_id": e} for e in extras]
        group_specs = [(1.0, spec)]
    else:
        mode = "none"
        analog = analog_for_question(
            {"question": question, "options": options},
            resources.evidence,
            resources.table,
            ANALOG,
            sources={"cis"},
        )
        national = analog["prior"]
        items = analog["analogs"]
        group_specs = [
            (
                weight,
                {
                    "study_id": c["study_id"],
                    "evidence_id": c["evidence_id"],
                    "mapping": c["mapping"],
                    "question_type": "single",
                },
            )
            for weight, c in analog["weighted"]
            if c.get("microdata") and c.get("study_id")
        ]
        if not analog["analogs"] or analog["best_similarity"] < 0.45:
            warnings.append(
                "Las preguntas más parecidas del CIS están poco relacionadas con la tuya: el resultado es poco informativo."
            )
    return mode, national, spec, group_specs, items, warnings


def _llm_path(resources, question, options, question_type, settings, service, n_segments):
    """Pipeline con LLM (grafo). ``service`` es un LLMService ya creado (Mock en los tests)."""
    from .graph import simulate

    result = asyncio.run(
        simulate(
            Question(question=question, options=options, question_type=question_type),
            resources.population,
            resources.evidence,
            service,
            n_segments,
            anchors=resources.anchors,
            calibration=resources.calibration,
        )
    )
    return result


def predict(
    question,
    options,
    question_type="single",
    preset="rapido_sin_llm",
    topic=None,
    resources=None,
    settings=None,
    service=None,
    n_segments=10,
):
    """Predicción completa. Devuelve un dict serializable (ver la clave de cada sección)."""
    started = time.monotonic()
    if preset not in PRESETS:
        raise ServiceError(f"Preset desconocido: {preset}")
    options = [o.strip() for o in options if o and o.strip()]
    try:
        Question(question=question, options=options, question_type=question_type)
    except ValueError as exc:
        raise ServiceError(
            "Revisa la pregunta y las opciones: mínimo dos opciones distintas y una pregunta de 5 caracteres."
        ) from exc
    settings = settings or Settings()
    resources = resources or load_resources()
    topic = topic if topic in TOPICS else classify_topic(question)
    warnings = []

    if preset == "llm_completo" or (preset == "recomendado" and service is not None):
        if service is None:
            raise ServiceError("El preset con LLM necesita un modelo disponible (Ollama).")
        result = _llm_path(resources, question, options, question_type, settings, service, n_segments)
        if result["execution_metadata"]["status"] != "completed":
            raise ServiceError("El modelo no pudo completar todos los segmentos; inténtalo de nuevo.")
        spec = result.get("anchor_spec")
        mode = spec["mode"] if spec else "none"
        national = result["national_distribution"]
        items = [
            {"evidence_id": h["evidence"]["evidence_id"], "similarity": None}
            for h in result["retrieved_evidence"][:5]
        ]
        group_specs = [(1.0, spec)] if spec else None
        warnings += [w for w in result.get("warnings", []) if "DEMO" not in w]
    else:
        if question_type != "single":
            raise ServiceError("Sin LLM solo se pueden estimar preguntas de respuesta única.")
        mode, national, spec, group_specs, items, extra = _fast(
            resources,
            question,
            options,
            topic,
            n_segments,
            min(settings.anchor_direct_similarity, DIRECT_SIMILARITY),
            settings.anchor_prior_similarity,
        )
        warnings += extra

    calibration = resources.calibration
    _, block = select_stratum(calibration, mode)
    radius = _radius(block["option_errors"]) if block else None
    sub_radius = _subgroup_radius(calibration, mode)
    if radius is None:
        warnings.append("Sin calibración suficiente: no se muestran intervalos.")
    evidence = _evidence_entries(resources.evidence, resources.table, items)
    subgroups = _subgroups(resources, options, mode, spec, group_specs, national, sub_radius)
    return {
        "question": question,
        "options": options,
        "question_type": question_type,
        "preset": preset,
        "topic": topic,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "explanation": _explain(mode, evidence),
        "confidence": _confidence(calibration, mode),
        "national": {
            "distribution": {o: float(national[o]) for o in options},
            "interval": _intervals(national, radius),
            "radius": radius,
        },
        "subgroups": subgroups,
        "subgroup_note": None if mode == "direct" else SUBGROUP_NOTE,
        "evidence": evidence,
        "warnings": warnings,
        "seconds": time.monotonic() - started,
    }
