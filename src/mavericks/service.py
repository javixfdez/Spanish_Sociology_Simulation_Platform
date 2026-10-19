"""Capa de servicio: una sola función ``predict`` que usan la interfaz, la CLI y el modo por lotes.

Presets:
  rapido_sin_llm  ancla directa (oleada más reciente) o prior por analogía + segmentos empíricos, sin GPU.
  recomendado     lo anterior y, si hay LLM disponible y calibración, mezcla calibrada para preguntas no repetidas.
  llm_completo    el pipeline completo con LLM (grafo de LangGraph).
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import assist, explain, fastcal
from . import categories as cat
from .analog import AnalogConfig, analog_for_question, load_national_table
from .analog_segments import group_estimates
from .anchor import AnchorStore, constraints_of, select_anchor
from .evidence import EvidenceRepository
from .population import PopulationLoader, PopulationRepository, build_segments
from .schemas import Prediction, Question, Settings
from .statistics import aggregate, load_calibration, select_stratum
from .topics import TOPICS, classify_topic

log = logging.getLogger("mavericks.service")
PRESETS = ("recomendado", "rapido_sin_llm", "llm_completo", "ajustado")
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
FT_MAX_OPTIONS = 7  # letras A-G del entrenamiento del modelo ajustado
CONTRACTION = 0.5  # fracción de la diferencia empírica de las análogas que se aplica a cada grupo
LLM_RANGE_NOTE = "Los rangos están calibrados para la estimación estadística; con el modelo de lenguaje no se han calibrado por separado."
TWO_OPTIONS_NOTE = "En preguntas de sí/no sin datos parecidos, el reparto real es difícil de anticipar; tómalo como orientación."
SUBGROUP_NOTE = "Diferencias estimadas a partir de preguntas parecidas del CIS; confianza baja."
# Umbral del modo directo validado con el backtest (153 casos) y con pares de preguntas del CIS: ver README
DIRECT_SIMILARITY = 0.85
# Configuración del prior por analogía elegida por validación cruzada (analog-eval)
ANALOG = AnalogConfig(k=5, power=4.0, min_sim=0.0, shrink=0.0, sim_mode="lex")


class ServiceError(ValueError):
    """Error de uso con un mensaje comprensible para quien utiliza la interfaz."""


class OutOfScopeError(ServiceError):
    """La pregunta no se puede responder con encuestas de opinión; ``coverage`` explica por qué y cómo reformularla."""

    def __init__(self, coverage):
        super().__init__(f"{coverage['explanation']} {coverage['reformulation']}".strip())
        self.coverage = coverage


@dataclass
class Paths:
    population: str = "data/processed/synthetic_population.parquet"
    evidence: str = "data/processed_v2/evidence_v2.sqlite"
    evidence_fallback: str = "data/processed/evidence.sqlite"
    national_table: str = "data/processed_v2/cis_national.json"
    cis_dir: str = "data/processed"
    calibration: str = "outputs/baseline_v1/calibration.json"
    calibration_fallback: str = "data/processed/calibration.json"
    fast_validation: str = "outputs/fast_validation.json"
    models: str = "outputs/models"


@dataclass
class Resources:
    population: PopulationRepository
    evidence: EvidenceRepository
    anchors: AnchorStore
    table: dict
    calibration: dict | None
    fast: dict | None = None
    paths: Paths = field(default_factory=Paths)
    prior_model: object = None  # MLP de producción (solo lo usa este servicio; el backtest nunca)
    prior_model_error: str | None = None
    prior_intervals: dict | None = None
    prior_validation: dict | None = None
    ft_intervals: dict | None = None  # rangos del modo ajustado (residuos de a1 en dev_grande)
    ft_validation: dict | None = None


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
    prior_model, prior_error = None, None
    try:
        from .prior_v2.production import ProductionPrior

        prior_model = ProductionPrior.load(paths.models)
    except Exception as exc:  # sin modelo entrenado el prior es el KNN y se indica
        prior_error = f"No hay modelo del prior aprendido ({type(exc).__name__}); ejecuta mavericks train-prior"
    return Resources(
        PopulationRepository(frame, metadata),
        EvidenceRepository(evidence_path),
        AnchorStore(paths.cis_dir),
        load_national_table(paths.national_table),
        calibration,
        fastcal.load(paths.fast_validation),
        paths,
        prior_model,
        prior_error,
        _read_json(Path(paths.models) / "prior_intervals.json"),
        _read_json(Path(paths.models) / "prior_validation.json"),
        _read_json(Path(paths.models) / "ft_intervals.json"),
        _read_json(Path(paths.models) / "ft_validation.json"),
    )


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


_EMBEDDER = {"obj": None, "error": None}
_EMBEDDER_LOCK = threading.Lock()


def get_embedder():
    """e5 multilingüe en CPU, cargado una sola vez (singleton). None si no está disponible (se usa el KNN)."""
    with _EMBEDDER_LOCK:
        if _EMBEDDER["obj"] is None and _EMBEDDER["error"] is None:
            try:
                from .embeddings import SentenceTransformerEmbedder
                from .prior_v2.embcache import MODEL

                _EMBEDDER["obj"] = SentenceTransformerEmbedder(MODEL, device="cpu", local_files_only=True)
            except Exception as exc:
                _EMBEDDER["error"] = f"{type(exc).__name__}: {exc}"
        return _EMBEDDER["obj"]


def embedder_error():
    return _EMBEDDER["error"]


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
    """Rango probable por opción; ``radius`` es un número o una función de la probabilidad de la opción."""
    if radius is None:
        return None
    fn = radius if callable(radius) else (lambda p: radius)
    return {o: [max(0.0, p - fn(p)), min(1.0, p + fn(p))] for o, p in distribution.items()}


def _prior_intervals(calib, group, options, national, subgroup_ratio):
    """Rangos del prior (media KNN+MLP): cuantil por grupo de nº de opciones (2 frente a 3 o más) con la escala de fastcal."""
    q = calib["groups"][group]

    def at(level):
        return lambda p: q[level] * float(fastcal.scale(np.asarray(p)))

    radius90, radius50 = at("q90"), at("q50")
    coverage = calib.get("coverage", {})

    def covered(level):
        return coverage.get(f"fuera_un_conjunto_{level}", {}).get(group, {}).get("cobertura")

    info = {
        "group": group,
        "combined": False,
        "cases": q["preguntas"],
        "mean_halfwidth_pp": float(np.mean([radius50(national[o]) for o in options]) * 100),
        "mean_halfwidth_pp_90": float(np.mean([radius90(national[o]) for o in options]) * 100),
        "coverage_50": covered(50),
        "coverage_90": covered(90),
        "calibrated_on": calib.get("calibrada_con"),
    }
    return radius90, (lambda p: radius90(p) * subgroup_ratio), radius50, info


def _confidence(calibration, mode, fast_rows=None, prior_validation=None, n_options=None):
    """Nivel de confianza por modo y error típico medido para ESE modo (no el global), con su número de casos."""
    if prior_validation and mode in ("prior", "none"):
        group = "2 opciones" if n_options == 2 else "3 o más"
        cell = prior_validation["error_tipico"][group]
        return {
            "level": CONFIDENCE[mode],
            "typical_error": cell["mae"],
            "backtest_cases": cell["preguntas"],
            "basis": cell.get("basis") or f"preguntas nuevas de {group} en tres conjuntos no vistos",
        }
    if fast_rows is not None:
        own = [r["mae"] for r in fast_rows if r["mode"] == mode]
        return {
            "level": CONFIDENCE[mode],
            "typical_error": float(np.mean(own)) if own else None,
            "backtest_cases": len(own),
            "basis": f"modo {MODE_LABELS[mode]}",
        }
    block = (calibration or {}).get("by_anchor_mode", {}).get(mode)
    return {
        "level": CONFIDENCE[mode],
        "typical_error": block["mae"] if block else None,
        "backtest_cases": block["n_cases"] if block else 0,
        "basis": f"modo {MODE_LABELS[mode]}",
    }


def _fast_intervals(model, mode, options, national, subgroup_ratio):
    """Rango probable por opción del modo rápido: (radio(p) nacional, radio(p) de subgrupos, info) o (None, None, None)."""
    if not model:
        return None, None, None, None
    n = len(options)
    probe, group = fastcal.radius(model, mode, n, 0.5)
    if probe is None:
        return None, None, None, None

    def national_radius(p):
        return fastcal.radius(model, mode, n, p)[0]

    def core_radius(p):
        return fastcal.radius(model, mode, n, p, level="q50")[0]

    def subgroup_radius(p):
        return national_radius(p) * subgroup_ratio

    info = {
        "combined": mode in model["combined"],
        "cases": group["cases"],
        "group": group["key"],
        "mean_halfwidth_pp": float(np.mean([national_radius(national[o]) for o in options]) * 100),
    }
    return national_radius, subgroup_radius, core_radius, info


def _spontaneous(spec, options):
    """Opciones que en la pregunta del CIS usada no se leen al entrevistado (espontáneas)."""
    if not spec:
        return []
    return [o for o in options if assist.is_spontaneous(spec["mapping"].get(o, ""))]


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
        "(media ponderada por similitud): es la situación con más incertidumbre."
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


def _subgroups(resources, options, mode, spec, group_specs, national, radius, override=None):
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
    if override is not None:  # modo ajustado: el modelo ajustado da cada subgrupo directamente
        estimated = override
    elif mode != "direct" and group_specs:
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
            elif estimated is not None and category in estimated.get(dimension, {}):
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


def _blend_prior(resources, question, options, knn, enabled):
    """Media 50/50 KNN + MLP; si el modelo o e5 no están disponibles (o es una validación) devuelve el KNN y lo indica."""
    knn = {o: float(knn[o]) for o in options}

    def fallback(reason):
        return knn, {"metodo": "knn", "motivo": reason}

    if not enabled:
        return fallback("validación con corte temporal: no se usa el modelo de producción")
    model = resources.prior_model
    if model is None:
        return fallback(resources.prior_model_error or "no hay modelo del prior aprendido")
    if not model.supports(options):
        return fallback(f"el modelo aprendido cubre de 2 a {model.meta.get('max_options', 7)} opciones")
    embedder = get_embedder()
    if embedder is None:
        return fallback(f"e5 no disponible ({embedder_error()})")
    try:
        mlp = model.predict(question, options, embedder)
    except Exception as exc:
        return fallback(f"el modelo aprendido falló ({type(exc).__name__})")
    mixed = {o: 0.5 * knn[o] + 0.5 * float(mlp[i]) for i, o in enumerate(options)}
    return mixed, {
        "metodo": "media_knn_mlp",
        "pesos": {"knn": 0.5, "mlp": 0.5},
        "modelo": {k: model.meta.get(k) for k in ("trained_at", "cutoff_date", "n_questions")},
    }


def _analog_prior(resources, question, options, topic, filters):
    """KNN actual (producción): el mismo que se validó, con el tema como bonificación de la búsqueda."""
    return analog_for_question(
        {"question": question, "options": options},
        resources.evidence,
        resources.table,
        ANALOG,
        {**filters, "topic": topic},
        sources={"cis"},
    )


def _fast(resources, question, options, topic, n_segments, direct_sim, prior_sim, filters=None):
    filters = filters or {}
    use_production_prior = not filters  # con filtros (validación) nunca se usa el modelo entrenado con todo el CIS
    query = f"{question} | " + "; ".join(options[:8])
    found = {}
    for text in (question, query):  # la pregunta literal recupera la misma pregunta en otras oleadas
        for hit in resources.evidence.search(
            text, limit=40, options=options, topic=topic, topic_filter=False, sources={"cis"}, **filters
        ):
            found.setdefault(hit["evidence"]["evidence_id"], hit)
    hits = list(found.values())
    spec = select_anchor(
        hits,
        {"question": question, "options": options, "question_type": "single"},
        direct_sim,
        prior_sim,
        prior_guard=assist.PRIOR_GUARD,
        topic=topic,
    )
    warnings = []
    similarity = None
    if spec is not None:
        similarity = spec["similarity"]
        mode = spec["mode"]
        items = [{"evidence_id": spec["evidence_id"], "similarity": spec["similarity"]}]
        extras = [
            h["evidence"]["evidence_id"] for h in hits if h["evidence"]["evidence_id"] != spec["evidence_id"]
        ][:4]
        items += [{"evidence_id": e} for e in extras]
        group_specs = [(1.0, spec)]
        if mode == "direct":
            national = _national_from_segments(resources, options, spec, n_segments)
            prior_used = {"metodo": "ancla_directa"}
        else:
            analog = _analog_prior(resources, question, options, topic, filters)
            national, prior_used = _blend_prior(resources, question, options, analog["prior"], use_production_prior)
    else:
        mode = "none"
        analog = _analog_prior(resources, question, options, topic, filters)
        national, prior_used = _blend_prior(resources, question, options, analog["prior"], use_production_prior)
        similarity = analog.get("best_similarity")
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
    return mode, national, spec, group_specs, items, warnings, similarity, prior_used


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


def _ft_estimate(question, options, shares_of, ft, year=None):
    """Reparto nacional y por subgrupo del modelo ajustado (un servidor aparte). Lanza ``FtError`` si no responde o responde mal."""
    from .ft import data as ft_data

    shown = [assist.display_label(o) for o in options]
    items = [{"question": question, "options": shown, "profile": ft_data.profile_text(None, None), "year": year}]
    keys = [(None, None)]
    for dimension, categories in shares_of.items():
        for category in categories:
            items.append({"question": question, "options": shown, "profile": ft_data.profile_text(dimension, category), "year": year})
            keys.append((dimension, category))
    probs, seconds = ft.predict_batch(items)
    national = dict(zip(options, probs[0], strict=True))
    groups = {}
    for (dimension, category), p in zip(keys[1:], probs[1:], strict=True):
        groups.setdefault(dimension, {})[category] = dict(zip(options, p, strict=True))
    return national, groups, seconds


def _shares(resources):
    frame = resources.population.frame
    out = {}
    for dimension in SUBGROUP_DIMENSIONS:
        if dimension in frame:
            present = set(frame[dimension].dropna())
            out[dimension] = {c: _population_share(frame, dimension, c) for c in _ORDER[dimension] if c in present}
    return out


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
    ft=None,
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
    scope = assist.scope_check(question, service if preset not in ("rapido_sin_llm", "ajustado") else None)
    if not scope["in_scope"]:
        raise OutOfScopeError(assist.coverage_card("none", None, scope))

    wants_llm = preset == "llm_completo" or (preset == "recomendado" and service is not None)
    if wants_llm and service is None:
        raise ServiceError("El modo con LLM necesita un modelo disponible (Ollama).")
    used_llm = False  # ¿las cifras mezclan la salida del LLM?
    ft_groups = ft_info = None
    ft_fallback = False
    legacy_llm = question_type != "single"  # preguntas de respuesta múltiple: solo el pipeline completo con LLM
    if legacy_llm:
        if not wants_llm:
            raise ServiceError("Sin LLM solo se pueden estimar preguntas de respuesta única.")
        result = _llm_path(resources, question, options, question_type, settings, service, n_segments)
        if result["execution_metadata"]["status"] != "completed":
            raise ServiceError("El modelo no pudo completar todos los segmentos; inténtalo de nuevo.")
        spec = result.get("anchor_spec")
        mode = spec["mode"] if spec else "none"
        national = result["national_distribution"]
        items = [
            {"evidence_id": h["evidence"]["evidence_id"], "similarity": None} for h in result["retrieved_evidence"][:5]
        ]
        group_specs = [(1.0, spec)] if spec else None
        prior_used = {"metodo": "llm"}
        used_llm = True
        warnings += [w for w in result.get("warnings", []) if "DEMO" not in w]
    else:
        mode, national, spec, group_specs, items, extra, similarity, prior_used = _fast(
            resources,
            question,
            options,
            topic,
            n_segments,
            min(settings.anchor_direct_similarity, DIRECT_SIMILARITY),
            settings.anchor_prior_similarity,
        )
        warnings += extra
        if preset == "ajustado" and mode != "direct" and len(options) > FT_MAX_OPTIONS:
            ft_fallback = True
            warnings.append(f"El modelo ajustado cubre de 2 a {FT_MAX_OPTIONS} opciones; se muestra la estimación del modo rápido.")
        elif preset == "ajustado" and mode != "direct":
            from .ft.client import FtClient, FtError

            try:
                ft_national, ft_groups, ft_seconds = _ft_estimate(question, options, _shares(resources), ft or FtClient())
                ft_info = {"seconds": ft_seconds, "rapid_national": {o: float(national[o]) for o in options}}
                national = ft_national
                prior_used = {"metodo": "modelo_ajustado", "prior_rapido": prior_used}
            except FtError as exc:
                log.warning("Modo ajustado: %s", exc)
                ft_fallback = True
                warnings.append("El modelo ajustado no respondió; se muestra la estimación del modo rápido.")
        if wants_llm and mode != "direct":
            # Modo con LLM: LLM (peso w) + prior estadístico (1 − w). Las preguntas repetidas ya son estadística directa.
            try:
                result = _llm_path(resources, question, options, question_type, settings, service, n_segments)
                if result["execution_metadata"]["status"] != "completed":
                    raise ServiceError("segmentos sin completar")
                w = settings.llm_blend_weight
                llm_national = result["national_distribution"]
                national = {o: w * float(llm_national[o]) + (1 - w) * float(national[o]) for o in options}
                prior_used = {"metodo": "mezcla_llm", "w_llm": w, "prior": prior_used}
                used_llm = True
            except Exception as exc:
                log.warning("El modo con LLM no pudo completar la estimación: %s", exc)
                warnings.append(
                    "El modelo de lenguaje no pudo completar la estimación; se muestra la estimación estadística (modo rápido)."
                )

    calibration = resources.calibration
    fast_payload = resources.fast if not legacy_llm else None
    fast_model = fast_payload["by_buckets" if fast_payload.get("chosen") == "by_buckets" else "by_mode"]["model"] if fast_payload else None
    radius = sub_radius = core_radius = interval_info = None
    group = "2 opciones" if len(options) == 2 else "3 o más"
    calib = resources.prior_intervals
    ft_calib = resources.ft_intervals
    if ft_groups is not None:  # nunca los rangos del prior con las cifras del modelo ajustado
        if ft_calib and group in ft_calib["groups"]:
            radius, sub_radius, core_radius, interval_info = _prior_intervals(ft_calib, group, options, national, 1.0)
        fast_model = None
    elif not legacy_llm and mode in ("prior", "none") and calib and group in calib["groups"]:
        _, block = select_stratum(calibration, mode, min_cases=1)
        ratio = 1.0
        if block:
            sub_overall = _subgroup_radius(calibration, mode)
            base = _radius(block["option_errors"])
            ratio = max(1.0, sub_overall / base) if sub_overall and base else 1.0
        radius, sub_radius, core_radius, interval_info = _prior_intervals(calib, group, options, national, ratio)
        fast_model = None
    elif fast_model:
        _, block = select_stratum(calibration, mode, min_cases=1)
        ratio = 1.0
        if block:
            sub_overall = _subgroup_radius(calibration, mode)
            base = _radius(block["option_errors"])
            ratio = max(1.0, sub_overall / base) if sub_overall and base else 1.0
        radius, sub_radius, core_radius, interval_info = _fast_intervals(fast_model, mode, options, national, ratio)
    if radius is None and not fast_model and interval_info is None and ft_groups is None:
        _, block = select_stratum(calibration, mode)
        radius = _radius(block["option_errors"]) if block else None
        sub_radius = _subgroup_radius(calibration, mode)
    if radius is None:
        warnings.append("Sin calibración suficiente: no se muestra el rango probable.")
    if interval_info and interval_info.get("combined"):
        warnings.append(
            "Hay pocas preguntas de validación de este tipo (prior); el rango probable se calcula junto con las preguntas "
            "sin equivalente en el CIS, por lo que es más amplio."
        )
    notes = []
    if used_llm:
        notes.append(LLM_RANGE_NOTE)
    if len(options) == 2 and mode in ("prior", "none"):
        notes.append(TWO_OPTIONS_NOTE)
    evidence = _evidence_entries(resources.evidence, resources.table, items)
    subgroups = _subgroups(resources, options, mode, spec, group_specs, national, sub_radius, ft_groups)
    output = {
        "question": question,
        "options": options,
        "question_type": question_type,
        "preset": preset,
        "topic": topic,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "explanation": _explain(mode, evidence),
        "confidence": _confidence(
            calibration, mode, fast_payload["rows"] if fast_payload else None,
            resources.ft_validation if ft_groups is not None and resources.ft_validation else resources.prior_validation, len(options),
        ),
        "fallback": ft_fallback,
        "ajustado": ft_info,
        "prior_usado": prior_used,
        "notes": notes,
        "national": {
            "distribution": {o: float(national[o]) for o in options},
            "interval": _intervals({o: float(national[o]) for o in options}, radius),
            "interval_core": _intervals({o: float(national[o]) for o in options}, core_radius),
            "interval_info": interval_info,
        },
        "spontaneous": _spontaneous(spec, options),
        "coverage": assist.coverage_card(mode, evidence[0] if evidence else None),
        "subgroups": subgroups,
        "subgroup_note": None if mode == "direct" else SUBGROUP_NOTE,
        "evidence": evidence,
        "warnings": warnings,
        "seconds": time.monotonic() - started,
    }
    return explain.add_template(output)


def suggest_options(question, resources=None, service=None, topic=None):
    """Opciones sugeridas para una pregunta escrita sin opciones (ver ``assist.suggest_options``)."""
    resources = resources or load_resources()
    return assist.suggest_options(question, resources, service, topic if topic in TOPICS else classify_topic(question))


def assess_coverage(question, resources=None, service=None, topic=None):
    """Clasificación previa: respaldada, basada en parecidas, sin datos comparables o fuera de alcance."""
    resources = resources or load_resources()
    return assist.assess_coverage(question, resources, service, topic if topic in TOPICS else classify_topic(question))
