"""Cargas útiles compactas para los agentes: solo lo que el LLM necesita, sin metadatos internos."""

PROFILE_KEYS = [
    "sexo",
    "edad",
    "ccaa",
    "tamuni",
    "nivel_estudios",
    "ingresos_hogar_tramo",
    "situacion_laboral",
    "nacionalidad",
]
COMPOSITION_DIMENSIONS = [
    "grupo_edad",
    "sexo",
    "nivel_estudios",
    "ccaa",
    "tamuni",
    "ingresos_hogar_tramo",
    "situacion_laboral",
    "nacionalidad",
]
MAX_EVIDENCE = 3


def _clip(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _summary(distribution, top=6):
    """Distribución resumida: las opciones más frecuentes con dos decimales."""
    items = sorted(distribution.items(), key=lambda kv: -kv[1])[:top]
    return {_clip(k, 60): round(v, 2) for k, v in items}


_SOURCE_NAMES = {"eurobarometro": "Eurobarómetro (muestra española)", "ess": "ESS (muestra española)"}


def build_examples(hits, anchors=None, n=3, exclude_ids=()):
    """Ejemplos reales (few-shot): preguntas recuperadas parecidas con su distribución nacional observada.

    Una por enunciado distinto (la oleada más reciente), para que el modelo vea la escala real de las respuestas.
    """
    chosen, seen = [], set()
    for hit in sorted(hits, key=lambda h: -h["retrieval_score"]):
        e = hit["evidence"]
        key = " ".join((e.get("question_text") or "").lower().split())
        if e["evidence_id"] in exclude_ids or not key or key in seen:
            continue
        stats = e.get("metadata", {}).get("empirical_statistics", {}).get("national")
        distribution, size = (stats["distribution"], stats.get("n_respondents")) if stats else (None, None)
        if distribution is None and anchors is not None and e.get("study_id"):
            original = e.get("metadata", {}).get("original_fields", {})
            try:
                dist, count = anchors.national(e["study_id"], e["evidence_id"], original.get("question_type", "single"))
            except Exception:
                continue
            ns = {k: v for k, v in dist.items() if k.lower().replace(".", "") not in {"ns", "nc", "no procede"}}
            mass = sum(ns.values())
            distribution, size = ({k: v / mass for k, v in ns.items()} if mass > 0 else None), count
        if not distribution or len(distribution) < 2:
            continue
        seen.add(key)
        chosen.append(
            {
                "question": _clip(e.get("question_text"), 220),
                "distribution": _summary(distribution),
                "n": size,
                "source": _SOURCE_NAMES.get(e.get("source"), "CIS"),
                "fieldwork": e.get("fieldwork_date"),
            }
        )
        if len(chosen) == n:
            break
    return chosen


def compact_segment(segment, total_weight):
    homogeneous = set(segment.get("homogeneous_dimensions", segment["demographic_profile"]))
    composition = {}
    for dimension in COMPOSITION_DIMENSIONS:
        shares = segment.get("socioeconomic_profile", {}).get(dimension)
        if not shares or dimension in homogeneous:
            continue
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:4]
        composition[dimension] = {k: round(v, 2) for k, v in top if v >= 0.05}
    return {
        "segment_id": segment["segment_id"],
        "population_share": round(segment["population_weight"] / total_weight, 4),
        "n_synthetic": segment["n_synthetic"],
        "profile": segment["demographic_profile"],
        "merged_categories": segment.get("merged_categories", {}),
        "composition": composition,
        "representative_profiles": [
            {k: r[k] for k in PROFILE_KEYS if k in r} for r in segment.get("representative_profiles", [])[:3]
        ],
    }


def compact_evidence(hits, assessment, anchors=None, limit=MAX_EVIDENCE, keep_ids=()):
    """Hasta ``limit`` evidencias compatibles, con su evaluación y el resultado nacional empírico."""
    items = {i["evidence_id"]: i for i in assessment.get("items", [])}
    compact = []
    # Las evidencias protegidas (la ancla) van siempre primero y no cuentan para el límite de contexto.
    ordered = sorted(hits, key=lambda h: (h["evidence"]["evidence_id"] not in keep_ids, -h["retrieval_score"]))
    for hit in ordered:
        e = hit["evidence"]
        item = items.get(e["evidence_id"], {})
        if item.get("relevance") == "incompatible" and e["evidence_id"] not in keep_ids:
            continue
        entry = {
            "evidence_id": e["evidence_id"],
            "question_text": _clip(e.get("question_text"), 240),
            "options": [_clip(o, 60) for o in (e.get("options") or [])[:8]],
            "study_id": e.get("study_id"),
            "fieldwork_date": e.get("fieldwork_date"),
            "relevance": item.get("relevance"),
            "differences": [_clip(d, 120) for d in item.get("differences", [])[:2]],
        }
        stats = e.get("metadata", {}).get("empirical_statistics", {}).get("national")
        if stats:
            entry["empirical_national"] = _summary(stats["distribution"])
        elif anchors is not None and e.get("study_id"):
            original = e.get("metadata", {}).get("original_fields", {})
            try:
                dist, n = anchors.national(e["study_id"], e["evidence_id"], original.get("question_type", "single"))
                if n:
                    entry["empirical_national"] = _summary(dist)
            except Exception:  # sin microdatos para esta evidencia: se omite la cifra
                pass
        compact.append(entry)
        if limit and len([c for c in compact if c["evidence_id"] not in keep_ids]) >= limit:
            break
    return compact
