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


def compact_evidence(hits, assessment, anchors=None, limit=MAX_EVIDENCE):
    """Hasta ``limit`` evidencias compatibles, con su evaluación y el resultado nacional empírico."""
    items = {i["evidence_id"]: i for i in assessment.get("items", [])}
    compact = []
    for hit in sorted(hits, key=lambda h: -h["retrieval_score"]):
        e = hit["evidence"]
        item = items.get(e["evidence_id"], {})
        if item.get("relevance") == "incompatible":
            continue
        entry = {
            "evidence_id": e["evidence_id"],
            "question_text": e.get("question_text"),
            "options": (e.get("options") or [])[:10],
            "study_id": e.get("study_id"),
            "fieldwork_date": e.get("fieldwork_date"),
            "relevance": item.get("relevance"),
            "differences": item.get("differences", [])[:2],
        }
        stats = e.get("metadata", {}).get("empirical_statistics", {}).get("national")
        if stats:
            entry["empirical_national"] = {k: round(v, 3) for k, v in stats["distribution"].items()}
        elif anchors is not None and e.get("study_id"):
            original = e.get("metadata", {}).get("original_fields", {})
            try:
                dist, n = anchors.national(e["study_id"], e["evidence_id"], original.get("question_type", "single"))
                if n:
                    entry["empirical_national"] = {k: round(v, 3) for k, v in dist.items()}
            except Exception:  # sin microdatos para esta evidencia: se omite la cifra
                pass
        compact.append(entry)
        if limit and len(compact) == limit:
            break
    return compact
