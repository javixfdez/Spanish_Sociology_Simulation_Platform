"""Prior por analogía: la distribución de una pregunta nueva se estima con las distribuciones nacionales reales de
las preguntas más parecidas (por enunciado y opciones), con alineación de escalas y contracción a la uniforme.

Todo el filtrado contra fugas (excluded_ids, excluded_texts, corte temporal estricto, reservados fuera) lo hace
``EvidenceRepository.search`` antes de ordenar, igual en tfidf, embed e hybrid; aquí solo se puntúa lo que sobrevive.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .anchor import option_key, option_mapping, question_similarity

NON_SUBSTANTIVE = {"ns", "nc", "np", "no sabe", "no contesta", "no procede", "no aplicable"}
KINDS = ("label", "semantic", "ordinal")
# La similitud coseno de E5 es alta incluso entre textos no relacionados: se reescala al rango útil observado.
EMB_LOW, EMB_HIGH = 0.78, 0.95


@dataclass(frozen=True)
class AnalogConfig:
    k: int = 8
    power: float = 2.0
    min_sim: float = 0.3
    shrink: float = 0.5
    sim_mode: str = "max"  # lex | emb | max
    kind_weights: tuple = (1.0, 0.8, 0.5)  # label, semantic, ordinal


def substantive(options):
    return [o for o in options if option_key(o) not in NON_SUBSTANTIVE]


def build_national_table(cis_dir, out):
    """Distribución nacional ponderada de cada pregunta de respuesta única del CIS: {uid: {n, dist}}."""
    cis_dir = Path(cis_dir)
    questions = pd.read_parquet(cis_dir / "cis_questions.parquet", columns=["question_uid", "estudio_id", "question_type"])
    singles = set(questions[questions.question_type == "single"].question_uid)
    respondents = pd.read_parquet(cis_dir / "cis_respondents.parquet", columns=["estudio_id", "respondent_uid", "peso"])
    table = {}
    for study in questions.estudio_id.unique():
        responses = pq.read_table(
            cis_dir / "cis_responses.parquet", columns=["respondent_uid", "question_uid", "answer"],
            filters=[("estudio_id", "=", study)],
        ).to_pandas()
        responses = responses[responses.question_uid.isin(singles)]
        people = respondents[respondents.estudio_id == study][["respondent_uid", "peso"]]
        merged = responses.merge(people, on="respondent_uid", how="left")
        merged["peso"] = merged.peso.fillna(0.0)
        for uid, group in merged.groupby("question_uid"):
            total = group.peso.sum()
            if total <= 0:
                continue
            sums = group.groupby("answer").peso.sum() / total
            table[uid] = {"n": int(group.respondent_uid.nunique()), "dist": {k: float(v) for k, v in sums.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    return len(table)


def load_national_table(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else {}


def align_options(question_options, evidence_options, embedder=None):
    """(mapping opción de la pregunta -> opción de la evidencia, tipo) o (None, None) si no se puede alinear.

    label: etiquetas equivalentes; semantic: biyección por embeddings (mejor pareja mutua); ordinal: mismo número
    de opciones sustantivas alineadas por posición (escalas de graduación).
    """
    mapping = option_mapping(question_options, evidence_options)
    if mapping is not None:
        return mapping, "label"
    sub = substantive(evidence_options)
    if len(sub) != len(question_options) or len(sub) < 2:
        return None, None
    if embedder is not None:
        q = np.asarray(embedder.encode_queries(list(question_options)), dtype=float)
        e = np.asarray(embedder.encode_passages(list(sub)), dtype=float)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        sim = q @ e.T
        rows, cols = sim.argmax(axis=1), sim.argmax(axis=0)
        if all(cols[rows[i]] == i for i in range(len(rows))) and len(set(rows)) == len(rows):
            return {question_options[i]: sub[rows[i]] for i in range(len(rows))}, "semantic"
    return dict(zip(question_options, sub)), "ordinal"


def national_distribution(evidence, table):
    stats = evidence.get("metadata", {}).get("empirical_statistics", {}).get("national")
    if stats and stats.get("distribution"):
        return stats["distribution"]
    entry = table.get(evidence["evidence_id"])
    return entry["dist"] if entry else None


def candidates(hits, question_text, options, table, embedder=None):
    """Análogas utilizables: con distribución nacional real y escala alineable. Una por enunciado distinto."""
    out, seen = [], set()
    for hit in hits:
        e = hit["evidence"]
        original = e.get("metadata", {}).get("original_fields", {})
        if original.get("question_type", e.get("question_type", "single")) not in (None, "single"):
            continue
        distribution = national_distribution(e, table)
        if not distribution:
            continue
        mapping, kind = align_options(options, e.get("options") or list(distribution), embedder)
        if mapping is None:
            continue
        values = {q: float(distribution.get(ev, 0.0)) for q, ev in mapping.items()}
        mass = sum(values.values())
        if mass <= 0:
            continue
        key = " ".join((e.get("question_text") or "").lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "evidence_id": e["evidence_id"],
                "question": " ".join(str(e.get("question_text") or "").split())[:220],
                "source": str(e.get("source", ""))[:20],
                "date": e.get("fieldwork_date"),
                "lex": question_similarity(question_text, e.get("question_text") or ""),
                "emb": hit.get("embedding_similarity"),
                "kind": kind,
                "dist": {q: v / mass for q, v in values.items()},
                "study_id": str(e["study_id"]) if e.get("study_id") is not None else None,
                "mapping": mapping,
                "microdata": e["evidence_id"] in table,
            }
        )
    return out


def similarity(candidate, mode):
    lex = candidate["lex"]
    emb = candidate.get("emb")
    scaled = None if emb is None else float(np.clip((emb - EMB_LOW) / (EMB_HIGH - EMB_LOW), 0, 1))
    if mode == "lex" or scaled is None:
        return lex
    if mode == "emb":
        return scaled
    return max(lex, scaled)


def combine(cands, options, config):
    """Prior = media de las distribuciones análogas ponderada por similitud^power × calidad de alineación,
    contraída hacia la uniforme con intensidad ``shrink`` / (peso total + ``shrink``)."""
    uniform = {o: 1 / len(options) for o in options}
    weights = dict(zip(KINDS, config.kind_weights))
    scored = []
    for c in cands:
        s = similarity(c, config.sim_mode)
        if s >= config.min_sim:
            scored.append((s**config.power * weights[c["kind"]], s, c))
    scored.sort(key=lambda t: -t[0])
    scored = scored[: config.k]
    total = sum(w for w, _, _ in scored)
    if total <= 0:
        return {"prior": uniform, "best_similarity": max([similarity(c, config.sim_mode) for c in cands], default=0.0),
                "weight": 0.0, "used": 0, "best_kind": None, "analogs": [], "trust": 0.0, "weighted": []}
    mix = {o: sum(w * c["dist"][o] for w, _, c in scored) / total for o in options}
    trust = total / (total + config.shrink) if config.shrink > 0 else 1.0
    prior = {o: trust * mix[o] + (1 - trust) * uniform[o] for o in options}
    best = max(scored, key=lambda t: t[1])
    return {
        "prior": prior,
        "best_similarity": best[1],
        "weight": total,
        "used": len(scored),
        "best_kind": best[2]["kind"],
        "analogs": [{"evidence_id": c["evidence_id"], "similarity": round(s, 3), "kind": c["kind"], "source": c["source"],
                     "question": c.get("question"), "distribution": {k: round(v, 3) for k, v in c["dist"].items()}}
                    for _, s, c in scored],
        "trust": trust,
        "weighted": [(w, c) for w, _, c in scored],
    }


def analog_for_question(question, repository, table, config, filters=None, sources=None, pool=40):
    """Prior por analogía de una pregunta con los mismos filtros de fuga que usa el grafo."""
    filters = dict(filters or {})
    options = question["options"]
    query = f"{question['question']} | " + "; ".join(options[:8])
    hits = repository.search(query, limit=pool, options=options, topic_filter=False, sources=sources, **filters)
    cands = candidates(hits, question["question"], options, table, repository.embedder)
    return combine(cands, options, config)


def collect_case(case, repository, table, embedder=None, strict=True, retriever=None, sources=None, pool=40):
    """Candidatas de un caso con todas las protecciones de fuga: se guardan para ajustar la configuración sin recuperar."""
    options = case["options"]
    query = f"{case['question']} | " + "; ".join(options[:8])
    excluded_texts = [case["question"]] if strict else []
    hits = repository.search(
        query, limit=pool, excluded_ids=[case["evidence_id"]], excluded_texts=excluded_texts,
        cutoff=case.get("fieldwork_date"), options=options, topic=case.get("topic"), topic_filter=False,
        retriever=retriever, sources=sources,
    )
    return {
        "evidence_id": case["evidence_id"],
        "estudio_id": case.get("estudio_id"),
        "topic": case.get("topic"),
        "split": case.get("split", "test"),
        "question": case["question"],
        "options": options,
        "observed": case["observed_distribution"],
        "candidates": candidates(hits, case["question"], options, table, embedder),
    }


def tvd(p, q):
    return 0.5 * sum(abs(p[k] - q[k]) for k in q)


def mae(p, q):
    return float(np.mean([abs(p[k] - q[k]) for k in q]))


def evaluate(collected, config):
    rows = []
    for c in collected:
        result = combine(c["candidates"], c["options"], config)
        u = {o: 1 / len(c["options"]) for o in c["options"]}
        rows.append(
            {
                **{k: c[k] for k in ("evidence_id", "estudio_id", "topic", "split")},
                "mae": mae(result["prior"], c["observed"]),
                "mae_uniform": mae(u, c["observed"]),
                "tvd": tvd(result["prior"], c["observed"]),
                "tvd_uniform": tvd(u, c["observed"]),
                "best_similarity": result["best_similarity"],
                "best_kind": result["best_kind"],
                "used": result["used"],
                "prior": result["prior"],
            }
        )
    return rows


def summarize(rows):
    if not rows:
        return None
    t_model, t_uniform = np.mean([r["tvd"] for r in rows]), np.mean([r["tvd_uniform"] for r in rows])
    return {
        "n": len(rows),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "uniform": float(np.mean([r["mae_uniform"] for r in rows])),
        "simbench": float(100 * (1 - t_model / t_uniform)) if t_uniform > 0 else None,
    }


def by_similarity(rows, edges=(0.0, 0.3, 0.5, 0.7, 1.01)):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = [r for r in rows if lo <= r["best_similarity"] < hi]
        out[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = summarize(group)
    return out


def by_kind(rows):
    return {k: summarize([r for r in rows if r["best_kind"] == k]) for k in (*KINDS, None)}


def tune(collected, grid, folds=5, seed=0):
    """Validación cruzada agrupada por estudio: elige la configuración con menor MAE en entrenamiento y la mide
    fuera de muestra. Devuelve la configuración final (ajustada con todos los casos) y las cifras CV."""
    groups = sorted({str(c["estudio_id"]) for c in collected})
    order = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {groups[g]: i % folds for i, g in enumerate(order)}
    held = []
    chosen = []
    for f in range(folds):
        train = [c for c in collected if fold_of[str(c["estudio_id"])] != f]
        test = [c for c in collected if fold_of[str(c["estudio_id"])] == f]
        if not train or not test:
            continue
        best = min(grid, key=lambda cfg: summarize(evaluate(train, cfg))["mae"])
        chosen.append(best)
        held += evaluate(test, best)
    final = min(grid, key=lambda cfg: summarize(evaluate(collected, cfg))["mae"])
    default = AnalogConfig()
    return {
        "final": final,
        "cv": summarize(held),
        "default": summarize(evaluate(collected, default)),
        "in_sample_best": summarize(evaluate(collected, final)),
        "fold_configs": [vars(replace(c)) for c in chosen],
        "held_rows": held,
    }


def default_grid():
    import itertools

    return [
        AnalogConfig(k=k, power=p, min_sim=m, shrink=s, sim_mode=mode)
        for k, p, m, s, mode in itertools.product((3, 5, 8, 10), (1.0, 2.0, 4.0), (0.0, 0.3, 0.5), (0.0, 0.25, 0.5, 1.0), ("lex", "emb", "max"))
    ]
