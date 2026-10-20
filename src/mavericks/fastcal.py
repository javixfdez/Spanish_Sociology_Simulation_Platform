"""Validación y calibración de intervalos del modo rápido (sin LLM) con los casos de test y de desarrollo.

``collect`` ejecuta la misma ruta que la interfaz (``service._fast``) con las protecciones anti-fuga del backtest
(la propia pregunta excluida y solo evidencia anterior a su trabajo de campo). ``fit`` calcula el radio del intervalo
del 90 % por grupo (modo y número de opciones) como cuantil conforme de |error| / escala(p), donde escala(p) hace los
intervalos más estrechos para probabilidades extremas. ``cross_validate`` mide cobertura y anchura dejando cada
pregunta fuera.
"""

import json
from pathlib import Path

import numpy as np

ALPHA = 0.1
MIN_CASES = 15  # casos por modo para tener radio propio; por debajo se combinan prior y sin ancla
FLOOR = 0.02


def scale(p):
    """Escala del error esperado: máxima en p=0,5 y pequeña en los extremos."""
    return np.sqrt(p * (1 - p) + FLOOR)


def n_bucket(n_options):
    return "2" if n_options <= 2 else "3-4" if n_options <= 4 else "5+"


def collect(cases, resources, split_of=None, progress=None):
    from . import service as svc

    rows = []
    for i, case in enumerate(cases):
        options = case["options"]
        filters = {"excluded_ids": [case["evidence_id"]], "cutoff": case.get("fieldwork_date")}
        mode, national, _spec, _groups, _items, _w, similarity, _prior = svc._fast(
            resources, case["question"], options, case.get("topic"), 10, svc.DIRECT_SIMILARITY, 0.55, filters
        )
        p = [float(national[o]) for o in options]
        obs = [float(case["observed_distribution"][o]) for o in options]
        rows.append(
            {
                "evidence_id": case["evidence_id"],
                "topic": case.get("topic"),
                "set": (split_of or {}).get(case["evidence_id"], "test"),
                "mode": mode,
                "similarity": similarity,
                "n_options": len(options),
                "predicted": p,
                "observed": obs,
                "mae": float(np.mean(np.abs(np.array(p) - np.array(obs)))),
                "mae_uniform": float(np.mean(np.abs(1 / len(options) - np.array(obs)))),
            }
        )
        if progress:
            progress(i + 1, len(cases))
    return rows


def _quantile(scores, level=1 - ALPHA):
    scores = np.sort(np.asarray(scores, dtype=float))
    if len(scores) == 0:
        return None
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * level)))
    return float(scores[rank - 1])


def _combined_modes(rows):
    counts = {}
    for r in rows:
        counts[r["mode"]] = counts.get(r["mode"], 0) + 1
    small = [m for m in ("prior", "none") if counts.get(m, 0) < MIN_CASES]
    # si prior no alcanza los casos mínimos se combina con sin ancla (más datos y más incertidumbre)
    return ["prior", "none"] if "prior" in small else small


def group_key(mode, bucket, combined):
    base = "prior+none" if mode in combined else mode
    return f"{base}|{bucket}" if bucket else base


def fit(rows, by_buckets=False):
    """Cuantiles por grupo: {"groups": {clave: {q, cases, options}}, "combined": [modos combinados]}."""
    combined = _combined_modes(rows)
    scores, cases = {}, {}
    for r in rows:
        bucket = n_bucket(r["n_options"]) if by_buckets else None
        keys = {group_key(r["mode"], None, combined)}
        if bucket:
            keys.add(group_key(r["mode"], bucket, combined))
        for key in keys:
            for p, o in zip(r["predicted"], r["observed"], strict=True):
                scores.setdefault(key, []).append(abs(p - o) / scale(p))
            cases.setdefault(key, set()).add(r["evidence_id"])
    groups = {
        k: {"q": _quantile(v), "q50": _quantile(v, 0.5), "cases": len(cases[k]), "options": len(v)}
        for k, v in scores.items()
    }
    return {"groups": groups, "combined": combined, "by_buckets": by_buckets}


def radius(model, mode, n_options, p, min_options=40, level="q"):
    """(radio, grupo) del rango del 90 % (``level="q"``) o del 50 % (``"q50"``) de una opción con probabilidad ``p``."""
    base = "prior+none" if mode in model["combined"] else mode
    keys = [f"{base}|{n_bucket(n_options)}"] if model["by_buckets"] else []
    keys.append(base)
    for key in keys:
        group = model["groups"].get(key)
        if group and group["options"] >= min_options:
            return group[level] * float(scale(p)), {"key": key, **group}
    return None, None


def cross_validate(rows, by_buckets=False):
    """Cobertura y anchura media (puntos) dejando cada pregunta fuera, por modo."""
    stats = {}
    for r in rows:
        model = fit([x for x in rows if x["evidence_id"] != r["evidence_id"]], by_buckets)
        for p, o in zip(r["predicted"], r["observed"], strict=True):
            rad, _ = radius(model, r["mode"], r["n_options"], p)
            if rad is None:
                continue
            s = stats.setdefault(r["mode"], {"hit": 0, "n": 0, "width": 0.0})
            s["n"] += 1
            s["hit"] += abs(p - o) <= rad
            s["width"] += 2 * min(rad, 1.0) * 100
    total_hit = sum(s["hit"] for s in stats.values())
    total_n = sum(s["n"] for s in stats.values())
    out = {
        m: {"coverage": s["hit"] / s["n"], "mean_width_pp": s["width"] / s["n"], "options": s["n"]}
        for m, s in stats.items()
    }
    out["global"] = {"coverage": total_hit / total_n, "options": total_n}
    return out


def refit(path):
    """Reajusta los modelos de rango a partir de las filas ya recogidas (sin volver a ejecutar los casos)."""
    payload = load(path)
    rows = payload["rows"]
    for buckets in (False, True):
        name = "by_buckets" if buckets else "by_mode"
        payload[name] = {"model": fit(rows, buckets), "cv": cross_validate(rows, buckets)}
    payload["chosen"] = "by_mode"
    save(path, payload)
    return payload


def save(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
