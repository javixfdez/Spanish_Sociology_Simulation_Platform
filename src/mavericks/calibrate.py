"""Calibración conjunta de la distribución final:  final = afilar(w·p_LLM + (1−w)·p_prior, α).

w (peso del LLM) y α (nitidez) se ajustan JUNTOS, después de mezclar con el prior, por tramos de similitud de la mejor
análoga, optimizando la JSD (no solo el MAE) y validando con validación cruzada dejando fuera un tema cada vez.
"""

import itertools

import numpy as np

from .analog import AnalogConfig, combine
from .sharpen import sharpen

W_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ALPHA_GRID = (0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0)
EPS = 1e-9


def blend(p_llm, p_prior, w, alpha):
    options = list(p_llm)
    mixed = {o: w * p_llm[o] + (1 - w) * p_prior[o] for o in options}
    return sharpen(mixed, alpha) if alpha != 1.0 else mixed


def jsd(p, q):
    a = np.array([max(p[k], EPS) for k in q])
    b = np.array([max(q[k], EPS) for k in q])
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    kl = lambda x, y: float((x * np.log2(x / y)).sum())  # noqa: E731
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def log_loss(p, q):
    """Entropía cruzada de lo observado q bajo lo predicho p (en nats)."""
    return float(-sum(q[k] * np.log(max(p[k], EPS)) for k in q))


def mae(p, q):
    return float(np.mean([abs(p[k] - q[k]) for k in q]))


def tvd(p, q):
    return 0.5 * sum(abs(p[k] - q[k]) for k in q)


def entropy(p):
    v = np.array([max(x, EPS) for x in p.values()])
    v /= v.sum()
    return float(-(v * np.log(v)).sum())


def build_rows(results, analog_cache, config=None, modes=("none", "prior"), split="test"):
    """Casos con predicción del LLM y prior por analogía: lo necesario para ajustar y evaluar la mezcla."""
    config = config or AnalogConfig()
    cache = {c["evidence_id"]: c for c in analog_cache}
    rows = []
    for r in results:
        if r["status"] != "completed" or r["anchor_mode"] not in modes or r["evidence_id"] not in cache:
            continue
        c = cache[r["evidence_id"]]
        result = combine(c["candidates"], c["options"], config)
        rows.append(
            {
                "evidence_id": r["evidence_id"],
                "topic": r.get("topic"),
                "split": split,
                "mode": r["anchor_mode"],
                "p_llm": r["predicted"],
                "p_prior": result["prior"],
                "sim": result["best_similarity"],
                "observed": r["observed"],
            }
        )
    return rows


class Blend:
    """Parámetros (w, α) por tramo de similitud: ``edges`` delimita los tramos."""

    def __init__(self, edges, params):
        self.edges, self.params = edges, params

    def bin(self, sim):
        return sum(sim >= e for e in self.edges[1:-1])

    def apply(self, row):
        w, alpha = self.params[self.bin(row["sim"])]
        return blend(row["p_llm"], row["p_prior"], w, alpha)

    def describe(self):
        labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(self.edges[:-1], self.edges[1:])]
        return [{"similarity": lb, "w": w, "alpha": a} for lb, (w, a) in zip(labels, self.params)]


def _objective(name):
    return {"jsd": jsd, "logloss": log_loss, "mae": mae}[name]


def fit(rows, edges=(0.0, 0.5, 1.01), objective="jsd", w_grid=W_GRID, alpha_grid=ALPHA_GRID, constant=False):
    """Mejor (w, α) por tramo (o único si ``constant``) sobre ``rows``."""
    loss = _objective(objective)
    probe = Blend(edges, [(1.0, 1.0)] * (len(edges) - 1))
    bins = [list() for _ in range(len(edges) - 1)] if not constant else [list(rows)]
    if not constant:
        for r in rows:
            bins[probe.bin(r["sim"])].append(r)
    params = []
    for group in bins:
        if not group:
            params.append((1.0, 1.0))
            continue
        best = min(
            itertools.product(w_grid, alpha_grid),
            key=lambda wa: np.mean([loss(blend(r["p_llm"], r["p_prior"], *wa), r["observed"]) for r in group]),
        )
        params.append(best)
    if constant:
        params = params * (len(edges) - 1)
    return Blend(edges, params)


def metrics(rows, predictions):
    t_model = np.mean([tvd(p, r["observed"]) for r, p in zip(rows, predictions)])
    t_uniform = np.mean([tvd(dict.fromkeys(r["observed"], 1 / len(r["observed"])), r["observed"]) for r in rows])
    ratios = [entropy(p) / entropy(r["observed"]) for r, p in zip(rows, predictions) if entropy(r["observed"]) > 0]
    return {
        "n": len(rows),
        "mae": float(np.mean([mae(p, r["observed"]) for r, p in zip(rows, predictions)])),
        "jsd": float(np.mean([jsd(p, r["observed"]) for r, p in zip(rows, predictions)])),
        "logloss": float(np.mean([log_loss(p, r["observed"]) for r, p in zip(rows, predictions)])),
        "simbench": float(100 * (1 - t_model / t_uniform)) if t_uniform > 0 else None,
        "entropy_ratio": float(np.mean(ratios)),
    }


def leave_one_topic_out(rows, edges=(0.0, 0.5, 1.01), objective="jsd", constant=False, **grids):
    """Predicciones fuera de muestra: cada tema se predice con parámetros ajustados con los demás."""
    topics = sorted({str(r["topic"]) for r in rows})
    held = [None] * len(rows)
    fitted = {}
    for t in topics:
        train = [r for r in rows if str(r["topic"]) != t]
        model = fit(train, edges, objective, constant=constant, **grids) if train else Blend(edges, [(1.0, 1.0)] * (len(edges) - 1))
        fitted[t] = model.describe()
        for i, r in enumerate(rows):
            if str(r["topic"]) == t:
                held[i] = model.apply(r)
    return held, fitted


def compare_variants(rows, objective="jsd"):
    """Tabla de variantes evaluadas fuera de muestra (dejando un tema fuera)."""
    variants = {
        "solo LLM": [r["p_llm"] for r in rows],
        "solo prior (analog)": [r["p_prior"] for r in rows],
    }
    out = {name: metrics(rows, preds) for name, preds in variants.items()}
    for name, kwargs in {
        "w constante, sin afilar": dict(constant=True, alpha_grid=(1.0,)),
        "α sobre LLM solo (w=1)": dict(constant=True, w_grid=(1.0,)),
        "w y α conjuntos, constantes": dict(constant=True),
        "w y α conjuntos por similitud": dict(),
    }.items():
        held, fitted = leave_one_topic_out(rows, objective=objective, **kwargs)
        out[name] = metrics(rows, held)
        out[name]["fitted_by_topic_left_out"] = fitted
    return out
