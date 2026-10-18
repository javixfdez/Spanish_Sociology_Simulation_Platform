"""Tarea 1: ajuste del KNN ponderado del prior por analogía con validación cruzada anidada sobre casos dev.

El prior es la media de las distribuciones de las k preguntas más parecidas, con peso similitud^potencia × calidad de
alineación de la escala (etiqueta 1,0; semántica 0,8; ordinal 0,5), sin contracción a la uniforme (como en producción).
Las similitudes son la léxica actual, el coseno e5 reescalado o su mezcla.
"""

import itertools

import numpy as np
from scipy.stats import wilcoxon

from ..analog import EMB_HIGH, EMB_LOW

KIND_WEIGHT = {"label": 1.0, "semantic": 0.8, "ordinal": 0.5}
TIE = 0.005  # diferencia de MAE (0,5 puntos) por debajo de la cual un caso se considera empate
K_GRID = (3, 5, 10)
POWER_GRID = (1, 2, 4)
SIMS = ("lex", "e5qq", "e5qp", "mix")  # mix = 0,5·léxica + 0,5·e5 query-query
CURRENT = ("lex", 5, 4)  # configuración de producción (ANALOG)


def scale_e5(cos, shift=0.0):
    """Reescala el coseno e5 al rango útil (como ``analog.similarity``); ``shift`` ajusta el rango de query-query."""
    return np.nan_to_num(np.clip((cos - EMB_LOW - shift) / (EMB_HIGH - EMB_LOW), 0, 1), nan=0.0)


def qq_shift(collected):
    """Diferencia media entre el coseno query-query y query-passage de las candidatas (no usa ninguna etiqueta)."""
    diffs = [c["cos_qq"] - c["cos_qp"] for case in collected for c in case["candidates"] if c.get("cos_qq") is not None]
    return float(np.mean(diffs)) if diffs else 0.0


class CaseArrays:
    """Candidatas de un caso como arrays para evaluar muchas configuraciones sin volver a recuperar."""

    def __init__(self, case, shift=0.0, forbid_ordinal_two=False):
        cands = case["candidates"]
        self.evidence_id, self.topic, self.study = case["evidence_id"], case["topic"], case["estudio_id"]
        self.options = case["options"]
        self.observed = np.array([case["observed"][o] for o in self.options], dtype=float)
        nan = lambda key: np.array([np.nan if c.get(key) is None else c[key] for c in cands], dtype=float)  # noqa: E731
        self.sim_lex = np.array([c["lex"] for c in cands], dtype=float)
        self.sim_qp = scale_e5(nan("cos_qp"))
        self.sim_qq = scale_e5(nan("cos_qq"), shift)
        self.pool = {
            "lex": np.array([c["in_lex"] for c in cands], dtype=bool),
            "e5qq": np.array([c["in_qq"] for c in cands], dtype=bool),
            "e5qp": np.array([c["in_qp"] for c in cands], dtype=bool),
        }
        self.pool["mix"] = self.pool["lex"] | self.pool["e5qq"]
        self.kinds = [c["kind"] for c in cands]
        self.kind_w = np.array([KIND_WEIGHT[k] for k in self.kinds], dtype=float)
        if forbid_ordinal_two and len(self.options) == 2:
            # variante decidida de antemano: con 2 opciones no se acepta la alineación por posición
            self.kind_w = np.where(np.array(self.kinds) == "ordinal", 0.0, self.kind_w)
        order = sorted({c["date"] or "" for c in cands})
        self.date_rank = np.array([order.index(c["date"] or "") for c in cands], dtype=float)
        self.dist = np.array([[c["dist"][o] for o in self.options] for c in cands], dtype=float).reshape(len(cands), len(self.options))

    def similarity(self, sim):
        return {
            "lex": self.sim_lex, "e5qq": self.sim_qq, "e5qp": self.sim_qp, "mix": 0.5 * self.sim_lex + 0.5 * self.sim_qq
        }[sim]

    def selected(self, sim, k, power):
        """Índices de las candidatas que entran en el prior (las k de mayor peso, con peso positivo)."""
        w = np.where(self.pool[sim], (self.similarity(sim) ** power) * self.kind_w, 0.0)
        order = np.lexsort((-self.date_rank, -w))[:k]
        return [int(i) for i in order if w[i] > 0]

    def prior(self, sim, k, power):
        n_options = len(self.options)
        uniform = np.full(n_options, 1 / n_options)
        if not len(self.sim_lex):
            return uniform
        w = np.where(self.pool[sim], (self.similarity(sim) ** power) * self.kind_w, 0.0)
        order = np.lexsort((-self.date_rank, -w))[:k]  # a igualdad de peso manda la más reciente
        total = w[order].sum()
        if total <= 0:
            return uniform
        return (w[order, None] * self.dist[order]).sum(axis=0) / total

    def mae(self, sim, k, power):
        return float(np.mean(np.abs(self.prior(sim, k, power) - self.observed)))


def grid():
    return list(itertools.product(SIMS, K_GRID, POWER_GRID))


def error_matrix(cases, configs):
    return np.array([[c.mae(*cfg) for c in cases] for cfg in configs])


def nested_grouped(matrix, groups):
    """Validación anidada agrupada: cada pliegue deja fuera un grupo entero (un estudio) y la mejor configuración
    se elige con los demás. Devuelve (error por caso fuera de muestra, configuración elegida por caso)."""
    n = matrix.shape[1]
    held = np.zeros(n)
    chosen = np.zeros(n, dtype=int)
    for g in sorted(set(groups)):
        test = [i for i in range(n) if groups[i] == g]
        train = [i for i in range(n) if groups[i] != g]
        best = int(np.argmin(matrix[:, train].mean(axis=1)))
        held[test] = matrix[best, test]
        chosen[test] = best
    return held, chosen


def paired(reference, other, seed=0, draws=5000):
    """Diferencia media (otra − referencia) con IC 95 % bootstrap, mejoran/empeoran/empatan y p-valor de Wilcoxon."""
    delta = np.asarray(other) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(axis=1)
    p = None
    if len(delta) >= 10 and np.any(delta != 0):
        p = float(wilcoxon(other, reference).pvalue)
    return {
        "mean_delta": float(delta.mean()),
        "ci": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "better": int((delta < -TIE).sum()),
        "worse": int((delta > TIE).sum()),
        "ties": int((np.abs(delta) <= TIE).sum()),
        "wilcoxon_p": p,
    }
