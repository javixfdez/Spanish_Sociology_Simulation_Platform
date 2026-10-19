"""MRP rápido y sin MCMC para los repartos por subgrupo o por cruce de subgrupos.

Modelo multinomial con efectos de edad (tramos del CIS), sexo, CCAA, estudios e ingresos (los que haya en la oleada) más cuatro variables por CCAA derivadas de la población
sintética calibrada con el INE (% con estudios superiores, % en municipios de más de 100.000 habitantes, renta mediana del hogar y edad media). La parte multinivel se aproxima con
una penalización L2 (un prior gaussiano por efecto): ``LogisticRegression`` multinomial con variables indicadoras, ``C`` elegido por validación cruzada de 5 pliegues DENTRO de los
microdatos de la oleada de entrenamiento y pesos del CIS. Después se postestratifica con la población sintética y se recalibra (inclinación exponencial del logit) para que el reparto
nacional coincida con el nacional de entrada. Si una dimensión no está en la oleada (más de la mitad de valores no válidos), se omite para esa oleada y se registra; no se imputa.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

DIMENSIONS = ("sexo", "grupo_edad", "ccaa", "nivel_estudios", "ingresos_hogar_tramo")
INVALID = {"nc", "otros", "n.s.", "n.c.", "ns", "nan", "none", ""}
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FOLDS, SEED, MIN_VALID_SHARE = 5, 0, 0.5
URBAN = ("100.001 a 400.000 habitantes", "400.001 a 1.000.000 habitantes", "Más de 1.000.000 habitantes")  # municipios de más de 100.000 habitantes
INCOME_MIDPOINT = {  # euros al mes (punto medio de cada tramo; el último abierto)
    "Menos de 1.100 €": 800, "De 1.100 a 1.800 €": 1450, "De 1.801 a 2.700 €": 2250, "De 2.701 a 3.900 €": 3300, "De 3.901 a 5.000 €": 4450, "Más de 5.000 €": 6000,
}


def valid(series):
    return series.notna() & ~series.astype(str).str.strip().str.lower().isin(INVALID)


def ccaa_covariates(population):
    """Variables por CCAA estandarizadas (z) a partir de la población sintética ponderada: [superiores, urbano, renta, edad media]."""
    p = population
    rows = {}
    for ccaa, g in p.groupby("ccaa"):
        gw = g["peso"]
        urban = g["tamuni"].isin(URBAN)
        income = g["ingresos_hogar_tramo"].map(INCOME_MIDPOINT)
        rows[ccaa] = {
            "superiores": float(((g["nivel_estudios"] == "superiores") * gw).sum() / gw.sum()),
            "urbano": float((urban * gw).sum() / gw.sum()),
            "renta": float((income.fillna(income.mean()) * gw).sum() / gw.sum()),
            "edad": float((g["edad"] * gw).sum() / gw.sum()),
        }
    frame = pd.DataFrame(rows).T
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, 1)


class Design:
    """Matriz de diseño: indicadoras de las dimensiones usadas (ceros para valores no válidos) + variables por CCAA."""

    def __init__(self, categories, used, covariates):
        self.categories, self.used, self.covariates = categories, tuple(used), covariates
        self.columns = [f"{d}={c}" for d in self.used for c in categories[d]]
        if "ccaa" in self.used:
            self.columns += [f"cov_{c}" for c in covariates.columns]

    def matrix(self, frame):
        parts = []
        for d in self.used:
            values = frame[d].astype(str).to_numpy()
            parts.append(np.stack([(values == c) for c in self.categories[d]], axis=1).astype(float) if self.categories[d] else np.zeros((len(frame), 0)))
        if "ccaa" in self.used:
            parts.append(self.covariates.reindex(frame["ccaa"].astype(str)).fillna(0.0).to_numpy())
        return np.hstack(parts)


class MRP:
    def __init__(self, population, dimensions=DIMENSIONS, seed=SEED):
        self.population, self.dimensions, self.seed = population, tuple(dimensions), seed
        self.covariates = ccaa_covariates(population)
        self.categories = {d: sorted(population[d].dropna().astype(str).unique()) for d in self.dimensions if d in population}
        self.omitted, self.C, self.cv_scores = [], None, {}

    def usable(self, frame):
        used = []
        for d in self.dimensions:
            if d in frame.columns and d in self.categories and valid(frame[d]).mean() >= MIN_VALID_SHARE:
                used.append(d)
            else:
                self.omitted.append(d)
        return used

    def fit(self, frame, options):
        """``frame``: respuestas de la oleada de entrenamiento con ``answer``, ``peso`` y las dimensiones; ``options``: opciones sustantivas (columnas del reparto)."""
        data = frame[frame["answer"].isin(options)].drop_duplicates("respondent_uid") if "respondent_uid" in frame else frame[frame["answer"].isin(options)]
        self.options = list(options)
        self.used = self.usable(data)
        self.design = Design(self.categories, self.used, self.covariates)
        X = self.design.matrix(data)
        y = data["answer"].map({o: i for i, o in enumerate(self.options)}).to_numpy()
        w = data["peso"].to_numpy(float)
        w = w / w.mean()
        self.prior = np.bincount(y, weights=w, minlength=len(self.options)) / w.sum()
        if len(np.unique(y)) < 2 or X.shape[1] == 0:
            self.model, self.C = None, None
            return self
        self.C = self._select_c(X, y, w)
        self.model = self._fit(X, y, w, self.C)
        return self

    def _fit(self, X, y, w, C):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(C=C, max_iter=2000).fit(X, y, sample_weight=w)

    def _proba(self, model, X):
        out = np.zeros((X.shape[0], len(self.options)))
        out[:, model.classes_] = model.predict_proba(X)
        return out

    def _select_c(self, X, y, w):
        n = len(y)
        losses = {c: 0.0 for c in C_GRID}
        for train, test in KFold(FOLDS, shuffle=True, random_state=self.seed).split(np.arange(n)):
            if len(np.unique(y[train])) < 2:
                continue
            for c in C_GRID:
                p = np.clip(self._proba(self._fit(X[train], y[train], w[train], c), X[test]), 1e-6, 1)
                p = p / p.sum(axis=1, keepdims=True)
                losses[c] += float(-(w[test] * np.log(p[np.arange(len(test)), y[test]])).sum())
        self.cv_scores = {c: v / w.sum() for c, v in losses.items()}
        return min(self.cv_scores, key=self.cv_scores.get)

    def person_probabilities(self):
        if self.model is None:
            return np.tile(self.prior, (len(self.population), 1))
        return self._proba(self.model, self.design.matrix(self.population))

    def postratify(self, goal, groups=None):
        """Reparto por subgrupo, recalibrado para que el nacional sea ``goal``. ``groups``: lista de tuplas ((dimensión, categoría), ...) (cruces); por defecto, las categorías de cada dimensión."""
        goal = np.clip(np.asarray(goal, float), 0.0, None)
        goal = goal / goal.sum()
        p = np.clip(self.person_probabilities(), 1e-6, None)
        p = p / p.sum(axis=1, keepdims=True)
        w = self.population["peso"].to_numpy(float)
        w = w / w.sum()
        q = tilt(p, w, goal)
        pop = self.population
        if groups is None:
            groups = [((d, c),) for d in self.categories for c in self.categories[d]]
        out = {}
        for group in groups:
            mask = np.ones(len(pop), bool)
            for dim, category in group:
                mask &= (pop[dim].astype(str) == category).to_numpy()
            if mask.any() and w[mask].sum() > 0:
                out[group] = (q[mask] * w[mask, None]).sum(axis=0) / w[mask].sum()
        return out


def tilt(p, w, goal, iterations=200, tol=1e-10):
    """Inclinación exponencial: δ tal que la media ponderada de softmax(log p + δ) es ``goal`` (las opciones con objetivo 0 quedan en 0)."""
    logp = np.log(p)
    delta = np.zeros(p.shape[1])
    active = goal > 0
    for _ in range(iterations):
        z = logp + delta
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q[:, ~active] = 0.0
        q /= q.sum(axis=1, keepdims=True)
        mean = (q * w[:, None]).sum(axis=0)
        gap = np.log(np.clip(goal[active], 1e-12, None)) - np.log(np.clip(mean[active], 1e-12, None))
        delta[active] += gap
        if np.abs(mean[active] - goal[active]).max() < tol:
            break
    return q
