"""Tarea 2: prior aprendido por extracción de características (e5 congelado + Ridge/MLP pequeños).

Unidad de entrenamiento: cada (pregunta, opción) del almacén del CIS con su proporción nacional observada (sin N.S./N.C.,
renormalizada). Variables: embeddings e5 de la pregunta («passage: ») y de la opción («query: »), reducidos con una PCA
ajustada solo con los datos de entrenamiento; su producto elemento a elemento (para que la opción se interprete en el
contexto de la pregunta: sin él, lo que solo depende de la pregunta se anula al normalizar con el softmax); coseno
pregunta–opción; posición en la escala (0,5 si hay 2 opciones); primera/última; espontánea; nº de opciones.
Objetivo: logit de la proporción. La distribución de una pregunta es el softmax de las salidas de sus opciones.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..analog import NON_SUBSTANTIVE
from ..anchor import option_key
from ..assist import is_spontaneous

MAX_OPTIONS = 7  # los casos dev tienen 2-7 opciones: no se entrena con escalas largas (partidos, provincias…)
EPS = 0.01


@dataclass
class TrainQuestion:
    evidence_id: str
    study: str
    date: str
    options: list
    props: list
    spontaneous: list


def train_questions(repository, table, emb, sources=("cis",), max_options=MAX_OPTIONS):
    """Preguntas del almacén utilizables como datos de entrenamiento (con sus fechas y proporciones)."""
    from ..evidence import source_family

    out = []
    for e in repository.all():
        if source_family(e) not in sources or e.evidence_id not in table or e.evidence_id not in emb.q_index:
            continue
        original = e.metadata.get("original_fields", {})
        if original.get("question_type", "single") not in (None, "single") or not e.fieldwork_date:
            continue
        dist = table[e.evidence_id]["dist"]
        options = [o for o in (e.options or []) if option_key(o) not in NON_SUBSTANTIVE and o in dist]
        if not (2 <= len(options) <= max_options):
            continue
        mass = sum(dist[o] for o in options)
        if mass <= 0:
            continue
        try:
            for o in options:
                emb.option_query(o)
        except KeyError:
            continue
        out.append(
            TrainQuestion(
                e.evidence_id, str(e.study_id), e.fieldwork_date[:10], options,
                [dist[o] / mass for o in options], [is_spontaneous(o) for o in options],
            )
        )
    return out


def raw_rows(q_vec, option_vecs, spontaneous):
    """(vector de pregunta repetido, vectores de opción, variables sueltas) de una pregunta."""
    n = len(option_vecs)
    pos = np.array([i / (n - 1) if n > 2 else 0.5 for i in range(n)])
    cos = option_vecs @ q_vec
    extras = np.stack(
        [cos, pos, pos**2, (np.arange(n) == 0).astype(float), (np.arange(n) == n - 1).astype(float),
         np.array(spontaneous, dtype=float), np.full(n, n / MAX_OPTIONS), np.full(n, float(n == 2))],
        axis=1,
    )
    return np.repeat(q_vec[None, :], n, axis=0), np.asarray(option_vecs), extras


class Featurizer:
    """PCA de preguntas y de opciones + estandarización, ajustadas solo con datos de entrenamiento."""

    def __init__(self, dims, seed=0):
        self.pq, self.po = PCA(dims, random_state=seed), PCA(dims, random_state=seed)
        self.scaler = StandardScaler()

    def _design(self, q, o, extras):
        qp, op = self.pq.transform(q), self.po.transform(o)
        return np.hstack([op, qp * op, qp, extras])

    def fit(self, q_unique, o_unique, q, o, extras):
        self.pq.fit(q_unique)
        self.po.fit(o_unique)
        return self.scaler.fit_transform(self._design(q, o, extras))

    def transform(self, q, o, extras):
        return self.scaler.transform(self._design(q, o, extras))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def build_training_matrix(questions, emb):
    qs, os_, ex, y = [], [], [], []
    for t in questions:
        q = emb.question(t.evidence_id)
        ov = np.stack([emb.option_query(o) for o in t.options])
        a, b, c = raw_rows(q, ov, t.spontaneous)
        qs.append(a), os_.append(b), ex.append(c), y.append(logit(t.props))
    return np.vstack(qs), np.vstack(os_), np.vstack(ex), np.concatenate(y)


def make_model(kind, param, seed=0):
    if kind == "ridge":
        return Ridge(alpha=param)
    return MLPRegressor(
        hidden_layer_sizes=param, early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=seed, alpha=1e-3
    )


def predict_case(featurizer, model, q_vec, option_vecs, spontaneous):
    q, o, ex = raw_rows(q_vec, option_vecs, spontaneous)
    return softmax(model.predict(featurizer.transform(q, o, ex)))
