"""«Ver la población sintética»: para la pregunta actual, la probabilidad de cada opción de cada persona de la población sintética y una respuesta simulada.

No cambia ninguna cifra de la predicción: parte del resultado ya calculado.
- Modo directo con MRP: probabilidades por persona del propio MRP (ajustado con los microdatos de la oleada de la pregunta) recalibradas al nacional mostrado.
- Resto de modos (método de subgrupos activo: diferencias de preguntas análogas o modelo ajustado): los repartos por grupo de cada dimensión se combinan de forma multiplicativa
  (producto de expertos: p_i ∝ nacional · Π_d reparto_d(grupo_i) / nacional), y se recalibra al nacional mostrado.
La respuesta simulada es un muestreo con semilla fija: agregando las 50.000 se recupera el reparto nacional dentro del error de muestreo.
"""

import numpy as np

from . import mrp_service
from .mrp import tilt

SEED = 20260630
SAMPLE = 200
PROFILE = ["person_id", "edad", "sexo", "grupo_edad", "ccaa", "nivel_estudios", "ingresos_hogar_tramo"]


def _goal(result):
    return np.array([result["national"]["distribution"][o] for o in result["options"]], float)


def person_probabilities(resources, result):
    """(matriz N×K de probabilidades por persona, método usado). La media ponderada por peso de las filas es el reparto nacional del resultado."""
    population = resources.population.frame
    options, goal = result["options"], _goal(result)
    w = population["peso"].to_numpy(float)
    w = w / w.sum()
    anchor = result.get("anchor")
    if anchor and (result.get("subgroups_method") or {}).get("clave") == "mrp" and mrp_service.usable(resources):
        spec = {**anchor, "question_type": "single"}
        model, _, _ = mrp_service.fitted(resources, spec, options)
        p = np.clip(model.person_probabilities(), 1e-6, None)
        p = p / p.sum(axis=1, keepdims=True)
        return tilt(p, w, goal), "MRP"
    logp = np.tile(np.log(np.clip(goal, 1e-6, None)), (len(population), 1))
    for dimension, node in (result.get("subgroups") or {}).items():
        if dimension not in population:
            continue
        ratio = {}
        for category, group in node["groups"].items():
            dist = np.array([group["distribution"][o] for o in options], float)
            ratio[str(category)] = np.log(np.clip(dist, 1e-6, None) / np.clip(goal, 1e-6, None))
        values = population[dimension].astype(str).to_numpy()
        for category, shift in ratio.items():
            logp[values == category] += shift
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p = p / p.sum(axis=1, keepdims=True)
    return tilt(p, w, goal), "producto de los repartos por grupo"


def simulate(probabilities, seed=SEED):
    """Una respuesta por persona (índice de la opción) por muestreo inverso con semilla fija."""
    rng = np.random.default_rng(seed)
    u = rng.random(len(probabilities))[:, None]
    return np.minimum((u > np.cumsum(probabilities, axis=1)).sum(axis=1), probabilities.shape[1] - 1)


def population_frame(resources, result, seed=SEED):
    """DataFrame con las 50.000 personas: perfil, probabilidad por opción y respuesta simulada. Devuelve (frame, método)."""
    population = resources.population.frame
    options = result["options"]
    q, method = person_probabilities(resources, result)
    frame = population[[c for c in PROFILE if c in population]].copy()
    frame["peso"] = population["peso"].to_numpy()
    for i, option in enumerate(options):
        frame[f"p({option})"] = np.round(q[:, i], 6)
    frame["respuesta_simulada"] = np.array(options, dtype=object)[simulate(q, seed)]
    return frame, method


def aggregate(frame, options):
    """Reparto ponderado de las respuestas simuladas."""
    w = frame["peso"].to_numpy(float)
    shares = {o: float(w[(frame["respuesta_simulada"] == o).to_numpy()].sum() / w.sum()) for o in options}
    return shares


def sample(frame, n=SAMPLE, seed=0):
    return frame.sample(min(n, len(frame)), random_state=seed).sort_values("person_id").reset_index(drop=True) if "person_id" in frame else frame.sample(min(n, len(frame)), random_state=seed)


def to_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")

