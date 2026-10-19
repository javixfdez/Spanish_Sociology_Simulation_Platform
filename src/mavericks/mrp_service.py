"""MRP en el servicio: subgrupos y cruces de varias variables a partir de los microdatos de la oleada ancla (modo directo).

Usa EXACTAMENTE la variante evaluada en ``docs/prereg_mrp.md`` (``mrp.MRP``, sin cambios). Aquí solo hay pegamento: preparar los microdatos de la ancla con las opciones de la pregunta, una caché en disco del ajuste
por pregunta ancla (``outputs/cache/mrp``, fuera de git; el ajuste con validación cruzada es lo caro) y los cruces. Los rangos de subgrupos y cruces son los actuales (no se recalibran con la evaluación del MRP).
"""

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from .mrp import DIMENSIONS, MRP, valid

log = logging.getLogger("mavericks.mrp")
CACHE_DIR = Path("outputs/cache/mrp")
CACHE_VERSION = "mrp-v1"
SMALL_CROSS = 50  # por debajo de este número de encuestados en el cruce, la estimación se apoya sobre todo en el modelo
ALIASES = {"edad": "grupo_edad", "estudios": "nivel_estudios", "ingresos": "ingresos_hogar_tramo", "comunidad": "ccaa", "comunidad_autonoma": "ccaa"}
DIMENSION_LABELS = {"sexo": "Sexo", "grupo_edad": "Edad", "ccaa": "Comunidad autónoma", "nivel_estudios": "Estudios", "ingresos_hogar_tramo": "Ingresos del hogar"}


class PlanError(ValueError):
    """Un valor o variable de un cruce no está en el esquema del CIS."""


def schema_of(population):
    """{variable: [categorías]} del esquema (las del CIS que usa la población sintética)."""
    return {d: sorted(population[d].dropna().astype(str).unique()) for d in DIMENSIONS if d in population}


def normalize_cross(items, schema):
    """Lista de {var, val} -> tupla ((variable, categoría), ...). Error si algo no está en el esquema; no se corrige ni se inventa ningún valor."""
    if not items:
        raise PlanError("cruce vacío")
    out, seen = [], set()
    for item in items:
        var = ALIASES.get(str(item.get("var", "")).strip().lower(), str(item.get("var", "")).strip().lower())
        if var not in schema:
            raise PlanError(f"variable fuera del esquema: {item.get('var')!r}")
        by_lower = {c.lower(): c for c in schema[var]}
        val = by_lower.get(str(item.get("val", "")).strip().lower())
        if val is None:
            raise PlanError(f"valor fuera del esquema para {var}: {item.get('val')!r}")
        if var in seen:
            raise PlanError(f"variable repetida: {var}")
        seen.add(var)
        out.append((var, val))
    return tuple(out)


def cross_label(cross):
    return " · ".join(f"{DIMENSION_LABELS.get(v, v)}: {c}" for v, c in cross)


def _key(spec, options, population_version):
    raw = json.dumps([CACHE_VERSION, str(spec["study_id"]), spec["evidence_id"], sorted(spec["mapping"].items()), list(options), population_version], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class CachedMRP(MRP):
    """MRP ya ajustado: las probabilidades por persona vienen de la caché (el resto es idéntico)."""

    def __init__(self, population, probabilities, omitted, c, used):
        super().__init__(population)
        self._p, self.omitted, self.C, self.used, self.model = probabilities, list(omitted), c, tuple(used), None

    def person_probabilities(self):
        return self._p


def anchor_frame(resources, spec, options):
    """Microdatos de la ancla con las respuestas ya expresadas con las opciones de la pregunta."""
    frame = resources.anchors._frame(spec["study_id"], spec["evidence_id"])
    back = {prev: option for option, prev in spec["mapping"].items()}
    return frame.assign(answer=frame["answer"].map(back))


def fitted(resources, spec, options):
    """(MRP ajustado, segundos, ¿desde caché?). El ajuste se cachea por pregunta ancla en outputs/cache/mrp."""
    population = resources.population.frame
    version = str(population["version_poblacion"].iloc[0]) if "version_poblacion" in population else str(len(population))
    path = CACHE_DIR / f"{_key(spec, options, version)}.npz"
    started = time.monotonic()
    if path.exists():
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            return CachedMRP(population, z["p"].astype(float), meta["omitted"], meta["C"], meta["used"]), time.monotonic() - started, True
        except Exception as exc:  # caché ilegible: se vuelve a ajustar
            log.warning("Caché del MRP ilegible (%s); se reajusta", exc)
    model = MRP(population).fit(anchor_frame(resources, spec, options), options)
    p = model.person_probabilities()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, p=p.astype(np.float32), meta=json.dumps({"omitted": model.omitted, "C": model.C, "used": list(model.used)}))
    except OSError as exc:
        log.warning("No se pudo escribir la caché del MRP: %s", exc)
    return model, time.monotonic() - started, False


def survey_size(resources, spec, options, cross):
    """Encuestados del CIS (oleada ancla) con respuesta sustantiva que cumplen el cruce."""
    frame = anchor_frame(resources, spec, options)
    frame = frame[frame["answer"].isin(options)].drop_duplicates("respondent_uid")
    for var, category in cross:
        if var not in frame.columns:
            return 0
        frame = frame[frame[var].astype(str) == category]
    return int(len(frame))


def subgroup_estimates(resources, spec, options, national, categories_by_dimension):
    """({dimensión: {categoría: reparto}}, {dimensión: {categoría: encuestados}}, info) con el MRP recalibrado al nacional dado."""
    model, seconds, cached = fitted(resources, spec, options)
    goal = np.array([national[o] for o in options], float)
    groups = [((d, c),) for d, cats in categories_by_dimension.items() for c in cats]
    post = model.postratify(goal, groups)
    frame = anchor_frame(resources, spec, options)
    frame = frame[frame["answer"].isin(options)].drop_duplicates("respondent_uid")
    estimates, sizes = {}, {}
    for d, cats in categories_by_dimension.items():
        estimates[d], sizes[d] = {}, {}
        counts = frame[d].astype(str).value_counts() if d in frame.columns else {}
        for c in cats:
            vec = post.get(((d, c),))
            if vec is not None:
                estimates[d][c] = dict(zip(options, (float(x) for x in vec), strict=True))
            sizes[d][c] = int(counts.get(c, 0)) if len(counts) else 0
    info = {"segundos": seconds, "cache": cached, "omitidas": list(model.omitted), "C": model.C}
    return estimates, sizes, info


def cross_estimates(resources, spec, options, national, crosses):
    """Reparto de cada cruce (tupla de (variable, categoría)) con su número de encuestados y el aviso de muestra pequeña."""
    model, seconds, cached = fitted(resources, spec, options)
    goal = np.array([national[o] for o in options], float)
    post = model.postratify(goal, list(crosses))
    out = []
    for cross in crosses:
        vec = post.get(cross)
        if vec is None:
            continue
        n = survey_size(resources, spec, options, cross)
        out.append({"cross": [{"var": v, "val": c} for v, c in cross], "label": cross_label(cross), "distribution": dict(zip(options, (float(x) for x in vec), strict=True)),
                    "n_cis": n, "small": n < SMALL_CROSS, "omitted_dimensions": [v for v, _ in cross if v in model.omitted]})
    return out, {"segundos": seconds, "cache": cached, "omitidas": list(model.omitted), "C": model.C}


def usable(resources):
    """¿Se puede usar el MRP? Hacen falta los microdatos y las columnas de la población que usa el modelo."""
    try:
        frame = resources.population.frame
        return resources.anchors is not None and all(c in frame.columns for c in (*DIMENSIONS, "peso", "tamuni", "edad"))
    except Exception:
        return False


__all__ = ["PlanError", "schema_of", "normalize_cross", "cross_label", "subgroup_estimates", "cross_estimates", "survey_size", "usable", "valid", "SMALL_CROSS"]
