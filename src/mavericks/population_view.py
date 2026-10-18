"""Comparación de los márgenes de la población sintética con los objetivos oficiales (para la interfaz)."""

import json
from pathlib import Path

import pandas as pd

from . import categories as cat
from .etl.population import load_ecp_targets

DIMENSIONS = [
    ("sexo", "Sexo", "INE · ECP"),
    ("grupo_edad", "Edad", "INE · ECP"),
    ("ccaa", "Comunidad autónoma", "INE · ECP"),
]


def build_targets(raw_dir, period, out):
    """Márgenes de adultos por sexo, grupo de edad y CCAA según la ECP (se guardan una vez)."""
    target = load_ecp_targets(Path(raw_dir) / "general" / "poblacion" / "ecp", period)
    total = float(target.sum())
    payload = {
        "period": period,
        "total": total,
        "sexo": dict(zip(cat.SEXO, (target.sum(axis=(0, 2, 3)) / total).tolist(), strict=True)),
        "grupo_edad": dict(zip(cat.GRUPOS_EDAD, (target.sum(axis=(0, 1, 2)) / total).tolist(), strict=True)),
        "ccaa": dict(zip(cat.CCAA, (target.sum(axis=(1, 2, 3)) / total).tolist(), strict=True)),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"out": str(out), "total": total}


def load_targets(path="data/processed/population_targets.json"):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def margin_table(frame, dimension, order, target=None):
    """Tabla categoría, % en la muestra, % objetivo y diferencia en puntos."""
    shares = frame.groupby(dimension).peso.sum().reindex(order).fillna(0)
    shares = shares / shares.sum()
    table = pd.DataFrame({"muestra": shares * 100})
    if target:
        table["objetivo"] = [target.get(c, float("nan")) * 100 for c in order]
        table["diferencia"] = table["muestra"] - table["objetivo"]
    return table


def summary(frame, targets):
    """Error medio y máximo (en puntos porcentuales) sobre las tres dimensiones con objetivo."""
    errors = []
    for dimension, _, _ in DIMENSIONS:
        order = getattr(cat, {"sexo": "SEXO", "grupo_edad": "GRUPOS_EDAD", "ccaa": "CCAA"}[dimension])
        table = margin_table(frame, dimension, order, (targets or {}).get(dimension))
        if "diferencia" in table:
            errors += table["diferencia"].abs().tolist()
    return {"mean_pp": sum(errors) / len(errors), "max_pp": max(errors)} if errors else None
