"""Tablas de validación para la interfaz, a partir de los resultados del backtest con estudios reservados."""

import json
from pathlib import Path

import pandas as pd

from .report import MODES, row
from .sharpen import interval_coverage

BACKTEST = "outputs/baseline_v1/backtest.json"
FLOORS = "outputs/case_floors.json"
MODE_NAMES = {key: label for key, label in MODES}


def load_backtest(path=BACKTEST, floors=FLOORS):
    file = Path(path)
    if not file.exists():
        return None
    results = json.loads(file.read_text(encoding="utf-8"))["results"]
    floor_file = Path(floors)
    floor_map = json.loads(floor_file.read_text(encoding="utf-8")) if floor_file.exists() else {}
    return {"results": results, "floors": floor_map}


def _table(results, floors, by, keys):
    rows = []
    for key in keys:
        items = [r for r in results if str(r.get(by)) == key]
        if not items:
            continue
        s = row(items, floors)
        rows.append(
            {
                "Grupo": MODE_NAMES.get(key, key),
                "Preguntas": s["n"],
                "Error del modelo": s["mae"],
                "Reparto uniforme": s["uniform"],
                "Resultado nacional previo": s["national"],
                "Suelo de ruido": s["floor"],
                "Mejora sobre uniforme (SimBench)": s["simbench"],
                "Gana al uniforme": f"{s['beats_uniform']}/{s['n'] - s['failed']}",
            }
        )
    return pd.DataFrame(rows)


def by_mode(data):
    return _table(data["results"], data["floors"], "anchor_mode", [m for m, _ in MODES])


def by_topic(data):
    topics = sorted({str(r["topic"]) for r in data["results"]})
    return _table(data["results"], data["floors"], "topic", topics)


def coverage(data):
    names = {"direct": "directo", "prior": "prior", "none": "sin ancla", "global": "global"}
    return [
        {"Grupo": names.get(group, group), "Preguntas": c["cases"], "Cobertura del IC 90 %": c.get("coverage")}
        for group, c in interval_coverage(data["results"]).items()
    ]


def points(data):
    """Pares predicho/observado por opción, para el gráfico de dispersión."""
    out = []
    for r in data["results"]:
        if r["status"] != "completed" or not r.get("predicted"):
            continue
        for option, value in r["observed"].items():
            out.append(
                {
                    "predicho": r["predicted"].get(option, 0) * 100,
                    "observado": value * 100,
                    "etiqueta": f"{r['evidence_id']} · {option}",
                    "modo": MODE_NAMES.get(r["anchor_mode"], r["anchor_mode"]),
                }
            )
    return pd.DataFrame(out)
