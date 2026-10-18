"""Tablas de validación para la interfaz: modo rápido (sin LLM) y línea base con LLM, todo en puntos porcentuales."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import tvd
from .sharpen import interval_coverage

BACKTEST = "outputs/baseline_v1/backtest.json"
FLOORS = "outputs/case_floors.json"
FAST = "outputs/fast_validation.json"
MODE_NAMES = {"direct": "Directo (pregunta repetida)", "prior": "Prior (pregunta parecida)", "none": "Sin ancla (pregunta nueva)"}
TOPIC_NAMES = {
    "vivienda": "Vivienda",
    "soledad": "Soledad",
    "ia_tecnologia": "IA y tecnología",
    "medio_ambiente": "Medio ambiente",
    "salud": "Salud",
    "consumo": "Consumo",
    "general": "General",
}
DEV_ONLY_NOTE = (
    "Vivienda, salud, medio ambiente y soledad solo se evalúan con casos de desarrollo: el CIS no ha publicado todavía "
    "estudios reservados de esos temas. Es una validación válida, pero se usó también para elegir la configuración."
)


def topic_name(topic):
    return TOPIC_NAMES.get(str(topic), str(topic).replace("_", " ").capitalize())


def load_floors(path=FLOORS):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def load_fast(path=FAST):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def load_baseline(path=BACKTEST):
    file = Path(path)
    if not file.exists():
        return None
    results = json.loads(file.read_text(encoding="utf-8"))["results"]
    rows = [
        {
            "evidence_id": r["evidence_id"], "mode": r["anchor_mode"], "topic": r["topic"], "set": "test",
            "mae": r["mae"], "uniform": r["mae_uniform_baseline"], "predicted": r["predicted"], "observed": r["observed"],
        }
        for r in results
        if r["status"] == "completed" and r.get("predicted")
    ]
    return {"rows": rows, "results": results}


def fast_rows(fast):
    rows = []
    for r in fast["rows"]:
        options = [str(i) for i in range(len(r["predicted"]))]
        rows.append(
            {
                "evidence_id": r["evidence_id"], "mode": r["mode"], "topic": r["topic"], "set": r["set"],
                "mae": r["mae"], "uniform": r["mae_uniform"],
                "predicted": dict(zip(options, r["predicted"], strict=True)),
                "observed": dict(zip(options, r["observed"], strict=True)),
            }
        )
    return rows


def _simbench(rows):
    model = np.mean([tvd(r["predicted"], r["observed"]) for r in rows])
    uniform = np.mean([tvd(dict.fromkeys(r["observed"], 1 / len(r["observed"])), r["observed"]) for r in rows])
    return float(100 * (1 - model / uniform)) if uniform > 0 else None


def summary(rows, floors, by):
    """Error medio en puntos porcentuales por modo o por tema (y SimBench aparte, para el detalle técnico)."""
    out = []
    keys = list(MODE_NAMES) if by == "mode" else sorted({str(r["topic"]) for r in rows})
    for key in keys:
        part = [r for r in rows if str(r[by]) == key]
        if not part:
            continue
        floor = [floors.get(r["evidence_id"], {}).get("national_floor") for r in part]
        floor = [f for f in floor if f is not None]
        out.append(
            {
                "Grupo": MODE_NAMES[key] if by == "mode" else topic_name(key),
                "Preguntas": len(part),
                "Error del modelo (pp)": 100 * np.mean([r["mae"] for r in part]),
                "Reparto uniforme (pp)": 100 * np.mean([r["uniform"] for r in part]),
                "Suelo de ruido (pp)": 100 * np.mean(floor) if floor else None,
                "SimBench": _simbench(part),
            }
        )
    return pd.DataFrame(out)


def configurations(fast, baseline, floors):
    """Filas por configuración y modo: modo rápido (153 preguntas) frente a la línea base con LLM (88 reservadas)."""
    frames = []
    if fast:
        frames.append(summary(fast_rows(fast), floors, "mode").assign(Configuración="Modo rápido sin LLM (test + desarrollo)"))
        test_only = [r for r in fast_rows(fast) if r["set"] == "test"]
        frames.append(summary(test_only, floors, "mode").assign(Configuración="Modo rápido sin LLM (solo test)"))
    if baseline:
        frames.append(summary(baseline["rows"], floors, "mode").assign(Configuración="Con LLM, línea base (test)"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def baseline_coverage(baseline):
    """Cobertura (%) del intervalo del 90 % de la línea base, dejando cada pregunta fuera."""
    coverage = interval_coverage(baseline["results"])
    return {k: (v["coverage"] * 100 if v.get("coverage") is not None else None, v["cases"]) for k, v in coverage.items()}


def fast_coverage(fast):
    key = fast.get("chosen", "by_mode")
    cv = fast[key]["cv"]
    return {m: (v["coverage"] * 100, v.get("options")) for m, v in cv.items()}


def points(rows, scale=100):
    out = []
    for r in rows:
        for option, value in r["observed"].items():
            out.append(
                {
                    "predicho": r["predicted"].get(option, 0) * scale,
                    "observado": value * scale,
                    "etiqueta": f"{r['evidence_id']} · {option}",
                    "modo": MODE_NAMES.get(r["mode"], r["mode"]),
                }
            )
    return pd.DataFrame(out)
