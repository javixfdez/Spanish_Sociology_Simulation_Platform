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


PRIOR_VALIDATION = "outputs/models/prior_validation.json"
CONFIG_NAMES = {"uniforme": "Reparto uniforme", "knn_actual": "KNN actual", "media_knn_mlp": "Media KNN + modelo aprendido"}


def load_prior_validation(path=PRIOR_VALIDATION):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def _ci(delta, ci):
    return "—" if delta is None else f"{100 * delta:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def _p(value):
    return "—" if value is None else ("<0,001" if value < 0.001 else f"{value:.3f}".replace(".", ","))


def prior_table(validation):
    """Una fila por conjunto y configuración: MAE, Δ frente al KNN con IC por estudio, Wilcoxon, preguntas y estudios."""
    rows = []
    for info in validation["sets"].values():
        cell = info["grupos"]["todas"]
        for name, entry in cell["configuraciones"].items():
            rows.append(
                {
                    "Conjunto": info["titulo"], "Configuración": CONFIG_NAMES[name], "Error (pp)": 100 * entry["mae"],
                    "Δ frente al KNN [IC 95 % por estudio]": _ci(entry.get("delta_vs_knn"), entry.get("ci", (0, 0))),
                    "Wilcoxon p": _p(entry.get("p")) if name != "knn_actual" else "—",
                    "Preguntas": cell["preguntas"], "Estudios": cell["estudios"],
                }
            )
    return pd.DataFrame(rows)


def prior_replication(validation):
    """Mejora de la media sobre el KNN en cada conjunto (para la cifra destacada)."""
    return [
        (info["titulo"], info["grupos"]["todas"]["configuraciones"]["media_knn_mlp"]["delta_vs_knn"],
         info["grupos"]["todas"]["configuraciones"]["media_knn_mlp"]["ci"], info["grupos"]["todas"]["preguntas"])
        for info in validation["sets"].values()
    ]


def prior_vs_uniform(validation):
    """Media KNN+modelo frente al reparto uniforme, por número de opciones y conjunto."""
    rows = []
    for info in validation["sets"].values():
        for group in ("2 opciones", "3 o más"):
            cell = info["grupos"].get(group)
            if not cell:
                continue
            entry = cell["configuraciones"]["media_knn_mlp"]
            rows.append(
                {
                    "Conjunto": info["titulo"], "Preguntas": group + f" ({cell['preguntas']})",
                    "Uniforme (pp)": 100 * cell["configuraciones"]["uniforme"]["mae"], "Media KNN + modelo (pp)": 100 * entry["mae"],
                    "Media − uniforme [IC 95 % por estudio]": _ci(entry["delta_vs_uniforme"], entry["ci_vs_uniforme"]),
                }
            )
    return pd.DataFrame(rows)


OFFICIAL = "outputs/final_eval/report.json"
OFFICIAL_NAMES = {
    "uniform": "Reparto uniforme", "knn": "Sistema anterior (directo / KNN)", "rapid": "Modo rápido",
    "with_llm": "Modo con LLM (0,3·LLM + 0,7·prior)",
}
MODE_ROWS = {"direct": "Directo (pregunta repetida)", "prior": "Prior (1 pregunta)", "none": "Sin ancla (pregunta nueva)"}


def load_official(path=OFFICIAL):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def official_table(report):
    """Error (pp) de cada configuración por modo y total, y diferencia frente al sistema anterior con IC por estudio."""
    rows = []
    for mode, label in MODE_ROWS.items():
        if mode in report["por_modo"]:
            rows.append({"Grupo": label, **{OFFICIAL_NAMES[k]: 100 * v for k, v in report["por_modo"][mode].items()}})
    rows.append({"Grupo": f"Total ({report['casos']} casos, {report['estudios']} estudios)", **{OFFICIAL_NAMES[k]: 100 * v for k, v in report["total"].items()}})
    return pd.DataFrame(rows)


def official_deltas(report):
    out = []
    for scope, node in (("Total", report["frente_al_sistema_anterior"]), (f"Preguntas no repetidas ({report['no_directos']['casos']})", report["no_directos"]["frente_al_sistema_anterior"])):
        for key in ("rapid", "with_llm"):
            v = node[key]
            out.append({"Conjunto": scope, "Modo": OFFICIAL_NAMES[key], "Δ frente al sistema anterior [IC 95 % por estudio]": _ci(v["delta"], v["ci"]),
                        "Mejoran / empeoran": f"{v['mejoran']} / {v['empeoran']}", "Wilcoxon p": _p(v["p"])})
    return pd.DataFrame(out)
