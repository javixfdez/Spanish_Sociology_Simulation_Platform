"""Informes de backtest: tablas por modo de ancla y tema (siempre separados) y comparación entre configuraciones."""

import numpy as np

from .sharpen import interval_coverage

MODES = [("direct", "directo (pregunta repetida)"), ("prior", "prior (pregunta parecida)"), ("none", "sin ancla (pregunta nueva)")]
TIE = 0.005  # diferencia de MAE por debajo de la cual un caso se considera empate


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def row(items):
    done = [r for r in items if r["status"] == "completed"]
    base = [r for r in done if r.get("mae_national_baseline") is not None]
    return {
        "n": len(items),
        "failed": len(items) - len(done),
        "mae": _mean([r["mae"] for r in done]),
        "uniform": _mean([r["mae_uniform_baseline"] for r in done]),
        "national_n": len(base),
        "national": _mean([r["mae_national_baseline"] for r in base]),
        "model_on_national": _mean([r["mae"] for r in base]),
        "subgroup_mae": _mean([r.get("subgroup_mae") for r in done]),
        "seconds": _mean([r.get("execution_time") for r in items]),
        "beats_uniform": sum(r["mae"] < r["mae_uniform_baseline"] for r in done),
        "beats_national": sum(r["mae"] < r["mae_national_baseline"] for r in base),
    }


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_table(results, by="anchor_mode"):
    header = (
        "| Grupo | casos | fallidos | MAE modelo | uniforme | nacional* | modelo en esos | gana a uniforme | "
        "gana a nacional* | MAE subgrupos | s/caso |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    keys = [(k, label) for k, label in MODES] if by == "anchor_mode" else [(t, t) for t in sorted({str(r["topic"]) for r in results})]
    for key, label in keys:
        items = [r for r in results if str(r.get(by)) == key]
        if not items:
            continue
        s = row(items)
        lines.append(
            f"| {label} | {s['n']} | {s['failed']} | {_fmt(s['mae'])} | {_fmt(s['uniform'])} | {_fmt(s['national'])} "
            f"(n={s['national_n']}) | {_fmt(s['model_on_national'])} | {s['beats_uniform']}/{s['n'] - s['failed']} | "
            f"{s['beats_national']}/{s['national_n']} | {_fmt(s['subgroup_mae'])} | {_fmt(s['seconds'], 0)} |"
        )
    return header + "\n".join(lines)


def mode_topic_table(results):
    header = "| Modo · tema | casos | MAE | uniforme | nacional* | modelo en esos | s/caso |\n|---|---|---|---|---|---|---|\n"
    lines = []
    for key, label in MODES:
        for topic in sorted({str(r["topic"]) for r in results if r["anchor_mode"] == key}):
            s = row([r for r in results if r["anchor_mode"] == key and str(r["topic"]) == topic])
            lines.append(
                f"| {label.split(' ')[0]} · {topic} | {s['n']} | {_fmt(s['mae'])} | {_fmt(s['uniform'])} | "
                f"{_fmt(s['national'])} (n={s['national_n']}) | {_fmt(s['model_on_national'])} | {_fmt(s['seconds'], 0)} |"
            )
    return header + "\n".join(lines)


def report(results, title="Backtest"):
    done = [r for r in results if r["status"] == "completed"]
    coverage = interval_coverage(results)
    cov_lines = [
        f"- {g}: {c['coverage']:.0%} de los valores reales dentro del intervalo del 90 % (n={c['cases']} casos)"
        if c.get("coverage") is not None
        else f"- {g}: sin cobertura calculable (n={c['cases']} casos, mínimo 15)"
        for g, c in coverage.items()
    ]
    overall = row(results)
    return "\n".join(
        [
            f"# {title}",
            "",
            f"{len(results)} casos · {overall['failed']} fallidos · {_fmt(overall['seconds'], 0)} s/caso de media",
            "",
            "## Por modo de ancla",
            markdown_table(results, "anchor_mode"),
            "",
            "## Por tema",
            markdown_table(results, "topic"),
            "",
            "## Por modo y tema",
            mode_topic_table(results),
            "",
            "## Cobertura de intervalos (dejando un caso fuera)",
            *cov_lines,
            "",
            "\\* línea base nacional: distribución sin segmentar de la pregunta recuperada más parecida (solo casos comparables).",
            f"Casos completados: {len(done)}.",
        ]
    )


def compare(base, others, modes=("prior", "none")):
    """Compara configuraciones caso a caso sobre los casos de ``modes`` (según el modo de la base)."""
    out = {}
    base_by = {r["evidence_id"]: r for r in base if r["status"] == "completed" and r["anchor_mode"] in modes}
    for name, results in others.items():
        by = {r["evidence_id"]: r for r in results if r["status"] == "completed"}
        shared = [k for k in base_by if k in by]
        deltas = np.array([by[k]["mae"] - base_by[k]["mae"] for k in shared])
        entry = {
            "cases": len(shared),
            "failed": sum(1 for r in results if r["status"] != "completed"),
            "mae_base": float(np.mean([base_by[k]["mae"] for k in shared])) if shared else None,
            "mae": float(np.mean([by[k]["mae"] for k in shared])) if shared else None,
            "median_delta": float(np.median(deltas)) if len(deltas) else None,
            "better": int((deltas < -TIE).sum()),
            "worse": int((deltas > TIE).sum()),
            "ties": int((np.abs(deltas) <= TIE).sum()),
            "seconds": _mean([by[k].get("execution_time") for k in shared]),
            "seconds_base": _mean([base_by[k].get("execution_time") for k in shared]),
            "by_mode": {},
            "mode_changes": sum(1 for k in shared if by[k]["anchor_mode"] != base_by[k]["anchor_mode"]),
        }
        for mode in modes:
            keys = [k for k in shared if by[k]["anchor_mode"] == mode]
            if keys:
                entry["by_mode"][mode] = {
                    "cases": len(keys),
                    "mae_base": float(np.mean([base_by[k]["mae"] for k in keys])),
                    "mae": float(np.mean([by[k]["mae"] for k in keys])),
                }
        out[name] = entry
    return out


def comparison_table(comparison):
    header = (
        "| Configuración | casos | MAE base | MAE | Δ mediano | mejoran | empeoran | empates | cambian de modo | s/caso (base) | fallidos |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {name} | {c['cases']} | {_fmt(c['mae_base'])} | {_fmt(c['mae'])} | {_fmt(c['median_delta'], 4)} | {c['better']} | "
        f"{c['worse']} | {c['ties']} | {c['mode_changes']} | {_fmt(c['seconds'], 0)} ({_fmt(c['seconds_base'], 0)}) | {c['failed']} |"
        for name, c in comparison.items()
    ]
    return header + "\n".join(lines)
