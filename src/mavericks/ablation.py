"""Tabla de ablación: todas las configuraciones sobre los mismos casos difíciles, por modo y tema.

Cada configuración se lleva a filas homogéneas (predicción, observado, error, errores por opción, segundos) para poder
usar el mismo informe, la misma cobertura de intervalos (dejando un caso fuera) y la misma comparación caso a caso
(mejoran/empeoran con IC 95 % bootstrap) para todas.
"""

import json
from pathlib import Path

import numpy as np

from .analog import AnalogConfig, combine
from .calibrate import blend as blend_distributions
from .report import bootstrap_ci, row
from .sharpen import interval_coverage

TIE = 0.005
MIN_WILCOXON = 10  # con menos casos pareados el contraste no tiene sentido
FAST_SECONDS = 2.0  # el modo rápido tarda unos 2 s por pregunta (medido en la interfaz)


def read_results(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["results"] if isinstance(raw, dict) else raw


def make_row(base, predicted, seconds, mode=None):
    """Fila homogénea a partir del caso de referencia ``base`` (resultado de baseline_v1/desarrollo)."""
    observed = base["observed"]
    p = {o: float(predicted[o]) for o in observed}
    errors = [abs(p[o] - observed[o]) for o in observed]
    return {
        "evidence_id": base["evidence_id"],
        "status": "completed",
        "anchor_mode": mode or base["anchor_mode"],
        "topic": base["topic"],
        "predicted": p,
        "observed": observed,
        "mae": float(np.mean(errors)),
        "option_errors": errors,
        "mae_uniform_baseline": base["mae_uniform_baseline"],
        "mae_national_baseline": base.get("mae_national_baseline"),
        "execution_time": seconds,
    }


def from_results(results, bases, ids):
    """Resultados de un backtest agrupados con el modo y tema del caso de referencia."""
    by = {r["evidence_id"]: r for r in results if r["status"] == "completed"}
    rows = []
    for i in ids:
        if i in by and i in bases:
            r = by[i]
            row_ = make_row(bases[i], r["predicted"], r.get("execution_time"))
            row_["subgroup_mae"] = r.get("subgroup_mae")
            rows.append(row_)
    return rows


def uniform_rows(bases, ids):
    rows = []
    for i in ids:
        if i in bases:
            observed = bases[i]["observed"]
            rows.append(make_row(bases[i], dict.fromkeys(observed, 1 / len(observed)), 0.0))
    return rows


def national_rows(bases, ids):
    rows = []
    for i in ids:
        b = bases.get(i)
        if b and b.get("baseline_national") and b.get("mae_national_baseline") is not None:
            rows.append(make_row(b, b["baseline_national"]["distribution"], 0.0))
    return rows


def fast_rows(fast_payload, bases, ids):
    rows = []
    by = {r["evidence_id"]: r for r in fast_payload["rows"]}
    for i in ids:
        if i in by and i in bases:
            options = list(bases[i]["observed"])
            predicted = dict(zip(options, by[i]["predicted"], strict=True))
            rows.append(make_row(bases[i], predicted, FAST_SECONDS))
    return rows


def blend_rows(results, bases, ids, analog_cache, w, config=None):
    """Mezcla offline w·LLM + (1−w)·prior por analogía a partir de las predicciones de ``results``."""
    config = config or AnalogConfig(5, 4.0, 0.0, 0.0, "lex")
    cache = {c["evidence_id"]: c for c in analog_cache}
    by = {r["evidence_id"]: r for r in results if r["status"] == "completed"}
    rows = []
    for i in ids:
        if i in by and i in bases and i in cache:
            prior = combine(cache[i]["candidates"], cache[i]["options"], config)["prior"]
            rows.append(
                make_row(bases[i], blend_distributions(by[i]["predicted"], prior, w, 1.0), by[i].get("execution_time"))
            )
    return rows


def compare_rows(base_rows, rows):
    """Mejoran/empeoran respecto a ``base_rows`` en los casos comunes, con IC 95 % bootstrap de la diferencia media."""
    base = {r["evidence_id"]: r for r in base_rows}
    shared = [r for r in rows if r["evidence_id"] in base]
    deltas = np.array([r["mae"] - base[r["evidence_id"]]["mae"] for r in shared])
    if not len(deltas):
        return None
    low, high = bootstrap_ci(deltas)
    p_value = None
    if len(deltas) >= MIN_WILCOXON and np.any(deltas != 0):
        from scipy.stats import wilcoxon

        p_value = float(wilcoxon(deltas).pvalue)  # contraste pareado sobre el MAE por caso
    return {
        "p_value": p_value,
        "cases": len(deltas),
        "mean_delta": float(deltas.mean()),
        "ci": (low, high),
        "better": int((deltas < -TIE).sum()),
        "worse": int((deltas > TIE).sum()),
        "ties": int((np.abs(deltas) <= TIE).sum()),
    }


def _pp(value):
    return "—" if value is None else f"{100 * value:.1f}"


def table(configs, reference="baseline_v1", floors=None):
    """Tabla markdown: error por modo y total (puntos), cobertura IC, s/caso y mejoran/empeoran frente a ``reference``."""
    header = (
        "| Configuración | casos | sin ancla (pp) | prior (pp) | total (pp) | suelo ruido (pp) | cobertura IC 90 % | s/caso "
        "| mejoran / empeoran / empatan | Δ medio [IC 95 %] (pp) | Wilcoxon p |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    base = configs.get(reference)
    for name, rows in configs.items():
        if not rows:
            continue
        by_mode = {m: row([r for r in rows if r["anchor_mode"] == m], floors) for m in ("none", "prior")}
        total = row(rows, floors)
        cov = interval_coverage(rows, min_cases=15).get("global", {}).get("coverage")
        cmp_ = compare_rows(base, rows) if base and name != reference else None
        better = f"{cmp_['better']} / {cmp_['worse']} / {cmp_['ties']}" if cmp_ else "—"
        delta = f"{_pp(cmp_['mean_delta'])} [{_pp(cmp_['ci'][0])}, {_pp(cmp_['ci'][1])}]" if cmp_ and cmp_["ci"][0] is not None else "—"
        if not cmp_:
            wilcoxon_p = "—"
        elif cmp_["p_value"] is None:
            wilcoxon_p = "n insuficiente"
        else:
            wilcoxon_p = "<0,001" if cmp_["p_value"] < 0.001 else f"{cmp_['p_value']:.3f}"
        seconds = total["seconds"]
        lines.append(
            f"| {name} | {len(rows)} | {_pp(by_mode['none']['mae'])} (n={by_mode['none']['n']}) | "
            f"{_pp(by_mode['prior']['mae'])} (n={by_mode['prior']['n']}) | {_pp(total['mae'])} | {_pp(total['floor'])} | "
            f"{'—' if cov is None else f'{100 * cov:.1f} %'} | {'—' if seconds is None else f'{seconds:.0f}'} | {better} | {delta} | {wilcoxon_p} |"
        )
    return header + "\n".join(lines)


def topic_table(configs, floors=None):
    """Error medio (pp) por tema y configuración."""
    topics = sorted({r["topic"] for rows in configs.values() for r in rows})
    header = "| Configuración | " + " | ".join(topics) + " |\n|---|" + "---|" * len(topics) + "\n"
    lines = []
    for name, rows in configs.items():
        cells = []
        for t in topics:
            part = [r for r in rows if r["topic"] == t]
            cells.append(f"{_pp(np.mean([r['mae'] for r in part]))} (n={len(part)})" if part else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return header + "\n".join(lines)


def _paired(results, bases, ids, analog_cache, config=None):
    """[(tema, p_llm, p_prior, observado)] de los casos con predicción del LLM y análogas."""
    config = config or AnalogConfig(5, 4.0, 0.0, 0.0, "lex")
    cache = {c["evidence_id"]: c for c in analog_cache}
    by = {r["evidence_id"]: r for r in results if r["status"] == "completed"}
    out = []
    for i in ids:
        if i in by and i in bases and i in cache:
            prior = combine(cache[i]["candidates"], cache[i]["options"], config)["prior"]
            out.append((bases[i]["topic"], by[i]["predicted"], prior, bases[i]["observed"]))
    return out


def _mean_metric(items, w, metric):
    return float(np.mean([metric(blend_distributions(p, q, w, 1.0), o) for _, p, q, o in items]))


def w_curve(results, bases, ids, analog_cache, grid=None):
    """Error medio (MAE y JSD) de la mezcla w·LLM + (1−w)·prior para cada w (w=0 es el prior solo, w=1 el LLM solo)."""
    from .calibrate import jsd, mae

    grid = grid if grid is not None else [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    items = _paired(results, bases, ids, analog_cache)
    return [{"w": w, "mae": _mean_metric(items, w, mae), "jsd": _mean_metric(items, w, jsd)} for w in grid], len(items)


def loto_w(results, bases, ids, analog_cache, grid=None, objective="mae"):
    """w elegido por validación cruzada dejando un tema fuera y el error resultante fuera de muestra."""
    from .calibrate import jsd, mae

    target = mae if objective == "mae" else jsd

    grid = grid if grid is not None else [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    items = _paired(results, bases, ids, analog_cache)
    chosen, held = {}, []
    for topic in sorted({t for t, *_ in items}):
        train = [x for x in items if x[0] != topic]
        w = min(grid, key=lambda g: _mean_metric(train, g, target))
        chosen[topic] = w
        held += [(w, x) for x in items if x[0] == topic]
    held_mae = float(np.mean([mae(blend_distributions(x[1], x[2], w, 1.0), x[3]) for w, x in held]))
    all_w = min(grid, key=lambda g: _mean_metric(items, g, target))
    return {"w_by_fold": chosen, "w_all": all_w, "mae_held_out": held_mae, "cases": len(items)}
