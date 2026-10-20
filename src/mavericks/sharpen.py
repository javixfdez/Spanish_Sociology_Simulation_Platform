"""Ajuste de nitidez (temperature scaling) de la distribución predicha, aprendido con validación cruzada.

El modelo tiende a repartir demasiado la masa cuando hay una opción dominante. Se aprende un exponente
``tau`` y se transforma  p_i -> p_i^tau / sum_j p_j^tau  (tau > 1 afila, tau < 1 aplana). ``tau`` se elige para
minimizar el MAE (la métrica de evaluación) y su beneficio se mide siempre con validación cruzada por caso:
cada caso se corrige con un tau aprendido sin él.
"""

import numpy as np

GRID = np.round(np.arange(0.5, 3.0001, 0.05), 2)
MIN_CASES = 15
_EPS = 1e-9


def sharpen(distribution, tau):
    values = {k: max(float(v), _EPS) ** tau for k, v in distribution.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def _mae(pred, obs):
    return float(np.mean([abs(pred[k] - obs[k]) for k in obs]))


def fit_tau(pairs, grid=GRID):
    """tau del grid que minimiza el MAE medio de los pares (predicho, observado)."""
    losses = [np.mean([_mae(sharpen(p, t), o) for p, o in pairs]) for t in grid]
    return float(grid[int(np.argmin(losses))])


def cross_validate(pairs, folds=5, seed=0, grid=GRID):
    """Evalúa el ajuste con validación cruzada. Devuelve tau final, MAE crudo y MAE corregido fuera de muestra.

    Si el ajuste no mejora fuera de muestra se devuelve tau = 1 (sin corrección).
    """
    n = len(pairs)
    if n < MIN_CASES:
        return None
    order = np.random.default_rng(seed).permutation(n)
    assignment = np.empty(n, dtype=int)
    assignment[order] = np.arange(n) % min(folds, n)
    corrected = [None] * n
    case_taus = [1.0] * n
    fold_taus = []
    for f in range(min(folds, n)):
        train = [pairs[i] for i in range(n) if assignment[i] != f]
        tau = fit_tau(train, grid)
        fold_taus.append(tau)
        for i in np.where(assignment == f)[0]:
            corrected[i] = sharpen(pairs[i][0], tau)
            case_taus[i] = tau
    raw = float(np.mean([_mae(p, o) for p, o in pairs]))
    cv = float(np.mean([_mae(c, o) for c, (_, o) in zip(corrected, pairs)]))
    full_tau = fit_tau(pairs, grid)
    useful = cv < raw
    return {
        "tau": full_tau if useful else 1.0,
        "fitted_tau": full_tau,
        "useful": useful,
        "n": n,
        "mae_raw": raw,
        "mae_cv": cv,
        "fold_taus": fold_taus,
        "corrected_cv": corrected if useful else [dict(p) for p, _ in pairs],
        "case_taus": case_taus if useful else [1.0] * n,
    }


def fit_by_mode(results, modes=("prior", "none"), min_cases=MIN_CASES, seed=0):
    """Un tau por modo de ancla (la estadística directa no se toca). Usa resultados completados de un backtest."""
    report = {}
    for mode in modes:
        items = [r for r in results if r["status"] == "completed" and r["anchor_mode"] == mode]
        pairs = [(r["predicted"], r["observed"]) for r in items]
        outcome = cross_validate(pairs, seed=seed) if len(pairs) >= min_cases else None
        report[mode] = (
            {k: v for k, v in outcome.items() if k not in ("corrected_cv", "case_taus")} if outcome else {"tau": 1.0, "n": len(pairs), "useful": False}
        )
        if outcome:
            report[mode]["_corrected"] = dict(zip([r["evidence_id"] for r in items], outcome["corrected_cv"]))
            report[mode]["_case_taus"] = dict(zip([r["evidence_id"] for r in items], outcome["case_taus"]))
    return report


def apply_to_results(results, report, cases):
    """Resultados de un backtest con las predicciones corregidas fuera de muestra (tau de cada caso).

    Recalcula MAE, JSD y errores nacionales y de subgrupos; conserva la predicción original en predicted_raw.
    ``cases`` aporta los subgrupos observados (no se guardan en los resultados).
    """
    from .evaluation import subgroup_errors
    from .statistics import js_divergence

    by_case = {c["evidence_id"]: c for c in cases}
    corrected_out = []
    for r in results:
        row = dict(r)
        mode_report = report.get(r.get("anchor_mode"), {})
        case_id = r["evidence_id"]
        if r["status"] == "completed" and case_id in mode_report.get("_corrected", {}):
            new, tau = mode_report["_corrected"][case_id], mode_report["_case_taus"][case_id]
            observed = r["observed"]
            row.update(
                predicted_raw=r["predicted"],
                predicted=new,
                sharpen_tau=tau,
                option_errors=[abs(new[o] - observed[o]) for o in observed],
                mae=_mae(new, observed),
                js_divergence=js_divergence(new, observed),
            )
            sharpened = {
                d: {c: sharpen(dist, tau) for c, dist in cats.items()}
                for d, cats in r.get("predicted_subgroups", {}).items()
            }
            observed_sub = by_case.get(case_id, {}).get("observed_subgroups", {})
            scores = subgroup_errors(sharpened, observed_sub)
            errors = []
            for d, cats in sharpened.items():
                for c, dist in cats.items():
                    target = observed_sub.get(d, {}).get(c)
                    if target and set(target["distribution"]) == set(dist):
                        errors.extend(abs(dist[o] - target["distribution"][o]) for o in dist)
            row.update(
                predicted_subgroups=sharpened,
                subgroup_errors=scores,
                subgroup_mae=float(np.mean([s["mae"] for s in scores])) if scores else None,
                subgroup_option_errors=errors,
            )
        corrected_out.append(row)
    return corrected_out


def interval_coverage(results, alpha=0.1, key="anchor_mode", min_cases=MIN_CASES):
    """Cobertura fuera de muestra (dejando un caso fuera): % de valores reales dentro del intervalo 1-alpha.

    El radio de cada caso sale de los errores por opción de los demás casos de su estrato.
    """
    out = {}
    done = [r for r in results if r["status"] == "completed" and r.get("option_errors")]
    for group in sorted({str(r.get(key)) for r in done}) + ["global"]:
        items = done if group == "global" else [r for r in done if str(r.get(key)) == group]
        if len(items) < min_cases:
            out[group] = {"cases": len(items), "coverage": None}
            continue
        inside = total = 0
        for r in items:
            others = np.sort([e for o in items if o is not r for e in o["option_errors"]])
            rank = min(len(others), int(np.ceil((len(others) + 1) * (1 - alpha))))
            radius = float(others[rank - 1])
            inside += sum(e <= radius for e in r["option_errors"])
            total += len(r["option_errors"])
        out[group] = {"cases": len(items), "coverage": inside / total, "target": 1 - alpha}
    return out
