"""Tarea 1: ajuste del KNN del prior. Solo casos dev (44 de modo prior y sin ancla).

Uso: ``python -m mavericks.prior_v2.task1`` (con OMP_NUM_THREADS=4). Escribe outputs/prior_v2/task1_report.md y .json.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import data, knn
from .embcache import EmbCache

OUT = Path("outputs/prior_v2")
EVIDENCE = "data/processed_v2/evidence_v2.sqlite"
TABLE = "data/processed_v2/cis_national.json"
NAMES = {"lex": "léxica", "e5qq": "e5 query-query", "e5qp": "e5 query-passage", "mix": "mezcla 0,5"}


def cfg_name(cfg):
    sim, k, p = cfg
    return f"{NAMES[sim]}, k={k}, pot={p}"


def modes_of_cases(ids):
    """Modo (prior / sin ancla) de cada caso según la línea base dev; solo sirve para el desglose."""
    results = json.loads(Path("outputs/baseline_v1/dev/backtest.json").read_text(encoding="utf-8"))["results"]
    by = {r["evidence_id"]: r["anchor_mode"] for r in results}
    return [by[i] for i in ids]


def fmt(v):
    return f"{100 * v:.2f}"


def paired_row(name, reference, errors):
    r = knn.paired(reference, errors)
    p = "n insuficiente" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.3f}"
    return (
        f"| {name} | {fmt(np.mean(errors))} | {100 * r['mean_delta']:+.2f} [{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}] "
        f"| {r['better']} / {r['worse']} / {r['ties']} | {p} |"
    )


def main():
    t0 = time.time()
    cases = data.load_dev_cases()
    emb = EmbCache(OUT / "emb_cache")
    vectors = data.case_vectors(OUT / "emb_cache" / "dev_cases.npz")
    pickle_path = OUT / "dev_collected.pkl"
    if pickle_path.exists():
        collected = data.load(pickle_path)
    else:
        repo = EvidenceRepository(EVIDENCE)
        collected = data.collect(cases, repo, load_national_table(TABLE), emb, vectors)
        data.save(pickle_path, collected)
    shift = knn.qq_shift(collected)
    arrays = [knn.CaseArrays(c, shift) for c in collected]
    configs = knn.grid()
    matrix = knn.error_matrix(arrays, configs)  # configuraciones × casos
    current = configs.index(knn.CURRENT)
    uniform = np.array([np.mean(np.abs(1 / len(a.options) - a.observed)) for a in arrays])
    studies = [a.study for a in arrays]
    held, chosen = knn.nested_grouped(matrix, studies)
    modes = modes_of_cases([a.evidence_id for a in arrays])
    n_opts = np.array([len(a.options) for a in arrays])

    means = matrix.mean(axis=1)
    top = np.argsort(means)[:5]
    wins = Counter()
    fold_rows = []
    for g in sorted(set(studies)):
        test = [i for i in range(len(arrays)) if studies[i] == g]
        train = [i for i in range(len(arrays)) if studies[i] != g]
        best = int(np.argmin(matrix[:, train].mean(axis=1)))
        wins[best] += 1
        fold_rows.append((g, len(test), cfg_name(configs[best]), 100 * matrix[best, test].mean(), 100 * matrix[current, test].mean()))

    lines = [
        "# Tarea 1 — KNN del prior (dev, 44 casos: 5 prior + 39 sin ancla)",
        "",
        f"Estudios dev en el conjunto: {len(set(studies))} (pliegues del bucle externo). Rejilla: {len(configs)} configuraciones "
        f"(similitud × k ∈ {knn.K_GRID} × potencia ∈ {knn.POWER_GRID}). Desplazamiento de escala e5 query-query: {shift:+.3f}.",
        "Candidatas: léxica = 40 mejores por TF-IDF (producción); e5 = 40 mejores por coseno sobre TODO el almacén filtrado; "
        "mezcla = unión de ambas.",
        "",
        "## Resultado honesto (bucle externo, agrupado por estudio)",
        "| Configuración | MAE (pp) | Δ frente a la actual [IC 95 %] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
        f"| Reparto uniforme | {fmt(uniform.mean())} | — | — | — |",
        f"| Actual ({cfg_name(knn.CURRENT)}) | {fmt(matrix[current].mean())} | — | — | — |",
        paired_row("Selección anidada (elegida con los otros estudios)", matrix[current], held),
        "",
        "## Las 5 mejores combinaciones y la actual (elegidas MIRANDO los 44 casos dev: optimistas)",
        "| Configuración | MAE (pp) | Δ frente a la actual [IC 95 %] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for idx in top:
        lines.append(paired_row(cfg_name(configs[idx]), matrix[current], matrix[idx]))
    lines += [
        paired_row(f"{cfg_name(knn.CURRENT)} (actual)", matrix[current], matrix[current]),
        "",
        "## Qué combinación ganó en el bucle interno (por pliegue de estudio)",
        "| Configuración elegida | pliegues | |",
        "|---|---|---|",
    ]
    for idx, n in wins.most_common():
        lines.append(f"| {cfg_name(configs[idx])} | {n} de {len(set(studies))} | {'actual' if idx == current else ''} |")
    lines += ["", "| Estudio fuera | casos | elegida con los demás | MAE elegida (pp) | MAE actual (pp) |", "|---|---|---|---|---|"]
    for g, n, name, e1, e0 in fold_rows:
        lines.append(f"| {g} | {n} | {name} | {e1:.2f} | {e0:.2f} |")

    def breakdown(title, groups):
        out = ["", f"## {title}", "| Grupo | casos | uniforme | actual | anidada | mejor global (optimista) |", "|---|---|---|---|---|---|"]
        for label, mask in groups:
            mask = np.asarray(mask)
            if mask.sum():
                out.append(
                    f"| {label} | {int(mask.sum())} | {fmt(uniform[mask].mean())} | {fmt(matrix[current][mask].mean())} | "
                    f"{fmt(held[mask].mean())} | {fmt(matrix[top[0]][mask].mean())} |"
                )
        return out

    lines += breakdown("Por modo", [(m, np.array(modes) == m) for m in ("prior", "none")])
    lines += breakdown("Por número de opciones", [("2 opciones", n_opts == 2), ("3 o más", n_opts >= 3)])
    best = configs[top[0]]
    edge = []
    if best[1] in (min(knn.K_GRID), max(knn.K_GRID)):
        edge.append(f"k={best[1]} está en el borde de la rejilla {knn.K_GRID}")
    if best[2] in (min(knn.POWER_GRID), max(knn.POWER_GRID)):
        edge.append(f"potencia={best[2]} está en el borde de la rejilla {knn.POWER_GRID}")
    lines += ["", "## Bordes de la rejilla", "; ".join(edge) if edge else "La mejor combinación no está en ningún borde."]
    lines += ["", f"Tiempo de cálculo: {time.time() - t0:.0f} s."]
    text = "\n".join(lines)
    (OUT / "task1_report.md").write_text(text, encoding="utf-8")
    (OUT / "task1_report.json").write_text(
        json.dumps(
            {
                "configs": [list(c) for c in configs], "mean_mae": means.tolist(), "current": list(knn.CURRENT),
                "nested_mae": float(held.mean()), "uniform_mae": float(uniform.mean()),
                "wins": {cfg_name(configs[i]): n for i, n in wins.items()},
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
