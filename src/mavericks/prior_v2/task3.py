"""Tarea 3: intervalos conformales Mondrian por tramos de similitud de la mejor análoga.

Score de no conformidad por opción: |observado − predicho| / escala(p), con la misma escala que ``fastcal`` (sqrt(p(1−p)+0,02)).
Los tramos son cuantiles de la similitud léxica de la mejor análoga (terciles; dos tramos como comprobación), decididos
antes de mirar la cobertura y recalculados en cada pliegue con los estudios de calibración. Un tramo con menos de 15
preguntas se une al contiguo. Validación: dejando fuera un estudio entero cada vez; cobertura y anchura por tramo.
"""

import numpy as np

from ..fastcal import _quantile, scale
from . import data
from .task1 import OUT

LEVEL = 0.9
MIN_TRAMO = 15


def case_scores(row, predictor):
    p = np.asarray(predictor(row))
    return p, np.abs(p - row["observed"]) / scale(p)


def make_edges(sims, n_tramos):
    if n_tramos <= 1:
        return []
    cuts = np.quantile(sims, [i / n_tramos for i in range(1, n_tramos)])
    return sorted(set(float(c) for c in cuts))


def merge_small(edges, sims, minimum=MIN_TRAMO):
    """Une al contiguo los tramos con menos de ``minimum`` preguntas (quita el corte que los separa)."""
    edges = list(edges)
    while edges:
        bins = np.digitize(sims, edges)
        counts = np.bincount(bins, minlength=len(edges) + 1)
        small = int(np.argmin(counts))
        if counts[small] >= minimum:
            break
        edges.pop(max(0, small - 1) if small == len(edges) else small)
    return edges


def fit(train_rows, predictor, n_tramos):
    sims = np.array([r["best_sim"] for r in train_rows])
    edges = merge_small(make_edges(sims, n_tramos), sims)
    bins = np.digitize(sims, edges)
    scores = [case_scores(r, predictor)[1] for r in train_rows]
    q = []
    for b in range(len(edges) + 1):
        pooled = np.concatenate([s for s, bb in zip(scores, bins, strict=True) if bb == b])
        q.append(_quantile(pooled, LEVEL))
    return {"edges": edges, "q": q, "n": [int((bins == b).sum()) for b in range(len(edges) + 1)]}


def radius(model, sim, p):
    b = int(np.digitize([sim], model["edges"])[0])
    return model["q"][b] * scale(np.asarray(p)), b


def leave_one_study_out(rows, predictor, n_tramos):
    """Cobertura y anchura por tramo (tramos de cada pliegue, numerados de menor a mayor similitud) y global."""
    studies = sorted({r["study"] for r in rows})
    stats = {}
    per_study = {}
    for s in studies:
        train = [r for r in rows if r["study"] != s]
        model = fit(train, predictor, n_tramos)
        for r in (r for r in rows if r["study"] == s):
            p = np.asarray(predictor(r))
            rad, b = radius(model, r["best_sim"], p)
            lo, hi = np.clip(p - rad, 0, 1), np.clip(p + rad, 0, 1)
            inside = (r["observed"] >= lo) & (r["observed"] <= hi)
            key = (min(b, len(model["edges"])), len(model["edges"]) + 1)
            for k in (key, "global", "2 opciones" if r["n_options"] == 2 else "3 o más"):
                e = stats.setdefault(k, {"hit": 0, "n": 0, "width": 0.0, "cases": 0})
                e["hit"] += int(inside.sum())
                e["n"] += len(p)
                e["width"] += float((hi - lo).sum())
                e["cases"] += 1
            e = per_study.setdefault(s, {"hit": 0, "n": 0})
            e["hit"] += int(inside.sum())
            e["n"] += len(p)
    return stats, per_study


def summarize(stats):
    return {
        k: {"coverage": v["hit"] / v["n"], "width_pp": 100 * v["width"] / v["n"], "cases": v["cases"], "options": v["n"]}
        for k, v in stats.items()
    }


def report_block(name, rows, predictor):
    out = [f"### Predictor: {name}", ""]
    for n_tramos in (1, 3, 2):
        stats, per_study = leave_one_study_out(rows, predictor, n_tramos)
        s = summarize(stats)
        title = "Sin tramos (un solo cuantil)" if n_tramos == 1 else f"{n_tramos} tramos de similitud (cuantiles)"
        out += [f"**{title}** — cobertura objetivo 90 %", "| Grupo | preguntas | cobertura | anchura media (pp) |", "|---|---|---|---|"]
        for k in sorted((k for k in s if isinstance(k, tuple)), key=lambda t: t[0]):
            label = f"tramo {k[0] + 1} de {k[1]}" if n_tramos > 1 else "todas"
            out.append(f"| {label} | {s[k]['cases']} | {100 * s[k]['coverage']:.1f} % | {s[k]['width_pp']:.1f} |")
        for k in ("global", "2 opciones", "3 o más"):
            out.append(f"| {k} | {s[k]['cases']} | {100 * s[k]['coverage']:.1f} % | {s[k]['width_pp']:.1f} |")
        cover = [v["hit"] / v["n"] for v in per_study.values()]
        out += [f"Cobertura por estudio dejado fuera: mínima {100 * min(cover):.0f} %, mediana {100 * np.median(cover):.0f} %, máxima {100 * max(cover):.0f} %.", ""]
    return out


def leave_one_study_out_by(rows, predictor, key_fn, level=LEVEL):
    """Igual, pero con tramos dados por ``key_fn(fila)`` (p. ej. nº de opciones). Exploratorio."""
    stats = {}
    for study in sorted({r["study"] for r in rows}):
        train = [r for r in rows if r["study"] != study]
        q = {}
        for key in {key_fn(r) for r in train}:
            pooled = np.concatenate([case_scores(r, predictor)[1] for r in train if key_fn(r) == key])
            q[key] = _quantile(pooled, level)
        for r in (r for r in rows if r["study"] == study):
            p = np.asarray(predictor(r))
            rad = q.get(key_fn(r), max(q.values())) * scale(p)
            lo, hi = np.clip(p - rad, 0, 1), np.clip(p + rad, 0, 1)
            inside = (r["observed"] >= lo) & (r["observed"] <= hi)
            for k in (key_fn(r), "global"):
                e = stats.setdefault(k, {"hit": 0, "n": 0, "width": 0.0, "cases": 0})
                e["hit"] += int(inside.sum())
                e["n"] += len(p)
                e["width"] += float((hi - lo).sum())
                e["cases"] += 1
    return summarize(stats)


def calibrate_and_test(train_rows, test_rows, predictor, key_fn, level):
    """Calibra un cuantil por grupo con ``train_rows`` y mide cobertura y anchura en ``test_rows`` (conjuntos disjuntos)."""
    q = {}
    for key in {key_fn(r) for r in train_rows}:
        q[key] = _quantile(np.concatenate([case_scores(r, predictor)[1] for r in train_rows if key_fn(r) == key]), level)
    stats = {}
    for r in test_rows:
        p = np.asarray(predictor(r))
        rad = q[key_fn(r)] * scale(p)
        lo, hi = np.clip(p - rad, 0, 1), np.clip(p + rad, 0, 1)
        inside = (r["observed"] >= lo) & (r["observed"] <= hi)
        for k in (key_fn(r), "global"):
            e = stats.setdefault(k, {"hit": 0, "n": 0, "width": 0.0, "cases": 0})
            e["hit"] += int(inside.sum())
            e["n"] += len(p)
            e["width"] += float((hi - lo).sum())
            e["cases"] += 1
    return summarize(stats), q


def exploratory_block(rows, predictor, name):
    out = [f"### Exploratorio (no registrado de antemano): tramos por número de opciones — predictor {name}",
           "| Partición | grupo | preguntas | cobertura | anchura media (pp) |", "|---|---|---|---|---|"]
    for title, fn in (("2 vs 3 o más opciones", lambda r: "2 opciones" if r["n_options"] == 2 else "3 o más"),
                      ("opciones × similitud (mediana)", lambda r: ("2" if r["n_options"] == 2 else "3+", "sim alta" if r["best_sim"] >= 0.35 else "sim baja"))):
        s = leave_one_study_out_by(rows, predictor, fn)
        for k in sorted((k for k in s if k != "global"), key=str):
            out.append(f"| {title} | {k} | {s[k]['cases']} | {100 * s[k]['coverage']:.1f} % | {s[k]['width_pp']:.1f} |")
        out.append(f"| {title} | global | {s['global']['cases']} | {100 * s['global']['coverage']:.1f} % | {s['global']['width_pp']:.1f} |")
    return out + [""]


def main():
    rows = [r for r in data.load(OUT / "bigdev_preds.pkl") if not r["repeated"]]
    sims = np.array([r["best_sim"] for r in rows])
    edges = np.quantile(sims, [1 / 3, 2 / 3])
    lines = [
        "# Tarea 3 — Intervalos conformales Mondrian (dev_grande, preguntas nuevas)",
        "",
        f"{len(rows)} preguntas nuevas de {len({r['study'] for r in rows})} estudios; validación dejando un estudio fuera. Terciles de la similitud léxica de "
        f"la mejor análoga: cortes en {edges[0]:.2f} y {edges[1]:.2f} (con todos los datos, solo descriptivo; en cada pliegue se recalculan).",
        "",
    ]
    predictors = {
        "media 50/50 candidata + MLP (la que gana en el punto 4)": lambda r: 0.5 * r["knn_cand"] + 0.5 * r["mlp"],
        "KNN actual (producción)": lambda r: r["knn_now"],
        "(exploratorio, no registrado) media 50/50 KNN actual + MLP": lambda r: 0.5 * r["knn_now"] + 0.5 * r["mlp"],
    }
    for name, fn in predictors.items():
        lines += report_block(name, rows, fn)
        lines += exploratory_block(rows, fn, name)
    text = "\n".join(lines)
    (OUT / "task3_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
