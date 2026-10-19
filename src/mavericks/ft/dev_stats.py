"""Fase 3, paso 4 (CPU): evaluación prerregistrada de a1 en dev_grande (docs/prereg_finetuning.md). Una sola vez.

    python -m mavericks.ft.dev_stats            # evaluación principal (escribe el marcador dev_grande_DONE)
    python -m mavericks.ft.dev_stats --subgroups  # descriptivo de subgrupos (a1 frente al método actual)

Entrada: outputs/prior_v2/bigdev_preds.pkl (priors por pregunta, con su corte temporal) y outputs/ft/phase3/dev_probs.pkl (dev_infer).
Prior de producción = media 50/50 de «KNN actual» y MLP por mes (como ``service._blend_prior``). Bootstrap por estudios enteros, 10.000 réplicas, semilla 0.
Criterio: mejora ≥ 0,5 puntos frente al prior con el IC 95 % entero por debajo de 0.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..prior_v2 import data
from ..prior_v2.bigdev_eval import mae, study_bootstrap
from ..prior_v2.knn import TIE

OUT = Path("outputs/ft/phase3")
DRAWS, SEED, MIN_GAIN_PP = 10_000, 0, 0.5
SYSTEMS = ["Reparto uniforme", "KNN actual", "Prior de producción (50/50 KNN+MLP)", "Qwen base sin ajustar", "a1", "Media 50/50 a1 + prior"]
REFERENCE = SYSTEMS[2]


def load_inputs():
    import pickle

    rows = [r for r in data.load("outputs/prior_v2/bigdev_preds.pkl") if not r["repeated"]]
    probs = pickle.loads((OUT / "dev_probs.pkl").read_bytes())
    return rows, probs


def predictions(rows, probs):
    preds = {name: [] for name in SYSTEMS}
    for r in rows:
        key = f"{r['evidence_id']}|nacional|"
        prior = 0.5 * np.asarray(r["knn_now"]) + 0.5 * np.asarray(r["mlp"])
        a1 = np.asarray(probs["probs"]["a1"][key], dtype=float)
        a1 = a1 / a1.sum()
        base = np.asarray(probs["probs"]["base"][key], dtype=float)
        base = base / base.sum()
        assert len(a1) == len(r["observed"]) == len(prior)
        preds["Reparto uniforme"].append(np.asarray(r["uniform"]))
        preds["KNN actual"].append(np.asarray(r["knn_now"]))
        preds[REFERENCE].append(prior)
        preds["Qwen base sin ajustar"].append(base)
        preds["a1"].append(a1)
        preds["Media 50/50 a1 + prior"].append(0.5 * a1 + 0.5 * prior)
    return preds


def errors(rows, preds):
    return {name: np.array([mae(p, r["observed"]) for p, r in zip(ps, rows, strict=True)]) for name, ps in preds.items()}


def compare(reference, err, studies):
    delta = err - reference
    low, high = study_bootstrap(delta, studies, draws=DRAWS, seed=SEED)
    p = float(wilcoxon(err, reference).pvalue) if len(delta) >= 10 and np.any(delta != 0) else None
    return {"mae": 100 * err.mean(), "delta": 100 * delta.mean(), "low": 100 * low, "high": 100 * high, "better": int((delta < -TIE).sum()),
            "worse": int((delta > TIE).sum()), "ties": int((np.abs(delta) <= TIE).sum()), "p": p, "n": len(err)}


def fmt_p(p):
    return "—" if p is None else ("<0,001" if p < 0.001 else f"{p:.3f}")


def table(rows, err, title):
    studies = [r["study"] for r in rows]
    ref = err[REFERENCE]
    lines = [f"### {title} ({len(rows)} preguntas, {len(set(studies))} estudios)",
             "| Sistema | MAE (pp) | Δ frente al prior [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|"]
    results = {}
    for name in SYSTEMS:
        if name == REFERENCE:
            lines.append(f"| {name} | {100 * err[name].mean():.2f} | — | — | — |")
            continue
        r = compare(ref, err[name], studies)
        results[name] = r
        lines.append(f"| {name} | {r['mae']:.2f} | {r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}] | {r['better']} / {r['worse']} / {r['ties']} | {fmt_p(r['p'])} |")
    return lines, results


def verdict(r):
    gain = -r["delta"]
    ok = gain >= MIN_GAIN_PP and r["high"] < 0
    return ("CUMPLE" if ok else "NO CUMPLE"), f"mejora {gain:+.2f} pp, IC 95 % por estudio [{r['low']:+.2f}, {r['high']:+.2f}]"


def leak_checks(rows, results):
    """Si algo mejora más de 4 puntos, primero se busca una fuga; aquí se comprueba siempre (fechas, ids, estudios y textos exactos)."""
    ex = pd.read_pickle("outputs/ft/data/examples.pkl")
    train = ex[ex.in_a]
    dev_ids, dev_studies = {r["evidence_id"] for r in rows}, {r["study"] for r in rows}
    norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()  # noqa: E731
    train_text = {norm(q) for q in train.question}
    dev = ex[(ex.scope == "nacional") & ex.evidence_id.isin(dev_ids)]
    exact = int(dev.question.map(norm).isin(train_text).sum())
    lines = [
        f"- fecha máxima de entrenamiento de a1: {train.date.max()} (< 2025-07-01); fecha mínima de dev_grande: {min(r['date'] for r in rows)}",
        f"- ids de dev_grande dentro del entrenamiento de a1: {len(dev_ids & set(train.evidence_id))}; estudios compartidos: {len(dev_studies & set(train.study.astype(str)))}",
        f"- preguntas de dev_grande con texto idéntico (normalizado) a una de entrenamiento (CIS, Eurobarómetro o ESS): {exact} de {len(dev)}",
    ]
    assert not dev_ids & set(train.evidence_id) and not dev_studies & set(train.study.astype(str))
    big = [(n, r["delta"]) for n, r in results.items() if r["delta"] < -4]
    lines.append("- mejora > 4 pp sobre el prior: " + (", ".join(f"{n} ({-d:.2f} pp)" for n, d in big) if big else "ninguna"))
    return lines, exact


def main_stats():
    marker = OUT / "dev_grande_DONE"
    assert not marker.exists(), "la evaluación de dev_grande ya se ejecutó (no se repite)"
    rows, probs = load_inputs()
    assert len(rows) == 703 and len({r["study"] for r in rows}) == 23
    preds = predictions(rows, probs)
    err = errors(rows, preds)
    out = ["# Fase 3 — evaluación prerregistrada de a1 en dev_grande (preguntas nuevas, reparto nacional)", "",
           f"{len(rows)} preguntas, {len({r['study'] for r in rows})} estudios. Prior de producción = media 50/50 de KNN actual y MLP por mes con su corte. "
           f"IC por bootstrap de estudios enteros ({DRAWS} réplicas, semilla {SEED}). Criterio: mejora ≥ {MIN_GAIN_PP} pp con el IC entero por debajo de 0.", ""]
    main, results = table(rows, err, "Todas las preguntas nuevas")
    out += main + [""]
    for label, sel in (("2 opciones", [i for i, r in enumerate(rows) if r["n_options"] == 2]), ("3 o más opciones", [i for i, r in enumerate(rows) if r["n_options"] >= 3])):
        lines, _ = table([rows[i] for i in sel], {k: v[sel] for k, v in err.items()}, label)
        out += lines + [""]
    out += ["## Criterio prerregistrado", ""]
    for name in ("a1", "Media 50/50 a1 + prior"):
        word, detail = verdict(results[name])
        out.append(f"- **{name} frente al prior de producción: {word}** ({detail})")
    out += ["", f"Segundos por pregunta (una a una, RTX 4060 portátil): a1 {probs['seconds']['a1']:.3f}; Qwen base {probs['seconds']['base']:.3f}", "",
            "## Comprobaciones de fuga", *leak_checks(rows, results)[0]]
    text = "\n".join(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_grande_report.md").write_text(text, encoding="utf-8")
    pd.DataFrame({"evidence_id": [r["evidence_id"] for r in rows], "study": [r["study"] for r in rows], "n_options": [r["n_options"] for r in rows],
                  **{k: 100 * v for k, v in err.items()}}).to_csv(OUT / "dev_grande_errors.csv", index=False)
    commit = subprocess.run(["git", "rev-list", "-1", "HEAD", "--", "docs/prereg_finetuning.md"], capture_output=True, text=True).stdout.strip()
    marker.write_text(f"evaluación de dev_grande ejecutada {time.strftime('%Y-%m-%d %H:%M')}; último commit de docs/prereg_finetuning.md: {commit}\n", encoding="utf-8")
    print(text)


def subgroup_descriptive():
    """a1 frente al método actual de subgrupos (``group_estimates``: diferencias empíricas de las análogas contraídas a la mitad, con raking)
    en los subgrupos del CIS (≥ 100 encuestados, ≤ 6 por pregunta y dimensión) de las mismas preguntas. Descriptivo, sin criterio."""
    import pickle

    from .. import service
    from ..analog import combine
    from ..analog_segments import group_estimates

    rows, probs = load_inputs()
    preds = predictions(rows, probs)
    prior = dict(zip([r["evidence_id"] for r in rows], preds[REFERENCE], strict=True))
    collected = {c["evidence_id"]: c for c in data.load("outputs/prior_v2/bigdev_collected.pkl")}
    ex = pd.read_pickle("outputs/ft/data/examples.pkl")
    truth = ex[(ex.scope != "nacional") & (ex.source == "CIS") & ex.evidence_id.isin(prior)]
    paths = service.Paths()
    from ..population import PopulationLoader

    frame, _ = PopulationLoader().load(paths.population)
    from ..anchor import AnchorStore

    anchors = AnchorStore(paths.cis_dir)
    shares = {}
    for dim in service.SUBGROUP_DIMENSIONS:
        present = set(frame[dim].dropna())
        shares[dim] = {c: service._population_share(frame, dim, c) for c in service._ORDER[dim] if c in present}
    cache_path = OUT / "subgroup_current.pkl"
    current = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    t0 = time.time()
    for n, (eid, group) in enumerate(truth.groupby("evidence_id")):
        if eid in current:
            continue
        case = collected[eid]
        options = case["options"]
        weighted = combine(case["candidates"], options, service.ANALOG)["weighted"]
        specs = [(w, {"study_id": c["study_id"], "evidence_id": c["evidence_id"], "mapping": c["mapping"], "question_type": "single"})
                 for w, c in weighted if c.get("microdata") and c.get("study_id")]
        total = dict(zip(options, prior[eid], strict=True))
        current[eid] = group_estimates(specs, options, shares, anchors, total, service.CONTRACTION) if specs else None
        if n % 25 == 0:
            print(f"{n}/{truth.evidence_id.nunique()} preguntas ({time.time() - t0:.0f} s)", flush=True)
    cache_path.write_bytes(pickle.dumps(current))
    recs = []
    for _, r in truth.iterrows():
        est = current.get(r["evidence_id"])
        category = r["category"]
        if est is None or r["dim"] not in est or category not in est[r["dim"]]:
            continue
        options = collected[r["evidence_id"]]["options"]
        cur = np.array([est[r["dim"]][category][o] for o in options])
        a1 = np.asarray(probs["probs"]["a1"][r["id"]], dtype=float)
        a1 = a1 / a1.sum()
        p = np.asarray(r["props"], dtype=float)
        recs.append({"evidence_id": r["evidence_id"], "study": r["study"], "dim": r["dim"], "n_options": len(p), "a1": 100 * mae(a1, p),
                     "actual": 100 * mae(cur, p), "nacional": 100 * mae(prior[r["evidence_id"]], p), "ambos": 100 * mae(0.5 * a1 + 0.5 * cur, p)})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "subgroups_errors.csv", index=False)
    lines = ["# Descriptivo de subgrupos — a1 frente al método actual (preguntas nuevas de dev_grande)", "",
             f"{len(df)} subgrupos del CIS (de {len(truth)} muestreados; el resto sin análogas con microdatos o con categoría ausente de la población sintética). "
             "Método actual = diferencias empíricas de las análogas (KNN actual) contraídas a la mitad con raking sobre el prior de producción. Sin criterio de adopción.", "",
             "| Ámbito | subgrupos | MAE a1 (pp) | MAE método actual (pp) | MAE «prior nacional sin diferencias» (pp) | Δ a1 − actual [IC 95 % por estudio] | a1 mejor / peor |", "|---|---|---|---|---|---|---|"]
    for label, part in [("todos", df), *[(f"dim {d}", g) for d, g in df.groupby("dim")], ("2 opciones", df[df.n_options == 2]), ("3+ opciones", df[df.n_options >= 3])]:
        delta = (part.a1 - part.actual).to_numpy()
        lo, hi = study_bootstrap(delta / 100, part.study.to_numpy(), draws=DRAWS, seed=SEED)
        lines.append(f"| {label} | {len(part)} | {part.a1.mean():.2f} | {part.actual.mean():.2f} | {part.nacional.mean():.2f} | {delta.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] | "
                     f"{int((delta < -TIE * 100).sum())} / {int((delta > TIE * 100).sum())} |")
    lines.append(f"\nMedia 50/50 a1 + método actual: {df.ambos.mean():.2f} pp.")
    text = "\n".join(lines)
    (OUT / "subgroups_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgroups", action="store_true")
    main_stats() if not parser.parse_args().subgroups else subgroup_descriptive()
