"""Evaluación sobre dev_grande: comparación de priors en las preguntas "nuevas" con IC bootstrap por estudio."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from . import data
from .knn import TIE
from .task1 import OUT

# variables de perfil sociodemográfico (no son opiniones): se usan solo en un análisis de sensibilidad
PROFILE = (
    "sexo", "edad", "ecivil", "estudios", "nivelestentrev", "cno11", "cno11r", "sitlab", "relalab", "religion", "clasesub",
    "clasesocial", "ingrehog", "recuerdo", "recuvotog", "participaciong", "escuela", "votosimg", "simpatia", "prefpte",
    "probvoto", "numpersonas", "perprin", "nacionalidad", "momennac", "nivcaste", "tamuni", "ccaa", "prov", "mun",
)


def is_profile(evidence_id):
    return evidence_id.split("_", 1)[-1].lower() in PROFILE


def mae(pred, observed):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(observed))))


def study_bootstrap(delta, studies, draws=5000, seed=0):
    """IC 95 % de la diferencia media remuestreando ESTUDIOS (las preguntas de un estudio están correlacionadas)."""
    delta, studies = np.asarray(delta, dtype=float), np.asarray(studies)
    names = sorted(set(studies))
    sums = np.array([delta[studies == s].sum() for s in names])
    counts = np.array([(studies == s).sum() for s in names])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(names), (draws, len(names)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(reference, errors, studies):
    delta = np.asarray(errors) - np.asarray(reference)
    low, high = study_bootstrap(delta, studies)
    p = float(wilcoxon(errors, reference).pvalue) if len(delta) >= 10 and np.any(delta != 0) else None
    return {
        "mae": float(np.mean(errors)), "delta": float(delta.mean()), "ci": (low, high),
        "better": int((delta < -TIE).sum()), "worse": int((delta > TIE).sum()), "ties": int((np.abs(delta) <= TIE).sum()), "p": p,
    }


def line(name, r):
    p = "—" if r["p"] is None else ("<0,001" if r["p"] < 0.001 else f"{r['p']:.3f}")
    return (
        f"| {name} | {100 * r['mae']:.2f} | {100 * r['delta']:+.2f} [{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}] "
        f"| {r['better']} / {r['worse']} / {r['ties']} | {p} |"
    )


def configs(rows):
    cand_mlp = [0.5 * r["knn_cand"] + 0.5 * r["mlp"] for r in rows]
    preds = {
        "Reparto uniforme": [r["uniform"] for r in rows], "KNN actual": [r["knn_now"] for r in rows],
        "KNN candidata (mezcla k=3 pot=4)": [r["knn_cand"] for r in rows], "MLP": [r["mlp"] for r in rows],
        "Media 50/50 candidata + MLP": cand_mlp,
    }
    return {n: np.array([mae(p, r["observed"]) for p, r in zip(ps, rows, strict=True)]) for n, ps in preds.items()}


def table(rows, title):
    e = configs(rows)
    studies = [r["study"] for r in rows]
    out = [f"### {title} ({len(rows)} preguntas, {len(set(studies))} estudios)",
           "| Configuración | MAE (pp) | Δ frente al KNN actual [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |",
           "|---|---|---|---|---|"]
    for n, err in e.items():
        out.append(line(n, compare(e["KNN actual"], err, studies)) if n != "KNN actual" else f"| {n} | {100 * err.mean():.2f} | — | — | — |")
    out += ["", "Frente a la candidata:", "| Configuración | MAE (pp) | Δ [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|"]
    for n in ("MLP", "Media 50/50 candidata + MLP"):
        out.append(line(n, compare(e["KNN candidata (mezcla k=3 pot=4)"], e[n], studies)))
    return out


def main():
    rows = data.load(OUT / "bigdev_preds.pkl")
    hard44 = {c["evidence_id"] for c in json.loads(Path("outputs/ablation/cases_dev_hard.json").read_text(encoding="utf-8-sig"))}
    new = [r for r in rows if not r["repeated"]]
    rep = [r for r in rows if r["repeated"]]
    lines = [
        "# dev_grande — preguntas del CIS (2-7 opciones) de estudios no reservados, 2025-07-01 a 2026-03-26",
        "",
        f"Preguntas: {len(rows)} en {len({r['study'] for r in rows})} estudios. Repetidas (existe en el pasado una con similitud léxica ≥0,90, "
        f"escala equivalente y mismo ítem): {len(rep)}; nuevas: {len(new)}.",
        "Por número de opciones (nuevas): " + ", ".join(f"{k}: {sum(1 for r in new if r['n_options'] == k)}" for k in range(2, 8)),
        "",
        "## Comparación principal: preguntas NUEVAS",
        *table(new, "Todas las nuevas"),
        "",
    ]
    for label, sel in (("2 opciones", [r for r in new if r["n_options"] == 2]), ("3 o más opciones", [r for r in new if r["n_options"] >= 3])):
        lines += [*table(sel, label), ""]
    indep = [r for r in new if r["evidence_id"] not in hard44]
    lines += ["## Sensibilidad", *table(indep, "Nuevas SIN los 44 casos dev con los que se eligió la candidata"), ""]
    lines += [*table([r for r in new if not is_profile(r["evidence_id"])], "Nuevas sin variables de perfil sociodemográfico"), ""]
    in44 = [r for r in new if r["evidence_id"] in hard44]
    lines += [*table(in44, "Solo los casos de los 44 que caen en la ventana (nuevas)"), ""]
    lines += ["## Descriptivo: preguntas repetidas (las cubriría el modo directo)", *table(rep, "Repetidas")]
    text = "\n".join(lines)
    (OUT / "bigdev_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
