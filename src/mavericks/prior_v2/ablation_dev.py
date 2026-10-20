"""Tabla final de ablación sobre los 44 casos dev difíciles (no se usa ningún caso oficial ni reservado).

Configuraciones: uniforme, KNN actual, prior nuevo (media 50/50 KNN + MLP), mezcla con el LLM w=0,3 (0,3·LLM + 0,7·prior nuevo) y el LLM
solo (baseline de desarrollo, qwen3:8b). El MLP se entrena por mes de trabajo de campo con el corte más restrictivo y sin los estudios del caso.
"""

import json
from pathlib import Path

import numpy as np

from ..analog import load_national_table
from ..calibrate import blend, mae
from ..evidence import EvidenceRepository
from . import bigdev, data, knn, learned
from . import bigdev_eval as ev
from .embcache import EmbCache
from .task1 import EVIDENCE, OUT, TABLE

W = 0.3
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def build_rows():
    cases = data.load_dev_cases()
    emb = EmbCache(OUT / "emb_cache")
    repo, table = EvidenceRepository(EVIDENCE), load_national_table(TABLE)
    mlp = bigdev.mlp_predictions(cases, learned.train_questions(repo, table, emb), emb)
    collected = {c["evidence_id"]: c for c in data.load(OUT / "dev_collected.pkl")}
    shift = knn.qq_shift(list(collected.values()))
    results = {
        r["evidence_id"]: r
        for r in json.loads(Path("outputs/baseline_v1/dev/backtest.json").read_text(encoding="utf-8"))["results"]
        if r["status"] == "completed"
    }
    rows = []
    for c in cases:
        a = knn.CaseArrays(collected[c["evidence_id"]], shift)
        options = a.options
        knn_p = dict(zip(options, a.prior(*knn.CURRENT), strict=True))
        new = {o: 0.5 * knn_p[o] + 0.5 * float(mlp[c["evidence_id"]][i]) for i, o in enumerate(options)}
        llm = results[c["evidence_id"]]
        rows.append(
            {
                "id": c["evidence_id"], "study": str(c["estudio_id"]), "topic": c["topic"], "n_options": len(options),
                "mode": llm["anchor_mode"], "observed": dict(zip(options, a.observed, strict=True)), "seconds": llm.get("execution_time"),
                "uniform": {o: 1 / len(options) for o in options}, "knn": knn_p, "prior": new, "llm": llm["predicted"],
            }
        )
    return rows


def loso_w(rows, key="prior"):
    """Error por caso con w elegido por validación cruzada dejando un estudio fuera (objetivo MAE)."""
    out = {}
    for study in sorted({r["study"] for r in rows}):
        train = [r for r in rows if r["study"] != study]
        w = min(GRID, key=lambda g: np.mean([mae(blend(r["llm"], r[key], g, 1.0), r["observed"]) for r in train]))
        for r in rows:
            if r["study"] == study:
                out[r["id"]] = (w, mae(blend(r["llm"], r[key], w, 1.0), r["observed"]))
    return out


def main():
    rows = build_rows()
    studies = [r["study"] for r in rows]
    fixed = {r["id"]: blend(r["llm"], r["prior"], W, 1.0) for r in rows}
    held = loso_w(rows)
    seconds = float(np.mean([r["seconds"] for r in rows if r["seconds"] is not None]))
    errors = {
        "Reparto uniforme": np.array([mae(r["uniform"], r["observed"]) for r in rows]),
        "KNN actual (prior anterior)": np.array([mae(r["knn"], r["observed"]) for r in rows]),
        "Prior nuevo = media KNN + MLP (modo rápido)": np.array([mae(r["prior"], r["observed"]) for r in rows]),
        "Mezcla con LLM, w=0,3 fijo (modo con LLM)": np.array([mae(fixed[r["id"]], r["observed"]) for r in rows]),
        "Mezcla con LLM, w elegido por CV (un estudio fuera)": np.array([held[r["id"]][1] for r in rows]),
        "LLM solo (qwen3:8b)": np.array([mae(r["llm"], r["observed"]) for r in rows]),
    }
    times = {"Reparto uniforme": "0", "KNN actual (prior anterior)": "≈0,3", "Prior nuevo = media KNN + MLP (modo rápido)": "≈2"}
    ref, prior = errors["KNN actual (prior anterior)"], errors["Prior nuevo = media KNN + MLP (modo rápido)"]
    lines = [
        f"# Ablación final sobre los {len(rows)} casos dev difíciles ({len(set(studies))} estudios)", "",
        "| Configuración | MAE (pp) | Δ frente al KNN actual [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p | s/pregunta |",
        "|---|---|---|---|---|---|",
    ]
    for name, err in errors.items():
        t = times.get(name, f"{seconds:.0f} (línea base dev)")
        if name == "KNN actual (prior anterior)":
            lines.append(f"| {name} | {100 * err.mean():.2f} | — | — | — | {t} |")
        else:
            lines.append(ev.line(name, ev.compare(ref, err, studies)).rstrip()[:-1].rstrip() + f" | {t} |")
    lines += ["", "### Mezcla con LLM frente al prior nuevo solo (¿aporta el LLM?)", "| Configuración | MAE (pp) | Δ frente al prior nuevo [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|"]
    for name in ("Mezcla con LLM, w=0,3 fijo (modo con LLM)", "Mezcla con LLM, w elegido por CV (un estudio fuera)", "LLM solo (qwen3:8b)"):
        lines.append(ev.line(name, ev.compare(prior, errors[name], studies)))
    chosen = sorted({round(w, 1) for w, _ in held.values()})
    lines += ["", f"w elegido por CV en cada pliegue: {chosen}.", "", "### Desglose descriptivo (MAE, pp)", "| Grupo | casos | uniforme | KNN actual | prior nuevo | mezcla w=0,3 | LLM solo |", "|---|---|---|---|---|---|---|"]
    keys = list(errors)
    groups = [("2 opciones", [r["n_options"] == 2 for r in rows]), ("3 o más", [r["n_options"] >= 3 for r in rows]),
              ("modo prior", [r["mode"] == "prior" for r in rows]), ("modo sin ancla", [r["mode"] == "none" for r in rows])]
    for label, mask in groups:
        m = np.array(mask)
        if m.sum():
            lines.append(f"| {label} | {int(m.sum())} | " + " | ".join(f"{100 * errors[keys[i]][m].mean():.2f}" for i in (0, 1, 2, 3, 5)) + " |")
    text = "\n".join(lines)
    (OUT / "ablation_dev.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
