"""Validación de los cruces de DOS variables del MRP (script aparte; no toca la aplicación ni ``mrp.py``).

    python -m mavericks.mrp_cross_eval [--smoke] [--no-docs]

Misma tarea que la evaluación prerregistrada de subgrupos (``mrp_eval``), pero sobre celdas de dos variables: con los microdatos de la oleada t−1 y el
reparto nacional REAL de t, predice el reparto de cada celda sexo×edad, edad×comunidad y estudios×edad de la oleada t y lo compara con lo medido
(ponderado) en las celdas con al menos 50 encuestados con respuesta sustantiva. Referencia: usar el reparto nacional de t para todas las celdas.
Solo preguntas repetidas del CIS en oleadas NO reservadas (``assert_no_reserved``; trabajo de campo de t hasta 2026-03-31).
IC 95 % remuestreando ESTUDIOS (las preguntas y celdas de un mismo estudio están correlacionadas).
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import mrp_eval
from .mrp import MRP
from .prior_v2.bigdev_eval import study_bootstrap

CROSSES = (("sexo", "grupo_edad"), ("grupo_edad", "ccaa"), ("nivel_estudios", "grupo_edad"))
MIN_CELL = 50
T_WINDOW = ("2024-07-01", "2026-03-31")
DRAWS, SEED = 10_000, 0
OUT = Path("outputs/mrp_cross")
DOC = Path("docs/mrp_resultados.md")
MARKER = "## Cruces de dos variables (validación posterior)"
N_BINS = ((50, 99), (100, 199), (200, 10**9))


def cell_truth(frame, pair, crosses=CROSSES, min_cell=MIN_CELL):
    """{((dim1, cat1), (dim2, cat2)): (n, reparto ponderado)} de la oleada t con al menos ``min_cell`` encuestados sustantivos."""
    f = frame[frame.answer.isin(pair["options"])].drop_duplicates("respondent_uid")
    out = {}
    for a, b in crosses:
        if a not in f.columns or b not in f.columns:
            continue
        valid = f[f[a].map(mrp_eval.valid_category) & f[b].map(mrp_eval.valid_category)]
        for (ca, cb), g in valid.groupby([a, b], dropna=True):
            if len(g) < min_cell:
                continue
            w = g.groupby("answer").peso.sum().reindex(pair["options"]).fillna(0.0)
            if w.sum() > 0:
                out[((a, str(ca)), (b, str(cb)))] = (int(len(g)), (w / w.sum()).to_numpy())
    return out


def evaluate(pairs, anchors, population, log=print, crosses=CROSSES):
    rows, info = [], []
    started = time.time()
    for n, pair in enumerate(pairs):
        frame_t = anchors._frame(pair["t_study"], pair["t_id"])
        frame_p = anchors._frame(pair["p_study"], pair["p_id"])
        truth = cell_truth(frame_t, pair, crosses)
        if truth:
            options = pair["options"]
            goal = mrp_eval.national(frame_t, pair)
            back = {prev: option for option, prev in pair["mapping"].items()}
            model = MRP(population).fit(frame_p.assign(answer=frame_p["answer"].map(back)), options)
            post = model.postratify(goal, list(truth))
            for cell, (size, observed) in truth.items():
                if cell not in post:
                    info.append({"t_id": pair["t_id"], "skipped": str(cell)})
                    continue
                rows.append({
                    "series": pair["series"], "t_id": pair["t_id"], "t_study": pair["t_study"], "cross": f"{cell[0][0]}×{cell[1][0]}", "cell": f"{cell[0][1]} × {cell[1][1]}",
                    "n": size, "n_options": len(options), "err_mrp": 100 * float(np.abs(post[cell] - observed).mean()), "err_national": 100 * float(np.abs(goal - observed).mean()),
                })
            info.append({"t_id": pair["t_id"], "omitted": list(model.omitted), "C": model.C})
        anchors._frames.pop((pair["t_study"], pair["t_id"]), None)
        anchors._frames.pop((pair["p_study"], pair["p_id"]), None)
        if (n + 1) % 25 == 0:
            log(f"{n + 1}/{len(pairs)} pares ({time.time() - started:.0f} s, {len(rows)} celdas)")
    return pd.DataFrame(rows), info


def block(d):
    delta = (d.err_mrp - d.err_national).to_numpy()
    low, high = study_bootstrap(delta, d.t_study.to_numpy(), draws=DRAWS, seed=SEED)
    return {"celdas": int(len(d)), "pares": int(d.t_id.nunique()), "estudios": int(d.t_study.nunique()), "series": int(d.series.nunique()),
            "mae_mrp": float(d.err_mrp.mean()), "mae_nacional": float(d.err_national.mean()), "delta": float(delta.mean()), "ic": [float(low), float(high)],
            "mrp_mejor": int((delta < 0).sum()), "mrp_peor": int((delta > 0).sum())}


def summarize(df):
    out = {"total": block(df), "por_cruce": {c: block(g) for c, g in df.groupby("cross")}, "por_n": {}}
    for low, high in N_BINS:
        part = df[(df.n >= low) & (df.n <= high)]
        if len(part):
            out["por_n"][f"{low}–{high if high < 10**9 else '∞'}"] = block(part)
    return out


def _row(name, b):
    return (f"| {name} | {b['celdas']} | {b['pares']} | {b['estudios']} | {b['mae_nacional']:.2f} | {b['mae_mrp']:.2f} | "
            f"{b['delta']:+.2f} [{b['ic'][0]:+.2f}, {b['ic'][1]:+.2f}] | {b['mrp_mejor']} / {b['mrp_peor']} |")


def report_text(res, window=T_WINDOW):
    t = res["total"]
    head = "| {} | Celdas | Pares | Estudios | Nacional (pp) | MRP (pp) | MRP − nacional [IC 95 %] | MRP mejor / peor |\n|---|---|---|---|---|---|---|---|"
    lines = [
        MARKER, "",
        "Script `python -m mavericks.mrp_cross_eval` (no cambia la aplicación ni `mrp.py`). Tarea: con los microdatos de la oleada t−1 y el reparto nacional real de t, predecir el reparto de cada celda de "
        f"dos variables de t y compararlo con lo medido (ponderado) en las celdas con ≥ {MIN_CELL} encuestados con respuesta sustantiva. Solo preguntas repetidas del CIS en oleadas no reservadas "
        f"(trabajo de campo de t entre {window[0]} y {window[1]}). Error = diferencia absoluta media entre opciones, en puntos porcentuales; «Nacional» es usar el reparto nacional de t para todas las celdas. "
        f"IC 95 % remuestreando estudios ({DRAWS} réplicas, semilla {SEED}).", "",
        f"**Total: {t['celdas']} celdas, {t['pares']} pares, {t['estudios']} estudios, {t['series']} series.** MRP {t['mae_mrp']:.2f} pp frente a {t['mae_nacional']:.2f} pp del reparto nacional "
        f"(diferencia {t['delta']:+.2f} pp, IC 95 % [{t['ic'][0]:+.2f}, {t['ic'][1]:+.2f}]).", "",
        "### Por cruce", head.format("Cruce")] + [_row(c, b) for c, b in res["por_cruce"].items()] + ["", "### Por tamaño de la celda en t", head.format("n encuestados")]
    lines += [_row(k, b) for k, b in res["por_n"].items()]
    lines += [
        "", "### Límites",
        "- El objetivo nacional es el **real** de t (como en la evaluación prerregistrada de subgrupos); con el nacional que estima el modelo para una pregunta nueva, el error de las celdas será mayor.",
        "- «Lo medido» es una muestra: el MAE incluye el ruido muestral de cada celda (más alto cuanto menor es n); no se ha descontado.",
        "- El IC remuestrea estudios (cubre la dependencia entre preguntas y celdas de un mismo estudio), no la dependencia entre oleadas consecutivas de la misma serie.",
        "- Para cruces no existe el «método anterior»: solo se compara con usar el reparto nacional para todas las celdas.",
    ]
    return "\n".join(lines) + "\n"


def write_doc(text, path=DOC):
    """Añade (o sustituye) la sección al final de ``docs/mrp_resultados.md`` sin tocar lo anterior."""
    old = path.read_bytes().decode("utf-8")  # sin traducir los saltos de línea: el documento conserva los suyos
    sep = "\r\n" if "\r\n" in old else "\n"
    old = old.replace("\r\n", "\n")
    if MARKER in old:
        old = old[: old.index(MARKER)].rstrip("\n") + "\n"
    path.write_bytes((old.rstrip("\n") + "\n\n" + text).replace("\n", sep).encode("utf-8"))


def run(smoke=False, docs=True):
    from .anchor import AnchorStore

    window = ("2024-04-01", "2024-06-30") if smoke else T_WINDOW
    pairs = mrp_eval.build_pairs(t_window=window)
    if smoke:
        pairs = pairs[:10]
    population = pd.read_parquet(Path(mrp_eval.CIS_DIR) / "synthetic_population.parquet")
    df, _ = evaluate(pairs, AnchorStore(mrp_eval.CIS_DIR), population)
    res = summarize(df)
    text = report_text(res, window)
    out = OUT / ("smoke" if smoke else ".")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_csv(out / "cells.csv", index=False)
    if docs and not smoke:
        write_doc(text)
    print(re.sub(r"[^\x00-\x7f]", "?", text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="prueba del código con 10 pares fuera de la ventana (no escribe la documentación)")
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args()
    run(args.smoke, not args.no_docs)
