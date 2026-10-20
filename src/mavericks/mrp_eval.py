"""Evaluación prerregistrada del MRP frente al método actual de subgrupos (docs/prereg_mrp.md). Solo preguntas repetidas del CIS entre 2024-01 y 2025-06.

    python -m mavericks.mrp_eval counts     # número de pares y de subgrupos (no calcula ningún error)
    python -m mavericks.mrp_eval run        # evaluación única; escribe outputs/mrp/DONE

Tarea: con los microdatos de la oleada t−1 y el reparto nacional REAL de t, predecir el reparto de cada subgrupo de t (≥ 100 encuestados con respuesta sustantiva en t).
Sistemas: método actual (``analog_segments.group_estimates``: desviaciones de t−1 contraídas al 50 % + raking), MRP (``mrp.py``) y «sin diferencias» (el nacional para todos).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .anchor import option_mapping, question_similarity, same_item, series_key
from .evidence import EvidenceRepository, source_family
from .ft.data import _substantive
from .prior_v2.bigdev_eval import PROFILE
from .prior_v2.data import assert_no_reserved, reserved_ids

OUT = Path("outputs/mrp")
EVIDENCE, TABLE, CIS_DIR = "data/processed_v2/evidence_v2.sqlite", "data/processed_v2/cis_national.json", "data/processed"
T_WINDOW = ("2024-07-01", "2025-06-30")  # trabajo de campo de la oleada t
PREVIOUS_FROM = "2024-01-01"  # la oleada t−1 es la anterior más reciente de la misma serie, no antes de esta fecha
MIN_SIMILARITY, MIN_SUBGROUP, MAX_OPTIONS = 0.85, 100, 7
DIMENSIONS = ("sexo", "grupo_edad", "ccaa", "nivel_estudios", "ingresos_hogar_tramo")
INVALID = {"nc", "otros", "n.s.", "n.c.", "ns", "nan", "none", ""}
DRAWS, SEED, MIN_GAIN_PP = 10_000, 0, 0.3
LOCK = OUT / "DONE"


def valid_category(value):
    return value is not None and str(value).strip().lower() not in INVALID


def build_pairs(repo=None, table=None, t_window=T_WINDOW):
    """Pares (t, t−1) de la misma serie. Sin leer ningún resultado: solo identificadores, fechas, textos y opciones."""
    from .analog import load_national_table

    repo = repo or EvidenceRepository(EVIDENCE)
    table = table or load_national_table(TABLE)
    cis = [e for e in repo.all() if source_family(e) == "cis" and e.evidence_id in table and e.fieldwork_date and e.question_text]
    cis = [e for e in cis if e.metadata.get("original_fields", {}).get("question_type", "single") in (None, "single")]
    by_series = {}
    for e in cis:
        key = series_key(e.evidence_id)
        if key and key not in PROFILE:
            by_series.setdefault(key, []).append(e)
    pairs = []
    for key, items in sorted(by_series.items()):
        items.sort(key=lambda e: e.fieldwork_date)
        for e in items:
            date = e.fieldwork_date[:10]
            if not t_window[0] <= date <= t_window[1]:
                continue
            keep = _substantive(list(e.options or []), table[e.evidence_id]["dist"])
            if not keep:
                continue
            earlier = [p for p in items if PREVIOUS_FROM <= p.fieldwork_date[:10] < date and str(p.study_id) != str(e.study_id)]
            for p in reversed(earlier):  # la más reciente primero
                mapping = option_mapping(keep, list(p.options or []), legacy_extras=True)  # el conjunto prerregistrado no cambia
                if mapping is None or question_similarity(e.question_text, p.question_text) < MIN_SIMILARITY or not same_item(e.question_text, p.question_text):
                    continue
                pairs.append({"series": key, "t_id": e.evidence_id, "t_study": str(e.study_id), "t_date": date, "p_id": p.evidence_id, "p_study": str(p.study_id),
                              "p_date": p.fieldwork_date[:10], "options": keep, "mapping": mapping, "question": " ".join(e.question_text.split())})
                break
    assert_no_reserved([x for p in pairs for x in (p["t_id"], p["p_id"])], "pares del MRP", reserved_ids(CIS_DIR))
    assert all(p["t_date"] <= t_window[1] and p["p_date"] < p["t_date"] for p in pairs)
    return pairs


def truth(frame, pair):
    """{(dimensión, categoría): (n, reparto ponderado sobre las opciones sustantivas)} de la oleada t."""
    f = frame[frame.answer.isin(pair["options"])]
    out = {}
    for dim in DIMENSIONS:
        if dim not in f.columns:
            continue
        for category, g in f.groupby(dim, dropna=True):
            if not valid_category(category):
                continue
            n = g.respondent_uid.nunique()
            if n < MIN_SUBGROUP:
                continue
            w = g.groupby("answer").peso.sum().reindex(pair["options"]).fillna(0.0)
            if w.sum() > 0:
                out[(dim, str(category))] = (int(n), (w / w.sum()).to_numpy())
    return out


def national(frame, pair):
    f = frame[frame.answer.isin(pair["options"])]
    w = f.groupby("answer").peso.sum().reindex(pair["options"]).fillna(0.0)
    return (w / w.sum()).to_numpy()


def counts():
    from .anchor import AnchorStore

    pairs = build_pairs()
    anchors = AnchorStore(CIS_DIR)
    per_dim = {d: 0 for d in DIMENSIONS}
    subgroups = 0
    for p in pairs:
        for (dim, _), _v in truth(anchors._frame(p["t_study"], p["t_id"]), p).items():
            per_dim[dim] += 1
            subgroups += 1
        anchors._frames.clear()
    return {"pares": len(pairs), "series": len({p["series"] for p in pairs}), "estudios_t": len({p["t_study"] for p in pairs}),
            "estudios_t_menos_1": len({p["p_study"] for p in pairs}), "subgrupos": subgroups, "subgrupos_por_dimension": per_dim,
            "fechas_t": [min(p["t_date"] for p in pairs), max(p["t_date"] for p in pairs)], "fechas_t_menos_1": [min(p["p_date"] for p in pairs), max(p["p_date"] for p in pairs)]}


def population_shares(population):
    w = population["peso"]
    return {d: (population.groupby(d)["peso"].sum() / w.sum()).to_dict() for d in DIMENSIONS}


def evaluate(pairs, anchors, population, log=print):
    """Una fila por subgrupo con los tres sistemas y el error de cada uno (puntos), más la información descriptiva por par."""
    from .analog_segments import group_estimates
    from .mrp import MRP

    shares = population_shares(population)
    rows, info = [], []
    started = time.time()
    for n, pair in enumerate(pairs):
        frame_t = anchors._frame(pair["t_study"], pair["t_id"])
        frame_p = anchors._frame(pair["p_study"], pair["p_id"])
        tr = truth(frame_t, pair)
        if not tr:
            continue
        options = pair["options"]
        goal = national(frame_t, pair)
        spec = {"study_id": pair["p_study"], "evidence_id": pair["p_id"], "mapping": pair["mapping"], "question_type": "single"}
        current = group_estimates([(1.0, spec)], options, shares, anchors, dict(zip(options, goal.tolist(), strict=True)), 0.5)
        back = {prev: option for option, prev in pair["mapping"].items()}
        train = frame_p.assign(answer=frame_p["answer"].map(back))
        model = MRP(population).fit(train, options)
        post = model.postratify(goal)
        for (dim, category), (size, observed) in tr.items():
            if category not in shares[dim] or dim not in current or category not in current[dim] or ((dim, category),) not in post:
                info.append({"series": pair["series"], "skipped": f"{dim}={category}"})
                continue
            cur = np.array([current[dim][category][o] for o in options])
            mrp = post[((dim, category),)]
            rows.append({"series": pair["series"], "t_id": pair["t_id"], "dim": dim, "category": category, "n": size, "n_options": len(options),
                         "err_current": 100 * float(np.abs(cur - observed).mean()), "err_mrp": 100 * float(np.abs(mrp - observed).mean()),
                         "err_none": 100 * float(np.abs(goal - observed).mean())})
        info.append({"series": pair["series"], "t_id": pair["t_id"], "omitted": list(model.omitted), "C": model.C, "used": list(model.used)})
        anchors._frames.pop((pair["t_study"], pair["t_id"]), None)
        anchors._frames.pop((pair["p_study"], pair["p_id"]), None)
        if (n + 1) % 20 == 0:
            log(f"{n + 1}/{len(pairs)} pares ({time.time() - started:.0f} s)")
    return pd.DataFrame(rows), info


def summarize(df, info, pairs):
    from scipy.stats import wilcoxon

    from .prior_v2.bigdev_eval import study_bootstrap

    def block(d):
        delta = (d.err_mrp - d.err_current).to_numpy()
        low, high = study_bootstrap(delta, d.series.to_numpy(), draws=DRAWS, seed=SEED)
        p = float(wilcoxon(d.err_mrp, d.err_current).pvalue) if len(d) >= 10 and np.any(delta != 0) else None
        return {"subgrupos": len(d), "series": int(d.series.nunique()), "actual": float(d.err_current.mean()), "mrp": float(d.err_mrp.mean()), "sin_diferencias": float(d.err_none.mean()),
                "delta": float(delta.mean()), "ic": [float(low), float(high)], "p": p, "mrp_mejor": int((delta < 0).sum()), "mrp_peor": int((delta > 0).sum())}

    total = block(df)
    by_dim = {d: block(g) for d, g in df.groupby("dim")}
    gain = -total["delta"]
    verdict = "CUMPLE" if gain >= MIN_GAIN_PP and total["ic"][1] < 0 else "NO CUMPLE"
    omitted = {d: sum(d in i.get("omitted", []) for i in info if "omitted" in i) for d in DIMENSIONS}
    c_values = pd.Series([i["C"] for i in info if i.get("C") is not None]).value_counts().sort_index().to_dict()
    return {"total": total, "por_dimension": by_dim, "veredicto": verdict, "mejora_pp": gain, "pares": len(pairs), "pares_evaluados": len({i["t_id"] for i in info if "t_id" in i}),
            "dimensiones_omitidas": omitted, "C_elegidos": {str(k): int(v) for k, v in c_values.items()}, "saltados": sum("skipped" in i for i in info)}


def fmt_p(p):
    return "—" if p is None else ("<0,001" if p < 0.001 else f"{p:.3f}")


def report_text(res):
    t = res["total"]
    lines = ["# MRP frente al método actual de subgrupos (evaluación prerregistrada)", "",
             f"{res['pares_evaluados']} pares ({res['pares']} en el recuento), {t['series']} series, {t['subgrupos']} subgrupos (con ≥ {MIN_SUBGROUP} encuestados en t). Error en puntos porcentuales por subgrupo; "
             f"IC 95 % remuestreando series ({DRAWS} réplicas, semilla {SEED}).", "",
             "| Sistema | MAE (pp) |", "|---|---|", f"| Sin diferencias (nacional para todos) | {t['sin_diferencias']:.2f} |", f"| Método actual | {t['actual']:.2f} |", f"| MRP | {t['mrp']:.2f} |", "",
             f"**MRP − método actual: {t['delta']:+.2f} pp, IC 95 % [{t['ic'][0]:+.2f}, {t['ic'][1]:+.2f}], Wilcoxon p = {fmt_p(t['p'])}; MRP mejor en {t['mrp_mejor']} y peor en {t['mrp_peor']} subgrupos.**", "",
             f"## Criterio (mejora ≥ {MIN_GAIN_PP} pp con el IC entero por debajo de 0): **{res['veredicto']}** (mejora de {res['mejora_pp']:+.2f} pp)", "",
             "## Por dimensión", "| Dimensión | Subgrupos | Sin diferencias | Actual | MRP | MRP − actual [IC 95 %] | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for d in DIMENSIONS:
        b = res["por_dimension"].get(d)
        if b:
            lines.append(f"| {d} | {b['subgrupos']} | {b['sin_diferencias']:.2f} | {b['actual']:.2f} | {b['mrp']:.2f} | {b['delta']:+.2f} [{b['ic'][0]:+.2f}, {b['ic'][1]:+.2f}] | {fmt_p(b['p'])} |")
    lines += ["", "## Descriptivo", f"- Dimensiones omitidas por oleada t−1 (menos del 50 % de valores válidos), en número de pares: {res['dimensiones_omitidas']}",
              f"- Valores de C elegidos por validación cruzada (número de pares): {res['C_elegidos']}", f"- Subgrupos saltados por no existir en la población o en el método: {res['saltados']}"]
    return "\n".join(lines)


def run(smoke=False):
    import subprocess

    from .anchor import AnchorStore

    if not smoke and LOCK.exists():
        sys.exit(f"La evaluación del MRP ya se ejecutó ({LOCK.read_text(encoding='utf-8').strip()}): no se repite.")
    window = ("2024-04-01", "2024-06-30") if smoke else T_WINDOW  # el modo --smoke usa pares FUERA del conjunto prerregistrado, solo para probar el código
    pairs = build_pairs(t_window=window)
    population = pd.read_parquet(Path(CIS_DIR) / "synthetic_population.parquet")
    anchors = AnchorStore(CIS_DIR)
    df, info = evaluate(pairs[:12] if smoke else pairs, anchors, population)
    res = summarize(df, info, pairs[:12] if smoke else pairs)
    text = report_text(res)
    out = OUT / ("smoke" if smoke else ".")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    df.to_csv(out / "subgroups.csv", index=False)
    if not smoke:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        LOCK.write_text(f"ejecutada {time.strftime('%Y-%m-%d %H:%M')}; commit {commit}", encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["counts", "run"])
    parser.add_argument("--smoke", action="store_true", help="prueba del código con pares FUERA del conjunto prerregistrado (no escribe el marcador)")
    args = parser.parse_args()
    if args.step == "counts":
        print(json.dumps(counts(), ensure_ascii=False, indent=1))
    else:
        run(args.smoke)
