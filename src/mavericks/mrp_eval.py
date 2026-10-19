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


def build_pairs(repo=None, table=None):
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
            if not T_WINDOW[0] <= date <= T_WINDOW[1]:
                continue
            keep = _substantive(list(e.options or []), table[e.evidence_id]["dist"])
            if not keep:
                continue
            earlier = [p for p in items if PREVIOUS_FROM <= p.fieldwork_date[:10] < date and str(p.study_id) != str(e.study_id)]
            for p in reversed(earlier):  # la más reciente primero
                mapping = option_mapping(keep, list(p.options or []))
                if mapping is None or question_similarity(e.question_text, p.question_text) < MIN_SIMILARITY or not same_item(e.question_text, p.question_text):
                    continue
                pairs.append({"series": key, "t_id": e.evidence_id, "t_study": str(e.study_id), "t_date": date, "p_id": p.evidence_id, "p_study": str(p.study_id),
                              "p_date": p.fieldwork_date[:10], "options": keep, "mapping": mapping, "question": " ".join(e.question_text.split())})
                break
    assert_no_reserved([x for p in pairs for x in (p["t_id"], p["p_id"])], "pares del MRP", reserved_ids(CIS_DIR))
    assert all(p["t_date"] <= T_WINDOW[1] and p["p_date"] < p["t_date"] for p in pairs)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["counts", "run"])
    args = parser.parse_args()
    if args.step == "counts":
        print(json.dumps(counts(), ensure_ascii=False, indent=1))
    else:
        sys.exit("la evaluación se ejecuta tras el preregistro")
