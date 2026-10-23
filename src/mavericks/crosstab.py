"""Cruce de dos preguntas del mismo estudio del CIS: tabla conjunta ponderada MEDIDA con los microdatos (sin modelo ni prior).

Filas: las opciones de la pregunta del resultado (ya expresadas como las escribió el usuario); columnas: las opciones de la otra pregunta, tal como
las recoge el CIS. Cuenta cada entrevistado una vez, con su peso, y solo si respondió de forma sustantiva a las dos preguntas.
"""

import pandas as pd

from .assist import clean_options, display_label
from .evidence import source_family
from .measured import SMALL
from .mrp_service import anchor_frame

MAX_OPTIONS = 12  # más columnas no caben en una tabla legible


def companions(resources, study_id, evidence_id):
    """Preguntas de respuesta única del mismo estudio del CIS con 2 a 12 opciones sustantivas (menos la propia): [{evidence_id, question, options}]."""
    out = []
    for e in resources.evidence.all():
        if source_family(e) != "cis" or str(e.study_id) != str(study_id) or e.evidence_id == evidence_id or not e.question_text:
            continue
        if e.metadata.get("original_fields", {}).get("question_type", "single") not in (None, "single"):
            continue
        options = clean_options(e.options)
        if 2 <= len(options) <= MAX_OPTIONS:
            out.append({"evidence_id": e.evidence_id, "question": " ".join(e.question_text.split()), "options": options})
    return sorted(out, key=lambda item: item["evidence_id"])


def joint(resources, anchor, options, other):
    """Tabla conjunta de la pregunta del resultado (``anchor`` + ``options``) con la pregunta ``other`` (elemento de ``companions``)."""
    frame_a = anchor_frame(resources, {**anchor, "question_type": "single"}, options)
    frame_b = resources.anchors._frame(anchor["study_id"], other["evidence_id"])
    a = frame_a[frame_a["answer"].isin(options)].drop_duplicates("respondent_uid")[["respondent_uid", "peso", "answer"]]
    b = frame_b[frame_b["answer"].isin(other["options"])].drop_duplicates("respondent_uid")[["respondent_uid", "answer"]]
    both = a.merge(b, on="respondent_uid", suffixes=("_a", "_b"))
    cells = [display_label(o) for o in other["options"]]
    rows = [display_label(o) for o in options]
    if both.empty or both["peso"].sum() <= 0:
        return {"n": 0, "small": True, "rows": rows, "columns": cells, "total": None, "by_row": None, "counts": None, "small_rows": rows, "n_eff": 0.0}
    weights = both.pivot_table(index="answer_a", columns="answer_b", values="peso", aggfunc="sum", fill_value=0.0).reindex(index=options, columns=other["options"], fill_value=0.0)
    counts = both.pivot_table(index="answer_a", columns="answer_b", values="respondent_uid", aggfunc="count", fill_value=0).reindex(index=options, columns=other["options"], fill_value=0)
    weights.index, weights.columns, counts.index, counts.columns = rows, cells, rows, cells
    total = 100 * weights / weights.to_numpy().sum()
    row_sums = weights.sum(axis=1)
    by_row = 100 * weights.div(row_sums.where(row_sums > 0), axis=0)
    w = both["peso"].to_numpy(float)
    row_n = counts.sum(axis=1)
    return {
        "n": int(len(both)), "small": len(both) < SMALL, "rows": rows, "columns": cells, "total": total, "by_row": by_row, "counts": counts,
        "small_rows": [r for r in rows if row_n[r] < SMALL], "n_eff": float(w.sum() ** 2 / (w**2).sum()),
    }


def to_frame_with_n(table, counts):
    """Tabla en % con el n entre paréntesis en cada celda (para mostrar y descargar)."""
    return pd.DataFrame({c: [f"{table.loc[r, c]:.1f} % (n={int(counts.loc[r, c])})" for r in table.index] for c in table.columns}, index=table.index)
