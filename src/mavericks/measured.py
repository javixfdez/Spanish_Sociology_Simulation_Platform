"""«Lo que midió el CIS»: reparto ponderado de los microdatos de la oleada de la pregunta, sin ningún modelo.

Para el total y para cada subgrupo o cruce pedido: reparto ponderado con los pesos del CIS sobre las respuestas sustantivas (sin N.S./N.C.), número de encuestados ``n``, n efectivo de Kish
``n_eff = (Σw)² / Σw²`` y margen de error muestral al 95 %: ``±1,96·√(p(1−p)/n_eff)`` (en puntos). Con ``n < 50`` el dato se marca como poco fiable.
"""

import numpy as np

from .mrp_service import anchor_frame, cross_label

Z = 1.96
SMALL = 50
NO_MEASURED = "No hay dato medido: esta pregunta no se ha hecho en el CIS; lo que se muestra es una estimación."
SMALL_NOTE = "Muestra pequeña: el dato medido es poco fiable; mira la estimación del modelo."


REFERENCE_DATE = "2026-06-30"  # fecha de corte de los datos del reto
MAX_AGE_MONTHS = 12
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def age_warning(fieldwork_end, reference=REFERENCE_DATE):
    """Aviso si el trabajo de campo tiene más de 12 meses respecto al 30/06/2026: «Dato de <mes año>: la opinión puede haber cambiado desde entonces»."""
    if not fieldwork_end or len(str(fieldwork_end)) < 7:
        return None
    year, month = int(str(fieldwork_end)[:4]), int(str(fieldwork_end)[5:7])
    ref_year, ref_month = int(reference[:4]), int(reference[5:7])
    if (ref_year - year) * 12 + (ref_month - month) > MAX_AGE_MONTHS:
        return f"Dato de {MONTHS[month - 1]} de {year}: la opinión puede haber cambiado desde entonces."
    return None


def tabulate(frame, options):
    """(reparto, margen, n, n_eff) de un conjunto de respuestas: una fila por encuestado, ``answer`` ya expresada con las opciones y ``peso``."""
    f = frame[frame["answer"].isin(options)].drop_duplicates("respondent_uid")
    n = int(len(f))
    if n == 0 or f["peso"].sum() <= 0:
        return None
    w = f["peso"].to_numpy(float)
    n_eff = float(w.sum() ** 2 / (w**2).sum())
    sums = f.groupby("answer")["peso"].sum().reindex(options).fillna(0.0).to_numpy(float)
    p = sums / sums.sum()
    margin = Z * np.sqrt(p * (1 - p) / n_eff)
    return p, margin, n, n_eff


def _entry(p, margin, n, n_eff, options):
    return {
        "distribution": dict(zip(options, (float(x) for x in p), strict=True)),
        "margin": dict(zip(options, (float(x) for x in margin), strict=True)),
        "interval": {o: [float(max(0.0, x - m)), float(min(1.0, x + m))] for o, x, m in zip(options, p, margin, strict=True)},
        "n": n, "n_eff": n_eff, "small": n < SMALL,
    }


def block(resources, spec, options, study_date, crosses=()):
    """Bloque medido de la oleada de la pregunta: total + los cruces pedidos (tuplas de (variable, categoría))."""
    frame = anchor_frame(resources, spec, options)
    total = tabulate(frame, options)
    if total is None:
        raise ValueError("la oleada de la pregunta no tiene respuestas sustantivas")
    out = {"study_id": str(spec["study_id"]), "evidence_id": spec["evidence_id"], "fieldwork_end": study_date, "national": _entry(*total, options), "crosses": [], "note": None,
           "age_warning": age_warning(study_date)}
    for cross in crosses:
        sub = frame
        for var, category in cross:
            if var not in sub.columns:
                sub = sub.iloc[0:0]
                break
            sub = sub[sub[var].astype(str) == category]
        got = tabulate(sub, options)
        item = {"cross": [{"var": v, "val": c} for v, c in cross], "label": cross_label(cross)}
        if got is None:
            item.update(distribution=None, margin=None, interval=None, n=0, n_eff=0.0, small=True)
        else:
            item.update(_entry(*got, options))
        out["crosses"].append(item)
    return out
