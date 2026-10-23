"""Comparador de formulaciones: la misma pregunta (o mensaje) escrita de 2 a 10 maneras, con las mismas opciones.

No hay cálculo propio: cada formulación pasa por ``service.predict`` en modo rápido y aquí solo se leen sus resultados
(reparto nacional, rangos, confianza, subgrupos) para ordenarlos por la opción elegida.
"""

import io

import pandas as pd

from . import service
from .assist import display_label

MIN_FORMULATIONS, MAX_FORMULATIONS = 2, 10
SUPPORT_DIMENSIONS = ("grupo_edad", "ccaa", "nivel_estudios")  # el grupo con menos apoyo se busca en edad, comunidad autónoma y estudios
COLUMNS = [
    "Formulación", "Opción", "% estimado", "Rango habitual mín. (%)", "Rango habitual máx. (%)", "Rango 90 % mín. (%)", "Rango 90 % máx. (%)",
    "Confianza", "Modo", "Grupo con menos apoyo", "% en ese grupo", "Estado",
]


def parse_lines(text):
    """Líneas no vacías, sin espacios sobrantes y sin repetidas (se conserva el orden)."""
    seen, out = set(), []
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
    return out


def least_support(result, option):
    """(«edad: 18-24», porcentaje) del grupo de edad, comunidad o estudios donde la opción elegida tiene menos apoyo; None si no hay subgrupos."""
    worst = None
    for dimension in SUPPORT_DIMENSIONS:
        block = (result.get("subgroups") or {}).get(dimension)
        for name, group in (block or {}).get("groups", {}).items():
            share = group["distribution"].get(option)
            if share is not None and (worst is None or share < worst[1]):
                worst = (f"{block['label']}: {name}", share)
    return worst


def _range(interval, option):
    pair = (interval or {}).get(option)
    return (None, None) if pair is None else (round(100 * pair[0], 1), round(100 * pair[1], 1))


def row_from_result(formulation, result, option):
    national = result["national"]
    support = least_support(result, option)
    core_lo, core_hi = _range(national.get("interval_core"), option)
    wide_lo, wide_hi = _range(national.get("interval"), option)
    return {
        "Formulación": formulation, "Opción": display_label(option), "% estimado": round(100 * national["distribution"][option], 1),
        "Rango habitual mín. (%)": core_lo, "Rango habitual máx. (%)": core_hi, "Rango 90 % mín. (%)": wide_lo, "Rango 90 % máx. (%)": wide_hi,
        "Confianza": result["confidence"]["level"], "Modo": result["mode_label"],
        "Grupo con menos apoyo": support[0] if support else None, "% en ese grupo": round(100 * support[1], 1) if support else None, "Estado": "ok",
    }


def error_row(formulation, option, message):
    return {**dict.fromkeys(COLUMNS), "Formulación": formulation, "Opción": display_label(option), "Estado": message}


def compare(formulations, options, option, resources=None, predict=None, progress=None):
    """Una fila por formulación, en el orden escrito. ``predict`` solo se cambia en los tests."""
    if not MIN_FORMULATIONS <= len(formulations) <= MAX_FORMULATIONS:
        raise ValueError(f"Escribe entre {MIN_FORMULATIONS} y {MAX_FORMULATIONS} formulaciones (una por línea).")
    if option not in options:
        raise ValueError("La opción por la que ordenar no está entre las opciones.")
    predict = predict or service.predict
    rows = []
    for n, text in enumerate(formulations, start=1):
        try:
            result = predict(text, options, "single", "rapido_sin_llm", resources=resources)
            rows.append(row_from_result(text, result, option))
        except service.OutOfScopeError as exc:
            rows.append(error_row(text, option, f"Fuera de alcance: {exc.coverage['explanation']}"))
        except service.ServiceError as exc:
            rows.append(error_row(text, option, str(exc)))
        if progress:
            progress(n, len(formulations))
    return rows


def to_frame(rows):
    """Ordenadas de mayor a menor % de la opción elegida; las que fallaron, al final."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["_fallo"] = frame["Estado"] != "ok"
    frame = frame.sort_values(["_fallo", "% estimado"], ascending=[True, False], kind="stable").drop(columns="_fallo")
    return frame.reset_index(drop=True)


def to_csv(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")
