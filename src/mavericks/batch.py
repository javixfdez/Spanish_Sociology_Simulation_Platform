"""Procesado por lotes y exportación de resultados (CSV/Excel/JSON) para la interfaz."""

import io
import json

import pandas as pd

from .service import ServiceError, predict

COLUMNS = {"pregunta": "pregunta", "opciones": "opciones", "tipo": "tipo"}
TYPES = {"unica": "single", "única": "single", "single": "single", "multiple": "multiple", "múltiple": "multiple"}
TEMPLATE = pd.DataFrame(
    {
        "pregunta": ["¿Le preocupa que la inteligencia artificial le quite su empleo?", "¿Está a favor de limitar el alquiler?"],
        "opciones": ["Mucho|Bastante|Poco|Nada", "Sí|No"],
        "tipo": ["unica", "unica"],
    }
)


def read_table(file, name):
    """Lee un CSV o Excel subido; devuelve las filas con pregunta, opciones (lista) y tipo."""
    try:
        frame = pd.read_excel(file) if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(file, sep=None, engine="python")
    except Exception as exc:
        raise ServiceError("No se pudo leer el archivo: usa un CSV o un Excel (.xlsx) con las columnas indicadas.") from exc
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in ("pregunta", "opciones") if c not in frame]
    if missing:
        raise ServiceError(f"Faltan columnas: {', '.join(missing)}. Se esperan: pregunta, opciones (separadas por |) y tipo.")
    rows = []
    for position, record in enumerate(frame.to_dict("records"), start=2):
        question = str(record.get("pregunta", "")).strip()
        options = [o.strip() for o in str(record.get("opciones", "")).split("|") if o.strip()]
        kind = TYPES.get(str(record.get("tipo", "unica")).strip().lower(), "single")
        rows.append({"row": position, "question": question, "options": options, "question_type": kind})
    if not rows:
        raise ServiceError("El archivo no tiene filas.")
    return rows


def run_batch(rows, preset, resources, service=None, on_progress=None):
    """Una predicción por fila; los errores de una fila no detienen el lote."""
    results = []
    for i, row in enumerate(rows):
        try:
            result = predict(
                row["question"], row["options"], row["question_type"], preset, resources=resources, service=service
            )
            results.append({"row": row["row"], "ok": True, "result": result})
        except ServiceError as exc:
            results.append({"row": row["row"], "ok": False, "error": str(exc), "question": row["question"]})
        if on_progress:
            on_progress(i + 1, len(rows))
    return results


def national_frame(result):
    node = result["national"]
    rows = []
    for option, value in node["distribution"].items():
        low, high = (node["interval"] or {}).get(option, (None, None))
        rows.append({"opcion": option, "porcentaje": value * 100, "ic90_min": _pct(low), "ic90_max": _pct(high)})
    return pd.DataFrame(rows)


def _pct(value):
    return None if value is None else value * 100


def subgroup_frame(result, dimension=None):
    rows = []
    for key, dim in result["subgroups"].items():
        if dimension and key != dimension:
            continue
        for group, info in dim["groups"].items():
            for option, value in info["distribution"].items():
                low, high = (info["interval"] or {}).get(option, (None, None))
                rows.append(
                    {
                        "dimension": dim["label"], "grupo": group, "opcion": option, "porcentaje": value * 100,
                        "ic90_min": _pct(low), "ic90_max": _pct(high), "peso_poblacion": info["population_share"] * 100,
                    }
                )
    return pd.DataFrame(rows)


def evidence_frame(result):
    return pd.DataFrame(
        [
            {
                "id": e["evidence_id"], "estudio": e["study_id"], "fecha": e["fieldwork_end"], "similitud": e["similarity"],
                "enunciado": e["question"],
                "distribucion_real": json.dumps(e["distribution"], ensure_ascii=False) if e["distribution"] else None,
            }
            for e in result["evidence"]
        ]
    )


def to_csv(result):
    nat = national_frame(result).assign(dimension="Total", grupo="Total")
    sub = subgroup_frame(result)
    return pd.concat([nat, sub], ignore_index=True).to_csv(index=False).encode("utf-8-sig")


def to_json(result):
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")


def to_excel(results):
    """Libro con un resumen por pregunta y, por cada pregunta, sus hojas de total, subgrupos y evidencia."""
    buffer = io.BytesIO()
    summary = []
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for item in results:
            if not item["ok"]:
                summary.append({"fila": item["row"], "pregunta": item["question"], "estado": item["error"]})
                continue
            r = item["result"]
            summary.append(
                {
                    "fila": item["row"], "pregunta": r["question"], "estado": "ok", "tema": r["topic"], "modo": r["mode_label"],
                    "confianza": r["confidence"]["level"], "error_tipico": r["confidence"]["typical_error"],
                    "opcion_mas_probable": max(r["national"]["distribution"], key=r["national"]["distribution"].get),
                }
            )
        pd.DataFrame(summary).to_excel(writer, sheet_name="Resumen", index=False)
        for item in results:
            if not item["ok"]:
                continue
            tag = f"P{item['row']}"
            national_frame(item["result"]).to_excel(writer, sheet_name=f"{tag} total", index=False)
            subgroup_frame(item["result"]).to_excel(writer, sheet_name=f"{tag} subgrupos", index=False)
            evidence_frame(item["result"]).to_excel(writer, sheet_name=f"{tag} evidencia", index=False)
    return buffer.getvalue()
