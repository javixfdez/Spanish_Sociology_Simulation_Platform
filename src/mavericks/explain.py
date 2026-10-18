"""Explicación escrita del resultado: plantilla (siempre) y redacción del LLM con comprobación de cifras.

La plantilla no usa ningún modelo. El LLM recibe SOLO las cifras ya calculadas, las diferencias por subgrupo que superan el error
típico y las preguntas del CIS de apoyo; su texto se acepta únicamente si todas las cifras con unidad (%, puntos) coinciden con las
calculadas (tolerancia de 1 punto) y no menciona grupos que no estén en los datos. Si no, si falla o tarda más del límite, se usa la
plantilla y se registra el motivo.
"""

import asyncio
import logging
import re
import time

from .schemas import ResultNarrative

log = logging.getLogger("mavericks.explain")
TOLERANCE = 1.0  # puntos
MAX_DIFFERENCES = 2

_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|por ciento|puntos porcentuales|puntos|pp)", re.I)
_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|y|a)\s*(\d+(?:[.,]\d+)?)\s*(?:%|por ciento)", re.I)


def _f(text):
    return float(text.replace(",", "."))


def majority(result):
    dist = result["national"]["distribution"]
    option = max(dist, key=dist.get)
    core = (result["national"].get("interval_core") or {}).get(option)
    return option, 100 * dist[option], (None if core is None else (100 * core[0], 100 * core[1]))


def subgroup_differences(result, top=MAX_DIFFERENCES):
    """Diferencias (en puntos) de la opción mayoritaria por subgrupo que superan el error típico, de mayor a menor."""
    option, national, _ = majority(result)
    threshold = 100 * (result["confidence"].get("typical_error") or 0.0)
    found = []
    for dim in result["subgroups"].values():
        for category, group in dim["groups"].items():
            diff = 100 * group["distribution"][option] - national
            if abs(diff) > threshold and abs(diff) >= 1.0:
                found.append({"dimension": dim["label"], "category": category, "option": option,
                              "percent": 100 * group["distribution"][option], "difference_points": diff})
    found.sort(key=lambda d: -abs(d["difference_points"]))
    return found[:top]


def support_sentence(result):
    mode, used = result["mode"], result.get("prior_usado") or {}
    first = (result.get("evidence") or [None])[0]
    if mode == "direct" and first:
        return f"Se apoya en la propia pregunta del CIS (estudio {first['study_id']}, {first['fieldwork_end'] or 's/f'}), con su resultado real más reciente."
    where = f"la más parecida es del estudio {first['study_id']}" if first else "no hay ninguna muy parecida"
    base = "preguntas parecidas del CIS" + (" y un modelo aprendido de todas las preguntas anteriores" if used.get("metodo") == "media_knn_mlp" else "")
    if mode == "prior":
        sim = f", similitud {100 * first['similarity']:.0f} %" if first and first.get("similarity") else ""
        return f"Es una pregunta parecida a otra del CIS ({where}{sim}); la estimación se apoya en {base}."
    return f"Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en {base} y tiene confianza baja."


def template_text(result):
    option, percent, core = majority(result)
    rango = f"; rango habitual {core[0]:.0f}–{core[1]:.0f} %" if core else ""
    sentences = [f"La opción más probable es «{option}», con un {percent:.0f} %{rango}."]
    diffs = subgroup_differences(result)
    if diffs:
        parts = [f"{d['dimension'].lower()} {d['category']} ({d['percent']:.0f} %, {d['difference_points']:+.0f} puntos)" for d in diffs]
        origin = "" if result["mode"] == "direct" else " (diferencias estimadas con preguntas parecidas)"
        sentences.append("Las mayores diferencias entre grupos: " + "; ".join(parts) + origin + ".")
    sentences.append(support_sentence(result))
    return " ".join(sentences)


def llm_payload(result):
    option, _, _ = majority(result)
    dist = result["national"]["distribution"]
    core = result["national"].get("interval_core") or {}
    return {
        "question": result["question"],
        "national_percent": {o: round(100 * v, 1) for o, v in dist.items()},
        "usual_range_percent": {o: [round(100 * core[o][0], 1), round(100 * core[o][1], 1)] for o in core},
        "group_differences": [{**d, "percent": round(d["percent"], 1), "difference_points": round(d["difference_points"], 1)}
                              for d in subgroup_differences(result, top=4)],
        "support": {
            "mode": result["mode_label"], "prior_method": (result.get("prior_usado") or {}).get("metodo"),
            "cis_questions": [{"study": e["study_id"], "date": e["fieldwork_end"], "text": (e["question"] or "")[:160],
                               "similarity_percent": None if not e.get("similarity") else round(100 * e["similarity"], 1)}
                              for e in (result.get("evidence") or [])[:3]],
        },
        "typical_error_points": None if result["confidence"].get("typical_error") is None else round(100 * result["confidence"]["typical_error"], 1),
    }


def allowed_numbers(payload):
    values = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(abs(float(x)))

    walk({k: v for k, v in payload.items() if k != "question"})
    return values


def check_text(text, payload):
    """None si el texto es válido; si no, el motivo. Comprueba cifras con unidad y grupos mencionados."""
    allowed = allowed_numbers(payload)
    quoted = [_f(m.group(1)) for m in _NUMBER.finditer(text)]
    for m in _RANGE.finditer(text):
        quoted += [_f(m.group(1)), _f(m.group(2))]
    for number in quoted:
        if not any(abs(number - a) <= TOLERANCE for a in allowed):
            return f"cifra que no cuadra con los datos: {number:g}"
    mentioned = {d["category"].lower() for d in payload["group_differences"]}
    labels = payload.get("_all_categories", [])
    lowered = text.lower()
    for label in labels:
        if label.lower() not in mentioned and len(label) >= 3 and re.search(rf"(?<!\w){re.escape(label.lower())}s?(?!\w)", lowered):
            return f"menciona un grupo sin diferencia en los datos: {label}"
    return None


def all_categories(result):
    return sorted({c for dim in result["subgroups"].values() for c in dim["groups"]})


async def _ask(service, payload, timeout):
    value = await asyncio.wait_for(service.call(ResultNarrative, {k: v for k, v in payload.items() if not k.startswith("_")}), timeout)
    return value.summary.strip()


def add_template(result):
    result["explanation_text"] = {"texto": template_text(result), "origen": "plantilla", "motivo": None}
    return result


def add_llm_text(result, service, timeout=20.0):
    """Sustituye la plantilla por el texto del LLM si es válido; si no, deja la plantilla y anota el motivo."""
    template = result.get("explanation_text") or add_template(result)["explanation_text"]
    payload = llm_payload(result)
    payload["_all_categories"] = all_categories(result)
    started = time.monotonic()
    reason = None
    try:
        text = asyncio.run(_ask(service, payload, timeout))
        reason = check_text(text, payload)
        if reason is None:
            result["explanation_text"] = {"texto": text, "origen": "llm", "motivo": None, "segundos": time.monotonic() - started}
            return result
    except (TimeoutError, asyncio.TimeoutError):
        reason = f"el modelo tardó más de {timeout:g} s"
    except Exception as exc:  # el LLM no responde o la salida no es válida
        reason = f"el modelo falló ({type(exc).__name__})"
    log.warning("Explicación del LLM descartada, se usa la plantilla: %s", reason)
    result["explanation_text"] = {**template, "origen": "plantilla", "motivo": reason, "segundos": time.monotonic() - started}
    return result
