"""Asistentes previos a la simulación: opciones sugeridas y alcance de la pregunta.

``suggest_options``: el usuario solo escribe la pregunta; las opciones salen, por este orden, de (a) la misma pregunta del
CIS, (b) una pregunta parecida, (c) una regla de pregunta cerrada, (d) el modelo de lenguaje o una escala estándar.
``assess_coverage``: clasifica si la pregunta se puede responder con las encuestas.
"""

import asyncio
import re

from .anchor import (
    _NON_SUBSTANTIVE,
    content_coverage,
    norm,
    option_key,
    prior_compatible,
    question_similarity,
    same_item,
)
from .schemas import OptionSuggestion, ScopeJudgement

DIRECT_SIMILARITY = 0.85
PRIOR_SIMILARITY = 0.55
ANALOG_SCALE_SIMILARITY = 0.40
PRIOR_GUARD = {"coverage": 0.6, "topic_match": True}

STANDARD_SCALES = {
    "acuerdo": ["Muy de acuerdo", "De acuerdo", "En desacuerdo", "Muy en desacuerdo"],
    "preocupacion": ["Mucho", "Bastante", "Poco", "Nada"],
    "valoracion": ["Muy buena", "Buena", "Regular", "Mala", "Muy mala"],
    "frecuencia": ["Nunca", "Casi nunca", "A veces", "Con frecuencia", "Siempre"],
    "confianza": ["Mucha confianza", "Bastante confianza", "Poca confianza", "Ninguna confianza"],
}
SCALE_NAMES = {
    "acuerdo": "acuerdo",
    "preocupacion": "preocupación",
    "valoracion": "valoración",
    "frecuencia": "frecuencia",
    "confianza": "confianza",
}
COVERAGE_LABELS = {
    "respaldada": "Respaldada por el CIS",
    "parecidas": "Basada en preguntas parecidas",
    "sin_datos": "Sin datos comparables",
    "fuera_de_alcance": "Fuera de alcance",
}

_INTERROGATIVES = re.compile(
    r"(cuánt\w+|cuál\w*|cómo|qué|quién\w*|dónde|cuándo|por qué|cuanto|cuanta|cuantos|cuantas|cual|cuales|quien|"
    r"donde|cuando|con que frecuencia|en que medida|hasta que punto)"
)
_CLOSED_START = re.compile(
    r"^(ha|han|has|he|hay|existe|es|esta|estan|tiene|tienen|usa|utiliza|suele|sabe|conoce|piensa|cree|considera|opina|"
    r"le parece|le gustaria|estaria|compraria|votaria|participo|vive|trabaja|fuma|bebe|lee|ve|escucha|dispone|usted ha|"
    r"ud ha|se ha|se siente|se encuentra|podria|deberia|deberian|puede|pueden)\b"
)
_SCALE_WORDS = re.compile(
    r"\b(de acuerdo|desacuerdo|preocup|valora|califica|satisfech|confia|confianza|frecuencia|importante|probable)"
)


def clean_options(options):
    """Opciones sustantivas de una pregunta del CIS: sin N.S./N.C./«No procede» ni la marca (NO LEER)."""
    out = []
    for option in options or []:
        key = option_key(option)
        if not key or key in _NON_SUBSTANTIVE or key.startswith("no procede") or key in {"nr", "no recuerda"}:
            continue
        label = re.sub(r"^\(NO LEER\)\s*", "", str(option), flags=re.I).strip()
        if label and label not in out:
            out.append(label)
    return out


def find_reference(resources, question, topic=None):
    """Pregunta del CIS más parecida (ignora las opciones, que aún no se conocen) y modo que le correspondería."""
    found = {}
    for hit in resources.evidence.search(
        question, limit=40, topic=topic, topic_filter=False, sources={"cis"}
    ):
        found.setdefault(hit["evidence"]["evidence_id"], hit)
    best = None
    for hit in found.values():
        e = hit["evidence"]
        original = e.get("metadata", {}).get("original_fields", {})
        qtype = original.get("question_type", e.get("metadata", {}).get("question_type"))
        text = e.get("question_text")
        if qtype != "single" or not e.get("study_id") or not text:
            continue
        if not same_item(question, text):
            continue
        options = clean_options(e.get("options"))
        if len(options) < 2:
            continue
        similarity = question_similarity(question, text)
        key = (similarity, e.get("fieldwork_date") or "")
        if best is None or key > best["key"]:
            best = {"key": key, "evidence": e, "similarity": similarity, "options": options}
    if best is None:
        return None
    e = best["evidence"]
    if best["similarity"] >= DIRECT_SIMILARITY:
        mode = "direct"
    elif best["similarity"] >= PRIOR_SIMILARITY and prior_compatible(question, topic, e, PRIOR_GUARD):
        mode = "prior"
    else:
        mode = "none"
    return {
        "mode": mode,
        "similarity": round(best["similarity"], 3),
        "options": best["options"],
        "evidence_id": e["evidence_id"],
        "study_id": str(e["study_id"]),
        "fieldwork_date": e.get("fieldwork_date"),
        "question": " ".join(str(e["question_text"]).split()),
        "coverage": round(content_coverage(question, e["question_text"]), 2),
    }


def closed_kind(question):
    """«a_favor» o «si_no» si la redacción es claramente una pregunta cerrada; None en otro caso."""
    plain = norm(question).strip(" ¿?")
    if "a favor" in plain and ("en contra" in plain or re.match(r"^(esta|estas)\b.*\ba favor\b", plain)):
        return "a_favor"
    if _INTERROGATIVES.search(question.lower()) or " o " in f" {plain} " or _SCALE_WORDS.search(plain):
        return None
    return "si_no" if _CLOSED_START.match(plain) else None


def scale_kind(question):
    plain = norm(question)
    if re.search(r"de acuerdo|desacuerdo", plain):
        return "acuerdo"
    if re.search(r"preocup|inquiet|miedo|temor", plain):
        return "preocupacion"
    if re.search(r"frecuencia|cuantas veces|a menudo", plain):
        return "frecuencia"
    if re.search(r"confia|confianza", plain):
        return "confianza"
    if re.search(r"valor|calific|satisf|funcion|nivel de|bueno|malo|calidad", plain):
        return "valoracion"
    return "acuerdo"


def _reference_info(ref):
    return (
        None
        if ref is None
        else {k: ref[k] for k in ("evidence_id", "study_id", "fieldwork_date", "similarity", "question")}
    )


def suggest_options(question, resources, service=None, topic=None, reference=None):
    """Opciones sugeridas + de dónde salen. Devuelve {options, source, explanation, reference, warnings}."""
    ref = reference if reference is not None else find_reference(resources, question, topic)
    info = _reference_info(ref)
    warnings = []

    def result(options, source, explanation, with_ref=False):
        return {
            "options": options, "source": source, "explanation": explanation,
            "reference": info if with_ref else None, "warnings": warnings,
        }

    if ref and ref["mode"] == "direct":
        return result(
            ref["options"], "cis_directo",
            f"Opciones oficiales de la misma pregunta del CIS (estudio {ref['study_id']}, {ref['evidence_id']}).", True,
        )
    if ref and ref["mode"] == "prior":
        return result(
            ref["options"], "cis_parecida",
            f"Escala de una pregunta parecida del CIS (estudio {ref['study_id']}, similitud {ref['similarity']:.0%}).",
            True,
        )
    closed = closed_kind(question)
    if closed == "a_favor":
        return result(["A favor", "En contra"], "cerrada", "Pregunta cerrada: a favor o en contra.")
    if closed == "si_no":
        return result(["Sí", "No"], "cerrada", "Pregunta cerrada de sí o no.")
    analog = ref if ref and ref["similarity"] >= ANALOG_SCALE_SIMILARITY and 3 <= len(ref["options"]) <= 6 else None
    if service is not None:
        try:
            payload = {
                "task": "suggest_options",
                "question": question,
                "reference_scales": [analog["options"]] if analog else [],
            }

            def check(value):
                keys = [option_key(o) for o in value.options]
                if len(set(keys)) != len(keys) or any(k in _NON_SUBSTANTIVE or not k for k in keys):
                    raise ValueError("Opciones repetidas o no sustantivas")

            value = asyncio.run(service.call(OptionSuggestion, payload, check))
            return result(
                [o.strip() for o in value.options], "modelo",
                "Propuesta del modelo de lenguaje, tomando como referencia escalas de preguntas parecidas del CIS.",
                analog is not None,
            )
        except Exception:  # el modelo no responde o devuelve opciones inválidas: se usa la escala sin modelo
            warnings.append("El modelo de lenguaje no pudo proponer opciones; se usa una escala del CIS.")
    if analog:
        return result(
            analog["options"], "escala_analoga",
            f"Escala de la pregunta más parecida del CIS (estudio {analog['study_id']}); revísala.", True,
        )
    kind = scale_kind(question)
    return result(
        list(STANDARD_SCALES[kind]), "escala_estandar",
        f"Escala estándar del CIS para preguntas de {SCALE_NAMES[kind]}; revísala.",
    )


# --- Alcance de la pregunta -------------------------------------------------------------------------------------

_COUNTRIES = (
    r"francia|alemania|italia|portugal|reino unido|inglaterra|estados unidos|eeuu|ee uu|mexico|argentina|chile|colombia|"
    r"peru|brasil|china|japon|rusia|marruecos|ucrania|venezuela|cuba|india|canada|australia"
)
_DEMONYMS = (
    r"franceses|alemanes|italianos|portugueses|britanicos|ingleses|estadounidenses|americanos|mexicanos|argentinos|"
    r"chilenos|colombianos|chinos|japoneses|rusos|marroquies|ucranianos|venezolanos|cubanos|indios|canadienses"
)
_RULES = [
    ("no_es_pregunta", r"^(dime|escribe|traduce|explica|calcula|resume|genera|haz|dame|cuenta|redacta)\b"),
    ("no_es_pregunta", r"^\s*[\d\s+\-*/x.=]+\??$"),
    ("persona_concreta", r"\b(mi|mis)\s+(vecin\w+|jef\w+|madre|padre|hij\w+|pareja|marid\w+|espos\w+|amig\w+|herman\w+|novi\w+|compan\w+|abuel\w+|tio|tia|primo|prima)\b"),
    ("prediccion", r"\b(ganara|ganaran|obtendra|obtendran|seran los resultados|cual sera el resultado|resultado de las proximas|"
                   r"que pasara|cuanto valdra|cuanto costara|en que ano (ocurrira|sera|habra))\b"),
    ("hechos", r"^(cuantos|cuantas)\b.*\b(habitantes|personas|votos|escanos|diputados|municipios|provincias|kilometros)\b"),
    ("hechos", r"^(cual es|cual fue|cuales son)\s+(la|el|los|las)\s+(capital|poblacion|superficie|pib|deuda|numero|fecha|altura|extension|distancia)\b"),
    ("hechos", r"^(cuando|donde)\s+(fue|es|esta|nacio|murio|se fundo|se firmo|empezo|termino)\b"),
    ("hechos", r"^(quien)\s+(es|fue|era|escribio|gano|invento|descubrio)\b"),
    ("hechos", r"^(cuanto)\s+(cuesta|mide|pesa|es|son|vale|dura|gana|cobra)\b"),
    ("hechos", r"^(que dia|que hora|que año|en que ano|en que año)\b"),
    ("poblacion_distinta", rf"\b(en|de)\s+({_COUNTRIES})\b"),
    ("poblacion_distinta", rf"\blos\s+({_DEMONYMS})\b"),
    ("poblacion_distinta", r"\b(los|las)\s+(ninos|ninas|menores|bebes|adolescentes|turistas|ancianos de mas de 100)\b"),
    ("poblacion_distinta", r"\b(menores de 18|de menos de 18|menores de edad)\b"),
]
_FUTURE_OPINION = re.compile(r"^(cree|piensa|considera|opina|espera|preve)\b")
_REFORMULATIONS = {
    "hechos": "Pregunta por una actitud o valoración, por ejemplo «¿Cree que hay demasiada gente en las grandes ciudades?».",
    "prediccion": "Pregunta por una expectativa u opinión, por ejemplo «¿Cree que …? Muy probable, probable, poco o nada probable».",
    "persona_concreta": "Pregunta por una actitud general de la población, por ejemplo «¿Con qué frecuencia se relaciona "
    "con sus vecinos?».",
    "poblacion_distinta": "Pregunta por la población adulta residente en España, por ejemplo «¿Qué opinan los españoles de …?».",
    "no_es_pregunta": "Escribe una pregunta de encuesta sobre una opinión o actitud, por ejemplo «¿Está de acuerdo con …?».",
}
_SCOPE_TEXT = {
    "hechos": "pide un dato objetivo, no una opinión ni una actitud de la población",
    "prediccion": "pide predecir un suceso futuro concreto, no recoger una opinión",
    "persona_concreta": "se refiere a una persona concreta, no a la población",
    "poblacion_distinta": "se refiere a una población distinta de los adultos residentes en España",
    "no_es_pregunta": "no es una pregunta de encuesta",
}


def scope_check(question, service=None):
    """Reglas y, si hay modelo, su juicio: {in_scope, kind, reason, reformulation}."""
    plain = norm(question).strip(" ¿?¡!.")
    kind = None
    if not _FUTURE_OPINION.match(plain):
        for name, pattern in _RULES:
            if re.search(pattern, plain):
                kind = name
                break
    reformulation = _REFORMULATIONS.get(kind, "")
    if kind is None and service is not None:
        try:
            value = asyncio.run(service.call(ScopeJudgement, {"task": "judge_scope", "question": question}))
            if value.category != "opinion":
                kind, reformulation = value.category, value.reformulation or _REFORMULATIONS.get(value.category, "")
        except Exception:  # sin respuesta útil del modelo: solo reglas
            pass
    if kind is None:
        return {"in_scope": True, "kind": None, "reason": "", "reformulation": ""}
    return {
        "in_scope": False, "kind": kind,
        "reason": f"La pregunta {_SCOPE_TEXT.get(kind, 'no es una pregunta de opinión')}.",
        "reformulation": reformulation,
    }


def coverage_card(mode, reference, scope=None):
    """Tarjeta de cobertura a partir del modo de la predicción (o del modo previsto) y la pregunta del CIS usada."""
    if scope and not scope["in_scope"]:
        return {
            "category": "fuera_de_alcance", "label": COVERAGE_LABELS["fuera_de_alcance"],
            "explanation": f"{scope['reason']} Las encuestas del CIS miden opiniones y actitudes de la población adulta de España.",
            "similarity": None, "reference": None, "reformulation": scope["reformulation"],
        }
    where = ""
    if reference:
        where = f" (estudio {reference['study_id']}, {reference['evidence_id']})"
    if mode == "direct" and reference:
        return {
            "category": "respaldada", "label": COVERAGE_LABELS["respaldada"],
            "explanation": f"Esta pregunta existe en el CIS{where}: la estimación parte de su resultado real.",
            "similarity": reference.get("similarity"), "reference": reference, "reformulation": "",
        }
    if mode == "prior" and reference:
        similarity = reference.get("similarity")
        shown = f"{similarity:.0%}" if similarity else "alta"
        return {
            "category": "parecidas", "label": COVERAGE_LABELS["parecidas"],
            "explanation": f"Hay una pregunta parecida en el CIS{where}, con similitud {shown}; añade incertidumbre.",
            "similarity": similarity, "reference": reference, "reformulation": "",
        }
    return {
        "category": "sin_datos", "label": COVERAGE_LABELS["sin_datos"],
        "explanation": "El CIS no tiene preguntas comparables: es una estimación de confianza baja.",
        "similarity": reference.get("similarity") if reference else None,
        "reference": reference, "reformulation": "",
    }


def assess_coverage(question, resources, service=None, topic=None, reference=None):
    """Clasifica si la pregunta se puede responder con las encuestas (antes de simular)."""
    ref = reference if reference is not None else find_reference(resources, question, topic)
    if ref and ref["mode"] == "direct":
        return coverage_card("direct", _reference_info(ref))
    scope = scope_check(question, service)
    return coverage_card(ref["mode"] if ref else "none", _reference_info(ref), scope)
