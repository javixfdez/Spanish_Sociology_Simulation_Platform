"""Taxonomía cerrada de temas, alineada con las carpetas de maverick_data/raw."""

import re
import unicodedata
from enum import Enum


class Topic(str, Enum):
    GENERAL = "general"
    VIVIENDA = "vivienda"
    SALUD = "salud"
    IA_TECNOLOGIA = "ia_tecnologia"
    CONSUMO = "consumo"
    MEDIO_AMBIENTE = "medio_ambiente"
    SOLEDAD = "soledad"


TOPICS = [t.value for t in Topic]

_KEYWORDS = {
    "vivienda": "vivienda,alquiler,hipoteca,inmobiliari,desahucio,okupa,piso,casa,propiedad,inquilino",
    "salud": "salud,sanidad,sanitari,medic,hospital,enfermedad,vacuna,mental,depresion,ansiedad,dieta,ejercicio,"
    "paciente,tratamiento",
    "ia_tecnologia": "inteligencia artificial,chatgpt,tecnolog,internet,digital,ordenador,movil,algoritmo,"
    "aplicacion,app,robot,redes sociales,ciberseguridad",
    "consumo": "consumo,consumidor,compra,precio,inflacion,ahorro,gasto,tienda,marca,pagar,cesta,cuesta",
    "medio_ambiente": "medio ambiente,clima,climatic,contamina,reciclaj,renovable,sostenib,emision,coche electrico,"
    "residuos,plastico",
    "soledad": "soledad,aislamiento,solo,sola,companhia,compania,amistad,apoyo social,relaciones personales,vejez,"
    "cuidados",
}
_ALIASES = {
    "tecnologia": "ia_tecnologia",
    "ia": "ia_tecnologia",
    "inteligencia artificial": "ia_tecnologia",
    "tecnologia e inteligencia artificial": "ia_tecnologia",
    "medioambiente": "medio_ambiente",
    "medio ambiente": "medio_ambiente",
    "housing": "vivienda",
    "health": "salud",
    "loneliness": "soledad",
    "consumption": "consumo",
    "environment": "medio_ambiente",
    "technology": "ia_tecnologia",
}


def _plain(text):
    text = unicodedata.normalize("NFKD", str(text or "")).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9ñ]+", " ", text).strip()


def classify_topic(text):
    """Tema por palabras clave; 'general' si no hay coincidencias claras."""
    plain = " " + _plain(text) + " "
    best, best_score = Topic.GENERAL.value, 0
    for topic, words in _KEYWORDS.items():
        score = sum(1 for w in words.split(",") if (" " + w) in plain)
        if score > best_score:
            best, best_score = topic, score
    return best


def normalize_topic(value, fallback_text=None):
    """Texto libre -> tema del enum, o None si no se reconoce y no hay texto de apoyo.

    Con ``fallback_text`` se clasifica por palabras clave cuando el valor no coincide.
    """
    plain = _plain(value)
    if plain.replace(" ", "_") in TOPICS:
        return plain.replace(" ", "_")
    if plain in _ALIASES:
        return _ALIASES[plain]
    if plain:
        guessed = classify_topic(plain)
        if guessed != Topic.GENERAL.value:
            return guessed
    if fallback_text:
        return classify_topic(fallback_text)
    return None
