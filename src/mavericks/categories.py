"""Esquema común de categorías (el del CIS) compartido por las ETL de CIS y población."""

import html
import re
import unicodedata

import pandas as pd

SEXO = ["hombre", "mujer"]
GRUPOS_EDAD = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
_EDAD_BINS = [18, 25, 35, 45, 55, 65, 75, 200]

NIVEL_ESTUDIOS = ["sin_estudios", "primaria", "secundaria_1", "secundaria_2", "fp", "superiores"]
NIVEL_ESTUDIOS_EXTRA = ["otros", "nc"]

CCAA = [
    "Andalucía",
    "Aragón",
    "Asturias (Principado de)",
    "Balears (Illes)",
    "Canarias",
    "Cantabria",
    "Castilla y León",
    "Castilla-La Mancha",
    "Cataluña",
    "Comunitat Valenciana",
    "Extremadura",
    "Galicia",
    "Madrid (Comunidad de)",
    "Murcia (Región de)",
    "Navarra (Comunidad Foral de)",
    "País Vasco",
    "Rioja (La)",
    "Ceuta (Ciudad Autónoma de)",
    "Melilla (Ciudad Autónoma de)",
]

TAMUNI = [
    "Menos o igual a 2.000 habitantes",
    "2.001 a 10.000 habitantes",
    "10.001 a 50.000 habitantes",
    "50.001 a 100.000 habitantes",
    "100.001 a 400.000 habitantes",
    "400.001 a 1.000.000 habitantes",
    "Más de 1.000.000 habitantes",
]

INGRESOS = [
    "Más de 5.000 €",
    "De 3.901 a 5.000 €",
    "De 2.701 a 3.900 €",
    "De 1.801 a 2.700 €",
    "De 1.100 a 1.800 €",
    "Menos de 1.100 €",
]
INGRESOS_EXTRA = ["N.S.", "N.C."]


def norm(text):
    """Minúsculas, sin acentos ni entidades HTML, espacios colapsados."""
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def age_group(ages):
    return pd.cut(
        pd.to_numeric(ages, errors="coerce"), _EDAD_BINS, right=False, labels=GRUPOS_EDAD
    ).astype("object")


def map_sexo(label):
    return {"hombre": "hombre", "mujer": "mujer"}.get(norm(label))


def map_estudios(label):
    """Etiqueta CIS de estudios -> categoría común. None si no procede (N.P.)."""
    n = norm(label)
    if n in ("n.p.", ""):
        return None
    if n.startswith("sin estudios"):
        return "sin_estudios"
    if n.startswith("primaria"):
        return "primaria"
    if n.startswith(("secundaria 1", "secundaria inferior")):
        return "secundaria_1"
    if n.startswith(("secundaria 2", "secundaria superior")):
        return "secundaria_2"
    if n in ("f.p.", "fp") or n.startswith("cualificacion profesional"):
        return "fp"
    if n.startswith(("superiores", "universitari")):
        return "superiores"
    if n in ("n.c.", "n.s."):
        return "nc"
    return "otros"


_CCAA_KEYS = {
    "andalucia": 0,
    "aragon": 1,
    "asturias": 2,
    "balears": 3,
    "canarias": 4,
    "cantabria": 5,
    "castilla y leon": 6,
    "castilla la mancha": 7,
    "cataluna": 8,
    "comunitat valenciana": 9,
    "extremadura": 10,
    "galicia": 11,
    "madrid": 12,
    "murcia": 13,
    "navarra": 14,
    "pais vasco": 15,
    "rioja": 16,
    "ceuta": 17,
    "melilla": 18,
}


def map_ccaa(label):
    n = norm(label)
    for key, i in _CCAA_KEYS.items():
        if n.startswith(key):
            return CCAA[i]
    return None


def map_tamuni(label):
    n = norm(label)
    for category in TAMUNI:
        if norm(category) == n:
            return category
    return None


_INGRESOS_NUMBERS = {
    (5000,): INGRESOS[0],
    (3901, 5000): INGRESOS[1],
    (2701, 3900): INGRESOS[2],
    (1801, 2700): INGRESOS[3],
    (1100, 1800): INGRESOS[4],
    (1100,): INGRESOS[5],
}


def map_ingresos(label):
    """Tramos de ingresos netos del hogar del CIS. Otras escalas (p. ej. personales) -> None."""
    n = norm(label)
    if n in ("n.s.", "n.c."):
        return n.upper()
    numbers = tuple(int(x.replace(".", "")) for x in re.findall(r"\d[\d.]*", n))
    category = _INGRESOS_NUMBERS.get(numbers)
    if category is None:
        return None
    # Distingue "Más de 5.000" y "Menos de 1.100" de los tramos con un único número.
    if numbers == (5000,) and not n.startswith("mas"):
        return None
    if numbers == (1100,) and not n.startswith("menos"):
        return None
    return category
