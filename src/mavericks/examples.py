"""Preguntas de ejemplo de la demo, precalculadas para que la interfaz responda al instante."""

import json
import time
from pathlib import Path

from .service import predict

EXAMPLES_PATH = "data/examples/ejemplos_demo.json"

SCALE_ACUERDO = ["Muy de acuerdo", "Bastante de acuerdo", "Regular", "Poco de acuerdo", "Nada de acuerdo"]
SI_NO = ["Sí", "No"]
FRECUENCIA = ["Nunca", "Casi nunca", "A veces", "Con frecuencia", "Siempre"]
VALORACION_REG = ["Muy buena", "Buena", "Regular", "Mala", "Muy mala"]
SUBIRA = ["Subirá", "Se mantendrá", "Bajará"]
MUCHO_REG = ["Mucho", "Bastante", "Regular", "Poco", "Nada"]

EXAMPLES = [
    # Directo: enunciado literal de preguntas que el CIS repite
    (
        "general",
        "Refiriéndonos a la situación económica general de España actualmente, ¿cómo la calificaría Ud.: muy buena, buena, mala o muy mala?",
        VALORACION_REG,
    ),
    (
        "vivienda",
        "En términos generales, ¿cree Ud. que el precio de la vivienda durante el próximo año subirá, se mantendrá o bajará?",
        SUBIRA,
    ),
    (
        "salud",
        "En general, la atención que recibió en las urgencias de la sanidad pública ¿fue muy buena, buena, regular, mala o muy mala?",
        VALORACION_REG,
    ),
    (
        "medio_ambiente",
        "¿Diría Ud. que en estos momentos el cambio climático le preocupa mucho, bastante, poco o nada?",
        MUCHO_REG,
    ),
    (
        "ia_tecnologia",
        "Indíqueme si le preocupan mucho, bastante poco o nada los siguientes riesgos "
        "[El riesgo de desaparición de ciertas profesiones debido al uso de la inteligencia artificial]",
        MUCHO_REG,
    ),
    # Prior: reformulaciones de preguntas parecidas del CIS
    (
        "vivienda",
        "¿Cree Ud. que el precio del alquiler durante el próximo año subirá, se mantendrá o bajará?",
        SUBIRA,
    ),
    (
        "consumo",
        "¿Cree que sus posibilidades de ahorrar el año que viene serán mayores, iguales o menores que las de este año?",
        ["Mayores", "Iguales", "Menores"],
    ),
    (
        "general",
        "¿Cómo calificaría hoy la situación económica de su hogar: muy buena, buena, regular, mala o muy mala?",
        VALORACION_REG,
    ),
    # Nuevas: sin equivalente en el CIS
    (
        "vivienda",
        "¿Está Ud. a favor o en contra de limitar por ley el precio del alquiler?",
        ["A favor", "En contra"],
    ),
    (
        "medio_ambiente",
        "¿Está de acuerdo con prohibir la venta de coches de combustión a partir de 2035?",
        SCALE_ACUERDO,
    ),
    ("soledad", "¿Se ha sentido solo/a en las últimas dos semanas?", SI_NO),
]


def build_examples(resources, preset="rapido_sin_llm", path=EXAMPLES_PATH, service=None):
    """Calcula y guarda los ejemplos. Con el preset recomendado y un LLM disponible se regeneran con él."""
    items = []
    for topic, question, options in EXAMPLES:
        items.append(
            predict(question, options, "single", preset, topic=topic, resources=resources, service=service)
        )
    payload = {"created": time.strftime("%Y-%m-%d %H:%M"), "preset": preset, "items": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_examples(path=EXAMPLES_PATH):
    file = Path(path)
    if not file.exists():
        return {"created": None, "preset": None, "items": []}
    return json.loads(file.read_text(encoding="utf-8"))
