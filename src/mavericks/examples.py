"""Preguntas de ejemplo de la demo, precalculadas para que la interfaz responda al instante."""

import json
import time
from pathlib import Path

from .service import predict, suggest_options

EXAMPLES_PATH = "data/examples/ejemplos_demo.json"

EXAMPLES = [
    # Directo: enunciado literal de preguntas que el CIS repite o hace en sus estudios
    (
        "general",
        "Refiriéndonos a la situación económica general de España actualmente, ¿cómo la calificaría Ud.: muy buena, buena, mala o muy mala?",
    ),
    (
        "vivienda",
        "En términos generales, ¿cree Ud. que el precio de la vivienda durante el próximo año subirá, se mantendrá o bajará?",
    ),
    (
        "salud",
        "En general, la atención que recibió en las urgencias de la sanidad pública ¿fue muy buena, buena, regular, mala o muy mala?",
    ),
    (
        "ia_tecnologia",
        "Indíqueme si le preocupan mucho, bastante poco o nada los siguientes riesgos "
        "[El riesgo de desaparición de ciertas profesiones debido al uso de la inteligencia artificial]",
    ),
    (
        "soledad",
        "De las siguientes frases que reflejan opiniones que algunas personas suelen tener, ¿podría decirme Ud. si está "
        "muy de acuerdo, de acuerdo, en desacuerdo o muy en desacuerdo con cada una de ellas? "
        "[Las personas mayores por lo general, sufren más soledad que las jóvenes]",
    ),
    # Prior: reformulaciones de preguntas parecidas del CIS
    ("vivienda", "¿Cree Ud. que el precio del alquiler durante el próximo año subirá, se mantendrá o bajará?"),
    (
        "consumo",
        "¿Cree que sus posibilidades de ahorrar el año que viene serán mayores, iguales o menores que las de este año?",
    ),
    (
        "general",
        "¿Cómo calificaría hoy la situación económica de su hogar: muy buena, buena, regular, mala o muy mala?",
    ),
    # Nuevas: sin equivalente en el CIS
    ("vivienda", "¿Está Ud. a favor o en contra de limitar por ley el precio del alquiler?"),
    ("medio_ambiente", "¿Está de acuerdo con prohibir la venta de coches de combustión a partir de 2035?"),
    ("ia_tecnologia", "¿Utiliza habitualmente herramientas de inteligencia artificial como ChatGPT?"),
]


def build_examples(resources, preset="rapido_sin_llm", path=EXAMPLES_PATH, service=None):
    """Calcula y guarda los ejemplos; las opciones salen de ``suggest_options`` como en la interfaz."""
    items = []
    for topic, question in EXAMPLES:
        suggestion = suggest_options(question, resources, service, topic)
        result = predict(
            question, suggestion["options"], "single", preset, topic=topic, resources=resources, service=service
        )
        items.append({**result, "suggestion": suggestion})
    payload = {"created": time.strftime("%Y-%m-%d %H:%M"), "preset": preset, "items": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_examples(path=EXAMPLES_PATH):
    file = Path(path)
    if not file.exists():
        return {"created": None, "preset": None, "items": []}
    return json.loads(file.read_text(encoding="utf-8"))
