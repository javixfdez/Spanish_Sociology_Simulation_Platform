"""Preguntas de ejemplo de la demo, precalculadas para que la interfaz responda al instante."""

import json
import time
from pathlib import Path

from .service import predict

EXAMPLES_PATH = "data/examples/ejemplos_demo.json"

SCALE_MUCHO = ["Mucho", "Bastante", "Poco", "Nada"]
SCALE_ACUERDO = ["Muy de acuerdo", "Bastante de acuerdo", "Regular", "Poco de acuerdo", "Nada de acuerdo"]
SI_NO = ["Sí", "No"]
FRECUENCIA = ["Nunca", "Casi nunca", "A veces", "Con frecuencia", "Siempre"]
VALORACION = ["Muy buena", "Buena", "Regular", "Mala", "Muy mala"]

EXAMPLES = [
    ("vivienda", "¿Considera que el acceso a la vivienda es un problema grave para los jóvenes en España?", SCALE_MUCHO),
    ("vivienda", "¿Está a favor de limitar por ley el precio del alquiler?", SI_NO),
    ("soledad", "¿Con qué frecuencia se siente solo/a?", FRECUENCIA),
    ("ia_tecnologia", "¿Le preocupa que la inteligencia artificial le quite su empleo?", SCALE_MUCHO),
    ("ia_tecnologia", "¿Utiliza habitualmente herramientas de inteligencia artificial como ChatGPT?", SI_NO),
    (
        "medio_ambiente",
        "¿Está de acuerdo con prohibir la venta de coches de combustión a partir de 2035?",
        SCALE_ACUERDO,
    ),
    (
        "medio_ambiente",
        "La Comisión Europea ha cambiado un acuerdo previo y ha decidido permitir la venta de coches de combustión "
        "después de 2035. ¿Está Ud. muy de acuerdo, bastante, poco o nada de acuerdo con esta decisión?",
        SCALE_ACUERDO,
    ),
    ("salud", "¿Cómo valora el funcionamiento de la sanidad pública en su comunidad autónoma?", VALORACION),
    ("consumo", "¿Está dispuesto/a a pagar más por productos sostenibles?", SCALE_ACUERDO),
    ("consumo", "¿Ha comprado productos de segunda mano en el último año?", SI_NO),
    ("general", "¿Cómo valora la situación económica de España actualmente?", VALORACION),
]


def build_examples(resources, preset="rapido_sin_llm", path=EXAMPLES_PATH, service=None):
    """Calcula y guarda los ejemplos. Con el preset recomendado y un LLM disponible se regeneran con él."""
    items = []
    for topic, question, options in EXAMPLES:
        items.append(predict(question, options, "single", preset, topic=topic, resources=resources, service=service))
    payload = {"created": time.strftime("%Y-%m-%d %H:%M"), "preset": preset, "items": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_examples(path=EXAMPLES_PATH):
    file = Path(path)
    if not file.exists():
        return {"created": None, "preset": None, "items": []}
    return json.loads(file.read_text(encoding="utf-8"))
