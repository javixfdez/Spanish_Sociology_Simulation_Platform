"""Guion de la demo: las mismas llamadas que hace la interfaz (``service.predict`` y la explicación) para 6 preguntas, una fuera de alcance y los tres modos.

    python -m mavericks.demo_run rapido        # sin Ollama ni GPU
    python -m mavericks.demo_run llm           # con Ollama (qwen3:8b)
    python -m mavericks.demo_run ajustado      # con el servidor del modelo ajustado (run_server.ps1) y, para la explicación, Ollama
    python -m mavericks.demo_run ajustado --fallback   # repite una pregunta con el servidor apagado: debe caer al modo rápido con aviso

Escribe ``outputs/demo/<modo>.json``; ``python -m mavericks.demo_run report`` compone ``docs/demo.md`` con los tres.
"""

import json
import sys
import time
from pathlib import Path

from . import explain, service
from .schemas import Settings
from .service import OutOfScopeError

OUT = Path("outputs/demo")
LIKERT4 = ["Mucho", "Bastante", "Poco", "Nada"]
AGREE5 = ["Muy de acuerdo", "Bastante de acuerdo", "Regular", "Poco de acuerdo", "Nada de acuerdo"]
QUESTIONS = [
    ("vivienda", "¿Cree que el precio del alquiler debería limitarse por ley?", AGREE5),
    ("soledad", "¿Con qué frecuencia se siente solo o sola?", ["Nunca", "Casi nunca", "A veces", "A menudo", "Siempre"]),
    ("ia", "¿Le preocupa que la inteligencia artificial le quite su empleo?", LIKERT4),
    ("medio ambiente", "¿Está a favor de prohibir la venta de coches de combustión a partir de 2035?", AGREE5),
    ("salud", "¿Cómo valora el funcionamiento de la sanidad pública?", ["Muy bueno", "Bueno", "Regular", "Malo", "Muy malo"]),
    ("consumo", "¿Piensa comprar menos productos en los próximos meses por la subida de precios?", ["Sí", "No"]),
    ("fuera de alcance", "¿Cuál es la capital de Francia?", ["París", "Londres"]),
]
PRESETS = {"rapido": "rapido_sin_llm", "llm": "llm_completo", "ajustado": "ajustado"}


def run_one(topic, question, options, kind, resources, llm):
    row = {"tema": topic, "pregunta": question, "opciones": options}
    started = time.monotonic()
    try:
        result = service.predict(question, options, "single", PRESETS[kind], resources=resources, service=llm if kind == "llm" else None)
    except OutOfScopeError as exc:
        return {**row, "resultado": "fuera de alcance", "segundos": time.monotonic() - started, "cobertura": exc.coverage["label"]}
    except Exception as exc:  # una pregunta rota no debe tumbar el guion
        return {**row, "resultado": f"error: {type(exc).__name__}: {exc}", "segundos": time.monotonic() - started}
    predict_seconds = time.monotonic() - started
    explanation_seconds = 0.0
    if llm is not None:  # igual que la interfaz: texto del LLM si es válido y llega a tiempo; si no, la plantilla
        t = time.monotonic()
        explain.add_llm_text(result, llm, timeout=Settings().explanation_timeout)
        explanation_seconds = time.monotonic() - t
    dist = result["national"]["distribution"]
    top = max(dist, key=dist.get)
    return {
        **row, "resultado": "ok", "modo_calculo": result["mode"], "producido_por": result["producido_por"], "segundos": predict_seconds,
        "segundos_explicacion": explanation_seconds, "primera": top, "porcentaje": 100 * dist[top],
        "distribucion": {o: round(100 * v, 1) for o, v in dist.items()},
        "rango_habitual": None if not result["national"].get("interval_core") else {o: [round(100 * a), round(100 * b)] for o, (a, b) in result["national"]["interval_core"].items()},
        "confianza": result["confidence"]["level"], "explicacion": result["explanation_text"], "avisos": result["warnings"], "fallback": result["fallback"],
    }


def run(kind, fallback=False):
    OUT.mkdir(parents=True, exist_ok=True)
    resources = service.load_resources()
    llm = None
    if kind in ("llm", "ajustado"):
        from .llm import LLMService, check_ollama, create_provider

        settings = Settings(llm_provider="ollama", ollama_model="qwen3:8b")
        check_ollama(settings)
        llm = LLMService(create_provider(settings), settings)
    started = time.monotonic()
    service.get_embedder()  # la interfaz lo carga al arrancar
    warm = time.monotonic() - started
    rows = []
    for topic, question, options in QUESTIONS:
        if fallback:
            if topic != "vivienda":
                continue
            rows.append(run_one(topic, question, options, kind, resources, llm))
            continue
        rows.append(run_one(topic, question, options, kind, resources, llm))
        print(topic, rows[-1].get("segundos"), rows[-1].get("producido_por", {}).get("modo"), flush=True)
    name = f"{kind}_fallback" if fallback else kind
    (OUT / f"{name}.json").write_text(json.dumps({"carga_e5_s": warm, "filas": rows}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def report():
    data = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in ("rapido", "llm", "ajustado")}
    fb = json.loads((OUT / "ajustado_fallback.json").read_text(encoding="utf-8"))
    lines = []
    for i, (topic, question, options) in enumerate(QUESTIONS):
        lines += [f"### {i + 1}. {topic.capitalize()}: «{question}»", f"Opciones: {' / '.join(options)}", "",
                  "| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |", "|---|---|---|---|---|---|"]
        for kind, label in (("rapido", "Rápido"), ("llm", "Con LLM"), ("ajustado", "Ajustado")):
            r = data[kind]["filas"][i]
            if r["resultado"] != "ok":
                lines.append(f"| {label} | {r['resultado']} ({r.get('cobertura', '')}) | — | — | {r['segundos']:.1f} | — |")
                continue
            p = r["producido_por"]
            how = f"{p['etiqueta']}" + (" · volvió al rápido" if p["fallback"] else "") + f" · {r['modo_calculo']}"
            text = r["explicacion"]
            lines.append(f"| {label} | {how} | {r['primera']} ({r['porcentaje']:.0f} %) | {r['confianza']} | {r['segundos']:.1f} + {r['segundos_explicacion']:.1f} | {text['origen']}"
                         + (f" ({text['motivo']})" if text.get("motivo") else "") + " |")
        lines.append("")
    lines += ["## Prueba de caída del servidor del modelo ajustado", ""]
    for r in fb["filas"]:
        p = r.get("producido_por", {})
        lines.append(f"- «{r['pregunta']}» con el servidor apagado: modo que produjo el resultado **{p.get('etiqueta')}**, fallback **{p.get('fallback')}**, motivo: {p.get('motivo')}; avisos: {r.get('avisos')}; {r['segundos']:.1f} s")
    return "\n".join(lines), data


if __name__ == "__main__":
    if sys.argv[1] == "report":
        print(report()[0])
    else:
        run(sys.argv[1], fallback="--fallback" in sys.argv)
