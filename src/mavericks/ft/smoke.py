r"""Prueba de humo de un adaptador: 10 prompts INVENTADOS (no son datos reales); probabilidades finitas, suma 1 y parecido entre dos adaptadores.

    .venv_ft\Scripts\python -m mavericks.ft.smoke outputs/ft/runs/b1/adapter_final outputs/ft/runs/a1/adapter_final
"""

import sys
from pathlib import Path

import numpy as np
import torch

from . import core, train

INVENTED = [
    ("¿Está de acuerdo con que los autobuses urbanos sean gratuitos en su ciudad?", ["Sí", "No"]),
    ("¿Cómo valora el estado de las aceras de su barrio?", ["Muy bueno", "Bueno", "Regular", "Malo", "Muy malo"]),
    ("¿Con qué frecuencia usa una bicicleta para desplazarse?", ["Todos los días", "Varias veces por semana", "Alguna vez al mes", "Nunca"]),
    ("¿Cree que los parques de su municipio tienen suficientes zonas de sombra?", ["Sí", "No"]),
    ("¿Qué le preocupa más del precio de los alimentos?", ["La fruta y verdura", "La carne y el pescado", "Los lácteos", "Otros productos", "Ninguno"]),
    ("¿Prefiere comprar en tiendas de barrio o en grandes superficies?", ["Tiendas de barrio", "Grandes superficies", "Me da igual"]),
    ("¿Ha usado alguna vez un patinete eléctrico de alquiler?", ["Sí", "No"]),
    ("¿Cómo calificaría el servicio de recogida de basuras de su calle?", ["Excelente", "Adecuado", "Mejorable", "Deficiente"]),
    ("¿Le parece bien que se amplíe el horario de las bibliotecas públicas?", ["Mucho", "Bastante", "Poco", "Nada"]),
    ("¿Dónde prefiere pasar las vacaciones de verano?", ["Playa", "Montaña", "Ciudad", "Pueblo", "En casa", "Otro lugar"]),
]


@torch.no_grad()
def read(model, tok, ids):
    out = []
    for question, options in INVENTED:
        prompt = core.prompt_text("Población: adultos residentes en España", "CIS", "2026", question, options, "es")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt", padding=True, add_special_tokens=False)
        log_q, _ = train.letter_log_probs(model, enc, ids, [len(options)])
        out.append(log_q[0].exp()[: len(options)].float().cpu().numpy())
    return out


def main(paths):
    results = {}
    for path in paths:
        tok, model = train.load_model(adapter=Path(path), trainable=False)
        model.eval()
        results[path] = read(model, tok, train.letter_ids(tok))
        del model
        torch.cuda.empty_cache()
    names = list(results)
    for i, (q, _) in enumerate(INVENTED):
        rows = [results[n][i] for n in names]
        assert all(np.isfinite(r).all() and abs(r.sum() - 1) < 1e-3 for r in rows)
        print(f"{i + 1}. " + " | ".join(np.array2string(r, precision=2) for r in rows))
    if len(names) == 2:
        diffs = [100 * np.abs(a - b).mean() for a, b in zip(*results.values(), strict=True)]
        print(f"todas finitas y suman 1; diferencia media entre adaptadores {np.mean(diffs):.1f} pp (máx. {np.max(diffs):.1f}); idénticas: {all(d < 1e-6 for d in diffs)}")


if __name__ == "__main__":
    main(sys.argv[1:])
