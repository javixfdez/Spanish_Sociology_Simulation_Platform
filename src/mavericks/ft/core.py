"""Piezas del fine-tuning que solo necesitan numpy/pandas/torch (se importan también desde .venv_ft, sin el resto del proyecto)."""

import random
import zlib

LETTERS = "ABCDEFG"
INSTRUCTION = "Estima cómo se reparten las respuestas de la población indicada a esta pregunta de encuesta. Responde solo con la letra de una opción."


def prompt_text(profile, source, year, question, options, language="es"):
    idioma = "pregunta en inglés" if language == "en" else "pregunta en español"
    lines = [
        INSTRUCTION,
        profile,
        f"Fuente: {source}, {year} ({idioma})",
        f"Pregunta: {' '.join(question.split())}",
        *[f"{LETTERS[i]}) {o}" for i, o in enumerate(options)],
        "Respuesta:",
    ]
    return "\n".join(lines)


def option_order(example_id, n_options, epoch=0, seed=0, shuffle=True):
    """Permutación de las opciones, determinista por (ejemplo, época, semilla); identidad si ``shuffle`` es falso."""
    order = list(range(n_options))
    if shuffle:
        random.Random(zlib.crc32(f"{example_id}|{epoch}|{seed}".encode())).shuffle(order)
    return order


def build_item(row, epoch=0, seed=0, shuffle=True):
    """(prompt, reparto) de un ejemplo con el orden de las opciones barajado al vuelo (el reparto se reordena igual)."""
    order = option_order(row["id"], len(row["options"]), epoch, seed, shuffle)
    options = [row["options"][i] for i in order]
    props = [float(row["props"][i]) for i in order]
    prompt = prompt_text(row["profile"], row["source"], row["year"], row["question"], options, row["language"])
    return prompt, props


def kl_loss(log_q, valid, props, weights):
    import torch

    device = log_q.device
    p = torch.zeros_like(log_q)
    for i, row in enumerate(props):
        p[i, : len(row)] = torch.tensor(row, device=device)
    kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q.masked_fill(~valid, 0.0))).sum(-1)
    weights = weights.to(device)
    return (kl * weights).sum() / weights.sum().clamp_min(1e-6), kl
