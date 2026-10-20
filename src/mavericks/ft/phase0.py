"""Informe de la Fase 0: número de ejemplos por fuente y tipo en los cortes A' y B, longitud de los prompts y opciones."""

import time
from pathlib import Path

import numpy as np

from . import data

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def counts(frame, cut):
    part = frame[frame[cut]]
    table = part.groupby(["source", "scope"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    header = "| fuente | " + " | ".join(table.columns) + " |\n|---|" + "---|" * len(table.columns)
    rows = [f"| {idx} | " + " | ".join(str(int(v)) for v in row) + " |" for idx, row in table.iterrows()]
    return "\n".join([header, *rows])


def token_lengths(frame, sample=3000, seed=0):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = frame.sample(min(sample, len(frame)), random_state=seed)
    lengths = [
        len(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True, return_dict=False))
        for p in rows.prompt
    ]
    letters = {c: tok.encode(c, add_special_tokens=False) for c in data.LETTERS}
    return np.array(lengths), letters


def main():
    t0 = time.time()
    frame = data.build()
    print(f"construido en {time.time() - t0:.0f} s: {len(frame)} ejemplos")
    lines = ["# Fase 0 — datos del fine-tuning", ""]
    for cut, label in (("in_a", "Corte A' (trabajo de campo < 2025-07-01)"), ("in_b", "Corte B (trabajo de campo < 2026-04-01)")):
        part = frame[frame[cut]]
        lines += [
            f"## {label}", counts(frame, cut), "",
            f"Total: {len(part)} ejemplos (el orden de las opciones se baraja al vuelo en cada paso; no hay copias en disco). "
            f"Estudios: {part.study.nunique()}; preguntas distintas: {part.evidence_id.nunique()}",
            "Número de opciones: " + ", ".join(f"{k}: {v}" for k, v in sorted(part.options.map(len).value_counts().items())),
            "Eurobarómetro y ESS filtrados (sin variables derivadas ni etiquetas de variable); peso de la pérdida ×0,5. "
            f"Preguntas externas distintas: {part[part.source != 'CIS'].evidence_id.nunique()}", "",
        ]
    lengths, letters = token_lengths(frame[frame.in_b])
    lines += [
        "## Longitud del prompt (tokens, con la plantilla de chat de " + BASE_MODEL + ")",
        f"media {lengths.mean():.0f}, mediana {np.median(lengths):.0f}, p95 {np.percentile(lengths, 95):.0f}, "
        f"p99 {np.percentile(lengths, 99):.0f}, máx {lengths.max()}",
        f"Letras A-G (un token cada una): {letters}",
    ]
    text = "\n".join(lines)
    (Path("outputs/ft") / "phase0_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
