"""Informe de la Fase 0: número de ejemplos por fuente y tipo en los cortes A y B, longitud de los prompts y opciones."""

import time

import numpy as np

from . import data

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def counts(frame, cut):
    part = frame[frame[cut]]
    table = part.groupby(["source", "scope"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    table["con copia barajada"] = 2 * table["total"]
    header = "| fuente | " + " | ".join(table.columns) + " |\n|---|" + "---|" * len(table.columns)
    rows = [f"| {idx} | " + " | ".join(str(int(v)) for v in row) + " |" for idx, row in table.iterrows()]
    return "\n".join([header, *rows])


def token_lengths(frame, sample=3000, seed=0):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = frame.sample(min(sample, len(frame)), random_state=seed)
    lengths = [len(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True, return_dict=False)) for p in rows.prompt]
    letters = {c: tok.encode(c, add_special_tokens=False) for c in data.LETTERS}
    return np.array(lengths), letters


def main():
    t0 = time.time()
    frame = data.build()
    print(f"construido en {time.time() - t0:.0f} s: {len(frame)} ejemplos")
    lines = ["# Fase 0 — datos del fine-tuning", ""]
    for cut, label in (("in_a", "Corte A (trabajo de campo < 2024-07-01)"), ("in_b", "Corte B (trabajo de campo < 2026-04-01)")):
        t = counts(frame, cut)
        lines += [f"## {label}", t, "", f"Total sin barajar: {int(frame[cut].sum())}; con la copia barajada: {2 * int(frame[cut].sum())}", ""]
        part = frame[frame[cut]]
        lines += ["Número de opciones: " + ", ".join(f"{k}: {v}" for k, v in sorted(part.options.map(len).value_counts().items())),
                  f"Estudios: {part.study.nunique()}; preguntas distintas: {part.evidence_id.nunique()}", ""]
    lengths, letters = token_lengths(frame[frame.in_b])
    lines += ["## Longitud del prompt (tokens, con la plantilla de chat de " + BASE_MODEL + ")",
              f"media {lengths.mean():.0f}, mediana {np.median(lengths):.0f}, p95 {np.percentile(lengths, 95):.0f}, p99 {np.percentile(lengths, 99):.0f}, máx {lengths.max()}",
              f"Letras A-G (un token cada una): {letters}", ""]
    a = frame[frame.in_a]
    total_a = 2 * len(a)
    lines += ["## Tamaño del corte A y propuesta", f"Con la copia barajada, A tiene {total_a} ejemplos (umbral de 80.000)."]
    if total_a > 80_000:
        national = int((a.scope == "nacional").sum()) * 2
        keep = max(0, 80_000 - national)
        lines.append(f"Propuesta: conservar todos los nacionales ({national}) y submuestrear los subgrupos hasta {keep} (estratificado por fuente, dimensión y número de opciones, semilla fija).")
    text = "\n".join(lines)
    (data.Path("outputs/ft") / "phase0_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
