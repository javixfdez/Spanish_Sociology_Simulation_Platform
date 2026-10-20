"""Traducción al español, una sola vez y cacheada en disco, de los enunciados y opciones de las fuentes europeas."""

import json
from pathlib import Path

from pydantic import BaseModel


class Translation(BaseModel):
    items: list[str]


def load_cache(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path, cache):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def texts_to_translate(records):
    seen = {}
    for r in records:
        for text in [r.get("question_text"), *(r.get("options") or [])]:
            if text and text not in seen:
                seen[text] = None
    return list(seen)


async def translate_texts(texts, service, cache, batch=25, on_batch=None):
    """Traduce los textos que no estén en ``cache`` en lotes; devuelve la caché ampliada."""
    pending = [t for t in texts if t not in cache]
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]

        def check(value, expected=len(chunk)):
            if len(value.items) != expected:
                raise ValueError(f"Se esperaban {expected} traducciones y llegaron {len(value.items)}")

        result = await service.call(Translation, {"task": "translate_to_spanish", "texts": chunk}, check)
        cache.update(dict(zip(chunk, result.items)))
        if on_batch:
            on_batch(cache)
    return cache


def apply_translation(record, cache):
    """Registro en español: enunciado, opciones y claves de las distribuciones; conserva el original en metadata."""

    def es(text):
        return cache.get(text, text)

    out = dict(record)
    out["question_text"] = es(record["question_text"]) if record.get("question_text") else record.get("question_text")
    out["options"] = [es(o) for o in record.get("options") or []]
    out["idioma"] = "es"
    metadata = json.loads(json.dumps(record.get("metadata", {})))
    metadata["original_en"] = {"question_text": record.get("question_text"), "options": record.get("options")}
    for block in metadata.get("empirical_statistics", {}).values():
        if not isinstance(block, dict):
            continue
        if "distribution" in block:
            block["distribution"] = {es(k): v for k, v in block["distribution"].items()}
        else:  # subgrupos: {categoría: {distribution, n}}
            for entry in block.values():
                if isinstance(entry, dict) and "distribution" in entry:
                    entry["distribution"] = {es(k): v for k, v in entry["distribution"].items()}
    out["metadata"] = metadata
    return out
