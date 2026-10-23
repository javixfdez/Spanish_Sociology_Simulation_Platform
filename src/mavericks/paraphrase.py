"""«¿Quizás querías decir…?»: sugerencia de una pregunta del CIS parecida a la escrita.

Solo sugiere; no cambia nada por su cuenta (ni la búsqueda del ancla, ni el prior, ni ninguna cifra). Compara la pregunta con las
preguntas de respuesta única del CIS del almacén de evidencia por coseno de embeddings e5 («query: » contra «query: »; tarea
simétrica) y propone la más parecida si supera ``THRESHOLD``. De cada variable del CIS se compara solo la oleada más reciente
(la que usaría el modo directo). Los vectores se calculan una vez y se guardan en ``outputs/cache``.
"""

import hashlib
from pathlib import Path

import numpy as np

from .anchor import series_key
from .assist import clean_options
from .evidence import source_family

THRESHOLD = 0.95  # coseno e5 mínimo para sugerir (calibrado: precisión 23/23 y cobertura 64 % en 36 paráfrasis, 0 de 52 negativas; ver docs/parafrasis.md)
CACHE_DIR = Path("outputs/cache")


def _latest_per_series(records):
    """Preguntas de respuesta única del CIS con al menos dos opciones sustantivas; de cada variable, la oleada más reciente."""
    latest = {}
    for e in records:
        if source_family(e) != "cis" or not e.question_text or not e.study_id:
            continue
        original = e.metadata.get("original_fields", {})
        if original.get("question_type", e.metadata.get("question_type", "single")) not in (None, "single"):
            continue
        if len(clean_options(e.options)) < 2:
            continue
        key = series_key(e.evidence_id) or e.evidence_id
        if key not in latest or (e.fieldwork_date or "") > (latest[key].fieldwork_date or ""):
            latest[key] = e
    return sorted(latest.values(), key=lambda e: e.evidence_id)


class ParaphraseIndex:
    def __init__(self, records, embedder, cache_dir=CACHE_DIR):
        self.items = _latest_per_series(records)
        self.texts = [" ".join(e.question_text.split()) for e in self.items]
        self.embedder = embedder
        digest = hashlib.sha1("\n".join(self.texts).encode("utf-8")).hexdigest()[:16]
        model = "".join(c if c.isalnum() else "_" for c in str(getattr(embedder, "name", "embedder")))[:50]
        self.cache = Path(cache_dir) / f"paraphrase_{model}_{digest}.npz"
        self.matrix = self._load_or_build()

    def _load_or_build(self):
        if self.cache.exists():
            with np.load(self.cache, allow_pickle=False) as stored:
                return stored["matrix"]
        matrix = np.asarray(self.embedder.encode_queries(self.texts), dtype=np.float32)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, matrix=matrix)
        return matrix

    def ranking(self, question, limit=5):
        """[(coseno, evidencia)] de las preguntas más parecidas, de mayor a menor."""
        q = np.asarray(self.embedder.encode_queries([" ".join(str(question).split())]), dtype=np.float32)[0]
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:limit]
        return [(float(scores[i]), self.items[i]) for i in order]


_INDEX = {"key": None, "obj": None}


def get_index(resources, embedder):
    key = (id(resources.evidence), id(embedder))
    if _INDEX["key"] != key:
        _INDEX.update(key=key, obj=ParaphraseIndex(resources.evidence.all(), embedder))
    return _INDEX["obj"]


def suggest(resources, question, embedder, threshold=None, index=None):
    """Sugerencia {question, evidence_id, study_id, fieldwork_date, options, similarity} o None si ninguna supera el umbral."""
    threshold = THRESHOLD if threshold is None else threshold
    if embedder is None or len(" ".join(str(question or "").split())) < 10:
        return None
    index = index or get_index(resources, embedder)
    best = index.ranking(question, limit=1)
    if not best or best[0][0] < threshold:
        return None
    score, e = best[0]
    return {
        "question": " ".join(e.question_text.split()), "evidence_id": e.evidence_id, "study_id": str(e.study_id),
        "fieldwork_date": e.fieldwork_date, "options": clean_options(e.options), "similarity": round(score, 3),
    }
