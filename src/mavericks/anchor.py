"""Ancla empírica: distribución CIS ponderada de las personas con el perfil de un segmento.

Para cada segmento se estima P(respuesta | perfil) con los microdatos del CIS de la misma pregunta,
con contracción jerárquica hacia niveles más amplios (nacional -> primera dimensión -> ...), de modo que
las celdas con pocos entrevistados se apoyan en el nivel superior. Solo se usan estudios que el RAG ha
recuperado (y por tanto filtrados por las exclusiones del backtesting).
"""

import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .categories import norm

DEMOGRAPHICS = ["grupo_edad", "sexo", "nivel_estudios", "ccaa", "tamuni", "ingresos_hogar_tramo"]
_NON_SUBSTANTIVE = {"ns", "nc", "no sabe", "no contesta", "np"}


def option_key(option):
    return re.sub(r"[^a-z0-9 ]", "", norm(option).replace("(no leer)", "")).strip()


def question_similarity(a, b):
    """Parecido de redacción entre dos preguntas (0-1): máximo de Jaccard de palabras y de secuencia."""
    ta, tb = norm(a), norm(b)
    if not ta or not tb:
        return 0.0
    wa, wb = set(re.findall(r"\w+", ta)), set(re.findall(r"\w+", tb))
    jaccard = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
    return max(jaccard, difflib.SequenceMatcher(None, ta, tb).ratio())


def option_mapping(question_options, evidence_options):
    """Opciones de la pregunta -> etiquetas de la evidencia, o None si no son equivalentes.

    Se admite que la evidencia tenga además N.S./N.C. (se renormaliza sin ellas).
    """
    by_key = {option_key(o): o for o in evidence_options}
    mapping = {}
    for option in question_options:
        key = option_key(option)
        if key not in by_key:
            return None
        mapping[option] = by_key[key]
    extras = set(by_key) - {option_key(o) for o in question_options}
    if not extras <= _NON_SUBSTANTIVE:
        return None
    return mapping


def select_anchor(hits, question, direct_similarity=0.9, prior_similarity=0.55):
    """Elige la evidencia que ancla la predicción: la de mayor parecido de redacción con opciones equivalentes
    y, a igualdad, la oleada más reciente (después la mejor puntuación de recuperación). Devuelve dict o None."""
    candidates = []
    for hit in hits:
        e = hit["evidence"]
        original = e.get("metadata", {}).get("original_fields", {})
        qtype = original.get("question_type", e.get("metadata", {}).get("question_type"))
        if not e.get("study_id") or qtype != question["question_type"] or not e.get("question_text"):
            continue
        mapping = option_mapping(question["options"], e.get("options") or [])
        if mapping is None:
            continue
        similarity = question_similarity(question["question"], e["question_text"])
        if similarity >= prior_similarity:
            candidates.append((round(similarity, 3), e.get("fieldwork_date") or "", hit["retrieval_score"], hit, mapping, qtype))
    if not candidates:
        return None
    similarity, _, _, hit, mapping, qtype = max(candidates, key=lambda c: (c[0], c[1], c[2]))
    e = hit["evidence"]
    return {
        "mode": "direct" if similarity >= direct_similarity else "prior",
        "evidence_id": e["evidence_id"],
        "study_id": str(e["study_id"]),
        "question_type": qtype,
        "similarity": similarity,
        "mapping": mapping,
        "evidence_options": e["options"],
        "fieldwork_date": e.get("fieldwork_date"),
    }


def project_to_band(values, low, high, total=1.0):
    """Proyecta ``values`` a [low, high] manteniendo suma ``total`` (desplazamiento común, bisección)."""
    values, low, high = (np.asarray(x, dtype=float) for x in (values, low, high))
    lo_t, hi_t = -1.0, 1.0
    for _ in range(60):
        mid = (lo_t + hi_t) / 2
        if np.clip(values + mid, low, high).sum() < total:
            lo_t = mid
        else:
            hi_t = mid
    result = np.clip(values + (lo_t + hi_t) / 2, low, high)
    return result * (total / result.sum()) if result.sum() > 0 else result


class AnchorStore:
    def __init__(self, directory="data/processed", shrinkage=25.0):
        directory = Path(directory)
        self.respondents = pd.read_parquet(directory / "cis_respondents.parquet")
        self.responses_path = directory / "cis_responses.parquet"
        self.shrinkage = shrinkage
        self._frames = {}

    @classmethod
    def available(cls, directory="data/processed"):
        directory = Path(directory)
        return (directory / "cis_respondents.parquet").exists() and (directory / "cis_responses.parquet").exists()

    def _frame(self, study_id, question_uid):
        key = (str(study_id), question_uid)
        if key not in self._frames:
            table = pq.read_table(
                self.responses_path,
                filters=[("estudio_id", "=", str(study_id)), ("question_uid", "=", question_uid)],
            ).to_pandas()
            people = self.respondents[self.respondents.estudio_id == str(study_id)]
            self._frames[key] = table.merge(
                people.drop(columns=["edad"], errors="ignore"), on=["estudio_id", "respondent_uid"], how="left"
            )
        return self._frames[key]

    @staticmethod
    def _distribution(frame, qtype):
        if frame.empty:
            return pd.Series(dtype=float), 0, 0.0
        persons = frame.drop_duplicates("respondent_uid")
        total = persons.peso.sum()
        if qtype == "single":
            sums = frame.groupby("answer").peso.sum()
        else:
            sums = frame[frame.selected.fillna(False).astype(bool)].groupby("answer").peso.sum()
        n_eff = total**2 / (persons.peso**2).sum() if total > 0 else 0.0
        return sums / total if total > 0 else sums, len(persons), float(n_eff)

    def _levels(self, frame, qtype, constraints):
        """Distribuciones por nivel de detalle creciente con su n efectivo."""
        levels = [("nacional", *self._distribution(frame, qtype))]
        current = frame
        applied = []
        for dimension, categories in constraints.items():
            if dimension not in current.columns or dimension not in DEMOGRAPHICS:
                continue
            current = current[current[dimension].isin(categories)]
            applied.append(dimension)
            levels.append((" + ".join(applied), *self._distribution(current, qtype)))
        return levels

    def segment_distribution(self, anchor, constraints):
        """Distribución {opción de la pregunta: prob.} del segmento con contracción jerárquica."""
        frame = self._frame(anchor["study_id"], anchor["evidence_id"])
        mapping, qtype = anchor["mapping"], anchor["question_type"]
        evidence_labels = list(mapping.values())
        levels = self._levels(frame, qtype, constraints)
        parent = None
        for name, dist, n, n_eff in levels:
            p = dist.reindex(evidence_labels).fillna(0.0)
            if qtype == "single":
                mass = p.sum()
                p = p / mass if mass > 0 else p  # renormaliza sin N.S./N.C.
            if parent is not None:
                p = (n_eff * p + self.shrinkage * parent) / (n_eff + self.shrinkage) if n_eff > 0 else parent
            parent = p
            last = {"level": name, "n": n, "n_eff": round(n_eff, 1)}
        probabilities = {q: float(parent[e]) for q, e in mapping.items()}
        national = levels[0]
        return {
            "probabilities": probabilities,
            "levels": [{"level": lv[0], "n": lv[2], "n_eff": round(lv[3], 1)} for lv in levels],
            "finest": last,
            "national": {
                q: float(national[1].reindex(evidence_labels).fillna(0.0)[e]) for q, e in mapping.items()
            },
        }

    def national(self, study_id, question_uid, qtype):
        dist, n, n_eff = self._distribution(self._frame(study_id, question_uid), qtype)
        return {k: round(float(v), 4) for k, v in dist.items()}, n


def constraints_of(segment):
    """Restricciones del segmento en orden de prioridad: {dimensión: [categorías]}."""
    return {d: list(c) for d, c in segment.get("constraints", {}).items()}


def load_anchor_store(settings, directory=None):
    """AnchorStore si existen los microdatos procesados del CIS; None en caso contrario."""
    directory = directory or settings.cis_dir
    return AnchorStore(directory, settings.anchor_shrinkage) if AnchorStore.available(directory) else None
