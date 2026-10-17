"""Generación de casos de backtest a partir de los estudios reservados del CIS (reservado_examen = si).

Los resultados observados se calculan aquí, con el peso muestral, y solo se usan para comparar después
de la simulación; nunca entran en el RAG ni en los prompts.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .anchor import option_key

SUBGROUP_DIMENSIONS = ["sexo", "grupo_edad", "nivel_estudios"]
_NON_SUBSTANTIVE = {"ns", "nc", "np", "no sabe", "no contesta", "no procede", "no aplicable"}
MIN_SUBGROUP = 100


def clean_option(option):
    return re.sub(r"^\(no leer\)\s*", "", str(option), flags=re.I).strip()


def _substantive(options):
    return [o for o in options if option_key(o) not in _NON_SUBSTANTIVE]


def _weighted(frame, options):
    """Distribución ponderada sobre las opciones sustantivas (renormalizada) y fracción NS/NC."""
    persons = frame.drop_duplicates("respondent_uid")
    total = persons.peso.sum()
    sums = frame.groupby("answer").peso.sum()
    chosen = sums.reindex(options).fillna(0.0)
    mass = chosen.sum()
    if total <= 0 or mass <= 0:
        return None, 1.0
    return {clean_option(o): float(v / mass) for o, v in chosen.items()}, float(1 - mass / total), len(persons)


def build_cases(cis_dir="data/processed", max_per_study=8, min_n=500, seed=42, previous_waves=True, studies=None):
    """Casos de backtest de los estudios reservados.

    ``previous_waves=True`` (modo temporal): se permite recuperar la misma pregunta de oleadas anteriores,
    como ocurriría en el uso real. ``False`` (modo estricto): se excluye cualquier pregunta con el mismo texto.
    """
    cis_dir = Path(cis_dir)
    questions = pd.read_parquet(cis_dir / "cis_questions.parquet")
    respondents = pd.read_parquet(cis_dir / "cis_respondents.parquet")
    studies_table = pd.read_parquet(cis_dir / "cis_studies.parquet")
    reserved = studies_table[studies_table.reservado_examen]
    if studies:
        reserved = reserved[reserved.estudio_id.isin([str(s) for s in studies])]
    rng = np.random.default_rng(seed)
    cases = []
    for study in reserved.itertuples():
        q = questions[
            (questions.estudio_id == study.estudio_id)
            & (questions.question_type == "single")
            & (questions.n_valid >= min_n)
            & (questions.question_text.str.len() >= 30)
        ]
        if q.empty:
            continue
        responses = pq.read_table(cis_dir / "cis_responses.parquet", filters=[("estudio_id", "=", study.estudio_id)]).to_pandas()
        people = respondents[respondents.estudio_id == study.estudio_id]
        candidates = []
        for row in q.itertuples():
            options = _substantive(list(row.options))
            if not 2 <= len(options) <= 8:
                continue
            frame = responses[responses.question_uid == row.question_uid].merge(
                people, on=["estudio_id", "respondent_uid"], how="left"
            )
            observed = _weighted(frame, options)
            if observed[0] is None or max(observed[0].values()) > 0.97:
                continue
            candidates.append((row, options, frame, observed))
        order = rng.permutation(len(candidates))[:max_per_study]
        for i in sorted(order):
            row, options, frame, (distribution, ns_nc_share, n) = candidates[i]
            subgroups = {}
            for dimension in SUBGROUP_DIMENSIONS:
                subgroups[dimension] = {}
                for category, part in frame.groupby(dimension):
                    result = _weighted(part, options)
                    if result[0] is not None and result[2] >= MIN_SUBGROUP:
                        subgroups[dimension][str(category)] = {"distribution": result[0], "n": result[2]}
            cases.append(
                {
                    "case_id": row.question_uid,
                    "evidence_id": row.question_uid,
                    "estudio_id": study.estudio_id,
                    "question": row.question_text,
                    "options": [clean_option(o) for o in options],
                    "question_type": "single",
                    "topic": study.topic,
                    "fieldwork_start": study.fieldwork_start,
                    "fieldwork_end": study.fieldwork_end,
                    # Corte temporal: solo evidencia cuyo trabajo de campo termina antes de este inicio.
                    "fieldwork_date": study.fieldwork_start,
                    "observed_distribution": distribution,
                    "observed_subgroups": subgroups,
                    "ns_nc_share": ns_nc_share,
                    "n_valid": int(n),
                    "allow_previous_waves": previous_waves,
                    "weight": "peso",
                }
            )
    return cases
