"""Fase 0 del fine-tuning: ejemplos (pregunta, perfil, repartos REALES) del CIS, Eurobarómetro y ESS, y formato del prompt.

Todos los objetivos son repartos observados (ponderados, sin N.S./N.C. y renormalizados); nada generado por un LLM. Ningún estudio
reservado (publicado ≥ 2026-04-01) entra en ningún conjunto (assert). Dos cortes por trabajo de campo: A < 2024-07-01 y B < 2026-04-01.
"""

import json
import random
import zlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..analog import NON_SUBSTANTIVE, load_national_table
from ..anchor import option_key
from ..assist import display_label
from ..evidence import EvidenceRepository, source_family
from ..prior_v2.data import assert_no_reserved, reserved_ids

CUT_A = "2024-07-01"
CUT_B = "2026-04-01"
MIN_SUBGROUP = 100
MAX_OPTIONS = 7
LETTERS = "ABCDEFG"
SOURCES = {"cis": "CIS", "eurobarometro": "Eurobarómetro", "ess": "ESS"}
DIMENSIONS = ("sexo", "grupo_edad", "ccaa", "nivel_estudios", "ingresos_hogar_tramo")
SEX = {"hombre": "hombres adultos", "mujer": "mujeres adultas"}
EDUCATION = {
    "sin_estudios": "sin estudios", "primaria": "con estudios primarios", "secundaria_1": "con la primera etapa de secundaria",
    "secundaria_2": "con bachillerato o secundaria superior", "fp": "con formación profesional", "superiores": "con estudios superiores",
}
EXCLUDED = {"nc", "otros"}


def profile_text(dim, category):
    """«Población: …» para el ámbito nacional (dim=None) o un subgrupo."""
    if dim is None:
        return "Población: adultos residentes en España"
    if dim == "sexo":
        return f"Población: {SEX.get(category, 'adultos')} residentes en España"
    if dim == "grupo_edad":
        who = "personas de 75 años o más" if category == "75+" else "personas de " + category.replace("-", " a ") + " años"
        return f"Población: {who} residentes en España"
    if dim == "ccaa":
        return f"Población: adultos residentes en {category}"
    if dim == "nivel_estudios":
        return f"Población: adultos {EDUCATION.get(category, category)} residentes en España"
    if dim == "ingresos_hogar_tramo":
        return f"Población: adultos con ingresos del hogar {category[0].lower() + category[1:]} residentes en España"
    return "Población: adultos residentes en España"


def prompt_text(profile, source, year, question, options):
    lines = [
        "Estima cómo se reparten las respuestas de la población indicada a esta pregunta de encuesta. Responde solo con la letra de una opción.",
        profile,
        f"Fuente: {source}, {year}",
        f"Pregunta: {' '.join(question.split())}",
        *[f"{LETTERS[i]}) {o}" for i, o in enumerate(options)],
        "Respuesta:",
    ]
    return "\n".join(lines)


def shuffled(example):
    """Copia con el orden de las opciones barajado (y el reparto reordenado), con semilla fija por ejemplo."""
    rng = random.Random(zlib.crc32(example["id"].encode()))
    order = list(range(len(example["options"])))
    rng.shuffle(order)
    if order == list(range(len(order))):
        order = order[::-1]
    return {**example, "id": example["id"] + "#s", "options": [example["options"][i] for i in order],
            "props": [example["props"][i] for i in order], "shuffled": True}


def _substantive(options, dist):
    kept = [o for o in options if option_key(o) not in NON_SUBSTANTIVE and o in dist]
    return kept if 2 <= len(kept) <= MAX_OPTIONS else []


def cis_examples(evidence_path, table_path, cis_dir="data/processed"):
    """Ejemplos nacionales y por subgrupo del CIS."""
    repo = EvidenceRepository(evidence_path)
    table = load_national_table(table_path)
    meta = {}
    for e in repo.all():
        if source_family(e) != "cis" or e.evidence_id not in table or not e.fieldwork_date or not e.question_text:
            continue
        if e.metadata.get("original_fields", {}).get("question_type", "single") not in (None, "single"):
            continue
        keep = _substantive(list(e.options or []), table[e.evidence_id]["dist"])
        if keep:
            meta[e.evidence_id] = (e, keep)
    assert_no_reserved(list(meta), "ejemplos del CIS")
    out = []
    for evidence_id, (e, keep) in meta.items():
        dist = table[evidence_id]["dist"]
        mass = sum(dist[o] for o in keep)
        out.append(_example(evidence_id, e, "CIS", display_options(keep), [dist[o] / mass for o in keep], table[evidence_id]["n"], None, None))
    out += cis_subgroups(meta, cis_dir)
    return out


def display_options(options):
    return [display_label(o) for o in options]


def _example(evidence_id, e, source, options, props, n, dim, category):
    date = e.fieldwork_date[:10]
    return {
        "id": f"{evidence_id}|{dim or 'nacional'}|{category or ''}", "evidence_id": evidence_id, "source": source, "study": str(e.study_id),
        "date": date, "year": date[:4], "topic": e.topic, "scope": "nacional" if dim is None else "subgrupo", "dim": dim, "category": category,
        "profile": profile_text(dim, category), "question": " ".join(e.question_text.split()), "options": options, "props": [float(p) for p in props],
        "n": int(n), "language": getattr(e, "idioma", None) or "es", "shuffled": False,
    }


def study_type(parquet_file, study):
    """El identificador de estudio con el tipo de la columna del parquet (texto o entero)."""
    import pyarrow as pa

    return int(study) if pa.types.is_integer(parquet_file.schema_arrow.field("estudio_id").type) else study


def cis_subgroups(meta, cis_dir):
    """Reparto de cada pregunta por sexo, edad, CCAA, estudios e ingresos con los microdatos (ponderados); ≥ 100 encuestados."""
    respondents = pd.read_parquet(Path(cis_dir) / "cis_respondents.parquet", columns=["estudio_id", "respondent_uid", "peso", *DIMENSIONS])
    by_study = {}
    for evidence_id, (e, keep) in meta.items():
        by_study.setdefault(str(e.study_id), []).append(evidence_id)
    examples = []
    responses = pq.ParquetFile(Path(cis_dir) / "cis_responses.parquet")
    for study, evidence_ids in sorted(by_study.items()):
        people = respondents[respondents.estudio_id.astype(str) == study]
        if people.empty:
            continue
        uids = {i: i for i in evidence_ids}  # en el CIS evidence_id y question_uid coinciden
        table = pq.read_table(
            Path(cis_dir) / "cis_responses.parquet", columns=["respondent_uid", "question_uid", "answer"],
            filters=[("estudio_id", "=", study_type(responses, study)), ("question_uid", "in", list(uids))],
        ).to_pandas()
        if table.empty:
            continue
        valid = pd.MultiIndex.from_tuples([(uid, a) for uid, i in uids.items() for a in meta[i][1]])
        table = table[pd.MultiIndex.from_arrays([table.question_uid, table.answer]).isin(valid)]
        merged = table.merge(people.drop(columns="estudio_id"), on="respondent_uid", how="inner")
        for dim in DIMENSIONS:
            part = merged.dropna(subset=[dim])
            part = part[~part[dim].astype(str).str.lower().isin(EXCLUDED)]
            grouped = part.groupby(["question_uid", dim, "answer"]).agg(w=("peso", "sum"), n=("respondent_uid", "nunique")).reset_index()
            for (uid, category), g in grouped.groupby(["question_uid", dim]):
                n = int(g.n.sum())
                if n < MIN_SUBGROUP:
                    continue
                evidence_id = uids[uid]
                e, keep = meta[evidence_id]
                weights = g.set_index("answer").w.reindex(keep).fillna(0.0)
                if weights.sum() <= 0:
                    continue
                examples.append(_example(evidence_id, e, "CIS", display_options(keep), (weights / weights.sum()).tolist(), n, dim, str(category)))
    return examples


def external_examples(path="data/processed_v2/external_evidence.json"):
    """Eurobarómetro y ESS (solo España, textos en inglés): nacionales y por sexo/edad si hay ≥ 100 encuestados."""
    out = []
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r.get("reservado_examen") in (True, "True") or not r.get("fieldwork_date"):
            continue
        stats = (r.get("metadata") or {}).get("empirical_statistics") or {}
        options = [o for o in r.get("options") or [] if option_key(o) not in NON_SUBSTANTIVE]
        national = (stats.get("national") or {}).get("distribution")
        if not national or not (2 <= len(options) <= MAX_OPTIONS) or any(o not in national for o in options):
            continue
        source = SOURCES[r["source"]]
        for dim, category, node in [(None, None, stats["national"]), *[(d, c, v) for d, cats in stats.items() if d in DIMENSIONS for c, v in cats.items()]]:
            if dim is not None and node.get("n_respondents", 0) < MIN_SUBGROUP:
                continue
            dist = node["distribution"]
            mass = sum(dist.get(o, 0.0) for o in options)
            if mass <= 0:
                continue
            date = r["fieldwork_date"][:10]
            out.append(
                {
                    "id": f"{r['evidence_id']}|{dim or 'nacional'}|{category or ''}", "evidence_id": r["evidence_id"], "source": source,
                    "study": str(r["study_id"]), "date": date, "year": date[:4], "topic": r.get("topic"), "scope": "nacional" if dim is None else "subgrupo",
                    "dim": dim, "category": category, "profile": profile_text(dim, category), "question": " ".join(r["question_text"].split()),
                    "options": options, "props": [dist.get(o, 0.0) / mass for o in options], "n": int(node.get("n_respondents") or r.get("sample_size") or 0),
                    "language": "en", "shuffled": False,
                }
            )
    return out


def build(evidence_path="data/processed_v2/evidence_v2.sqlite", table_path="data/processed_v2/cis_national.json", out_dir="outputs/ft/data"):
    examples = cis_examples(evidence_path, table_path) + external_examples()
    reserved = reserved_ids()
    assert not {e["evidence_id"] for e in examples} & reserved, "entra un ejemplo de un estudio reservado"
    examples = [e for e in examples if e["date"] < CUT_B]  # el corte B es estricto: lo posterior (p. ej. 2026-04-01) queda fuera
    for e in examples:
        e["in_a"] = e["date"] < CUT_A
        e["in_b"] = e["date"] < CUT_B
        e["prompt"] = prompt_text(e["profile"], e["source"], e["year"], e["question"], e["options"])
    frame = pd.DataFrame(examples)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    frame.to_pickle(Path(out_dir) / "examples.pkl")
    return frame
