"""Casos de desarrollo y candidatas con las protecciones anti-fuga del backtest (se reutiliza ``EvidenceRepository.search``).

Regla del experimento: solo casos dev; cualquier identificador de estudio reservado (publicado ≥ 2026-04-01) detiene el
programa. El corte temporal estricto, la exclusión de la propia pregunta y el filtro de reservados los aplica ``search``.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..analog import candidates
from ..topics import normalize_topic  # noqa: F401

RESERVED_FROM = "2026-04-01"
DEV_CASES = "outputs/ablation/cases_dev_hard.json"  # dev de modo prior y sin ancla (44), generado desde la línea base dev


def reserved_ids(cis_dir="data/processed"):
    q = pd.read_parquet(Path(cis_dir) / "cis_questions.parquet", columns=["evidence_id", "reservado_examen", "fecha_publicacion"])
    return set(q[q.reservado_examen | (q.fecha_publicacion >= RESERVED_FROM)].evidence_id)


def assert_no_reserved(evidence_ids, what, reserved=None):
    reserved = reserved if reserved is not None else reserved_ids()
    bad = sorted(set(evidence_ids) & reserved)
    assert not bad, f"{what}: entran evidence_id de estudios reservados: {bad[:5]}"


def load_dev_cases(path=DEV_CASES):
    cases = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assert cases and all(c.get("split") == "dev" for c in cases), "solo se admiten casos de desarrollo"
    assert_no_reserved([c["evidence_id"] for c in cases], "casos")
    return cases


POOL = 40  # tamaño del conjunto de candidatas de cada método de recuperación (como en producción)


def collect(cases, repository, table, emb, case_vectors, sources=("cis",), progress=None, strict=True, full_text=None):
    """Candidatas de cada caso con los filtros anti-fuga de ``search`` (corte temporal, sin la propia pregunta, sin reservados).

    ``search(keep_zero=True)`` devuelve TODO el almacén ya filtrado. Sobre él se definen tres conjuntos de candidatas:
    las 40 mejores por TF-IDF (las de producción, con coincidencia léxica), las 40 mejores por coseno e5 query-query y
    las 40 mejores por e5 query-passage. ``case_vectors``: {id del caso: {"query": vector, "passage": vector}}.
    """
    reserved = reserved_ids()
    out = []
    for n, case in enumerate(cases):
        options = case["options"]
        hits = repository.search(
            f"{case['question']} | " + "; ".join(options[:8]),
            limit=1_000_000,
            excluded_ids=[case["evidence_id"]],
            excluded_texts=[case["question"]] if strict else [],
            cutoff=case["fieldwork_date"],
            options=options,
            topic=case["topic"],  # como en producción: el tema solo bonifica el orden, no filtra
            topic_filter=False,
            retriever="tfidf",
            sources=set(sources),
            keep_zero=True,
        )
        ids_all = [h["evidence"]["evidence_id"] for h in hits]
        assert_no_reserved(ids_all, f"almacén filtrado de {case['evidence_id']}", reserved)
        assert case["evidence_id"] not in ids_all
        pool_lex = [h["evidence"]["evidence_id"] for h in hits if h["cosine_similarity"] > 0][:POOL]
        embeddable = [i for i in ids_all if i in emb.q_index and i in emb.qq_index]
        q = case_vectors[case["evidence_id"]]["query"]
        cos_qp = {i: float(emb.question(i) @ q) for i in embeddable}
        cos_qq = {i: float(emb.question_query(i) @ q) for i in embeddable}
        pool_qp = sorted(embeddable, key=lambda i: -cos_qp[i])[:POOL]
        pool_qq = sorted(embeddable, key=lambda i: -cos_qq[i])[:POOL]
        union = set(pool_lex) | set(pool_qp) | set(pool_qq)
        cands = candidates([h for h in hits if h["evidence"]["evidence_id"] in union], case["question"], options, table)
        for c in cands:
            i = c["evidence_id"]
            c.update(
                cos_qq=cos_qq.get(i), cos_qp=cos_qp.get(i), in_lex=i in pool_lex, in_qq=i in pool_qq, in_qp=i in pool_qp
            )
        assert_no_reserved([c["evidence_id"] for c in cands], f"candidatas de {case['evidence_id']}", reserved)
        assert all((c["date"] or "")[:10] < case["fieldwork_date"][:10] for c in cands), "candidata posterior al caso"
        repeated = None
        if full_text is not None:  # ¿existe en el pasado la misma pregunta (similitud léxica ≥ 0,90, escala equivalente, mismo ítem)?
            from ..anchor import same_item

            repeated = any(
                c["kind"] == "label" and c["lex"] >= 0.90 and same_item(case["question"], full_text.get(c["evidence_id"], ""))
                for c in cands
            )
        out.append(
            {
                "repeated": repeated,
                **{k: case[k] for k in ("evidence_id", "estudio_id", "topic", "fieldwork_date", "question", "options")},
                "observed": case["observed_distribution"],
                "candidates": cands,
                "store_size": len(ids_all),
            }
        )
        if progress:
            progress(n + 1, len(cases))
    return out


def save(path, collected):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(collected, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def case_vectors(npz_path):
    z = np.load(npz_path)
    return {
        str(i): {"query": q, "passage": p} for i, q, p in zip(z["ids"], z["as_query"], z["as_passage"], strict=True)
    }
