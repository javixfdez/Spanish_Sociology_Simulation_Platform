"""Caché en disco de los embeddings e5 de las preguntas del almacén y de los textos de sus opciones.

Prefijos de e5: lo almacenado (preguntas y opciones del CIS) se codifica con «passage: »; la pregunta que se consulta,
con «query: ». La construcción necesita sentence-transformers (``.venv_embed``); la lectura solo numpy.
"""

import json
from pathlib import Path

import numpy as np

MODEL = "intfloat/multilingual-e5-base"
NON_SUBSTANTIVE = {"ns", "nc", "np", "no sabe", "no contesta", "no procede", "no aplicable", "nsnc"}


def clean_label(option):
    from ..assist import display_label

    return display_label(option)


def store_items(evidence_path, table_path, sources=("cis",)):
    """[(evidence_id, texto, opciones)] de las preguntas de respuesta única del CIS con distribución nacional."""
    from ..analog import load_national_table
    from ..evidence import EvidenceRepository, source_family

    table = load_national_table(table_path)
    out = []
    for e in EvidenceRepository(evidence_path).all():
        if source_family(e) not in sources or e.evidence_id not in table or not e.question_text:
            continue
        original = e.metadata.get("original_fields", {})
        if original.get("question_type", "single") not in (None, "single"):
            continue
        out.append((e.evidence_id, " ".join(e.question_text.split()), list(e.options or [])))
    return out


def build(evidence_path, table_path, out_dir, device="cpu", batch_size=32):
    from ..embeddings import SentenceTransformerEmbedder

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = store_items(evidence_path, table_path)
    embedder = SentenceTransformerEmbedder(MODEL, device=device, batch_size=batch_size)
    ids = [i for i, _, _ in items]
    questions = embedder.encode_passages([t for _, t, _ in items]).astype(np.float32)
    labels = sorted({clean_label(o) for _, _, opts in items for o in opts if clean_label(o)})
    options = embedder.encode_passages(labels).astype(np.float32)
    np.savez_compressed(out / "questions.npz", ids=np.array(ids), matrix=questions)
    np.savez_compressed(out / "options.npz", labels=np.array(labels), matrix=options)
    (out / "meta.json").write_text(
        json.dumps({"model": MODEL, "device": device, "prefix": "passage: ", "questions": len(ids), "options": len(labels)}),
        encoding="utf-8",
    )
    return len(ids), len(labels)


def build_query_questions(evidence_path, table_path, out_dir, device="cpu", batch_size=32):
    """Las mismas preguntas del almacén codificadas como «query: » (tarea simétrica pregunta contra pregunta)."""
    from ..embeddings import SentenceTransformerEmbedder

    items = store_items(evidence_path, table_path)
    embedder = SentenceTransformerEmbedder(MODEL, device=device, batch_size=batch_size)
    matrix = embedder.encode_queries([t for _, t, _ in items]).astype(np.float32)
    np.savez_compressed(Path(out_dir) / "questions_query.npz", ids=np.array([i for i, _, _ in items]), matrix=matrix)
    return len(items)


def build_query_options(evidence_path, table_path, out_dir, dev_cases=(), device="cpu", batch_size=32):
    """Etiquetas de opción del almacén y de los casos dev codificadas como «query: » (para el prior aprendido)."""
    from ..embeddings import SentenceTransformerEmbedder

    items = store_items(evidence_path, table_path)
    labels = sorted(
        {clean_label(o) for _, _, opts in items for o in opts if clean_label(o)}
        | {clean_label(o) for c in dev_cases for o in c["options"]}
    )
    embedder = SentenceTransformerEmbedder(MODEL, device=device, batch_size=batch_size)
    np.savez_compressed(
        Path(out_dir) / "options_query.npz", labels=np.array(labels), matrix=embedder.encode_queries(labels).astype(np.float32)
    )
    return len(labels)


class EmbCache:
    """Lectura de la caché: vectores de preguntas (por evidence_id) y de opciones (por etiqueta limpia)."""

    def __init__(self, directory):
        directory = Path(directory)
        self.directory = directory
        q = np.load(directory / "questions.npz")
        o = np.load(directory / "options.npz")
        self.q_index = {str(i): k for k, i in enumerate(q["ids"])}
        self.q = q["matrix"]
        self.o_index = {str(label): k for k, label in enumerate(o["labels"])}
        self.o = o["matrix"]
        qq = directory / "questions_query.npz"
        if qq.exists():
            z = np.load(qq)
            self.qq_index = {str(i): k for k, i in enumerate(z["ids"])}
            self.qq = z["matrix"]

    def question(self, evidence_id):
        return self.q[self.q_index[evidence_id]]

    def option_query(self, label):
        if not hasattr(self, "oq_index"):
            z = np.load(self.directory / "options_query.npz")
            self.oq_index = {str(x): k for k, x in enumerate(z["labels"])}
            self.oq = z["matrix"]
        return self.oq[self.oq_index[clean_label(label)]]

    def question_query(self, evidence_id):
        return self.qq[self.qq_index[evidence_id]]

    def option(self, label):
        return self.o[self.o_index[clean_label(label)]]


def encode_cases(cases, out_path, device="cpu"):
    """Embeddings de las preguntas de los casos (como «query: » y como «passage: ») y de sus opciones («passage: »)."""
    from ..embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(MODEL, device=device)
    texts = [" ".join(c["question"].split()) for c in cases]
    labels = sorted({clean_label(o) for c in cases for o in c["options"]})
    np.savez_compressed(
        out_path,
        ids=np.array([c["evidence_id"] for c in cases]),
        as_query=embedder.encode_queries(texts).astype(np.float32),
        as_passage=embedder.encode_passages(texts).astype(np.float32),
        labels=np.array(labels),
        options=embedder.encode_passages(labels).astype(np.float32),
    )
