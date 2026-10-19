"""Prior aprendido de producción: MLP (PCA 64, una capa de 64) sobre embeddings e5, entrenado UNA vez con todo el CIS no reservado.

Este modelo solo lo carga el servicio (``service.py``). El backtest y todas las validaciones entrenan por caso con su propio
corte temporal y NUNCA usan este modelo (hay tests que lo vigilan). Se guarda fuera de git, en ``outputs/models/``, con metadatos.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import data, learned
from .embcache import MODEL as E5_MODEL
from .embcache import EmbCache, clean_label

MODEL_DIR = "outputs/models"
MODEL_FILE = "prior_mlp.pkl"
META_FILE = "prior_mlp.json"
PCA_DIMS = 64
HIDDEN = (64,)
SEED = 0


def file_hash(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fit(questions, emb, dims=PCA_DIMS, hidden=HIDDEN, seed=SEED):
    """(featurizer, modelo) entrenados con ``questions``; ``emb`` da los vectores e5 de preguntas y opciones."""
    q, o, ex, y = learned.build_training_matrix(questions, emb)
    unique_q = np.stack([emb.question(t.evidence_id) for t in questions])
    labels = sorted({clean_label(x) for t in questions for x in t.options})
    unique_o = np.stack([emb.option_query(x) for x in labels])
    featurizer = learned.Featurizer(dims, seed)
    x = featurizer.fit(unique_q, unique_o, q, o, ex)
    return featurizer, learned.make_model("mlp", hidden, seed).fit(x, y)


def ensure_embeddings(emb_dir, evidence_path, table_path):
    """Crea la caché de embeddings e5 (preguntas y etiquetas de opción) si no existe: en una instalación nueva no hay nada que reutilizar."""
    from . import embcache

    directory = Path(emb_dir)
    if not (directory / "questions.npz").exists():
        embcache.build(evidence_path, table_path, directory)
    if not (directory / "options_query.npz").exists():
        embcache.build_query_options(evidence_path, table_path, directory)


def train(evidence_path, table_path, emb_dir, out_dir=MODEL_DIR, dims=PCA_DIMS, hidden=HIDDEN, seed=SEED):
    """Entrena con todo el almacén del CIS y guarda modelo + metadatos. Devuelve los metadatos."""
    ensure_embeddings(emb_dir, evidence_path, table_path)
    repository = EvidenceRepository(evidence_path)
    emb = EmbCache(emb_dir)
    questions = learned.train_questions(repository, load_national_table(table_path), emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento de producción")
    if not questions:
        raise ValueError("No hay preguntas con las que entrenar")
    featurizer, model = fit(questions, emb, dims, hidden, seed)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "cutoff_date": max(t.date for t in questions),
        "n_questions": len(questions),
        "n_rows": int(sum(len(t.options) for t in questions)),
        "e5": {"model": E5_MODEL, "question_prefix": "passage: ", "option_prefix": "query: "},
        "seed": seed,
        "pca_dims": dims,
        "hidden": list(hidden),
        "max_options": learned.MAX_OPTIONS,
        "store_hash": file_hash(evidence_path),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / MODEL_FILE, "wb") as f:
        pickle.dump({"featurizer": featurizer, "model": model}, f)
    (out / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


class ProductionPrior:
    def __init__(self, featurizer, model, meta):
        self.featurizer, self.model, self.meta = featurizer, model, meta

    @classmethod
    def load(cls, directory=MODEL_DIR):
        directory = Path(directory)
        with open(directory / MODEL_FILE, "rb") as f:
            parts = pickle.load(f)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(parts["featurizer"], parts["model"], meta)

    def supports(self, options):
        return 2 <= len(options) <= self.meta.get("max_options", learned.MAX_OPTIONS)

    def predict(self, question, options, embedder):
        """Distribución sobre ``options`` (softmax de las salidas); la pregunta se codifica como «passage: » y las opciones como «query: »."""
        labels = [clean_label(o) for o in options]
        q_vec = np.asarray(embedder.encode_passages([" ".join(question.split())]))[0]
        o_vecs = np.asarray(embedder.encode_queries(labels))
        return learned.predict_case(self.featurizer, self.model, q_vec, o_vecs, [False] * len(options))
