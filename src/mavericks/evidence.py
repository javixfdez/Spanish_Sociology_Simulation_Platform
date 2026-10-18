import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import Evidence
from .topics import Topic, normalize_topic


def read_records(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, list) else [data]
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return [
            {
                "source": str(path),
                "question_summary": text or None,
                "metadata": {"document_type": "unstructured", "requires_ocr": not bool(text.strip())},
            }
        ]
    readers = {
        ".csv": lambda path: pd.read_csv(
            path, dtype={"estudio_id": str, "respondent_uid": str, "question_uid": str}
        ),
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    if suffix == ".sav":
        raise ValueError(
            "Los .sav del CIS no se ingieren directamente (un registro por entrevistado no es evidencia): "
            "use el comando etl-cis y luego ingest-surveys sobre cis_questions.parquet"
        )
    if suffix in readers:
        frame = readers[suffix](path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
    return json.loads(frame.to_json(orient="records", date_format="iso"))


RETRIEVERS = ("tfidf", "embed", "hybrid")
RRF_K = 60


def source_family(e):
    """cis | eurobarometro | ess: permite activar o desactivar familias de fuentes sin reindexar."""
    return "cis" if str(e.source or "").upper().startswith("CIS") else str(e.source or "").lower()


def evidence_text(e):
    """Texto que se indexa: enunciado literal más opciones (sin N.S./N.C.)."""
    options = "; ".join((e.options or [])[:8])
    return f"{e.question_text or e.question_summary or ''} | {options}"[:900]


class EvidenceRepository:
    """Almacén de evidencia con recuperación TF-IDF, por embeddings multilingües o híbrida (RRF).

    Las tres rutas comparten el filtrado previo (excluded_ids, excluded_texts, corte temporal estricto y
    exclusión de reservados): solo cambia cómo se ordenan los candidatos que lo superan.
    """

    def __init__(self, path="data/processed/evidence.sqlite", semantic_index=None, embedder=None, retriever="tfidf"):
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever debe ser uno de {RETRIEVERS}")
        if retriever != "tfidf" and embedder is None:
            raise ValueError("Los recuperadores embed/hybrid necesitan un embedder")
        self.semantic_index = semantic_index
        self.embedder = embedder
        self.retriever = retriever
        self._cache = None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def ingest(self, path, include_reserved=False):
        """Ingiere registros. Los estudios reservados para el examen (reservado_examen) se omiten
        salvo ``include_reserved``: no pueden alimentar el RAG, solo el backtesting."""
        count = 0
        self.skipped_reserved = 0
        with sqlite3.connect(self.path) as db:
            for record in read_records(path):
                # Explicit aliases only; never infer respondent joins or survey percentages.
                record = dict(record)
                if record.get("reservado_examen") and not include_reserved:
                    self.skipped_reserved += 1
                    continue
                for source, target in [("question_uid", "evidence_id"), ("estudio_id", "study_id")]:
                    if source in record and target not in record:
                        record[target] = record[source]
                record.setdefault("source", str(path))
                record.setdefault("original_file", str(path))
                record.setdefault(
                    "evidence_id",
                    "FILE_"
                    + hashlib.sha256(
                        json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()[:24],
                )
                extras = {k: v for k, v in record.items() if k not in Evidence.model_fields}
                record["metadata"] = {**record.get("metadata", {}), "original_fields": extras}
                raw_topic = record.get("topic")
                topic = normalize_topic(
                    raw_topic, " ".join(filter(None, [record.get("question_text"), record.get("question_summary")]))
                )
                if raw_topic and topic != raw_topic:
                    record["metadata"]["original_topic"] = raw_topic
                record["topic"] = topic or Topic.GENERAL.value
                evidence = Evidence.model_validate(record)
                db.execute(
                    "INSERT INTO evidence VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                    (evidence.evidence_id, evidence.model_dump_json()),
                )
                count += 1
        if self.semantic_index is not None:
            self.semantic_index.upsert(self.all())
        return count

    def attach_microdata(
        self,
        respondents_path,
        responses_path,
        weight="peso",
        option_column="answer",
        question_type="single",
        demographics=("grupo_edad", "sexo", "nivel_estudios"),
    ):
        respondents = pd.DataFrame(read_records(respondents_path))
        responses = pd.DataFrame(read_records(responses_path))
        updated = 0
        answered = set(responses.question_uid)
        with sqlite3.connect(self.path) as db:
            for evidence in self.all():
                if evidence.evidence_id not in answered:
                    continue
                stats = {
                    "national": survey_distribution(
                        respondents,
                        responses,
                        evidence.evidence_id,
                        weight=weight,
                        option_column=option_column,
                        question_type=question_type,
                    )["national"]
                }
                for column in demographics:
                    if column in respondents:
                        stats[column] = survey_distribution(
                            respondents,
                            responses,
                            evidence.evidence_id,
                            weight=weight,
                            option_column=option_column,
                            question_type=question_type,
                            group_by=column,
                        )
                evidence.metadata["empirical_statistics"] = stats
                evidence.metadata["microdata_files"] = [str(respondents_path), str(responses_path)]
                evidence.metadata["question_type"] = question_type
                evidence.sample_size = stats["national"]["n_respondents"]
                db.execute(
                    "UPDATE evidence SET payload=? WHERE id=?",
                    (evidence.model_dump_json(), evidence.evidence_id),
                )
                updated += 1
        return updated

    def _index(self):
        """Registros y matriz TF-IDF en memoria; se recalculan solo si cambia el fichero SQLite."""
        stat = Path(self.path).stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._cache is None or self._cache["signature"] != signature:
            with sqlite3.connect(self.path) as db:
                records = [
                    Evidence.model_validate_json(row[0])
                    for row in db.execute("SELECT payload FROM evidence ORDER BY id")
                ]
            texts = [" ".join(filter(None, [e.question_text, e.question_summary, e.topic])) for e in records]
            vectorizer = matrix = None
            try:
                vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2))
                matrix = vectorizer.fit_transform(texts)
            except ValueError:  # vocabulario vacío
                vectorizer = matrix = None
            self._cache = {
                "signature": signature,
                "records": records,
                "vectorizer": vectorizer,
                "matrix": matrix,
                "position": {e.evidence_id: i for i, e in enumerate(records)},
                "embeddings": None,
            }
        if self.embedder is not None and self.retriever != "tfidf" and self._cache["embeddings"] is None:
            self._cache["embeddings"] = self._embeddings(self._cache["records"])
        return self._cache

    def _embeddings(self, records):
        """Matriz de embeddings normalizados con caché incremental en disco (por modelo y por texto indexado)."""
        texts = [evidence_text(e) for e in records]
        model = getattr(self.embedder, "name", "embedder")
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        safe = "".join(c if c.isalnum() else "_" for c in model)[:60]
        cache = Path(self.path).with_suffix(f".emb-{safe}.npz")
        known = {}
        if cache.exists():
            with np.load(cache, allow_pickle=False) as stored:
                known = dict(zip(stored["keys"].tolist(), stored["vectors"]))
        missing = [i for i, k in enumerate(keys) if k not in known]
        if missing:
            fresh = np.asarray(self.embedder.encode_passages([texts[i] for i in missing]), dtype=np.float32)
            fresh = fresh / np.maximum(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-9)
            known.update({keys[i]: fresh[j] for j, i in enumerate(missing)})
            used = sorted(set(keys))
            np.savez(cache, keys=np.array(used), vectors=np.stack([known[k] for k in used]))
        return np.stack([known[k] for k in keys])

    def all(self):
        return list(self._index()["records"])

    def search(
        self,
        query,
        limit=5,
        excluded_ids=(),
        excluded_texts=(),
        cutoff=None,
        target_population=None,
        options=(),
        topic=None,
        topic_filter=True,
        include_reserved=False,
        retriever=None,
        sources=None,
        keep_zero=False,
    ):
        """Búsqueda híbrida. Con ``topic`` se filtra a ese tema y a 'general'; si no hay resultados
        se repite sin filtro y los resultados llevan ``topic_fallback=True``.
        ``keep_zero=True`` conserva también los registros sin ninguna palabra en común con la consulta (con tfidf se
        descartan por defecto): sirve para recuperar sobre TODO el almacén ya filtrado y reordenar con otro criterio."""

        def normalize(t):
            return " ".join((t or "").casefold().split())

        excluded = {normalize(t) for t in excluded_texts}
        records = [
            e
            for e in self.all()
            if e.evidence_id not in excluded_ids
            and normalize(e.question_text) not in excluded
            and not set(e.metadata.get("equivalent_ids", [])) & set(excluded_ids)
            and (include_reserved or not e.metadata.get("original_fields", {}).get("reservado_examen"))
            and (not sources or source_family(e) in sources)
        ]
        if cutoff:
            # Corte estricto: la evidencia debe haber terminado su trabajo de campo (fieldwork_date, fin)
            # antes de la fecha de corte. Las fechas desconocidas se excluyen en lugar de darse por válidas.
            records = [e for e in records if e.fieldwork_date and e.fieldwork_date[:10] < cutoff]
        if not records:
            return []
        as_of = date.fromisoformat(cutoff[:10]) if cutoff else date.today()
        mode = retriever or self.retriever
        topic = normalize_topic(topic) if topic else None
        general = Topic.GENERAL.value
        if topic and topic != general and topic_filter:
            narrowed = [e for e in records if (e.topic or general) in (topic, general)]
            hits = self._rank(narrowed, query, limit, target_population, options, topic, as_of, mode, keep_zero) if narrowed else []
            if hits:
                return hits
            hits = self._rank(records, query, limit, target_population, options, topic, as_of, mode, keep_zero)
            return [{**h, "topic_fallback": True} for h in hits]
        return self._rank(records, query, limit, target_population, options, topic, as_of, mode, keep_zero)

    def _rank(self, records, query, limit, target_population, options, topic, as_of, mode="tfidf", keep_zero=False):
        index = self._index()
        rows = [index["position"][e.evidence_id] for e in records]
        if index["matrix"] is None:
            tfidf = np.zeros(len(records))
        else:
            tfidf = cosine_similarity(index["vectorizer"].transform([query]), index["matrix"][rows])[0]
        embedded = None
        if mode != "tfidf":
            q = np.asarray(self.embedder.encode_queries([query]), dtype=np.float32)
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
            embedded = (index["embeddings"][rows] @ q[0]).astype(float)
        if mode == "tfidf":
            similarities = tfidf
        elif mode == "embed":
            low, high = float(embedded.min()), float(embedded.max())
            similarities = (embedded - low) / (high - low) if high > low else np.ones_like(embedded)
        else:  # hybrid: fusión por rango recíproco de TF-IDF y embeddings
            fused = np.zeros(len(records))
            for values in (tfidf, embedded):
                order = np.argsort(-values, kind="stable")
                ranks = np.empty(len(values), dtype=int)
                ranks[order] = np.arange(len(values))
                fused += 1.0 / (RRF_K + ranks + 1)
            similarities = fused / fused.max() if fused.max() > 0 else fused
        semantic_ranks = {}
        if self.semantic_index is not None:
            hits = self.semantic_index.search(query, limit=max(limit * 4, 20))
            if hits:
                semantic_ranks = {e_id: rank for rank, e_id in enumerate(hits["ids"][0])}
        results = []
        for position, (e, similarity) in enumerate(zip(records, similarities)):
            if mode == "tfidf" and similarity <= 0 and not keep_zero and e.evidence_id not in semantic_ranks:
                continue
            universe = float(target_population is not None and e.target_population == target_population)
            option_match = len(set(options) & set(e.options)) / max(1, len(set(options) | set(e.options)))
            thematic = 0.0
            if topic:
                thematic = 1.0 if e.topic == topic else 0.3 if e.topic == Topic.GENERAL.value else 0.0
            recency = 0.0
            if e.fieldwork_date:
                try:
                    recency = 1 / (
                        1 + max(0, (as_of - date.fromisoformat(e.fieldwork_date[:10])).days) / 365
                    )
                except ValueError:
                    pass
            score = (
                0.65 * float(similarity)
                + 0.1 * universe
                + 0.1 * option_match
                + 0.1 * thematic
                + 0.05 * recency
            )
            if e.evidence_id in semantic_ranks:
                score = 0.7 * score + 0.3 / (1 + semantic_ranks[e.evidence_id])
            hit = {"evidence": e.model_dump(), "retrieval_score": score, "cosine_similarity": float(tfidf[position])}
            if embedded is not None:
                hit["embedding_similarity"] = float(embedded[position])
            results.append(hit)
        return sorted(results, key=lambda r: (-r["retrieval_score"], r["evidence"]["evidence_id"]))[:limit]


class SemanticIndex:
    """Optional persistent multilingual index. Only accepts an already downloaded local model."""

    def __init__(self, model_path, persist_path):
        if not Path(model_path).is_dir():
            raise ValueError("Se requiere un modelo de embeddings local; no se descarga automáticamente")
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), local_files_only=True)
        self.collection = chromadb.PersistentClient(path=str(persist_path)).get_or_create_collection(
            "evidence"
        )

    def upsert(self, records):
        if not records:
            return
        texts = [e.question_text or e.question_summary or "" for e in records]
        self.collection.upsert(
            ids=[e.evidence_id for e in records],
            documents=texts,
            embeddings=self.model.encode(texts).tolist(),
        )

    def search(self, query, limit=5):
        count = self.collection.count()
        if not count:
            return []
        return self.collection.query(
            query_embeddings=self.model.encode([query]).tolist(), n_results=min(limit, count)
        )


def survey_distribution(
    respondents,
    responses,
    question_uid,
    weight="peso",
    group_by=None,
    question_type="single",
    option_column="answer",
    selected_column="selected",
):
    keys = ["estudio_id", "respondent_uid"]
    required_r = set(keys + [weight] + ([group_by] if group_by else []))
    if not required_r <= set(respondents) or not set(keys + ["question_uid", option_column]) <= set(
        responses
    ):
        raise ValueError("Contrato CIS incompleto; indique columnas explícitas")
    if respondents.duplicated(keys).any():
        raise ValueError("Entrevistados duplicados dentro de un estudio")
    selected = responses[responses.question_uid == question_uid]
    if selected.empty:
        raise ValueError("Pregunta no encontrada")
    if selected.estudio_id.nunique() != 1:
        raise ValueError("question_uid ambiguo entre estudios")
    if question_type == "single" and selected.duplicated(keys).any():
        raise ValueError("Respuesta única duplicada")
    if question_type == "multiple":
        if selected_column not in selected or selected.duplicated(keys + [option_column]).any():
            raise ValueError("Respuesta múltiple requiere selected y una fila por persona/opción")
        if not selected[selected_column].isin([0, 1, False, True]).all():
            raise ValueError("selected debe ser booleano")
    if question_type == "multiple":
        option_sets = selected.groupby(keys)[option_column].apply(lambda values: frozenset(values))
        if len(set(option_sets)) != 1:
            raise ValueError(
                "Respuesta múltiple requiere todas las opciones, incluso selected=0, por entrevistado"
            )
    merged = selected.merge(respondents, on=keys, how="left", validate="many_to_one", indicator=True)
    if (merged._merge != "both").any():
        raise ValueError("Respuestas sin entrevistado del mismo estudio")
    weights = pd.to_numeric(merged[weight], errors="raise")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Peso muestral inválido")
    if merged[option_column].isna().any():
        raise ValueError("Codifique NS/NC explícitamente; hay respuestas ausentes")
    merged[weight] = weights
    groups = [("national", merged)] if not group_by else merged.groupby(group_by, dropna=False)
    result = {}
    for key, group in groups:
        individuals = group.drop_duplicates(keys)
        denominator = individuals[weight].sum()
        numerator = group if question_type == "single" else group[group[selected_column].astype(bool)]
        counts = numerator.groupby(option_column, dropna=False)[weight].sum()
        result[str(key)] = {
            "weighted_frequencies": {str(k): float(v) for k, v in counts.items()},
            "distribution": {str(k): float(v / denominator) for k, v in counts.items()},
            "n_respondents": len(individuals),
            "weight_sum": float(denominator),
            "n_eff": float(denominator**2 / (individuals[weight] ** 2).sum()),
            "source": {
                "question_uid": question_uid,
                "estudio_id": str(group.estudio_id.iloc[0]),
                "weight_column": weight,
            },
            "warnings": ["n_eff es diagnóstico de Kish, no precisión bajo diseño complejo"]
            + (["Muestra de segmento insuficiente"] if len(individuals) < 30 else []),
        }
    return result
