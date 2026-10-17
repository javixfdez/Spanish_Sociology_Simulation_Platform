"""Contexto de actualidad con corte temporal: notas y documentos de contexto fechados y una instantánea del
último barómetro del CIS anterior al trabajo de campo. Todo lo posterior al corte se descarta."""

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CUTOFF = date(2026, 6, 30)
_MONTHS = {
    m: i + 1
    for i, m in enumerate("enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split())
}
_DATE = re.compile(r"\b(\d{1,2}) de (" + "|".join(_MONTHS) + r") de (\d{4})\b")
_MAX_PDF_PAGES = 30
SNIPPET_CHARS = 300


def infer_date(name, text):
    """Fecha de publicación: la primera fecha completa del texto (notas del INE) o el cierre del año del nombre."""
    match = _DATE.search(text[:3000])
    if match:
        return date(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))
    year = re.search(r"(20\d{2})", name)
    # Sin fecha explícita se asume el último día del año del nombre (hipótesis conservadora para el corte).
    return date(int(year.group(1)), 12, 31) if year else None


def chunk(text, size=SNIPPET_CHARS):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > size and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [c[: size + 120] for c in chunks if len(c) >= 40]


def read_document(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:_MAX_PDF_PAGES])
    return path.read_text(encoding="utf-8", errors="ignore")


def build_context_index(raw_dir):
    """Lista de fragmentos {doc, topic, published, text} de raw/*/contexto/ (omite lo posterior al 30/06/2026)."""
    entries, skipped = [], []
    for path in sorted(Path(raw_dir).glob("*/contexto/*")):
        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        text = read_document(path)
        published = infer_date(path.stem, text)
        if published is None or published > CUTOFF:
            skipped.append(path.name)
            continue
        body = text.split("Principales resultados", 1)[-1] if path.suffix == ".txt" else text
        topic = path.parent.parent.name
        for fragment in chunk(body):
            entries.append({"doc": path.name, "topic": topic, "published": published.isoformat(), "text": fragment})
    return {"entries": entries, "skipped": skipped}


class ContextStore:
    def __init__(self, entries, snapshot_dir=None):
        self.entries = entries
        self.snapshot_dir = snapshot_dir
        self._studies = None

    @classmethod
    def load(cls, path, snapshot_dir=None):
        path = Path(path)
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []
        return cls(entries, snapshot_dir)

    def snippets(self, question, topic, before, k=3):
        """Hasta k fragmentos relevantes publicados estrictamente antes de ``before`` (tema propio o general)."""
        eligible = [
            e
            for e in self.entries
            if e["published"] < before and (e["topic"] in (topic, "general") or not topic)
        ]
        if not eligible:
            return []
        try:
            vectorizer = TfidfVectorizer(strip_accents="unicode")
            matrix = vectorizer.fit_transform([e["text"] for e in eligible])
            scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
        except ValueError:
            return []
        order = [i for i in scores.argsort()[::-1] if scores[i] > 0][:k]
        return [
            {"source": eligible[i]["doc"], "date": eligible[i]["published"], "text": eligible[i]["text"][:SNIPPET_CHARS]}
            for i in order
        ]

    def snapshot(self, anchors, before):
        """Instantánea del último barómetro del CIS con fin de campo anterior a ``before``."""
        if anchors is None or self.snapshot_dir is None:
            return None
        directory = Path(self.snapshot_dir)
        if self._studies is None:
            self._studies = pd.read_parquet(directory / "cis_studies.parquet")
        studies = self._studies
        pool = studies[
            studies.titulo.str.contains("baromet", case=False, na=False)
            & ~studies.reservado_examen
            & (studies.fieldwork_end < before)
        ]
        if pool.empty:
            return None
        latest = pool.sort_values("fieldwork_end").iloc[-1]
        parts = []
        for variable, label in (("PESPANNA1", "principal problema de España"), ("ECOESP", "situación económica de España")):
            try:
                dist, n = anchors.national(latest.estudio_id, f"CIS{latest.estudio_id}_{variable}", "single")
            except Exception:
                continue
            if not n:
                continue
            ns = {k: v for k, v in dist.items() if k.lower().strip(". ") not in {"n.s", "n.c", "ns", "nc"}}
            mass = sum(ns.values())
            if mass <= 0:
                continue
            top = sorted(ns.items(), key=lambda kv: -kv[1])[:3]
            parts.append(f"{label}: " + ", ".join(f"{k} {v / mass:.0%}" for k, v in top))
        if not parts:
            return None
        return {
            "source": f"Barómetro CIS {latest.estudio_id}",
            "date": latest.fieldwork_end,
            "text": f"Campo del {latest.fieldwork_start} al {latest.fieldwork_end}; " + "; ".join(parts),
        }

    def build(self, question, topic, before, anchors=None, k=3):
        items = []
        snap = self.snapshot(anchors, before)
        if snap:
            items.append(snap)
        return items + self.snippets(question, topic, before, k)
