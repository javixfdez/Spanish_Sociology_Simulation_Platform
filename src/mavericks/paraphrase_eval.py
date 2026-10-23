"""Calibración del umbral de «¿Quizás querías decir…?» (``python -m mavericks.paraphrase_eval``).

Positivas: 30 paráfrasis escritas a mano de preguntas reales del CIS (``tests/data/parafrasis_calibracion.json``) y las 6 de la batería
(``subtipo: parafrasis``). Negativas: las 40 preguntas «de predicción» de la batería (no existen en el CIS) y 12 preguntas cercanas
pero distintas. Una sugerencia es correcta si la pregunta propuesta es la misma variable del CIS (misma serie) o contiene el enunciado de la pregunta original
(la misma pregunta se repite con otros identificadores en distintos barómetros).
"""

import json
import sys
from pathlib import Path

from . import battery
from . import service as svc
from .anchor import series_key
from .paraphrase import ParaphraseIndex

DATA = Path("tests/data/parafrasis_calibracion.json")
THRESHOLDS = (0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)


def build_sets(index):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    positives = []
    for item in data["positivas"]:
        fragments = [f.strip().lower() for f in item["fragmento"].split("|")]
        matches = [e for e, t in zip(index.items, index.texts, strict=True) if any(f in t.lower() for f in fragments)]
        if not matches:
            raise SystemExit(f"No hay pregunta del CIS con el fragmento «{item['fragmento']}»")
        positives.append((item["pregunta"], (matches, fragments), "mía"))
    cases = battery.load_cases()
    by_id = {c["id"]: c for c in cases}
    for c in cases:
        if c.get("subtipo") == "parafrasis":
            base = battery.resolve(by_id[c["base"]], by_id)["pregunta"]
            target = max(zip(index.items, index.texts, strict=True), key=lambda p: _overlap(p[1], base))[0]
            positives.append((c["pregunta"], ([target], None), "batería"))
    negatives = [(battery.resolve(c, by_id)["pregunta"], "predicción batería") for c in cases if c["categoria"] == "prediccion"]
    negatives += [(q, "cercana distinta") for q in data["cercanas_distintas"]]
    return positives, negatives


def _overlap(a, b):
    from .anchor import question_similarity

    return question_similarity(a, b)


def is_correct(best, target):
    """``target`` = (preguntas del CIS aceptables, fragmento de enunciado o None)."""
    accepted, fragments = target
    text = " ".join(best.question_text.split()).lower()
    if any(f in text for f in fragments or []):
        return True
    if not fragments and any(_overlap(text, " ".join(a.question_text.split()).lower()) >= 0.8 for a in accepted):  # paráfrasis de la batería
        return True
    for a in accepted:
        ka, kb = series_key(a.evidence_id), series_key(best.evidence_id)
        if a.evidence_id == best.evidence_id or (ka is not None and ka == kb):
            return True
    return False


def evaluate(index, positives, negatives):
    pos = []
    for question, target, origin in positives:
        score, best = index.ranking(question, limit=1)[0]
        pos.append((score, is_correct(best, target), origin, question, best.evidence_id, target[0][0].evidence_id))
    neg = []
    for q, origin in negatives:
        score, best = index.ranking(q, limit=1)[0]
        neg.append((score, origin, q, " ".join(best.question_text.split())[:90]))
    return pos, neg


def table(pos, neg):
    rows = []
    for t in THRESHOLDS:
        shown_pos = [p for p in pos if p[0] >= t]
        correct = [p for p in shown_pos if p[1]]
        false_neg = [n for n in neg if n[0] >= t]
        shown = len(shown_pos) + len(false_neg)
        rows.append({
            "umbral": t, "sugerencias": shown, "correctas": len(correct), "incorrectas": shown - len(correct),
            "precision": round(len(correct) / shown, 3) if shown else None,
            "cobertura": round(len(correct) / len(pos), 3), "negativas_con_sugerencia": len(false_neg),
        })
    return rows


def main():
    res = svc.load_resources()
    index = ParaphraseIndex(res.evidence.all(), svc.get_embedder())
    positives, negatives = build_sets(index)
    pos, neg = evaluate(index, positives, negatives)
    print(f"positivas: {len(pos)} · negativas: {len(neg)} · preguntas indexadas: {len(index.items)}")
    print(f"{'umbral':>7} {'sugeridas':>10} {'correctas':>10} {'incorrectas':>12} {'precisión':>10} {'cobertura':>10} {'neg. con sugerencia':>20}")
    for r in table(pos, neg):
        print(f"{r['umbral']:>7} {r['sugerencias']:>10} {r['correctas']:>10} {r['incorrectas']:>12} {str(r['precision']):>10} {r['cobertura']:>10} {r['negativas_con_sugerencia']:>20}")
    if "--detalle" in sys.argv:
        print("\nPositivas (coseno, ¿misma variable?):")
        for score, ok, origin, q, got, want in sorted(pos, reverse=True):
            print(f"  {score:.3f} {'OK ' if ok else 'MAL'} [{origin}] {q[:70]} -> {got} (esperada {want})")
        print("\nNegativas (coseno máximo):")
        for score, origin, q, got in sorted(neg, reverse=True)[:15]:
            print(f"  {score:.3f} [{origin}] {q[:70]} -> {got}")


if __name__ == "__main__":
    main()
