"""dev_grande: todas las preguntas del CIS (2-7 opciones) de estudios NO reservados con trabajo de campo entre 2025-07-01 y la
última fecha anterior a 2026-04-01, cada una predicha solo con lo anterior a su fecha y sin su propio estudio.

``python -m mavericks.prior_v2.bigdev`` recupera las candidatas, entrena un MLP por mes y guarda las predicciones.
"""

import time
from collections import defaultdict

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import data, knn, learned
from .embcache import EmbCache, clean_label
from .task1 import EVIDENCE, OUT, TABLE

WINDOW = ("2025-07-01", "2026-04-01")
CANDIDATE = ("mix", 3, 4)
MLP = ("mlp (64,) pca64", "mlp", (64,), 64)  # la que eligió la validación anidada en la Tarea 2: no se reajusta


def build_cases(repository, table, emb, window=WINDOW):
    """Casos de dev_grande con la distribución observada nacional (sin N.S./N.C., renormalizada)."""
    topics = {e.evidence_id: e for e in repository.all()}
    questions = [t for t in learned.train_questions(repository, table, emb) if window[0] <= t.date < window[1]]
    data.assert_no_reserved([t.evidence_id for t in questions], "dev_grande")
    cases = []
    for t in sorted(questions, key=lambda t: t.evidence_id):
        e = topics[t.evidence_id]
        options = [clean_label(o) for o in t.options]
        if len(set(options)) != len(options):
            continue
        cases.append(
            {
                "evidence_id": t.evidence_id, "estudio_id": t.study, "topic": e.topic, "fieldwork_date": t.date,
                "question": " ".join(e.question_text.split()), "options": options,
                "observed_distribution": dict(zip(options, t.props, strict=True)), "split": "dev_grande",
            }
        )
    return cases


def mlp_predictions(cases, questions, emb):
    """Un MLP por mes de trabajo de campo, con el corte más restrictivo del mes y sin los estudios de sus casos."""
    by_month = defaultdict(list)
    for c in cases:
        by_month[c["fieldwork_date"][:7]].append(c)
    out = {}
    for month, group in sorted(by_month.items()):
        cutoff = min(c["fieldwork_date"] for c in group)
        excluded = {str(c["estudio_id"]) for c in group}
        train = [t for t in questions if t.date < cutoff and t.study not in excluded]
        assert all(t.date < cutoff and t.study not in excluded for t in train)
        assert not {c["evidence_id"] for c in group} & {t.evidence_id for t in train}
        q, o, ex, y = learned.build_training_matrix(train, emb)
        uq = np.stack([emb.question(t.evidence_id) for t in train])
        labels = sorted({clean_label(x) for t in train for x in t.options})
        uo = np.stack([emb.option_query(x) for x in labels])
        featurizer = learned.Featurizer(MLP[3])
        x = featurizer.fit(uq, uo, q, o, ex)
        model = learned.make_model(MLP[1], MLP[2]).fit(x, y)
        for c in group:
            ov = np.stack([emb.option_query(opt) for opt in c["options"]])
            out[c["evidence_id"]] = learned.predict_case(featurizer, model, emb.question(c["evidence_id"]), ov, [False] * len(ov))
        print(f"MLP {month}: {len(group)} casos, corte {cutoff}, {len(train)} preguntas de entrenamiento", flush=True)
    return out


def make_rows(collected, predictions):
    """Predicciones de cada configuración y datos de la pregunta, listos para evaluar."""
    shift = knn.qq_shift(collected)
    rows = []
    for c in collected:
        a = knn.CaseArrays(c, shift)
        sel = a.selected(*knn.CURRENT)
        best_sim = float(a.sim_lex[sel].max()) if sel else 0.0
        rows.append(
            {
                "evidence_id": c["evidence_id"], "study": str(c["estudio_id"]), "topic": c["topic"], "date": c["fieldwork_date"],
                "n_options": len(c["options"]), "repeated": c["repeated"], "observed": a.observed, "best_sim": best_sim,
                "uniform": np.full(len(a.options), 1 / len(a.options)),
                "knn_now": a.prior(*knn.CURRENT), "knn_cand": a.prior(*CANDIDATE), "mlp": predictions[c["evidence_id"]],
            }
        )
    return rows


def main():
    t0 = time.time()
    emb = EmbCache(OUT / "emb_cache")
    repo = EvidenceRepository(EVIDENCE)
    table = load_national_table(TABLE)
    cases = build_cases(repo, table, emb)
    print(f"dev_grande: {len(cases)} preguntas, {len({c['estudio_id'] for c in cases})} estudios", flush=True)
    vectors = {c["evidence_id"]: {"query": emb.question_query(c["evidence_id"])} for c in cases}
    full_text = {e.evidence_id: " ".join((e.question_text or "").split()) for e in repo.all()}
    collected_path = OUT / "bigdev_collected.pkl"
    if collected_path.exists():
        collected = data.load(collected_path)
    else:
        collected = data.collect(
            cases, repo, table, emb, vectors, strict=False, full_text=full_text,
            progress=lambda n, total: print(f"candidatas {n}/{total} ({time.time() - t0:.0f} s)", flush=True) if n % 100 == 0 else None,
        )
        data.save(collected_path, collected)
    questions = learned.train_questions(repo, table, emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento")
    predictions = mlp_predictions(cases, questions, emb)
    rows = make_rows(collected, predictions)
    data.save(OUT / "bigdev_preds.pkl", rows)
    print(f"listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
