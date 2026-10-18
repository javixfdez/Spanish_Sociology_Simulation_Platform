"""Segunda confirmación PRERREGISTRADA: prior de 2 opciones.

Fijado en git antes de ejecutar:
- Alternativa única: «2 opciones → MLP solo; 3 o más → media 50/50 KNN actual + MLP» frente a la base «media 50/50 KNN actual + MLP en
  todas». Sin rejillas, sin pesos estimados, sin otras variantes. El MLP es el mismo (PCA 64, una capa de 64, un modelo por mes de
  trabajo de campo con el corte más restrictivo y sin el estudio del caso).
- Conjunto NUEVO: preguntas "nuevas" (sin pregunta pasada con similitud léxica ≥ 0,90, escala equivalente y mismo ítem) de estudios
  del CIS no reservados con trabajo de campo entre 2024-07-01 y 2024-12-31, de 2 a 7 opciones, sin los estudios de los casos dev, de
  dev_grande ni de la primera confirmación.
- Criterio de adopción: en las preguntas de 2 opciones, mejora ≥ 0,5 puntos con el IC 95 % bootstrap por estudio por debajo de cero; y
  en el total de preguntas nuevas, que no empeore (el IC por estudio incluye el cero o queda por debajo).
- Referencias (no son variantes): reparto uniforme y KNN actual.
- Descriptivo previo en dev_grande (outputs/prior_v2/two_options_dev_grande.txt): allí el MLP solo NO mejora a la media en 2 opciones
  (Δ +0,79 [−0,66; +5,36]) y el KNN SÍ gana al uniforme (27,8 frente a 32,1); la premisa de que el KNN pierde con el uniforme solo se
  observó en la primera confirmación (28,6 frente a 22,8).
"""

import time

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import bigdev, bigdev_eval, confirm, data, learned
from .embcache import EmbCache
from .task1 import EVIDENCE, OUT, TABLE

WINDOW = ("2024-07-01", "2025-01-01")  # trabajo de campo hasta el 31 de diciembre de 2024
MIN_GAIN = 0.005


def alternative(row):
    return row["mlp"] if row["n_options"] == 2 else confirm.average(row)


def banned_studies():
    previous = {r["study"] for r in data.load(OUT / "confirm_preds.pkl")}
    return confirm.excluded_studies() | previous


def main():
    t0 = time.time()
    emb = EmbCache(OUT / "emb_cache")
    repo = EvidenceRepository(EVIDENCE)
    table = load_national_table(TABLE)
    banned = banned_studies()
    cases = [c for c in bigdev.build_cases(repo, table, emb, WINDOW) if str(c["estudio_id"]) not in banned]
    data.assert_no_reserved([c["evidence_id"] for c in cases], "confirmación 2")
    assert not {str(c["estudio_id"]) for c in cases} & banned, "entra un estudio ya usado"
    assert all(WINDOW[0] <= c["fieldwork_date"] < WINDOW[1] for c in cases)
    vectors = {c["evidence_id"]: {"query": emb.question_query(c["evidence_id"])} for c in cases}
    full_text = {e.evidence_id: " ".join((e.question_text or "").split()) for e in repo.all()}
    collected = data.collect(cases, repo, table, emb, vectors, strict=False, full_text=full_text)
    questions = learned.train_questions(repo, table, emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento")
    predictions = bigdev.mlp_predictions(cases, questions, emb)
    rows = bigdev.make_rows(collected, predictions)
    data.save(OUT / "confirm2_preds.pkl", rows)
    new = [r for r in rows if not r["repeated"]]
    studies = [r["study"] for r in new]
    two = np.array([r["n_options"] == 2 for r in new])

    def errors(fn):
        return np.array([bigdev_eval.mae(fn(r), r["observed"]) for r in new])

    e_base, e_alt = errors(confirm.average), errors(alternative)
    e_unif, e_knn = errors(lambda r: r["uniform"]), errors(lambda r: r["knn_now"])
    sub = [r["study"] for r, k in zip(new, two, strict=True) if k]
    c2 = bigdev_eval.compare(e_base[two], e_alt[two], sub)
    ct = bigdev_eval.compare(e_base, e_alt, studies)
    ok2 = c2["delta"] <= -MIN_GAIN and c2["ci"][1] < 0
    oktot = ct["ci"][0] <= 0
    lines = [
        "# Segunda confirmación prerregistrada (2024-07-01 a 2024-12-31)",
        "",
        f"Preguntas de la ventana tras excluir estudios previos: {len(rows)} de {len({r['study'] for r in rows})} estudios; nuevas: "
        f"**{len(new)}** de **{len(set(studies))}** estudios, de las que **{int(two.sum())}** de 2 opciones ({len(set(sub))} estudios).",
        "Nuevas por nº de opciones: " + ", ".join(f"{k}: {sum(1 for r in new if r['n_options'] == k)}" for k in range(2, 8)),
        "",
        "## Preguntas de 2 opciones",
        "| Configuración | MAE (pp) | Δ frente a la base [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
        bigdev_eval.line("Reparto uniforme (referencia)", bigdev_eval.compare(e_base[two], e_unif[two], sub)),
        bigdev_eval.line("KNN actual (referencia)", bigdev_eval.compare(e_base[two], e_knn[two], sub)),
        f"| Base: media 50/50 KNN+MLP | {100 * e_base[two].mean():.2f} | — | — | — |",
        bigdev_eval.line("**Alternativa: MLP solo**", c2),
        "",
        "## Total de preguntas nuevas",
        "| Configuración | MAE (pp) | Δ frente a la base [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
        bigdev_eval.line("Reparto uniforme (referencia)", bigdev_eval.compare(e_base, e_unif, studies)),
        bigdev_eval.line("KNN actual (referencia)", bigdev_eval.compare(e_base, e_knn, studies)),
        f"| Base: media 50/50 KNN+MLP en todas | {100 * e_base.mean():.2f} | — | — | — |",
        bigdev_eval.line("**Alternativa: 2 opciones → MLP; 3+ → media**", ct),
        "",
        f"**Criterio, 2 opciones** (mejora ≥0,5 pp y IC superior <0): Δ={100 * c2['delta']:+.2f}, IC superior={100 * c2['ci'][1]:+.2f} → "
        + ("cumple" if ok2 else "NO cumple"),
        f"**Criterio, total** (el IC por estudio no queda por encima de cero): IC inferior={100 * ct['ci'][0]:+.2f} → " + ("cumple" if oktot else "NO cumple"),
        "",
        "## " + ("SE ADOPTA la alternativa" if ok2 and oktot else "NO se adopta: se queda la media 50/50 en todas"),
        "",
        f"Tiempo: {time.time() - t0:.0f} s.",
    ]
    text = "\n".join(lines)
    (OUT / "confirm2_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
