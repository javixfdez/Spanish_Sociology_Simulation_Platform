"""Confirmación PRERREGISTRADA (una sola comparación) sobre datos no vistos.

Fijado antes de ejecutar:
- Conjunto: preguntas "nuevas" (sin ninguna pregunta pasada con similitud léxica ≥ 0,90, escala equivalente y mismo ítem) de
  estudios del CIS no reservados con trabajo de campo entre 2025-01-01 y 2025-06-30, de 2 a 7 opciones, SIN ningún estudio de los
  casos dev ni de dev_grande.
- Comparación: «media 50/50 KNN actual + MLP» frente a «KNN actual»; MLP con PCA 64 y una capa de 64 (sin reajustar), un modelo
  por mes de trabajo de campo con el corte más restrictivo y sin el estudio del caso. No se prueba ninguna otra variante.
- Criterio de adopción: mejora ≥ 0,5 puntos con el IC 95 % bootstrap por estudio por debajo de cero.
- Intervalos: cuantiles por grupo (2 frente a 3 o más opciones) calibrados con las nuevas de dev_grande y validados aquí.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import bigdev, bigdev_eval, data, learned, task3
from .embcache import EmbCache
from .task1 import EVIDENCE, OUT, TABLE

WINDOW = ("2025-01-01", "2025-07-01")  # trabajo de campo hasta el 30 de junio de 2025
ADOPT_MIN_GAIN = 0.005  # 0,5 puntos


def excluded_studies():
    """Estudios de los casos dev (65 y 44) y de dev_grande: no pueden entrar en la confirmación."""
    dev = json.loads(Path("data/processed_v2/backtest_cases_dev.json").read_text(encoding="utf-8-sig"))
    hard = json.loads(Path("outputs/ablation/cases_dev_hard.json").read_text(encoding="utf-8-sig"))
    big = data.load(OUT / "bigdev_preds.pkl")
    return {str(c["estudio_id"]) for c in dev + hard} | {r["study"] for r in big}


def average(row):
    return 0.5 * row["knn_now"] + 0.5 * row["mlp"]


def group_of(row):
    return "2 opciones" if row["n_options"] == 2 else "3 o más"


def main():
    t0 = time.time()
    emb = EmbCache(OUT / "emb_cache")
    repo = EvidenceRepository(EVIDENCE)
    table = load_national_table(TABLE)
    banned = excluded_studies()
    cases = [c for c in bigdev.build_cases(repo, table, emb, WINDOW) if str(c["estudio_id"]) not in banned]
    data.assert_no_reserved([c["evidence_id"] for c in cases], "confirmación")
    assert not {str(c["estudio_id"]) for c in cases} & banned, "entra un estudio dev o de dev_grande"
    assert all(WINDOW[0] <= c["fieldwork_date"] < WINDOW[1] for c in cases)
    print(f"candidatas a confirmación: {len(cases)} preguntas, {len({c['estudio_id'] for c in cases})} estudios", flush=True)
    vectors = {c["evidence_id"]: {"query": emb.question_query(c["evidence_id"])} for c in cases}
    full_text = {e.evidence_id: " ".join((e.question_text or "").split()) for e in repo.all()}
    collected = data.collect(cases, repo, table, emb, vectors, strict=False, full_text=full_text)
    questions = learned.train_questions(repo, table, emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento")
    predictions = bigdev.mlp_predictions(cases, questions, emb)
    rows = bigdev.make_rows(collected, predictions)
    data.save(OUT / "confirm_preds.pkl", rows)
    new = [r for r in rows if not r["repeated"]]
    studies = [r["study"] for r in new]

    def errors(fn, subset):
        return np.array([bigdev_eval.mae(fn(r), r["observed"]) for r in subset])

    e_now = errors(lambda r: r["knn_now"], new)
    e_avg = errors(average, new)
    e_unif = errors(lambda r: r["uniform"], new)
    result = bigdev_eval.compare(e_now, e_avg, studies)
    ref_unif = bigdev_eval.compare(e_now, e_unif, studies)
    lines = [
        "# Confirmación prerregistrada (2025-01-01 a 2025-06-30)",
        "",
        f"Preguntas de la ventana tras excluir estudios dev y de dev_grande: {len(rows)} de {len({r['study'] for r in rows})} estudios; "
        f"nuevas: **{len(new)}** de **{len(set(studies))}** estudios; repetidas: {len(rows) - len(new)}.",
        "Nuevas por nº de opciones: " + ", ".join(f"{k}: {sum(1 for r in new if r['n_options'] == k)}" for k in range(2, 8)),
        "",
        "## Comparación única (nuevas)",
        "| Configuración | MAE (pp) | Δ frente al KNN actual [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
        bigdev_eval.line("Reparto uniforme (referencia)", ref_unif),
        f"| KNN actual | {100 * e_now.mean():.2f} | — | — | — |",
        bigdev_eval.line("**Media 50/50 KNN actual + MLP**", result),
    ]
    adopted = result["delta"] <= -ADOPT_MIN_GAIN and result["ci"][1] < 0
    lines += [
        "",
        f"**Criterio** (mejora ≥0,5 pp y límite superior del IC por estudio <0): Δ={100 * result['delta']:+.2f}, "
        f"IC superior={100 * result['ci'][1]:+.2f} → " + ("**SE CONFIRMA**" if adopted else "**NO se confirma**"),
        "",
        "## Desglose descriptivo (no se usa para decidir)",
        "| Grupo | preguntas | uniforme | KNN actual | media KNN+MLP | Δ [IC 95 % por estudio] |",
        "|---|---|---|---|---|---|",
    ]
    for label in ("2 opciones", "3 o más"):
        mask = np.array([group_of(r) == label for r in new])
        sub = [r for r, k in zip(new, mask, strict=True) if k]
        c = bigdev_eval.compare(e_now[mask], e_avg[mask], [r["study"] for r in sub])
        lines.append(
            f"| {label} | {int(mask.sum())} | {100 * e_unif[mask].mean():.2f} | {100 * e_now[mask].mean():.2f} | "
            f"{100 * e_avg[mask].mean():.2f} | {100 * c['delta']:+.2f} [{100 * c['ci'][0]:+.2f}, {100 * c['ci'][1]:+.2f}] |"
        )

    big_new = [r for r in data.load(OUT / "bigdev_preds.pkl") if not r["repeated"]]
    lines += ["", "## Intervalos con calibración separada por nº de opciones (calibrada con las nuevas de dev_grande)"]
    for name, fn in (("media KNN actual + MLP", average), ("KNN actual", lambda r: r["knn_now"])):
        lines += [
            f"### Predictor: {name}",
            "| Nivel | Conjunto | grupo | preguntas | cobertura | anchura media (pp) |",
            "|---|---|---|---|---|---|",
        ]
        for level in (0.9, 0.5):
            loso = task3.leave_one_study_out_by(big_new, fn, group_of, level)
            conf, _ = task3.calibrate_and_test(big_new, new, fn, group_of, level)
            for title, s in (("dev_grande (dejando un estudio fuera)", loso), ("confirmación (datos no vistos)", conf)):
                for k in ("2 opciones", "3 o más", "global"):
                    lines.append(
                        f"| {int(level * 100)} % | {title} | {k} | {s[k]['cases']} | {100 * s[k]['coverage']:.1f} % | {s[k]['width_pp']:.1f} |"
                    )
    lines += ["", f"Tiempo: {time.time() - t0:.0f} s."]
    text = "\n".join(lines)
    (OUT / "confirm_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
