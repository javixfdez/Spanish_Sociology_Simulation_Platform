"""Evaluación final ÚNICA sobre los 41 casos oficiales difíciles, con la configuración congelada y los DOS modos.

NO se ha ejecutado. Pasos (cada uno exige ``--confirm-final`` y la evaluación se bloquea tras el informe: no se puede repetir):

1. ``python -m mavericks.final_eval plan``  (no lee ningún resultado): escribe la configuración congelada con el commit y su hash.
2. ``python -m mavericks.final_eval priors --confirm-final``  (CPU): predice los 41 casos con el prior de validación, no el de
   producción: KNN actual con los filtros del backtest (la propia pregunta y su texto excluidos, corte temporal estricto) y un MLP
   entrenado por mes de trabajo de campo con el corte más restrictivo del mes y sin los estudios de sus casos.
3. Ejecutar el LLM (GPU) con el comando que imprime ``plan``: mismo pipeline y modelo que la línea base de desarrollo con la que se fijó w.
4. ``python -m mavericks.final_eval report --confirm-final``: combina y crea el bloqueo ``outputs/final_eval/DONE``.

Modos: **rápido** = media 50/50 KNN + MLP; **con LLM** = 0,3·LLM + 0,7·(prior rápido). Referencias: uniforme, KNN actual y LLM solo.
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("outputs/final_eval")
CASES = "outputs/ablation/cases_official_hard.json"
LLM_RESULTS = OUT / "llm" / "backtest.json"
W_LLM = 0.3
W_MLP = 0.5
MODEL = "qwen3:8b"
LOCK = OUT / "DONE"


class FinalEvalError(RuntimeError):
    pass


def guard(confirmed):
    """Solo se ejecuta con ``--confirm-final`` y una única vez."""
    if not confirmed:
        raise FinalEvalError("La evaluación final sobre los 41 oficiales exige --confirm-final (se ejecuta una sola vez, junto con quien la revisa).")
    if LOCK.exists():
        raise FinalEvalError(f"La evaluación final ya se ejecutó ({LOCK.read_text(encoding='utf-8').strip()}): no se repite.")


def frozen_config():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "cases": CASES,
        "knn": {"k": 5, "power": 4.0, "similarity": "lex", "shrink": 0.0, "min_sim": 0.0, "pool": 40, "topic_bonus": True},
        "mlp": {"pca_dims": 64, "hidden": [64], "max_options": 7, "seed": 0, "granularity": "mes de trabajo de campo, corte más restrictivo"},
        "rapid_mode": {"w_mlp": W_MLP, "w_knn": 1 - W_MLP},
        "llm_mode": {"model": MODEL, "w_llm": W_LLM, "w_prior": 1 - W_LLM, "pipeline": "línea base: sin analogía, sin ejemplos, sin corrector"},
        "never_use": "el MLP de producción (outputs/models)",
    }


def llm_command():
    return (
        f"python main.py backtest --provider ollama --model {MODEL} --path data/processed/synthetic_population.parquet "
        f"--cases {CASES} --segments 10 --concurrency 2 --no-llm-assessment --calibration off "
        f"--output {LLM_RESULTS.as_posix()}"
    )


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    config = frozen_config()
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=1))
    print("\nComando del LLM (paso 3):\n" + llm_command())


def blend(prior, other, w):
    return {o: w * float(other[o]) + (1 - w) * float(prior[o]) for o in prior}


def build_rows(cases, knn, mlp, llm=None):
    """Una fila por caso con todas las predicciones; ``llm`` = {id: distribución} (puede faltar)."""
    rows = []
    for c in cases:
        i, options = c["evidence_id"], c["options"]
        observed = np.array([c["observed_distribution"][o] for o in options])
        k = np.array([knn[i][o] for o in options])
        m = np.array(mlp[i])
        rapid = (1 - W_MLP) * k + W_MLP * m
        row = {
            "evidence_id": i, "study": str(c["estudio_id"]), "topic": c["topic"], "n_options": len(options), "observed": observed,
            "uniform": np.full(len(options), 1 / len(options)), "knn": k, "rapid": rapid,
        }
        if llm is not None and i in llm:
            p = np.array([llm[i][o] for o in options])
            row.update(llm=p, with_llm=W_LLM * p + (1 - W_LLM) * rapid)
        rows.append(row)
    return rows


def priors():
    """Paso 2: KNN actual y MLP por mes para los 41 casos, con las mismas protecciones que el backtest."""
    from . import service as svc
    from .analog import analog_for_question, load_national_table
    from .evidence import EvidenceRepository
    from .prior_v2 import data, learned
    from .prior_v2.bigdev import MLP
    from .prior_v2.embcache import EmbCache, clean_label
    from .prior_v2.task1 import EVIDENCE, TABLE

    cases = json.loads(Path(CASES).read_text(encoding="utf-8-sig"))
    repo, table = EvidenceRepository(EVIDENCE), load_national_table(TABLE)
    emb = EmbCache("outputs/prior_v2/emb_cache")
    embedder = svc.get_embedder()
    if embedder is None:
        raise FinalEvalError(f"e5 no disponible: {svc.embedder_error()}")
    knn = {}
    for c in cases:
        filters = {"excluded_ids": [c["evidence_id"]], "excluded_texts": [c["question"]], "cutoff": c["fieldwork_date"], "topic": c["topic"]}
        knn[c["evidence_id"]] = analog_for_question(
            {"question": c["question"], "options": c["options"]}, repo, table, svc.ANALOG, filters, sources={"cis"}
        )["prior"]
    questions = learned.train_questions(repo, table, emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento de la evaluación final")
    by_month = {}
    for c in cases:
        by_month.setdefault(c["fieldwork_date"][:7], []).append(c)
    mlp = {}
    for month, group in sorted(by_month.items()):
        cutoff = min(c["fieldwork_date"] for c in group)
        banned = {str(c["estudio_id"]) for c in group}
        train = [t for t in questions if t.date < cutoff and t.study not in banned]
        assert all(t.date < cutoff and t.study not in banned for t in train), "fuga temporal o de estudio"
        q, o, ex, y = learned.build_training_matrix(train, emb)
        uq = np.stack([emb.question(t.evidence_id) for t in train])
        labels = sorted({clean_label(x) for t in train for x in t.options})
        featurizer = learned.Featurizer(MLP[3])
        x = featurizer.fit(uq, np.stack([emb.option_query(x) for x in labels]), q, o, ex)
        model = learned.make_model(MLP[1], MLP[2]).fit(x, y)
        for c in group:
            q_vec = np.asarray(embedder.encode_passages([" ".join(c["question"].split())]))[0]
            o_vecs = np.asarray(embedder.encode_queries([clean_label(opt) for opt in c["options"]]))
            mlp[c["evidence_id"]] = learned.predict_case(featurizer, model, q_vec, o_vecs, [False] * len(o_vecs))
        print(f"MLP {month}: {len(group)} casos, corte {cutoff}, {len(train)} preguntas de entrenamiento", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "priors.pkl", "wb") as f:
        pickle.dump({"knn": knn, "mlp": mlp}, f)


def report():
    """Paso 4: tablas finales y bloqueo."""
    from .prior_v2 import bigdev_eval as ev

    cases = json.loads(Path(CASES).read_text(encoding="utf-8-sig"))
    with open(OUT / "priors.pkl", "rb") as f:
        stored = pickle.load(f)
    results = json.loads(LLM_RESULTS.read_text(encoding="utf-8"))["results"]
    llm = {r["evidence_id"]: r["predicted"] for r in results if r["status"] == "completed"}
    rows = build_rows(cases, stored["knn"], stored["mlp"], llm)
    rows = [r for r in rows if "llm" in r]  # los mismos casos en todas las configuraciones
    seconds = {r["evidence_id"]: r.get("execution_time") for r in results}
    names = {"uniform": "Reparto uniforme", "knn": "KNN actual", "rapid": "Modo rápido (KNN + MLP)", "with_llm": "Modo con LLM (0,3·LLM + 0,7·prior)", "llm": "LLM solo"}
    studies = [r["study"] for r in rows]
    errors = {k: np.array([ev.mae(r[k], r["observed"]) for r in rows]) for k in names}
    lines = [f"# Evaluación final única (41 oficiales difíciles; {len(rows)} casos, {len(set(studies))} estudios)", "",
             "| Configuración | MAE (pp) | Δ frente al KNN actual [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p | s/caso |", "|---|---|---|---|---|---|"]
    llm_seconds = float(np.mean([seconds[r["evidence_id"]] for r in rows if seconds.get(r["evidence_id"]) is not None]))
    for key, label in names.items():
        t = {"with_llm": f"{llm_seconds:.0f}", "llm": f"{llm_seconds:.0f}", "rapid": "≈2", "knn": "≈0,3"}.get(key, "0")
        if key == "knn":
            lines.append(f"| {label} | {100 * errors[key].mean():.2f} | — | — | — | {t} |")
        else:
            lines.append(ev.line(label, ev.compare(errors["knn"], errors[key], studies)).rstrip()[:-1].rstrip() + f" | {t} |")
    for label, mask in (("2 opciones", [r["n_options"] == 2 for r in rows]), ("3 o más opciones", [r["n_options"] >= 3 for r in rows])):
        m = np.array(mask)
        if m.sum():
            lines += ["", f"**{label}** ({int(m.sum())} casos): " + "; ".join(f"{names[k]} {100 * errors[k][m].mean():.2f}" for k in names)]
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    LOCK.write_text(f"ejecutada {time.strftime('%Y-%m-%d %H:%M')}", encoding="utf-8")
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["plan", "priors", "report"])
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "plan":
        return plan()
    guard(args.confirm_final)
    return {"priors": priors, "report": report}[args.step]()


if __name__ == "__main__":
    try:
        main()
    except FinalEvalError as exc:
        sys.exit(str(exc))
