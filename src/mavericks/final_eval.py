"""Evaluación final ÚNICA sobre los 88 casos oficiales (estudios reservados), con la configuración congelada y los DOS modos.

Pasos (cada uno exige ``--confirm-final``; el informe bloquea la evaluación, que no se puede repetir):

1. ``plan``: escribe la configuración congelada (commit, pesos, comandos). No lee ningún resultado.
2. ``priors --confirm-final`` (CPU): para cada caso, modo y estimación base del servicio con los filtros del backtest (la propia pregunta
   fuera, corte temporal estricto): estadística directa si la pregunta se repite; si no, KNN actual. Para los no directos, un MLP entrenado
   por mes de trabajo de campo con el corte más restrictivo y sin los estudios de sus casos (nunca el modelo de producción, que ni se carga).
3. ``reuse-llm --confirm-final``: reutiliza la ejecución del LLM (qwen3:8b) de los 41 casos no directos si todas las comprobaciones de
   configuración y de hashes coinciden; si no, ``plan`` indica el comando para ejecutarla de nuevo.
4. ``report --confirm-final``: tablas finales por modo y total, y bloqueo ``outputs/final_eval/DONE``.

Modo rápido = media 50/50 KNN + MLP; modo con LLM = 0,3·LLM + 0,7·(prior rápido). Las preguntas repetidas usan la estadística directa en los dos modos.
"""

import argparse
import ast
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path("outputs/final_eval")
CASES = "data/processed/backtest_cases.json"  # los 88 oficiales
HARD_CASES = "outputs/ablation/cases_official_hard.json"  # los 41 no directos (las preguntas con LLM)
LLM_SOURCE = Path("outputs/queue/model_qwen3_8b/backtest.json")  # ejecución de control de hoy
LLM_RESULTS = OUT / "llm" / "backtest.json"
LLM_DB = "data/processed_v2/frozen/evidence_frozen.sqlite"
W_LLM = 0.3
W_MLP = 0.5
MODEL = "qwen3:8b"
LOCK = OUT / "DONE"
MODES = {"direct": "Directo (pregunta repetida)", "prior": "Prior (pregunta parecida)", "none": "Sin ancla (pregunta nueva)"}
CHOICE_NOTE = "La elección de qwen3:8b frente a qwen3.5:9b usó estos 41 casos; el prior, w y los rangos no."


class FinalEvalError(RuntimeError):
    pass


def guard(confirmed):
    """Solo se ejecuta con ``--confirm-final`` y una única vez."""
    if not confirmed:
        raise FinalEvalError("La evaluación final sobre los oficiales exige --confirm-final (se ejecuta una sola vez, junto con quien la revisa).")
    if LOCK.exists():
        raise FinalEvalError(f"La evaluación final ya se ejecutó ({LOCK.read_text(encoding='utf-8').strip()}): no se repite.")


def llm_command():
    return (
        f"python main.py backtest --provider ollama --model {MODEL} --path data/processed/synthetic_population.parquet --db {LLM_DB} "
        f"--cases {HARD_CASES} --segments 10 --concurrency 2 --no-llm-assessment --sources cis --calibration off "
        f"--output {LLM_RESULTS.as_posix()}   (Ollama 0.40.6, OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_FLASH_ATTENTION=1, OLLAMA_NUM_PARALLEL=2)"
    )


def frozen_config():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "cases": CASES,
        "knn": {"k": 5, "power": 4.0, "similarity": "lex", "shrink": 0.0, "min_sim": 0.0, "pool": 40, "topic_bonus": True},
        "direct": {"similarity": 0.85, "item_guard": True},
        "mlp": {"pca_dims": 64, "hidden": [64], "max_options": 7, "seed": 0, "granularity": "mes de trabajo de campo, corte más restrictivo"},
        "rapid_mode": {"w_mlp": W_MLP, "w_knn": 1 - W_MLP},
        "llm_mode": {"model": MODEL, "w_llm": W_LLM, "w_prior": 1 - W_LLM, "pipeline": "línea base: sin analogía, sin ejemplos, sin corrector",
                     "ollama": "0.40.6", "kv": "q8_0", "flash": True, "parallel": 2, "evidence": LLM_DB, "sources": "cis"},
        "never_use": "el modelo de producción del prior",
    }


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    config = frozen_config()
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=1))
    print("\nComando del LLM (solo si no se puede reutilizar la ejecución de control):\n" + llm_command())


# --- comprobación de que la ejecución de control es la configuración congelada ------------------------------------------------------------

def sha(path, n=16):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()[:n]


def prompt_hash(llm_py_text):
    tree = ast.parse(llm_py_text)
    found = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body if isinstance(t, ast.Assign) and getattr(t.targets[0], "id", "") in ("SYSTEM", "ROLES")}
    roles = {k: v for k, v in found["ROLES"].items() if k in ("Methodology", "Assessment", "Prediction", "Narrative")}
    return hashlib.sha256(json.dumps({"SYSTEM": found["SYSTEM"], "ROLES": roles}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def provenance(source=LLM_SOURCE):
    """Hechos de la ejecución de control y de la configuración actual; ``ok`` solo si todo coincide con lo congelado."""
    results = json.loads(Path(source).read_text(encoding="utf-8"))["results"]
    runtime = results[0].get("runtime") or {}
    tags = json.load(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5))
    version = json.load(urllib.request.urlopen("http://localhost:11434/api/version", timeout=5))["version"]
    digest = next((m["digest"][:12] for m in tags["models"] if m["name"] == MODEL), None)
    log = Path.home() / "AppData/Local/Ollama/server.log"
    server = re.findall(r"OLLAMA_FLASH_ATTENTION:(\w+).*?OLLAMA_KV_CACHE_TYPE:(\w+).*?OLLAMA_NUM_PARALLEL:(\d+)", log.read_text(encoding="utf-8", errors="ignore")) if log.exists() else []
    status = Path("outputs/queue/status.log")
    started = re.search(r"(\d\d:\d\d:\d\d) model_qwen3_8b INICIO", status.read_text(encoding="utf-8")) if status.exists() else None
    start_iso = time.strftime("%Y-%m-%d", time.localtime(status.stat().st_mtime)) + " " + started.group(1) if started else None
    before = subprocess.run(["git", "rev-list", "-1", f"--before={start_iso}", "HEAD"], capture_output=True, text=True).stdout.strip() if start_iso else ""

    def at_start(path):
        return subprocess.run(["git", "show", f"{before}:{path}"], capture_output=True, text=True, encoding="utf-8").stdout

    files = ("graph.py", "payloads.py", "anchor.py", "population.py", "statistics.py", "analog.py", "evaluation.py", "context.py", "categories.py", "topics.py", "evidence.py")
    unchanged = {
        f: bool(before) and hashlib.sha256(at_start(f"src/mavericks/{f}").encode()).hexdigest() == hashlib.sha256(Path("src/mavericks", f).read_text(encoding="utf-8").encode()).hexdigest()
        for f in files
    }
    same_prompts = bool(before) and prompt_hash(at_start("src/mavericks/llm.py")) == prompt_hash(Path("src/mavericks/llm.py").read_text(encoding="utf-8"))
    facts = {
        "modelo": runtime.get("model"), "digest_control": runtime.get("model_digest"), "digest_actual": digest,
        "ollama_control": runtime.get("ollama_version"), "ollama_actual": version,
        "servidor_flash_kv_paralelo": server[-1] if server else None, "arranques_del_servidor": len(server),
        "hash_evidencia": sha(LLM_DB), "hash_poblacion": sha("data/processed/synthetic_population.parquet"), "hash_casos_difíciles": sha(HARD_CASES),
        "inicio_de_la_ejecucion": start_iso, "commit_al_inicio": before,
        "prompts_iguales_a_los_del_inicio": same_prompts,
        "ficheros_del_pipeline_iguales_a_los_del_inicio": unchanged,
        "casos": len(results), "completados": sum(r["status"] == "completed" for r in results),
    }
    facts["ok"] = (
        facts["modelo"] == MODEL and facts["digest_control"] == facts["digest_actual"] and facts["ollama_control"] == facts["ollama_actual"] == "0.40.6"
        and facts["servidor_flash_kv_paralelo"] == ("true", "q8_0", "2") and facts["arranques_del_servidor"] == 1
        and same_prompts and all(unchanged.values()) and facts["casos"] == facts["completados"] == 41
    )
    return facts


def reuse_llm():
    facts = provenance()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm_provenance.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=1))
    if not facts["ok"]:
        raise FinalEvalError("La ejecución de control no coincide con la configuración congelada: hay que ejecutar el LLM de nuevo.\n" + llm_command())
    LLM_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    LLM_RESULTS.write_bytes(Path(LLM_SOURCE).read_bytes())
    print("Ejecución del LLM reutilizada:", LLM_RESULTS)


# --- predicciones -------------------------------------------------------------------------------------------------------------------------

def blend(prior, other, w):
    return {o: w * float(other[o]) + (1 - w) * float(prior[o]) for o in prior}


def build_rows(cases, stored, llm=None):
    """Una fila por caso; ``stored`` = {"modes", "base", "mlp"}; ``llm`` = {id: distribución}. Directo: estadística directa en todo."""
    rows = []
    for c in cases:
        i, options = c["evidence_id"], c["options"]
        observed = np.array([c["observed_distribution"][o] for o in options])
        mode = stored["modes"][i]
        base = np.array([stored["base"][i][o] for o in options])
        row = {
            "evidence_id": i, "study": str(c["estudio_id"]), "topic": c["topic"], "n_options": len(options), "mode": mode,
            "observed": observed, "uniform": np.full(len(options), 1 / len(options)), "knn": base,
        }
        if mode == "direct":
            row.update(rapid=base, with_llm=base)
        else:
            rapid = (1 - W_MLP) * base + W_MLP * np.array(stored["mlp"][i])
            row["rapid"] = rapid
            if llm is not None and i in llm:
                p = np.array([llm[i][o] for o in options])
                row.update(llm=p, with_llm=W_LLM * p + (1 - W_LLM) * rapid)
        rows.append(row)
    return rows


def priors():
    """Paso 2: modo y base de los 88 casos con los filtros del backtest; MLP por mes para los no directos."""
    from . import service as svc
    from .analog import load_national_table
    from .evidence import EvidenceRepository
    from .prior_v2 import data, learned
    from .prior_v2.bigdev import MLP
    from .prior_v2.embcache import EmbCache, clean_label
    from .prior_v2.task1 import EVIDENCE, TABLE

    cases = json.loads(Path(CASES).read_text(encoding="utf-8-sig"))
    resources = svc.load_resources(svc.Paths(models="outputs/final_eval/sin_modelo_de_produccion"))
    assert resources.prior_model is None, "el modelo de producción no puede cargarse en la evaluación final"
    modes, base = {}, {}
    for c in cases:
        filters = {"excluded_ids": [c["evidence_id"]], "cutoff": c["fieldwork_date"]}
        mode, national, *_ = svc._fast(resources, c["question"], c["options"], c["topic"], 10, svc.DIRECT_SIMILARITY, 0.55, filters)
        modes[c["evidence_id"]], base[c["evidence_id"]] = mode, {o: float(national[o]) for o in c["options"]}
    print({m: sum(1 for v in modes.values() if v == m) for m in MODES}, flush=True)
    hard = [c for c in cases if modes[c["evidence_id"]] != "direct"]
    embedder = svc.get_embedder()
    if embedder is None:
        raise FinalEvalError(f"e5 no disponible: {svc.embedder_error()}")
    repo, table, emb = EvidenceRepository(EVIDENCE), load_national_table(TABLE), EmbCache("outputs/prior_v2/emb_cache")
    questions = learned.train_questions(repo, table, emb)
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento de la evaluación final")
    by_month = {}
    for c in hard:
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
        pickle.dump({"modes": modes, "base": base, "mlp": mlp}, f)


# --- informe -------------------------------------------------------------------------------------------------------------------------------

NAMES = {
    "uniform": "Reparto uniforme", "knn": "Sistema anterior (directo / KNN actual)", "rapid": "Modo rápido (directo / KNN + MLP)",
    "with_llm": "Modo con LLM (directo / 0,3·LLM + 0,7·prior)",
}


def table(rows, ev, title):
    studies = [r["study"] for r in rows]
    errors = {k: np.array([ev.mae(r[k], r["observed"]) for r in rows]) for k in NAMES}
    out = [f"### {title} ({len(rows)} casos, {len(set(studies))} estudios)",
           "| Configuración | MAE (pp) | Δ frente al sistema anterior [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|"]
    for key, label in NAMES.items():
        out.append(f"| {label} | {100 * errors[key].mean():.2f} | — | — | — |" if key == "knn" else ev.line(label, ev.compare(errors["knn"], errors[key], studies)))
    return out


def report():
    """Paso 4: tablas finales por modo y total, y bloqueo."""
    from .prior_v2 import bigdev_eval as ev

    cases = json.loads(Path(CASES).read_text(encoding="utf-8-sig"))
    with open(OUT / "priors.pkl", "rb") as f:
        stored = pickle.load(f)
    results = json.loads(LLM_RESULTS.read_text(encoding="utf-8"))["results"]
    llm = {r["evidence_id"]: r["predicted"] for r in results if r["status"] == "completed"}
    rows = build_rows(cases, stored, llm)
    missing = [r["evidence_id"] for r in rows if r["mode"] != "direct" and "llm" not in r]
    if missing:
        raise FinalEvalError(f"Faltan resultados del LLM para {len(missing)} casos no directos")
    seconds = float(np.mean([r["execution_time"] for r in results if r["status"] == "completed"]))
    lines = [f"# Evaluación final única — {len(rows)} casos oficiales (estudios reservados)", "", CHOICE_NOTE, ""]
    lines += table(rows, ev, "Total")
    for mode, label in MODES.items():
        part = [r for r in rows if r["mode"] == mode]
        if part and mode != "direct":
            lines += ["", *table(part, ev, label)]
        elif part:
            e = np.array([ev.mae(r["knn"], r["observed"]) for r in part])
            u = np.array([ev.mae(r["uniform"], r["observed"]) for r in part])
            lines += ["", f"### {label} ({len(part)} casos): estadística directa en ambos modos", f"MAE directo {100 * e.mean():.2f} pp frente a {100 * u.mean():.2f} del reparto uniforme."]
    hard = [r for r in rows if r["mode"] != "direct"]
    errors = {k: np.array([ev.mae(r[k], r["observed"]) for r in hard]) for k in ("knn", "rapid", "llm", "with_llm")}
    studies = [r["study"] for r in hard]
    lines += ["", f"### Referencia: LLM solo en los {len(hard)} no directos", ev.line("LLM solo", ev.compare(errors["knn"], errors["llm"], studies)),
              ev.line("Modo con LLM frente a modo rápido", ev.compare(errors["rapid"], errors["with_llm"], studies)),
              "", f"Segundos por caso del LLM (ejecución de control): {seconds:.0f}.", "", "### Por número de opciones (no directos)"]
    for label, mask in (("2 opciones", [r["n_options"] == 2 for r in hard]), ("3 o más", [r["n_options"] >= 3 for r in hard])):
        m = np.array(mask)
        if m.sum():
            lines.append(f"- {label} ({int(m.sum())}): " + "; ".join(f"{k} {100 * errors[k][m].mean():.2f}" for k in errors))
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    summary = {
        "casos": len(rows), "estudios": len({r["study"] for r in rows}), "nota": CHOICE_NOTE,
        "por_modo": {m: {k: float(np.mean([ev.mae(r[k], r["observed"]) for r in rows if r["mode"] == m])) for k in NAMES} for m in MODES if any(r["mode"] == m for r in rows)},
        "total": {k: float(np.mean([ev.mae(r[k], r["observed"]) for r in rows])) for k in NAMES},
    }
    studies_all = [r["study"] for r in rows]
    base_errors = np.array([ev.mae(r["knn"], r["observed"]) for r in rows])
    for k in ("rapid", "with_llm", "uniform"):
        c = ev.compare(base_errors, np.array([ev.mae(r[k], r["observed"]) for r in rows]), studies_all)
        summary.setdefault("frente_al_sistema_anterior", {})[k] = {"delta": c["delta"], "ci": list(c["ci"]), "mejoran": c["better"], "empeoran": c["worse"], "p": c["p"]}
    nd = {}
    for k in ("rapid", "with_llm", "uniform"):
        c = ev.compare(errors["knn"], np.array([ev.mae(r[k], r["observed"]) for r in hard]), studies)
        nd[k] = {"delta": c["delta"], "ci": list(c["ci"]), "mejoran": c["better"], "empeoran": c["worse"], "p": c["p"]}
    summary["no_directos"] = {"casos": len(hard), "estudios": len(set(studies)), "mae": {k: float(errors[k].mean()) for k in errors}, "frente_al_sistema_anterior": nd}
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    LOCK.write_text(f"ejecutada {time.strftime('%Y-%m-%d %H:%M')}", encoding="utf-8")
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["plan", "priors", "reuse-llm", "report"])
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "plan":
        return plan()
    guard(args.confirm_final)
    return {"priors": priors, "reuse-llm": reuse_llm, "report": report}[args.step]()


if __name__ == "__main__":
    try:
        main()
    except FinalEvalError as exc:
        sys.exit(str(exc))
