"""Evaluación ÚNICA del modo ajustado (modelo B) en los 88 casos oficiales (estudios reservados). Definida en docs/prereg_finetuning.md (anexo del 2026-10-19).

    python -m mavericks.final_eval_ft --confirm-final        # con el servidor de B arrancado (scripts/run_server.ps1)

Reutiliza los casos, el modo de cada caso, el prior (KNN y MLP por mes) y la métrica de ``final_eval`` (``outputs/final_eval/priors.pkl``, solo lectura; esa evaluación no se
toca ni se repite). Modo ajustado = estadística directa si hay ancla directa; si no, el reparto del modelo B (primer token, orden original, sin fallback). Se niega a correr
si existe ``outputs/final_eval_ft/DONE``.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import fastcal, final_eval
from .assist import display_label
from .ft.client import FtClient, FtError
from .prior_v2.bigdev_eval import mae
from .prior_v2.knn import TIE

OUT = Path("outputs/final_eval_ft")
LOCK = OUT / "DONE"
ADAPTER_SHA256 = "f6a4dd9453f3233933b14a12af70d64ef0db197c37dedf24dbab0e832f339c6b"  # fijado en el anexo del preregistro
ADAPTER = Path("outputs/ft/runs/b1/adapter_final/adapter_model.safetensors")
INTERVALS = Path("outputs/models/ft_intervals.json")
DRAWS, SEED, MIN_GAIN_PP, MIN_STUDIES = 10_000, 0, 0.5, 8
NAMES = {
    "uniform": "Reparto uniforme", "knn": "Sistema anterior (directo / KNN actual)", "rapid": "Modo rápido (directo / KNN + MLP)",
    "with_llm": "Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia", "ajustado": "Modo AJUSTADO (directo / modelo B)",
}
SECONDARY = {"b_all": "B solo (incluso con ancla directa)", "b_mix": "Media 50/50 B + prior rápido"}
MAX_OPTIONS = 7  # letras A-G del entrenamiento: con más opciones B no puede responder y el modo ajustado usa el modo rápido (como el MLP, max_options)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def guard(confirmed):
    if not confirmed:
        raise final_eval.FinalEvalError("La evaluación del modo ajustado exige --confirm-final (se ejecuta una sola vez).")
    if LOCK.exists():
        raise final_eval.FinalEvalError(f"La evaluación del modo ajustado ya se ejecutó ({LOCK.read_text(encoding='utf-8').strip()}): no se repite.")
    if sha256(ADAPTER) != ADAPTER_SHA256:
        raise final_eval.FinalEvalError("El adaptador no es el fijado en el preregistro (sha256 distinto).")


class VramWatch:
    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            self.peak = max(self.peak, int(out or 0))
            time.sleep(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()


def query_b(cases, client):
    """Reparto de B para los 88 casos, uno a uno (prompt del entrenamiento; año = año del trabajo de campo del caso). Cualquier fallo aborta."""
    probs, wall, server = {}, [], []
    for c in cases:
        if len(c["options"]) > MAX_OPTIONS:
            probs[c["evidence_id"]] = None
            continue
        started = time.time()
        p, seconds = client.predict(c["question"], [display_label(o) for o in c["options"]], year=c["fieldwork_date"][:4])
        wall.append(time.time() - started)
        server.append(seconds)
        probs[c["evidence_id"]] = p
    return probs, wall, server


def bootstrap(delta, groups, draws=DRAWS, seed=SEED):
    delta, groups = np.asarray(delta, dtype=float), np.asarray(groups)
    names = sorted(set(groups))
    if len(names) < MIN_STUDIES:  # pocos estudios: se remuestrean casos (se indica)
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), "casos"
    sums = np.array([delta[groups == s].sum() for s in names])
    counts = np.array([(groups == s).sum() for s in names])
    pick = np.random.default_rng(seed).integers(0, len(names), (draws, len(names)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), "estudios"


def compare(reference, errors, studies):
    from scipy.stats import wilcoxon

    delta = np.asarray(errors) - np.asarray(reference)
    low, high, unit = bootstrap(delta, studies)
    p = float(wilcoxon(errors, reference).pvalue) if len(delta) >= 10 and np.any(delta != 0) else None
    return {"mae": 100 * float(np.mean(errors)), "delta": 100 * float(delta.mean()), "low": 100 * low, "high": 100 * high, "unit": unit, "p": p,
            "better": int((delta < -TIE).sum()), "worse": int((delta > TIE).sum()), "ties": int((np.abs(delta) <= TIE).sum()), "n": len(delta)}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def fmt_p(p):
    return "—" if p is None else ("<0,001" if p < 0.001 else f"{p:.3f}")


def rows_with_b(cases, stored, llm, b_probs):
    rows = final_eval.build_rows(cases, stored, llm)
    for r in rows:
        raw = b_probs[r["evidence_id"]]
        r["supported"] = raw is not None
        r["ajustado"] = r["rapid"]  # directo, o sin respuesta de B (más de 7 opciones): modo rápido
        if raw is not None:
            b = np.asarray(raw, dtype=float)
            r["b_all"] = b
            r["b_mix"] = 0.5 * b + 0.5 * r["rapid"]
            if r["mode"] != "direct":
                r["ajustado"] = b
    return rows


def table(rows, title, reference="rapid", keys=None):
    studies = [r["study"] for r in rows]
    errors = {k: np.array([mae(r[k], r["observed"]) for r in rows]) for k in (keys or NAMES) if all(k in r for r in rows)}
    lines = [f"### {title} ({len(rows)} casos, {len(set(studies))} estudios)",
             f"| Sistema | MAE (pp) | Δ frente al {NAMES[reference].split(' (')[0].lower()} [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|"]
    results = {}
    for k, err in errors.items():
        if k == reference:
            lines.append(f"| {NAMES[k]} | {100 * err.mean():.2f} | — | — | — |")
            continue
        r = compare(errors[reference], err, studies)
        results[k] = r
        lines.append(f"| {NAMES[k]} | {r['mae']:.2f} | {r['delta']:+.2f} [{r['low']:+.2f}, {r['high']:+.2f}] | {r['better']} / {r['worse']} / {r['ties']} | {fmt_p(r['p'])} |")
    return lines, results


def coverage(rows):
    """Cobertura de los rangos del 50 % y del 90 % por opción: casos directos con sus rangos actuales; el resto, B con la calibración Mondrian de a1 en dev_grande."""
    ft = json.loads(INTERVALS.read_text(encoding="utf-8"))
    payload = fastcal.load("outputs/fast_validation.json")
    model = payload["by_buckets" if payload.get("chosen") == "by_buckets" else "by_mode"]["model"] if payload else None
    hits = {"direct": {"q": [0, 0], "q50": [0, 0]}, "no_direct": {"q": [0, 0], "q50": [0, 0]}}
    for r in rows:
        key = "direct" if r["mode"] == "direct" else "no_direct"
        p = np.asarray(r["ajustado"], dtype=float)
        group = "2 opciones" if len(p) == 2 else "3 o más"
        for level in ("q", "q50"):
            if key == "direct":
                rad = np.array([fastcal.radius(model, "direct", len(p), x, level=level)[0] for x in p], dtype=object) if model else None
                if rad is None or any(x is None for x in rad):
                    continue
                rad = rad.astype(float)
            else:
                rad = ft["groups"][group]["q90" if level == "q" else "q50"] * fastcal.scale(p)
            hits[key][level][0] += int((np.abs(p - r["observed"]) <= rad).sum())
            hits[key][level][1] += len(p)
    out = {}
    for key, levels in {**hits, "todos": {lv: [sum(hits[k][lv][i] for k in hits) for i in (0, 1)] for lv in ("q", "q50")}}.items():
        out[key] = {("90 %" if lv == "q" else "50 %"): (h / n if n else None) for lv, (h, n) in levels.items()}
    return out


def report(cases, stored, llm, b_probs, wall, server, vram, health):
    rows = rows_with_b(cases, stored, llm, b_probs)
    studies = sorted({r["study"] for r in rows})
    main, results = table(rows, "Total (comparación principal: modo ajustado frente a modo rápido)")
    lines = [f"# Evaluación única del modo ajustado (modelo B) — {len(rows)} casos oficiales (estudios reservados)", "",
             f"Adaptador B sha256 `{ADAPTER_SHA256}`. Bootstrap por {('estudios' if len(studies) >= MIN_STUDIES else 'casos')} ({len(studies)} estudios; {DRAWS} réplicas, semilla {SEED}). "
             f"Criterio: mejora ≥ {MIN_GAIN_PP} pp frente al modo rápido con el IC 95 % entero por debajo de 0.", "", *main, ""]
    segments = {"Directos (estadística directa en ambos modos)": [r for r in rows if r["mode"] == "direct"],
                "Sin ancla (pregunta nueva)": [r for r in rows if r["mode"] == "none"], "Resto (prior: pregunta parecida)": [r for r in rows if r["mode"] == "prior"],
                "No directos (sin ancla + resto)": [r for r in rows if r["mode"] != "direct"]}
    for title, part in segments.items():
        if part:
            lines += [*(table(part, title)[0] if title.startswith(("Sin", "Resto", "No")) else _direct_line(part, title)), ""]
    unsupported = [r for r in rows if not r["supported"]]
    lines += [f"Casos con más de {MAX_OPTIONS} opciones (B no puede responder; el modo ajustado usa el modo rápido): {len(unsupported)} "
              f"({sum(r['mode'] != 'direct' for r in unsupported)} no directos).", ""]
    sup_all = [r for r in rows if r["supported"]]
    sup_hard = [r for r in sup_all if r["mode"] != "direct"]
    for title, part, keys in (("Secundaria: B en todos los casos con ≤ 7 opciones, incluso con ancla directa", sup_all, ["rapid", "ajustado", "b_all"]),
                              ("Secundaria: media 50/50 B + prior rápido en los no directos con ≤ 7 opciones", sup_hard, ["rapid", "ajustado", "b_mix"])):
        names = {**NAMES, **SECONDARY}
        studies_p = [r["study"] for r in part]
        errs = {k: np.array([mae(r[k], r["observed"]) for r in part]) for k in keys}
        lines += [f"### {title} ({len(part)} casos, {len(set(studies_p))} estudios)", "| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 %] | mejoran / empeoran / empatan |", "|---|---|---|---|"]
        for k in keys:
            if k == "rapid":
                lines.append(f"| {names[k]} | {100 * errs[k].mean():.2f} | — | — |")
            else:
                c = compare(errs["rapid"], errs[k], studies_p)
                lines.append(f"| {names[k]} | {c['mae']:.2f} | {c['delta']:+.2f} [{c['low']:+.2f}, {c['high']:+.2f}] | {c['better']} / {c['worse']} / {c['ties']} |")
        lines.append("")
    lines += ["### Por número de opciones (no directos)"]
    hard = segments["No directos (sin ancla + resto)"]
    for label, sel in (("2 opciones", [r for r in hard if r["n_options"] == 2]), ("3 o más", [r for r in hard if r["n_options"] >= 3])):
        if sel:
            lines += ["", *table(sel, label, keys=["uniform", "knn", "rapid", "with_llm", "ajustado"])[0]]
    word = {}
    main_result = results["ajustado"]
    gain = -main_result["delta"]
    word["ajustado"] = "CUMPLE" if gain >= MIN_GAIN_PP and main_result["high"] < 0 else "NO CUMPLE"
    cov = coverage(rows)
    lines += ["", "## Criterio prerregistrado", f"- **Modo ajustado frente a modo rápido en los 88: {word['ajustado']}** (mejora {gain:+.2f} pp, IC 95 % [{main_result['low']:+.2f}, {main_result['high']:+.2f}]; {main_result['unit']} remuestreados)",
              "", "## Cobertura de los rangos (descriptiva; por opción)", "| Casos | 50 % | 90 % |", "|---|---|---|",
              *[f"| {k} | {pct(v['50 %'])} | {pct(v['90 %'])} |" for k, v in cov.items()],
              "", "## Coste", f"- segundos por pregunta de B: servidor {np.mean(server):.3f} s, ida y vuelta {np.mean(wall):.3f} s (88 consultas una a una)",
              f"- VRAM: pico total de la GPU durante la evaluación {vram} MiB de 8188 (con el servidor de B y el modelo de Ollama cargados); máximo asignado por el servidor {health.get('vram_gb', 0):.2f} GB"]
    text = "\n".join(lines)
    summary = {"principal": main_result, "veredicto": word["ajustado"], "cobertura": cov, "segundos_servidor": float(np.mean(server)), "segundos_total": float(np.mean(wall)),
               "vram_pico_mib": vram, "mae": {k: 100 * float(np.mean([mae(r[k], r["observed"]) for r in rows])) for k in NAMES},
               "estudios": len(studies), "casos": len(rows)}
    return text, summary, rows


def _direct_line(part, title):
    e = np.array([mae(r["rapid"], r["observed"]) for r in part])
    b = np.array([mae(r["b_all"], r["observed"]) for r in part if r["supported"]])
    u = np.array([mae(r["uniform"], r["observed"]) for r in part])
    return [f"### {title} ({len(part)} casos)", f"MAE de la estadística directa {100 * e.mean():.2f} pp (uniforme {100 * u.mean():.2f}). Secundaria: B en los {len(b)} con ≤ 7 opciones {100 * b.mean():.2f} pp."]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-final", action="store_true")
    args = parser.parse_args(argv)
    guard(args.confirm_final)
    client = FtClient()
    if not client.healthy():
        raise final_eval.FinalEvalError("El servidor de B no responde (arráncalo con scripts/run_server.ps1).")
    cases = json.loads(Path(final_eval.CASES).read_text(encoding="utf-8-sig"))
    assert len(cases) == 88
    stored = pickle.loads((final_eval.OUT / "priors.pkl").read_bytes())
    llm = {r["evidence_id"]: r["predicted"] for r in json.loads(final_eval.LLM_RESULTS.read_text(encoding="utf-8"))["results"] if r["status"] == "completed"}
    with VramWatch() as watch:
        try:
            b_probs, wall, server = query_b(cases, client)
        except FtError as exc:
            raise final_eval.FinalEvalError(f"Fallo del servidor de B: {exc}. No hay fallback en la evaluación; nada se ha escrito.") from exc
    health = client._call("/health")
    text, summary, rows = report(cases, stored, llm, b_probs, wall, server, watch.peak, health)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b_probs.json").write_text(json.dumps(b_probs), encoding="utf-8")
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    LOCK.write_text(f"ejecutada {time.strftime('%Y-%m-%d %H:%M')}; commit {commit}; adaptador {ADAPTER_SHA256}", encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    try:
        main()
    except final_eval.FinalEvalError as exc:
        sys.exit(str(exc))
