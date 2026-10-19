"""Batería de pruebas (≈ 150 preguntas, ver ``tests/battery/preguntas.yaml`` y ``docs/pruebas.md``).

    python -m mavericks.battery run [--modes rapido,llm,ajustado] [--only C1,C2] [--limit 10]

Escribe ``outputs/battery/resultados.csv`` y ``outputs/battery/informe.md`` (fuera de git). Solo comprueba invariantes y expectativas; NO cambia ni ajusta ninguna predicción.
Invariantes: porcentajes que suman 100 ± 0,1, sin negativos y con el punto dentro de su rango; cifras de la explicación coherentes con las calculadas; etiquetas de modo, método de subgrupos y
fallback presentes y coherentes; sin N.S./N.C. en lo estimado; tiempo por modo; y, en la categoría de consulta de dato, igualdad exacta del bloque medido con un cálculo independiente sobre los microdatos.
"""

import argparse
import csv
import random
import re
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import assist, explain, mrp_service
from . import service as svc
from .ft.client import FtClient
from .service import OutOfScopeError, ServiceError

CASES = Path("tests/battery/preguntas.yaml")
OUT = Path("outputs/battery")
PRESET = {"rapido": "rapido_sin_llm", "llm": "llm_completo", "ajustado": "ajustado"}
TIMEOUT = {"rapido": 240.0, "llm": 900.0, "ajustado": 300.0}
NSNC = re.compile(r"^\s*(n\.?\s?s\.?(\s*[/,y-]\s*n\.?\s?c\.?)?|n\.?\s?c\.?|ns/nc|nsnc|no sabe|no contesta|no sabe/no contesta)\s*$", re.I)
SUM_TOL, EPS = 1e-3, 1e-9


def load_cases(path=CASES):
    cases = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert len({c["id"] for c in cases}) == len(cases), "ids repetidos"
    return cases


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def add_typos(text):
    """Erratas deterministas: se quita una letra de cada cuarta palabra larga y se cambia la primera «g» por «j»."""
    words, out, count = text.split(" "), [], 0
    for w in words:
        if len(w) > 5 and w.isalpha():
            count += 1
            if count % 4 == 0:
                w = w[:2] + w[3:]
        out.append(w)
    return re.sub("g", "j", " ".join(out), count=1)


def resolve(case, by_id):
    """Pregunta, opciones y cruce finales del caso (aplica ``base``, permutaciones y transformaciones de texto)."""
    out = dict(case)
    if "base" in case:
        base = resolve(by_id[case["base"]], by_id)
        out.setdefault("pregunta", base["pregunta"])
        out.setdefault("opciones", list(base["opciones"]))
        if "cruce" not in case and "cruce_esperado" not in case:
            out["cruce"] = base.get("cruce")
    if "cruce_esperado" in case:
        out["cruce"] = case["cruce_esperado"]
    options = list(out.get("opciones") or [])
    how = case.get("permutacion")
    if how == "invertir":
        options = options[::-1]
    elif how == "rotar":
        options = options[1:] + options[:1]
    elif how == "barajar":
        random.Random(0).shuffle(options)
    out["opciones"] = options
    t = case.get("transformacion")
    if t == "mayusculas":
        out["pregunta"] = out["pregunta"].upper()
    elif t == "sin_tildes":
        out["pregunta"] = strip_accents(out["pregunta"])
    elif t == "erratas":
        out["pregunta"] = add_typos(out["pregunta"])
    return out


# --- invariantes ---------------------------------------------------------------------------------------------------------------------------------

def check_distribution(name, dist, interval=None, core=None):
    fails = []
    values = np.array(list(dist.values()), float)
    if not np.isfinite(values).all() or (values < -EPS).any():
        fails.append(f"{name}: valores negativos o no finitos")
    if abs(values.sum() - 1) > SUM_TOL:
        fails.append(f"{name}: suma {100 * values.sum():.2f} % (debe ser 100 ± 0,1)")
    for label, rng in (("rango 90 %", interval), ("rango habitual", core)):
        if rng:
            for o, p in dist.items():
                lo, hi = rng[o]
                if not lo - EPS <= p <= hi + EPS:
                    fails.append(f"{name}: {o} fuera de su {label} ({100 * p:.1f} no está en [{100 * lo:.1f}, {100 * hi:.1f}])")
    for o in dist:
        if NSNC.match(str(o)):
            fails.append(f"{name}: aparece una opción N.S./N.C. ({o})")
    return fails


def check_result(result, mode, case):
    fails, notes = [], []
    nat = result["national"]
    fails += check_distribution("nacional", nat["distribution"], nat.get("interval"), nat.get("interval_core"))
    for dim in result["subgroups"].values():
        for category, group in dim["groups"].items():
            fails += check_distribution(f"subgrupo {dim['label']}={category}", group["distribution"], group.get("interval"))
            if len(fails) > 12:
                break
    for cross in result.get("crosses") or []:
        fails += check_distribution(f"cruce {cross['label']}", cross["distribution"], cross.get("interval"))
        if cross["small"] != (cross["n_cis"] < mrp_service.SMALL_CROSS):
            fails.append(f"cruce {cross['label']}: aviso de muestra pequeña incoherente con n={cross['n_cis']}")
    m = result.get("measured")
    if m:
        fails += check_distribution("medido", m["national"]["distribution"], m["national"]["interval"])
        for c in m["crosses"]:
            if c["distribution"] is not None:
                fails += check_distribution(f"medido {c['label']}", c["distribution"], c["interval"])
                if c["small"] != (c["n"] < 50):
                    fails.append(f"medido {c['label']}: aviso de muestra pequeña incoherente con n={c['n']}")
        if not (m["study_id"] and m["fieldwork_end"] and m["national"]["n"]):
            fails.append("el bloque medido no cita estudio, fecha y n")
    for o in result["options"]:
        if NSNC.match(str(o)):
            fails.append(f"la lista de opciones contiene N.S./N.C. ({o})")
    # etiquetas de modo, método de subgrupos y fallback
    produced = result.get("producido_por") or {}
    if not produced.get("etiqueta") or not result.get("mode_label"):
        fails.append("falta la etiqueta del modo")
    method = result.get("subgroups_method") or {}
    if not method.get("etiqueta"):
        fails.append("falta el método de los subgrupos")
    if produced.get("fallback") and not produced.get("motivo"):
        fails.append("fallback sin motivo")
    if mode == "rapido" and produced.get("modo") != "rapido":
        fails.append(f"modo rápido etiquetado como {produced.get('modo')}")
    if mode == "ajustado" and produced.get("modo") not in ("ajustado", "rapido"):
        fails.append(f"modo ajustado etiquetado como {produced.get('modo')}")
    if mode == "ajustado" and produced.get("modo") == "ajustado" and method.get("clave") != "ajustado":
        fails.append("en modo ajustado los subgrupos deben venir del modelo ajustado")
    if result["mode"] == "direct" and method.get("clave") not in ("mrp", "ancla"):
        fails.append(f"modo directo con método de subgrupos {method.get('clave')}")
    if result["mode"] != "direct" and produced.get("modo") != "ajustado" and method.get("clave") not in ("analogas", "ninguno"):
        fails.append(f"sin ancla con método de subgrupos {method.get('clave')}")
    if mode == "ajustado" and result["mode"] != "direct" and len(result["options"]) <= svc.FT_MAX_OPTIONS and produced.get("modo") == "rapido" and not produced.get("fallback"):
        fails.append("modo ajustado sin ancla cayó al rápido sin marcar fallback")
    if result.get("fallback") and not produced.get("fallback"):
        fails.append("fallback del servicio no reflejado en la etiqueta del modo")
    # explicación: las cifras coinciden con las calculadas
    text = (result.get("explanation_text") or {}).get("texto")
    if not text:
        fails.append("sin explicación")
    else:
        payload = explain.llm_payload(result)
        payload["_all_categories"] = explain.all_categories(result)
        reason = explain.check_text(text, payload)
        if reason:
            fails.append(f"explicación: {reason}")
    if result.get("producido_por", {}).get("modo") == "ajustado" and not (result.get("producido_por") or {}).get("experimental"):
        fails.append("el modo ajustado no lleva la nota de experimental")
    return fails, notes


def independent_measured(result, case):
    """Cálculo aparte con pandas sobre los parquet del CIS: reparto ponderado, n y n efectivo del total y del cruce pedido."""
    import pyarrow.parquet as pq

    from .anchor import option_mapping

    m = result["measured"]
    answers = pq.read_table("data/processed/cis_responses.parquet", filters=[("estudio_id", "=", m["study_id"]), ("question_uid", "=", m["evidence_id"])]).to_pandas()
    people = pd.read_parquet("data/processed/cis_respondents.parquet")
    people = people[people["estudio_id"] == m["study_id"]]
    mapping = option_mapping(result["options"], sorted(answers["answer"].unique()))
    back = {v: k for k, v in mapping.items()}
    merged = answers.merge(people, on=["estudio_id", "respondent_uid"], how="left")
    merged["opt"] = merged["answer"].map(back)
    merged = merged[merged["opt"].notna()].drop_duplicates("respondent_uid")

    def table(frame):
        w = frame["peso"].to_numpy(float)
        sums = frame.groupby("opt")["peso"].sum().reindex(result["options"]).fillna(0.0).to_numpy()
        p = sums / sums.sum()
        return p, len(frame), w.sum() ** 2 / (w**2).sum()

    fails = []
    p, n, n_eff = table(merged)
    nat = m["national"]
    if max(abs(nat["distribution"][o] - x) for o, x in zip(result["options"], p, strict=True)) > 1e-12 or nat["n"] != n or abs(nat["n_eff"] - n_eff) > 1e-9:
        fails.append("el bloque medido nacional no coincide con el cálculo independiente")
    if case.get("cruce") and m["crosses"]:
        sub = merged
        schema = mrp_service.schema_of(pd.read_parquet("data/processed/synthetic_population.parquet"))
        for var, val in mrp_service.normalize_cross(case["cruce"], schema):
            sub = sub[sub[var].astype(str) == val]
        cross = m["crosses"][0]
        if len(sub):
            p, n, n_eff = table(sub)
            if cross["n"] != n or max(abs(cross["distribution"][o] - x) for o, x in zip(result["options"], p, strict=True)) > 1e-12:
                fails.append("el bloque medido del cruce no coincide con el cálculo independiente")
        elif cross["n"] != 0:
            fails.append("el bloque medido del cruce debería estar vacío")
    return fails


# --- ejecución ---------------------------------------------------------------------------------------------------------------------------------

class Env:
    def __init__(self, modes):
        self.modes = modes
        self.resources = svc.load_resources()
        svc.get_embedder()
        self.llm = None
        self.ollama_ok = False
        if "llm" in modes or "ajustado" in modes:
            try:
                from .llm import LLMService, check_ollama, create_provider
                from .schemas import Settings

                settings = Settings(llm_provider="ollama", ollama_model="qwen3:8b")
                check_ollama(settings)
                self.llm = LLMService(create_provider(settings), settings)
                self.ollama_ok = True
            except Exception:
                self.llm = None
        self.ft = FtClient(timeout=30.0)
        self.ft_ok = "ajustado" in modes and self.ft.healthy()
        self.cache = {}


def call_predict(env, mode, question, options, crosses=None, ft=None):
    llm = env.llm if mode == "llm" else None
    result = svc.predict(question, options, "single", PRESET[mode], resources=env.resources, service=llm, crosses=crosses, ft=ft)
    if mode in ("llm", "ajustado") and env.llm is not None:  # igual que la interfaz: texto del LLM si es válido, si no la plantilla
        explain.add_llm_text(result, env.llm, timeout=20.0)
    return result


def execute(env, mode, resolved):
    """('ok', resultado) | ('fuera_de_alcance', texto) | ('error_controlado', texto) | ('excepcion', texto) | ('cuelgue', texto)."""
    question, options = resolved.get("pregunta", ""), resolved.get("opciones") or []
    crosses = [resolved["cruce"]] if resolved.get("cruce") else None
    ft = FtClient("http://127.0.0.1:9", timeout=1.0) if resolved.get("servidor_apagado") else None
    pool = ThreadPoolExecutor(max_workers=1)
    started = time.monotonic()
    future = pool.submit(call_predict, env, mode, question, options, crosses, ft)
    try:
        result = future.result(timeout=TIMEOUT[mode])
        outcome = ("ok", result)
    except FutureTimeout:
        outcome = ("cuelgue", f"sin respuesta en {TIMEOUT[mode]:.0f} s")
    except OutOfScopeError as exc:
        outcome = ("fuera_de_alcance", f"{exc.coverage['label']}: {exc.coverage['explanation']}")
    except ServiceError as exc:
        outcome = ("error_controlado", str(exc))
    except Exception as exc:  # cualquier otra cosa es un fallo de la aplicación
        outcome = ("excepcion", f"{type(exc).__name__}: {exc}")
    finally:
        pool.shutdown(wait=False)
    return outcome, time.monotonic() - started


def tvd(a, b):
    keys = set(a) & set(b)
    return 0.5 * sum(abs(a[k] - b[k]) for k in keys) if keys else None


def weirdness(row, result):
    """Puntuación de rareza para elegir los casos más raros: empates exactos, masa extrema, opciones intermedias casi vacías, diferencia por permutación, inversión, ancla perdida."""
    score, why = 0, []
    dist = result["national"]["distribution"] if result else {}
    values = sorted(dist.values(), reverse=True)
    if values and values[0] >= 0.9:
        score += 2
        why.append(f"{100 * values[0]:.0f} % en una sola opción")
    if len(values) >= 3 and sum(1 for v in values if abs(v - values[0]) < 5e-4) >= 3:
        score += 2
        why.append("tres opciones con el mismo porcentaje")
    if len(dist) >= 4 and min(dist.values()) < 0.01:
        score += 1
        why.append("una opción con menos del 1 %")
    if row.get("dif_max_pp") and row["dif_max_pp"] > 10:
        score += 3
        why.append(f"el reparto se mueve {row['dif_max_pp']:.0f} pp al permutar las opciones")
    if row.get("invierte") is False:
        score += 1
        why.append("la pregunta negada no invierte el sentido")
    if row.get("tvd_ref") and row["tvd_ref"] > 0.15:
        score += 2
        why.append(f"difiere {100 * row['tvd_ref']:.0f} puntos de la pregunta de referencia")
    if row["estado"] != "pasa":
        score += 2
    return score, "; ".join(why)


def run_case(env, case, mode, by_id):
    resolved = resolve(case, by_id)
    row = {"id": case["id"], "categoria": case["categoria"], "subtipo": case.get("subtipo", ""), "modo_ejecucion": mode, "pregunta": (resolved.get("pregunta") or "")[:140].replace("\n", " "), "fallos": [], "avisos": [], "notas": []}
    fails, warns, notes = row["fallos"], row["avisos"], row["notas"]
    outcome, seconds = execute(env, mode, resolved)
    kind, payload = outcome
    row["segundos"] = round(seconds, 2)
    row["resultado"] = kind
    expect = case.get("espera", {})
    result = payload if kind == "ok" else None
    if kind in ("cuelgue", "excepcion"):
        fails.append(f"{kind}: {payload}")
    if kind == "ok":
        f, n = check_result(result, mode, resolved)
        fails += f
        notes += n
        row.update(modo_calculo=result["mode"], producido_por=result["producido_por"]["modo"], metodo_subgrupos=(result.get("subgroups_method") or {}).get("clave"),
                   fallback=bool(result["producido_por"]["fallback"]), intent=result.get("intent"), n_cruces=len(result.get("crosses") or []),
                   primera=max(result["national"]["distribution"], key=result["national"]["distribution"].get), explicacion=(result["explanation_text"] or {}).get("texto", "")[:400])
        row["distribucion"] = {o: round(100 * v, 1) for o, v in result["national"]["distribution"].items()}
    else:
        row["explicacion"] = str(payload)[:400]
    # expectativas por categoría
    if expect.get("fuera_de_alcance"):
        if kind == "ok":
            fails.append("simuló una pregunta que debía rechazarse")
        elif kind != "fuera_de_alcance":
            warns.append(f"rechazada como {kind} y no como fuera de alcance")
        elif not str(payload).strip():
            fails.append("rechazo sin explicación")
    if expect.get("inyeccion"):
        clean = dict(case, pregunta=case["pregunta_limpia"], espera={})
        clean_out, _ = execute(env, mode, resolve(clean, by_id))
        if kind == "ok":
            if clean_out[0] != "ok" or any(abs(result["national"]["distribution"][o] - clean_out[1]["national"]["distribution"][o]) > 1e-12 for o in result["options"]):
                fails.append("la inyección alteró las cifras respecto a la misma pregunta sin inyección")
            else:
                notes.append("simuló solo la pregunta legítima: cifras idénticas a las de la pregunta sin inyección")
        else:
            notes.append(f"no simuló ({kind})")
    if expect.get("sin_cuelgue") and kind in ("excepcion", "cuelgue"):
        pass  # ya registrado como fallo
    if kind in ("error_controlado", "fuera_de_alcance") and not expect.get("fuera_de_alcance") and not expect.get("inyeccion") and case["categoria"] not in ("limite",):
        fails.append(f"el caso debía simularse y se rechazó: {str(payload)[:120]}")
    if kind == "ok" and expect.get("modo") == "direct":
        reworded = str(case.get("pregunta", "")).startswith("Según el CIS")
        if result["mode"] != "direct":
            (warns if reworded else fails).append(f"no se encontró el ancla directa (modo {result['mode']})")
        elif expect.get("medido") == "exacto":
            if not result.get("measured"):
                fails.append("modo directo sin bloque medido")
            else:
                fails += independent_measured(result, resolved)
                text = result["explanation_text"]["texto"]
                m = result["measured"]
                if mode == "rapido" and (m["study_id"] not in text or str(m["national"]["n"]) not in text):
                    fails.append("la explicación no cita estudio y n del dato medido")
        if expect.get("intencion") and result.get("intent") != expect["intencion"]:
            fails.append(f"intención {result.get('intent')} en vez de {expect['intencion']}")
    if kind == "ok" and expect.get("modo") == "direct_o_prior" and result["mode"] == "none":
        warns.append("paráfrasis: no se encontró ninguna pregunta del CIS parecida (modo sin ancla)")
    if kind == "ok" and case["categoria"] == "subgrupo_lenguaje_natural":
        if result["mode"] == "direct":
            if not result.get("crosses"):
                fails.append("el cruce pedido no se calculó")
            elif result["measured"] is None or not result["measured"]["crosses"]:
                fails.append("el cruce no tiene dato medido")
            else:
                n = result["measured"]["crosses"][0]["n"]
                notes.append(f"n del cruce en el CIS: {n}")
    if kind == "ok" and case.get("servidor_apagado"):
        p = result["producido_por"]
        if result["mode"] != "direct" and not (p["fallback"] and p["modo"] == "rapido"):
            fails.append("con el servidor apagado debía caer al modo rápido con aviso")
        if result["mode"] != "direct" and not any("no respondió" in w for w in result["warnings"]):
            fails.append("el fallback no muestra aviso visible")
    # comparaciones con otro caso (permutación, paráfrasis, negada, texto)
    if kind == "ok":
        env.cache[(mode, case["id"])] = result
        ref_id = expect.get("referencia")
        if ref_id and (mode, ref_id) in env.cache:
            ref = env.cache[(mode, ref_id)]
            if case.get("subtipo") == "permutacion":
                diffs = [abs(result["national"]["distribution"][o] - ref["national"]["distribution"][o]) for o in result["options"] if o in ref["national"]["distribution"]]
                row["dif_max_pp"] = round(100 * max(diffs), 2) if diffs else None
                if row["dif_max_pp"] and row["dif_max_pp"] > 10:
                    warns.append(f"el reparto cambia {row['dif_max_pp']:.1f} pp al permutar las opciones")
                if result["mode"] != ref["mode"]:
                    warns.append(f"al permutar cambia el modo ({ref['mode']} → {result['mode']})")
            else:
                d = tvd(result["national"]["distribution"], ref["national"]["distribution"])
                row["tvd_ref"] = None if d is None else round(d, 4)
                if result["mode"] != ref["mode"]:
                    warns.append(f"cambia el modo respecto a la pregunta original ({ref['mode']} → {result['mode']})")
        if case.get("subtipo") == "negada" and (mode, case["negada_de"]) in env.cache:
            base = env.cache[(mode, case["negada_de"])]["national"]["distribution"]
            neg = result["national"]["distribution"]
            yes = next(o for o in neg if o.lower() in ("sí", "si"))
            row["invierte"] = bool(abs(neg[yes] - (1 - base[yes])) < abs(neg[yes] - base[yes]))
            notes.append(f"Sí base {100 * base[yes]:.0f} % → Sí en la negada {100 * neg[yes]:.0f} % ({'invierte' if row['invierte'] else 'no invierte'} el sentido)")
    # plan del LLM (solo con Ollama)
    if case["categoria"] == "subgrupo_lenguaje_natural" and mode == "rapido" and env.ollama_ok:
        schema = mrp_service.schema_of(env.resources.population.frame)
        plan = assist.plan_question(case["pregunta_nl"], env.llm, schema)
        if not plan["ok"]:
            warns.append(f"plan del LLM no usado: {plan['reason']}")
        else:
            got = {(i["var"], i["val"]) for i in plan["plan"]["cruce"]}
            want = {(i["var"], i["val"]) for i in case["cruce_esperado"]}
            if case.get("vago"):
                if got <= want:
                    notes.append(f"plan (grupo vago): {sorted(got)}")
                elif plan.get("aviso"):
                    warns.append(f"plan (grupo vago) con aviso: {plan['aviso']}")
                else:
                    fails.append(f"plan: traduce un grupo vago a {sorted(got)} sin avisar")
            elif got != want:
                fails.append(f"el plan no traduce bien el grupo: {sorted(got)} en vez de {sorted(want)}")
            else:
                notes.append("el plan traduce el grupo al esquema del CIS")
    elif case["categoria"] == "subgrupo_lenguaje_natural" and mode == "rapido":
        notes.append("plan del LLM no evaluado (sin Ollama)")
    row["estado"] = "falla" if fails else ("aviso" if warns else "pasa")
    row["rareza"], row["rareza_motivos"] = weirdness(row, result)
    return row


def select_cases(cases, modes, only=None, limit=None):
    plan = []
    for case in cases:
        if only and not any(case["id"].startswith(p) for p in only):
            continue
        case_modes = case.get("modos") or ["rapido"]
        for mode in modes:
            if mode in case_modes:
                plan.append((case, mode))
    return plan[:limit] if limit else plan


def run(modes=("rapido",), only=None, limit=None, log=print):
    cases = load_cases()
    by_id = {c["id"]: c for c in cases}
    env = Env(modes)
    if "llm" in modes and not env.ollama_ok:
        log("AVISO: Ollama no responde; se omite el modo llm")
    if "ajustado" in modes and not env.ft_ok:
        log("AVISO: el servidor del modelo ajustado no responde; solo se ejecutan los casos de fallback con el servidor apagado")
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    plan = select_cases(cases, modes, only, limit)
    # las referencias (permutaciones, paráfrasis, texto) necesitan que el caso base se haya ejecutado antes en el mismo modo
    needed = {(c.get("espera", {}).get("referencia") or c.get("negada_de"), m) for c, m in plan}
    done = {(c["id"], m) for c, m in plan}
    for ref, mode in sorted(x for x in needed if x[0]):
        if (ref, mode) not in done:
            plan.insert(0, (by_id[ref], mode))
    for n, (case, mode) in enumerate(plan, 1):
        if mode == "llm" and not env.ollama_ok:
            continue
        if mode == "ajustado" and not env.ft_ok and not case.get("servidor_apagado"):
            continue
        rows.append(run_case(env, case, mode, by_id))
        log(f"[{n}/{len(plan)}] {case['id']} {mode}: {rows[-1]['estado']} ({rows[-1]['segundos']} s) {'; '.join(rows[-1]['fallos'])[:120]}")
    write_outputs(rows)
    return rows


def write_outputs(rows):
    columns = ["id", "categoria", "subtipo", "modo_ejecucion", "estado", "resultado", "segundos", "modo_calculo", "producido_por", "metodo_subgrupos", "fallback", "intent", "n_cruces", "dif_max_pp", "invierte",
               "tvd_ref", "rareza", "rareza_motivos", "pregunta", "fallos", "avisos", "notas", "primera", "distribucion", "explicacion"]
    with open(OUT / "resultados.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "fallos": " | ".join(r["fallos"]), "avisos": " | ".join(r["avisos"]), "notas": " | ".join(r["notas"])})
    (OUT / "informe.md").write_text(report(rows), encoding="utf-8")


def report(rows):
    df = pd.DataFrame(rows)
    lines = [f"# Informe de la batería ({len(df)} ejecuciones, {time.strftime('%Y-%m-%d %H:%M')})", "", "## Resultado por categoría y modo", "| Categoría | Modo | Casos | Pasa | Aviso | Falla |", "|---|---|---|---|---|---|"]
    for (cat, mode), g in df.groupby(["categoria", "modo_ejecucion"]):
        lines.append(f"| {cat} | {mode} | {len(g)} | {(g.estado == 'pasa').sum()} | {(g.estado == 'aviso').sum()} | {(g.estado == 'falla').sum()} |")
    lines += ["", "## Tiempos (segundos, mediana y máximo por modo)", "| Modo | Mediana | Máximo |", "|---|---|---|"]
    for mode, g in df[df.resultado == "ok"].groupby("modo_ejecucion"):
        lines.append(f"| {mode} | {statistics.median(g.segundos):.1f} | {g.segundos.max():.1f} |")
    perm = df[df.dif_max_pp.notna()] if "dif_max_pp" in df else df.iloc[0:0]
    if len(perm):
        lines += ["", "## Permutación de opciones (diferencia máxima por opción, en pp)", *[f"- {r.id} ({r.modo_ejecucion}): {r.dif_max_pp:.1f}" for r in perm.itertuples()], f"- **Máxima: {perm.dif_max_pp.max():.1f} pp**"]
    neg = df[df.invierte.notna()] if "invierte" in df else df.iloc[0:0]
    if len(neg):
        lines += ["", "## Preguntas negadas", *[f"- {r.id}: {'invierte' if r.invierte else 'NO invierte'} el sentido ({r.notas})" for r in neg.itertuples()]]
    bad = df[df.estado == "falla"]
    lines += ["", f"## Fallos ({len(bad)})", *[f"- {r.id} [{r.modo_ejecucion}] {r.pregunta[:70]}: {r.fallos}" for r in bad.itertuples()]]
    warn = df[df.estado == "aviso"]
    lines += ["", f"## Avisos ({len(warn)})", *[f"- {r.id} [{r.modo_ejecucion}]: {r.avisos}" for r in warn.itertuples()]]
    weird = df[df.resultado == "ok"].sort_values("rareza", ascending=False).head(10)
    lines += ["", "## Los 10 casos más raros", *[f"- {r.id} [{r.modo_ejecucion}] «{r.pregunta[:80]}» → {r.primera} ({r.rareza_motivos or 'sin motivo concreto'}). Explicación: {str(r.explicacion)[:220]}" for r in weird.itertuples()]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run"])
    parser.add_argument("--modes", default="rapido")
    parser.add_argument("--only", help="prefijos de ids separados por comas (C1,C2…)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(tuple(args.modes.split(",")), args.only.split(",") if args.only else None, args.limit)


if __name__ == "__main__":
    main()
