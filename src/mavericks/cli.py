import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .anchor import load_anchor_store
from .context import ContextStore
from .evaluation import backtest, save_json, summarize_backtest
from .evidence import EvidenceRepository, SemanticIndex, read_records, survey_distribution
from .graph import simulate
from .llm import LLMService, check_ollama, create_provider
from .population import PopulationLoader, PopulationRepository, build_segments, create_demo
from .report import compare, comparison_table
from .report import report as build_report
from .schemas import Question, Settings
from .sharpen import apply_to_results, fit_by_mode, interval_coverage
from .statistics import build_calibration, load_calibration


def parser():
    root = argparse.ArgumentParser(description="GenAI Mavericks — simulación poblacional fundamentada")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("create-demo")
    check = commands.add_parser("check-model")
    check.add_argument("--model")
    for name in ["inspect-population", "build-segments", "simulate", "backtest"]:
        sub = commands.add_parser(name)
        sub.add_argument("--path", default="data/fixtures/population.csv")
        sub.add_argument("--metadata")
        sub.add_argument("--segments", type=int, default=5)
        if name == "build-segments":
            sub.add_argument("--dimensions", nargs="*", help="Dimensiones de segmentación por orden de prioridad")
        sub.add_argument("--output")
        if name in {"simulate", "backtest"}:
            sub.add_argument("--provider", choices=["ollama", "mock", "external"])
            sub.add_argument("--model")
            sub.add_argument("--db", default="data/processed/evidence.sqlite")
            sub.add_argument("--concurrency", type=int)
            sub.add_argument("--anchor", choices=["auto", "off"], default="auto", help="Ancla empírica CIS")
            sub.add_argument("--stability-runs", type=int, help="Repeticiones por segmento para medir estabilidad")
            sub.add_argument("--retriever", choices=["tfidf", "embed", "hybrid"], help="Recuperador de evidencia")
            sub.add_argument("--embedding-model", help="Modelo multilingüe local para embed/hybrid")
            sub.add_argument("--embedding-device", help="cpu o cuda")
            sub.add_argument("--context", action="store_true", help="Añade contexto de actualidad fechado")
            sub.add_argument("--sharpen", action="store_true", help="Aplica el ajuste de nitidez de calibration.json")
            sub.add_argument("--analog", action="store_true", help="Prior por analogía (y sus ejemplos) en el prompt")
            sub.add_argument("--corrector", type=float, help="LLM corrector: ajuste máximo ±X por opción sobre el prior por analogía")
            sub.add_argument("--blend", action="store_true", help="Mezcla calibrada LLM+prior de calibration.json")
            sub.add_argument("--calibration-in", help="calibration.json a LEER en backtest (--calibration es de escritura)")
            sub.add_argument("--sources", help="Familias de evidencia permitidas: cis,eurobarometro,ess")
            sub.add_argument("--few-shot", type=int, help="Nº de ejemplos reales en el prompt (0 = apagado)")
            sub.add_argument("--no-llm-assessment", action="store_true", help="Evaluador de evidencia determinista (sin LLM)")
            sub.add_argument("--calibration", help="JSON de calibración (por defecto data/processed/calibration.json) u off")
            sub.add_argument("--cis-dir", help="Carpeta con cis_respondents/cis_responses (por defecto data/processed)")
        if name == "simulate":
            sub.add_argument("--question", required=True)
            sub.add_argument("--options", nargs="+", required=True)
            sub.add_argument("--question-type", choices=["single", "multiple"], default="single")
            sub.add_argument("--rag", choices=["on", "off"], default="on")
        if name == "backtest":
            sub.add_argument("--cases", required=True)
            sub.add_argument("--resume", action="store_true", help="Continúa desde el .partial.jsonl")
    ingest = commands.add_parser("ingest-surveys")
    ingest.add_argument("--path", required=True)
    ingest.add_argument("--db", default="data/processed/evidence.sqlite")
    ingest.add_argument("--respondents")
    ingest.add_argument("--responses")
    ingest.add_argument("--weight", choices=["peso", "peso_ccaa"], default="peso")
    ingest.add_argument("--answer-column", default="answer")
    ingest.add_argument("--question-type", choices=["single", "multiple"], default="single")
    ingest.add_argument("--include-reserved", action="store_true", help="Incluye estudios reservados (solo backtesting)")
    cases = commands.add_parser("make-backtest-cases", help="Casos de backtest desde los estudios reservados")
    cases.add_argument("--cis-dir", default="data/processed")
    cases.add_argument("--out", default="data/processed/backtest_cases.json")
    cases.add_argument("--max-per-study", type=int, default=8)
    cases.add_argument("--min-n", type=int, default=500)
    cases.add_argument("--seed", type=int, default=42)
    cases.add_argument("--mode", choices=["temporal", "strict"], default="temporal")
    cases.add_argument("--studies", nargs="*")
    cases.add_argument("--split", choices=["test", "dev"], default="test", help="test = reservados; dev = no reservados recientes")
    cases.add_argument("--topics", nargs="*", help="Solo con --split dev")
    cases.add_argument("--since", help="Solo con --split dev: inicio de campo mínimo (AAAA-MM-DD)")
    search = commands.add_parser("search-evidence")
    search.add_argument("--query", required=True)
    search.add_argument("--db", default="data/processed/evidence.sqlite")
    etl = commands.add_parser("etl-cis", help="Convierte los zips del CIS al contrato de la Fase 1")
    etl.add_argument("--raw", default="maverick_data/raw")
    etl.add_argument("--catalog", default="maverick_data/catalogo.csv")
    etl.add_argument("--out", default="data/processed")
    etl.add_argument("--only", nargs="*", help="Números de estudio CIS (p. ej. 3546 3535)")
    pop = commands.add_parser("etl-population", help="Genera la población sintética (Censo + ECP + EPA + ECV)")
    pop.add_argument("--raw", default="maverick_data/raw")
    pop.add_argument("--out", default="data/processed")
    pop.add_argument("--n", type=int, default=50_000)
    pop.add_argument("--seed", type=int, default=42)
    pop.add_argument("--period", default="1 de enero de 2026", help="Periodo de la ECP (anterior al 30/06/2026)")
    ext = commands.add_parser("etl-external", help="Eurobarómetro y ESS (solo España) como evidencia adicional")
    ext.add_argument("--raw", default="maverick_data/raw")
    ext.add_argument("--out", default="data/processed_v2")
    ext.add_argument("--only", nargs="*", help="Eurobarómetros (ZA7848) o ESS (ESS11, ESS10SC) concretos")
    ext.add_argument("--catalog", default="maverick_data/catalogo.csv", help="Catálogo con las fechas de campo en el título")
    ctx = commands.add_parser("etl-context", help="Indexa raw/*/contexto con fecha de publicación")
    ctx.add_argument("--raw", default="maverick_data/raw")
    ctx.add_argument("--out", default="data/processed_v2/context.json")
    fit = commands.add_parser("fit-sharpening", help="Aprende el tau de nitidez por modo (validación cruzada) y recalibra")
    fit.add_argument("--results", required=True, help="JSON de un backtest")
    fit.add_argument("--cases", default="data/processed/backtest_cases.json")
    fit.add_argument("--out", default="data/processed/calibration.json")
    fit.add_argument("--seed", type=int, default=0)
    nat = commands.add_parser("build-national", help="Tabla de distribuciones nacionales del CIS (para análogas)")
    nat.add_argument("--cis-dir", default="data/processed")
    nat.add_argument("--out", default="data/processed_v2/cis_national.json")
    ana = commands.add_parser("analog-eval", help="Prior por analogía sin LLM: calidad por similitud y ajuste por validación cruzada")
    ana.add_argument("--cases", nargs="+", required=True)
    ana.add_argument("--db", default="data/processed_v2/evidence_v2.sqlite")
    ana.add_argument("--table", default="data/processed_v2/cis_national.json")
    ana.add_argument("--out", required=True, help="Caché de candidatas (JSON)")
    ana.add_argument("--retriever", choices=["tfidf", "embed", "hybrid"], default="tfidf")
    ana.add_argument("--embedding-model")
    ana.add_argument("--sources", help="cis,eurobarometro,ess")
    ana.add_argument("--previous-waves", action="store_true", help="No excluir el texto idéntico (modo temporal)")
    ana.add_argument("--reuse", action="store_true", help="Reutilizar la caché de candidatas de --out")
    ana.add_argument("--tune", action="store_true")
    full = commands.add_parser("analog-full", help="Pipeline sin LLM: total por analogía + segmentos desde microdatos + raking")
    full.add_argument("--cases", nargs="+", required=True)
    full.add_argument("--db", default="data/processed_v2/evidence_v2.sqlite")
    full.add_argument("--table", default="data/processed_v2/cis_national.json")
    full.add_argument("--population", default="data/processed/synthetic_population.parquet")
    full.add_argument("--cis-dir", default="data/processed")
    full.add_argument("--segments", type=int, default=10)
    full.add_argument("--sources", default="cis")
    full.add_argument("--out", required=True)
    full.add_argument("--baseline", help="backtest.json de la línea base para comparar subgrupos en los mismos casos")
    full.add_argument("--k", type=int, default=5)
    full.add_argument("--power", type=float, default=4.0)
    full.add_argument("--min-sim", type=float, default=0.0)
    full.add_argument("--shrink", type=float, default=0.0)
    full.add_argument("--sim-mode", default="lex")
    tr = commands.add_parser("translate-external", help="Traduce al español (una vez, con caché) las fuentes europeas")
    tr.add_argument("--in", dest="source", default="data/processed_v2/external_evidence.json")
    tr.add_argument("--out", default="data/processed_v2/external_evidence_es.json")
    tr.add_argument("--cache", default="data/processed_v2/translations_es.json")
    tr.add_argument("--model")
    tr.add_argument("--batch", type=int, default=25)
    tr.add_argument("--limit", type=int, help="Solo los primeros N registros (prueba)")
    nz = commands.add_parser("noise-floor", help="Suelo de ruido de los subgrupos frente a plano, desviaciones empíricas y LLM")
    nz.add_argument("--cases", nargs="+", required=True)
    nz.add_argument("--baseline", nargs="+", required=True, help="backtest.json de la línea base (oficial y, si existe, desarrollo)")
    nz.add_argument("--db", default="data/processed_v2/evidence_v2.sqlite")
    nz.add_argument("--table", default="data/processed_v2/cis_national.json")
    nz.add_argument("--cis-dir", default="data/processed")
    nz.add_argument("--draws", type=int, default=200)
    nz.add_argument("--direct-k", type=int, default=1, help="Oleadas de la misma pregunta a mezclar en los directos")
    nz.add_argument("--sources", default="cis")
    nz.add_argument("--out", required=True)
    cf = commands.add_parser("case-floors", help="Suelo de ruido (nacional y de subgrupos) de cada caso")
    cf.add_argument("--cases", nargs="+", required=True)
    cf.add_argument("--cis-dir", default="data/processed")
    cf.add_argument("--draws", type=int, default=100)
    cf.add_argument("--out", required=True)
    blend_ = commands.add_parser("fit-blend", help="Mezcla LLM+prior y nitidez conjuntas, validadas dejando un tema fuera")
    blend_.add_argument("--results", nargs="+", required=True, help="backtest.json oficial (primero) y de desarrollo")
    blend_.add_argument("--analog-cache", nargs="+", required=True)
    blend_.add_argument("--k", type=int, default=5)
    blend_.add_argument("--power", type=float, default=4.0)
    blend_.add_argument("--min-sim", type=float, default=0.0)
    blend_.add_argument("--shrink", type=float, default=0.0)
    blend_.add_argument("--sim-mode", default="lex")
    blend_.add_argument("--objective", choices=["jsd", "logloss", "mae"], default="jsd")
    blend_.add_argument("--out", help="Escribe/actualiza la sección blend de este calibration.json")
    rep = commands.add_parser("report-backtest", help="Tablas por modo y tema de un backtest")
    rep.add_argument("--results", required=True)
    rep.add_argument("--title", default="Backtest")
    rep.add_argument("--out")
    cmp_ = commands.add_parser("compare-backtests", help="Compara configuraciones caso a caso contra una base")
    cmp_.add_argument("--base", required=True)
    cmp_.add_argument("--others", nargs="+", required=True, help="nombre=ruta.json")
    cmp_.add_argument("--out")
    stats = commands.add_parser("survey-stats")
    stats.add_argument("--respondents", required=True)
    stats.add_argument("--responses", required=True)
    stats.add_argument("--question-uid", required=True)
    stats.add_argument("--weight", choices=["peso", "peso_ccaa"], default="peso")
    stats.add_argument("--group-by")
    stats.add_argument("--answer-column", default="answer")
    stats.add_argument("--question-type", choices=["single", "multiple"], default="single")
    stats.add_argument("--output")
    return root


def load_results(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["results"] if isinstance(raw, dict) else raw


def run(args):
    updates = {}
    if getattr(args, "provider", None):
        updates["llm_provider"] = args.provider
    if getattr(args, "model", None):
        updates["ollama_model"] = args.model
    if getattr(args, "analog", False):
        updates["analog"] = True
    if getattr(args, "corrector", None):
        updates["corrector"] = args.corrector
    if getattr(args, "blend", False):
        updates["blend"] = True
    if getattr(args, "sharpen", False):
        updates["sharpen"] = True
    if getattr(args, "context", False):
        updates["context"] = True
    if getattr(args, "sources", None):
        updates["sources"] = args.sources
    if getattr(args, "few_shot", None) is not None:
        updates["few_shot"] = args.few_shot
    for flag in ("retriever", "embedding_model", "embedding_device"):
        if getattr(args, flag, None):
            updates[flag] = getattr(args, flag)
    if getattr(args, "no_llm_assessment", False):
        updates["llm_assessment"] = False
    if getattr(args, "stability_runs", None):
        updates["stability_runs"] = args.stability_runs
    if getattr(args, "concurrency", None) is not None:
        updates["llm_max_concurrency"] = args.concurrency
    settings = Settings(**updates)

    def evidence_repository(path):
        index = (
            SemanticIndex(settings.embedding_model_path, settings.chroma_path)
            if settings.embedding_model_path
            else None
        )
        embedder = None
        if settings.retriever != "tfidf":
            from .embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
        return EvidenceRepository(path, index, embedder=embedder, retriever=settings.retriever)

    if args.command == "survey-stats":
        import pandas as pd

        value = survey_distribution(
            pd.DataFrame(read_records(args.respondents)),
            pd.DataFrame(read_records(args.responses)),
            args.question_uid,
            weight=args.weight,
            group_by=args.group_by,
            question_type=args.question_type,
            option_column=args.answer_column,
        )
        if args.output:
            save_json(args.output, value)
        return value
    if args.command == "etl-cis":
        from .etl.cis import run_etl

        return run_etl(args.raw, args.catalog, args.out, set(args.only) if args.only else None)
    if args.command == "etl-population":
        from .etl.population import run_population_etl

        return run_population_etl(args.raw, args.out, args.n, args.seed, args.period)
    if args.command == "make-backtest-cases":
        from .backtest_cases import build_cases

        built = build_cases(
            args.cis_dir, args.max_per_study, args.min_n, args.seed, args.mode == "temporal", args.studies,
            args.split, args.topics, args.since,
        )
        save_json(args.out, built)
        import collections

        return {
            "cases": len(built),
            "out": args.out,
            "split": args.split,
            "by_topic": dict(collections.Counter(c["topic"] for c in built)),
            "studies": sorted({c["estudio_id"] for c in built}),
        }
    if args.command == "translate-external":
        from .translate import apply_translation, load_cache, save_cache, texts_to_translate, translate_texts

        records = json.loads(Path(args.source).read_text(encoding="utf-8"))[: args.limit]
        cache = load_cache(args.cache)
        wanted = texts_to_translate(records)
        todo = len([t for t in wanted if t not in cache])
        if todo:
            service = LLMService(create_provider(settings), settings)
            cache = asyncio.run(
                translate_texts(wanted, service, cache, args.batch, on_batch=lambda c: save_cache(args.cache, c))
            )
            save_cache(args.cache, cache)
        translated = [apply_translation(r, cache) for r in records]
        save_json(args.out, translated)
        return {"records": len(translated), "texts": len(wanted), "translated_now": todo, "out": args.out}
    if args.command == "case-floors":
        import numpy as np

        from .anchor import AnchorStore
        from .noise import case_floors

        anchors = AnchorStore(args.cis_dir)
        rng = np.random.default_rng(0)
        floors = {}
        for case in (c for path in args.cases for c in json.loads(Path(path).read_text(encoding="utf-8-sig"))):
            floors[case["evidence_id"]] = case_floors(case, anchors, rng, args.draws)
        save_json(args.out, floors)
        values = [f["national_floor"] for f in floors.values() if f["national_floor"] is not None]
        return {"cases": len(floors), "mean_national_floor": float(np.mean(values)), "out": args.out}
    if args.command == "noise-floor":
        import numpy as np

        from .analog import AnalogConfig, collect_case, combine, load_national_table
        from .anchor import AnchorStore
        from .noise import subgroup_rows
        from .noise import summarize as summarize_rows

        configs = {"hard": AnalogConfig(5, 4.0, 0.0, 0.0, "lex"), "direct": AnalogConfig(args.direct_k, 4.0, 0.0, 0.0, "lex")}
        repo = evidence_repository(args.db)
        table = load_national_table(args.table)
        anchors = AnchorStore(args.cis_dir)
        sources = set(args.sources.split(",")) if args.sources else None
        base = {r["evidence_id"]: r for path in args.baseline for r in load_results(path) if r["status"] == "completed"}
        rng = np.random.default_rng(0)
        rows = []
        for case in (c for path in args.cases for c in json.loads(Path(path).read_text(encoding="utf-8-sig"))):
            result_base = base.get(case["evidence_id"])
            if result_base is None:
                continue
            group = "direct" if result_base["anchor_mode"] == "direct" else "hard"
            # difíciles: modo estricto (texto idéntico excluido); directos: oleadas anteriores de la misma pregunta
            collected = collect_case(case, repo, table, repo.embedder, group == "hard", sources=sources)
            result = combine(collected["candidates"], collected["options"], configs[group])
            options = collected["options"]

            def empirical(dimension, category, result=result, options=options):
                uniform = np.full(len(options), 1 / len(options))
                acc, total = np.zeros(len(options)), 0.0
                for weight, c in result["weighted"]:
                    dist = c["dist"]
                    if c.get("microdata") and c.get("study_id"):
                        spec = {"study_id": c["study_id"], "evidence_id": c["evidence_id"], "mapping": c["mapping"], "question_type": "single"}
                        try:
                            dist = anchors.segment_distribution(spec, {dimension: [category]})["probabilities"]
                        except Exception:
                            dist = c["dist"]
                    vector = np.array([dist[o] for o in options], dtype=float)
                    acc += weight * vector / max(vector.sum(), 1e-12)
                    total += weight
                if total <= 0:
                    return dict(zip(options, uniform))
                mix = acc / total
                return dict(zip(options, (result["trust"] * mix + (1 - result["trust"]) * uniform).tolist()))

            llm = result_base.get("predicted_subgroups") or {}
            predictors = {
                "flat": lambda d, c, result=result: result["prior"],
                "empirical": empirical,
                "llm": lambda d, c, llm=llm: (llm.get(d) or {}).get(c),
            }
            for row in subgroup_rows(case, anchors, predictors, rng, args.draws):
                rows.append({**row, "group": group, "split": case.get("split", "test")})
        output = {}
        for group in ("hard", "direct"):
            part = [r for r in rows if r["group"] == group]
            with_llm = [r for r in part if r["llm"] is not None]
            output[group] = {
                "all_subgroups": summarize_rows(part, ["flat", "empirical"]),
                "subgroups_the_llm_published": summarize_rows(with_llm, ["flat", "empirical", "llm"]),
            }
        save_json(args.out, {"summary": output, "rows": rows})
        return output
    if args.command == "analog-full":
        import time

        import numpy as np

        from .analog import AnalogConfig, collect_case, combine, load_national_table
        from .analog_segments import predict
        from .anchor import AnchorStore
        from .evaluation import subgroup_errors

        config = AnalogConfig(args.k, args.power, args.min_sim, args.shrink, args.sim_mode)
        frame, metadata = PopulationLoader().load(args.population)
        population = PopulationRepository(frame, metadata)
        repo = evidence_repository(args.db)
        table = load_national_table(args.table)
        anchors = AnchorStore(args.cis_dir)
        sources = set(args.sources.split(",")) if args.sources else None
        cases = [c for path in args.cases for c in json.loads(Path(path).read_text(encoding="utf-8-sig"))]
        rows = []
        for case in cases:
            started = time.monotonic()
            collected = collect_case(case, repo, table, repo.embedder, True, sources=sources)
            result = combine(collected["candidates"], collected["options"], config)
            observed = case["observed_distribution"]
            row = {"evidence_id": case["evidence_id"], "topic": case["topic"], "split": case.get("split", "test"),
                   "best_similarity": result["best_similarity"], "used": result["used"]}
            for name, flag in (("dev", True), ("flat", False)):
                out = predict(result, collected["options"], population, anchors, args.segments, deviations=flag)
                scores = subgroup_errors(out["subgroups"], case.get("observed_subgroups", {}))
                row[f"{name}_subgroup_mae"] = float(np.mean([s["mae"] for s in scores])) if scores else None
                row[f"{name}_subgroups"] = len(scores)
                row["national_mae"] = float(np.mean([abs(out["national"][o] - observed[o]) for o in observed]))
            row["seconds"] = time.monotonic() - started
            rows.append(row)
        base = {}
        if args.baseline:
            base = {r["evidence_id"]: r for r in load_results(args.baseline) if r["status"] == "completed"}
        summary = {}
        for split in sorted({r["split"] for r in rows}):
            part = [r for r in rows if r["split"] == split and r["dev_subgroup_mae"] is not None]
            shared = [r for r in part if r["evidence_id"] in base and base[r["evidence_id"]].get("subgroup_mae") is not None]
            summary[split] = {
                "cases": len([r for r in rows if r["split"] == split]),
                "national_mae": float(np.mean([r["national_mae"] for r in rows if r["split"] == split])),
                "subgroup_mae_with_deviations": float(np.mean([r["dev_subgroup_mae"] for r in part])) if part else None,
                "subgroup_mae_flat": float(np.mean([r["flat_subgroup_mae"] for r in part])) if part else None,
                "seconds_per_case": float(np.mean([r["seconds"] for r in rows if r["split"] == split])),
                "baseline_same_cases": {
                    "n": len(shared),
                    "baseline_subgroup_mae": float(np.mean([base[r["evidence_id"]]["subgroup_mae"] for r in shared])) if shared else None,
                    "analog_subgroup_mae": float(np.mean([r["dev_subgroup_mae"] for r in shared])) if shared else None,
                    "analog_flat_subgroup_mae": float(np.mean([r["flat_subgroup_mae"] for r in shared])) if shared else None,
                },
            }
        save_json(args.out, {"summary": summary, "rows": rows})
        return summary
    if args.command == "fit-blend":
        from .analog import AnalogConfig
        from .calibrate import build_rows, compare_variants

        config = AnalogConfig(args.k, args.power, args.min_sim, args.shrink, args.sim_mode)
        cache = [c for path in args.analog_cache for c in json.loads(Path(path).read_text(encoding="utf-8"))]
        out = {}
        for i, path in enumerate(args.results):
            split = "test" if i == 0 else "dev"
            out[split] = {"rows": len(build_rows(load_results(path), cache, config, split=split))}
        official = build_rows(load_results(args.results[0]), cache, config, split="test")
        dev = [r for path in args.results[1:] for r in build_rows(load_results(path), cache, config, split="dev")]
        out["official_only"] = compare_variants(official, args.objective)
        if dev:
            out["official_plus_dev_cv"] = compare_variants(official + dev, args.objective)
        if args.out:
            from .calibrate import fit as fit_blend_model

            edges = (0.0, 0.5, 1.01)
            model = fit_blend_model(official + dev, edges, args.objective)
            target = Path(args.out)
            calibration = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            calibration["blend"] = {
                "edges": list(edges),
                "params": [list(p) for p in model.params],
                "objective": args.objective,
                "fitted_on": {"official": len(official), "dev": len(dev)},
                "analog_config": {"k": args.k, "power": args.power, "min_sim": args.min_sim, "shrink": args.shrink, "sim_mode": args.sim_mode},
            }
            save_json(target, calibration)
            out["blend_written"] = {"path": str(target), "params": calibration["blend"]["params"]}
        return out
    if args.command == "build-national":
        from .analog import build_national_table

        return {"questions": build_national_table(args.cis_dir, args.out), "out": args.out}
    if args.command == "analog-eval":
        from .analog import (
            AnalogConfig,
            by_kind,
            by_similarity,
            collect_case,
            default_grid,
            evaluate,
            load_national_table,
            summarize,
            tune,
        )

        if args.reuse and Path(args.out).exists():
            collected = json.loads(Path(args.out).read_text(encoding="utf-8"))
        else:
            cases = [c for path in args.cases for c in json.loads(Path(path).read_text(encoding="utf-8-sig"))]
            repo = evidence_repository(args.db)
            table = load_national_table(args.table)
            sources = set(args.sources.split(",")) if args.sources else None
            collected = [
                collect_case(c, repo, table, repo.embedder, not args.previous_waves, sources=sources) for c in cases
            ]
            save_json(args.out, collected)
        output = {"cases": len(collected), "by_split": {}}
        for split in sorted({c["split"] for c in collected}):
            part = [c for c in collected if c["split"] == split]
            rows = evaluate(part, AnalogConfig())
            output["by_split"][split] = {
                "default": summarize(rows),
                "by_best_similarity": by_similarity(rows),
                "by_alignment": {str(k): v for k, v in by_kind(rows).items()},
                "without_any_analog": sum(1 for c in part if not c["candidates"]),
            }
        if args.tune:
            result = tune(collected, default_grid())
            output["tuned"] = {
                "final_config": vars(result["final"]),
                "cv": result["cv"],
                "default": result["default"],
                "in_sample_best": result["in_sample_best"],
                "by_split_cv": {
                    s: summarize([r for r in result["held_rows"] if r["split"] == s]) for s in sorted({c["split"] for c in collected})
                },
                "cv_by_best_similarity": by_similarity(result["held_rows"]),
            }
        return output
    if args.command == "report-backtest":
        raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
        text = build_report(raw["results"] if isinstance(raw, dict) else raw, args.title)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return {"report": args.out}
    if args.command == "compare-backtests":
        others = {spec.split("=", 1)[0]: load_results(spec.split("=", 1)[1]) for spec in args.others}
        comparison = compare(load_results(args.base), others)
        text = comparison_table(comparison)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return comparison
    if args.command == "fit-sharpening":
        raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
        results = raw["results"] if isinstance(raw, dict) else raw
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        report = fit_by_mode(results, seed=args.seed)
        corrected = apply_to_results(results, report, cases)
        calibration = build_calibration(
            corrected, None, None, settings.stability_runs, settings.min_calibration_cases, sharpening=report
        )
        save_json(args.out, calibration)
        return {
            "calibration": args.out,
            "sharpening": {
                m: {k: v for k, v in info.items() if not k.startswith("_") and k != "corrected_cv"}
                for m, info in report.items()
            },
            "interval_coverage_leave_one_out": interval_coverage(corrected),
        }
    if args.command == "etl-context":
        from .context import build_context_index

        index = build_context_index(args.raw)
        save_json(args.out, index)
        return {"fragments": len(index["entries"]), "documents": len({e["doc"] for e in index["entries"]}), "skipped": index["skipped"]}
    if args.command == "etl-external":
        from .etl.external import run_external_etl

        return run_external_etl(args.raw, args.out, set(args.only) if args.only else None, args.catalog)
    if args.command == "create-demo":
        path = create_demo("data/fixtures")
        EvidenceRepository().ingest(path.parent / "evidence.json")
        return {"label": "DEMO / SYNTHETIC", "population_path": str(path)}
    if args.command == "check-model":
        return check_ollama(settings)
    if args.command == "ingest-surveys":
        if bool(args.respondents) != bool(args.responses):
            raise ValueError("Indique --respondents y --responses juntos")
        repo = evidence_repository(args.db)
        processed = repo.ingest(args.path, include_reserved=args.include_reserved)
        attached = (
            repo.attach_microdata(
                args.respondents, args.responses, args.weight, args.answer_column, args.question_type
            )
            if args.respondents
            else 0
        )
        return {
            "processed": processed,
            "skipped_reserved": repo.skipped_reserved,
            "questions_with_microdata": attached,
        }
    if args.command == "search-evidence":
        return evidence_repository(args.db).search(args.query)
    frame, metadata = PopulationLoader().load(args.path, args.metadata)
    repository = PopulationRepository(frame, metadata)
    if args.command == "inspect-population":
        return {
            "n_synthetic": len(frame),
            "total_population_weight": float(frame.peso.sum()),
            "metadata": metadata.model_dump(),
            "missing_values": frame.isna().sum().to_dict(),
            "warnings": []
            if metadata.categories
            else ["Sin catálogo: categorías no verificadas contra Fase 1"],
        }
    if args.command == "build-segments":
        return build_segments(repository, args.segments, dimensions=args.dimensions)
    provider = create_provider(settings)

    def factory():
        return LLMService(provider, settings)

    evidence = evidence_repository(args.db)
    anchors = None if args.anchor == "off" else load_anchor_store(settings, args.cis_dir)
    context_store = ContextStore.load(settings.context_path, args.cis_dir or settings.cis_dir) if settings.context else None
    calibration_path = None if args.calibration == "off" else (args.calibration or settings.calibration_path)
    if args.command == "simulate":
        calibration = load_calibration(calibration_path)
    else:
        calibration = load_calibration(getattr(args, "calibration_in", None))
    if args.command == "simulate":
        result = asyncio.run(
            simulate(
                Question(question=args.question, options=args.options, question_type=args.question_type),
                repository,
                evidence,
                factory(),
                args.segments,
                args.rag == "on",
                anchors=anchors,
                calibration=calibration,
                context_store=context_store,
            )
        )
        output = args.output or f"outputs/{result['simulation_id']}.json"
    else:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8-sig"))
        total_cases = len(cases)
        output = args.output or "outputs/backtest.json"
        partial = Path(output).with_suffix(".partial.jsonl")
        done = []
        if args.resume and partial.exists():
            done = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
            finished = {r["evidence_id"] for r in done}
            cases = [c for c in cases if c["evidence_id"] not in finished]
        elif partial.exists():
            partial.unlink()
        partial.parent.mkdir(parents=True, exist_ok=True)

        counter = {"n": len(done)}

        def keep(row):
            counter["n"] += 1
            label = row["evidence_id"].removeprefix("CIS")
            mae_text = "FALLIDO" if row["mae"] is None else f"MAE {row['mae']:.3f}"
            logging.info(
                "[caso %d/%d] %s · %s · %s · %s · %.0f s",
                counter["n"], total_cases, label, row.get("topic"), row.get("anchor_mode"), mae_text, row.get("execution_time", 0),
            )
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        results = done + asyncio.run(
            backtest(cases, repository, evidence, factory, args.segments, anchors=anchors, calibration=calibration, on_result=keep, context_store=context_store)
        )
        result = {"summary": summarize_backtest(results), "results": results}
        if calibration_path:
            built = build_calibration(
                results, provider.name, repository.metadata.population_version, settings.stability_runs, settings.min_calibration_cases
            )
            save_json(calibration_path, built)
            result["calibration"] = {"path": str(calibration_path), "n_cases": built["overall"]["n_cases"]}
    save_json(output, result)
    print(f"Resultado guardado: {output}", file=sys.stderr)
    return result


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # logs y JSON siempre en UTF-8, aunque la consola sea cp1252
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = parser().parse_args()
    try:
        result = run(args)
        if getattr(args, "output", None) and args.command in {"inspect-population", "build-segments"}:
            save_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and result.get("execution_metadata", {}).get("status") == "failed":
            return 2
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1
