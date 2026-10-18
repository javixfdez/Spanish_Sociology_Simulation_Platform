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
    calibration = load_calibration(calibration_path) if args.command == "simulate" else None
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
            backtest(cases, repository, evidence, factory, args.segments, anchors=anchors, on_result=keep, context_store=context_store)
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
