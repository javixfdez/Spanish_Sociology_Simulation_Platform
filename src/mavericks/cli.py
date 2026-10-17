import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .evaluation import backtest, save_json
from .evidence import EvidenceRepository, SemanticIndex, read_records, survey_distribution
from .graph import simulate
from .llm import LLMService, check_ollama, create_provider
from .population import PopulationLoader, PopulationRepository, build_segments, create_demo
from .schemas import Question, Settings


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
        if name == "simulate":
            sub.add_argument("--question", required=True)
            sub.add_argument("--options", nargs="+", required=True)
            sub.add_argument("--question-type", choices=["single", "multiple"], default="single")
            sub.add_argument("--rag", choices=["on", "off"], default="on")
        if name == "backtest":
            sub.add_argument("--cases", required=True)
    ingest = commands.add_parser("ingest-surveys")
    ingest.add_argument("--path", required=True)
    ingest.add_argument("--db", default="data/processed/evidence.sqlite")
    ingest.add_argument("--respondents")
    ingest.add_argument("--responses")
    ingest.add_argument("--weight", choices=["peso", "peso_ccaa"], default="peso")
    ingest.add_argument("--answer-column", default="answer")
    ingest.add_argument("--question-type", choices=["single", "multiple"], default="single")
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


def run(args):
    updates = {}
    if getattr(args, "provider", None):
        updates["llm_provider"] = args.provider
    if getattr(args, "model", None):
        updates["ollama_model"] = args.model
    if getattr(args, "concurrency", None) is not None:
        updates["llm_max_concurrency"] = args.concurrency
    settings = Settings(**updates)

    def evidence_repository(path):
        index = (
            SemanticIndex(settings.embedding_model_path, settings.chroma_path)
            if settings.embedding_model_path
            else None
        )
        return EvidenceRepository(path, index)

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
        processed = repo.ingest(args.path)
        attached = (
            repo.attach_microdata(
                args.respondents, args.responses, args.weight, args.answer_column, args.question_type
            )
            if args.respondents
            else 0
        )
        return {"processed": processed, "questions_with_microdata": attached}
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
    if args.command == "simulate":
        result = asyncio.run(
            simulate(
                Question(question=args.question, options=args.options, question_type=args.question_type),
                repository,
                evidence,
                factory(),
                args.segments,
                args.rag == "on",
            )
        )
        output = args.output or f"outputs/{result['simulation_id']}.json"
    else:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8-sig"))
        result = asyncio.run(backtest(cases, repository, evidence, factory, args.segments))
        output = args.output or "outputs/backtest.json"
    save_json(output, result)
    print(f"Resultado guardado: {output}", file=sys.stderr)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
