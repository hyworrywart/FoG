"""Public CLI for FoG-V and FoG-T."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

from .cache import load_relations, require_relation_caches
from .datasets import load_dataset
from .knowledge_graph.freebase import FreebaseStore
from .models import OpenAIClient
from .pipelines import FoGPipeline, PipelineConfig
from .relation_selection import RelationSelector
from .results import ResultStore


def build_parser():
    parser = argparse.ArgumentParser(description="Run the FoG-V or FoG-T pipeline")
    parser.add_argument("--pipeline", choices=("vision", "text"), required=True)
    parser.add_argument("--dataset", choices=("webqsp", "cwq", "grailqa"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--relations-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--sentence-model", default="sentence-transformers/msmarco-distilbert-base-tas-b")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--random-n", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def select_questions(data, args):
    selected = data[args.start_index:args.end_index]
    if args.random_n is not None:
        if args.random_n < 0 or args.random_n > len(selected):
            raise ValueError("--random-n must be between zero and the selected dataset size")
        selected = random.Random(args.seed).sample(selected, args.random_n)
    return selected


def error_result(item, config, started, error):
    import time
    return {
        "schema_version": "1.0", "question_id": str(item["id"]),
        "question": item["question"], "topic_entities": item["topic_entities"],
        "thinking_cot": "", "retrieved_paths": [], "LLM_answer": "",
        "prediction": [], "iterations": 0, "LLM_calls": 0, "token_stats": {},
        "elapsed_time": time.monotonic() - started, "run_config": asdict(config),
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv=None):
    import time
    args = build_parser().parse_args(argv)
    data = select_questions(load_dataset(args.data_file), args)
    require_relation_caches(args.relations_dir, data)
    output_dir = Path(args.output_dir)
    results = ResultStore(output_dir / "results.json")
    if not args.resume and results.path.exists():
        raise FileExistsError(f"{results.path} exists; pass --resume or choose another output directory")
    completed = results.completed_ids() if args.resume else set()
    pending = [item for item in data if str(item["id"]) not in completed]
    config = PipelineConfig(
        pipeline=args.pipeline, dataset=args.dataset, text_model=args.text_model,
        vision_model=args.vision_model if args.pipeline == "vision" else None,
        sentence_model=args.sentence_model,
    )
    selector = RelationSelector(args.sentence_model)

    def process(item):
        started = time.monotonic()
        try:
            llm = OpenAIClient(config.text_model, config.vision_model)
            pipeline = FoGPipeline(config, llm, FreebaseStore(), selector, output_dir / "graphs")
            return pipeline.run(item, load_relations(args.relations_dir, item["id"]))
        except Exception as error:  # API failures are recorded; method switching is forbidden.
            return error_result(item, config, started, error)

    for item in pending:
        results.append(process(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
