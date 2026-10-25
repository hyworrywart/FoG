# FoG

This is the official implementation of the EMNLP 2026 paper **[Foresight-on-Graph: Vision-Guided Knowledge Graph Reasoning with Large Language Models](docs/foresight-on-graph.pdf)**.

## Install

Python 3.10 or later, Graphviz (`dot`), and a local Freebase SPARQL service are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Export `OPENAI_API_KEY` and optionally `OPENAI_API_BASE`. See [docs/freebase.md](docs/freebase.md) for Freebase setup and [docs/data.md](docs/data.md) for the normalized data and relation cache formats.

## Prepare data

```bash
python utils/prepare_dataset.py --dataset webqsp --input data/webqsp.json --output prepared/webqsp.json --relations-dir relations/webqsp
```

## Run

```bash
./scripts/run_fog_v.sh --dataset webqsp --data-file prepared/webqsp.json --relations-dir relations/webqsp --output-dir outputs/webqsp-v --text-model gpt-3.5-turbo-0125 --vision-model gpt-4o-2024-11-20
./scripts/run_fog_t.sh --dataset webqsp --data-file prepared/webqsp.json --relations-dir relations/webqsp --output-dir outputs/webqsp-t --text-model gpt-3.5-turbo-0125
```

Use `--start-index`, `--end-index`, or `--random-n` with `--seed` to select questions. `--resume` skips question IDs already present in `results.json`. Results are atomically replaced after each completed question. An API error is recorded in the result and never causes a pipeline switch.

## Evaluate

```bash
python -m FoG.evaluate --data-file prepared/webqsp.json --results-file outputs/webqsp-v/results.json
```

`hit` means any intersection between normalized predicted and gold answer sets; it is not Hits@1. F1 is calculated from normalized answer-set precision and recall.

Apache-2.0 licensed. This repository does not distribute Freebase dumps or generated relation caches.

## Citation

If you use FoG in your research, please cite our paper:

```bibtex
@inproceedings{hu2026foresight,
  title     = {Foresight-on-Graph: Vision-Guided Knowledge Graph Reasoning with Large Language Models},
  author    = {Hu, Yuwei and Lei, Runlin and Wei, Zhewei and Liu, Yongchao and Hong, Chuntao and Zheng, Jiajun and Zhang, Juelu},
  booktitle = {EMNLP 2026},
  year      = {2026}
}
```
