# Data format

FoG reads a UTF-8 JSON array. Every entry must contain a non-empty question ID, question, topic-entity mapping, and at least one gold answer:

```json
{
  "id": "question-id",
  "question": "question text",
  "topic_entities": {"m.x": "Entity name"},
  "answers": [{"id": "m.y", "name": "Answer name"}]
}
```

The `data/` directory contains the WebQSP, CWQ, and GrailQA source JSON files. `utils/prepare_dataset.py` converts them, validates all four fields, writes the normalized dataset to a separate output path, and then builds the required two-hop relation cache in `--relations-dir`.

Each selected question must have a two-hop cache named `<sanitized_question_id>.txt` in `--relations-dir`. It contains sorted, unique Freebase relation names, one per line. `manifest.json` records the dataset, hop count, total question count, and completed IDs. The runner checks every selected ID before starting and never falls back to a global relation list.

Generated relation caches and Freebase dumps are not part of this release.
