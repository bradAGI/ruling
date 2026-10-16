# Development

## Layout

```
ruling/            the package: engine.py (MLX scoring), vision.py (images, through mlx-vlm),
                   playground.html and history.html (the served pages),
                   hosted.py (OpenAI-compatible endpoints), cascade.py, calibration.py, evals.py,
                   train.py, retrain.py, ledger.py, registry.py, server.py, cli.py, sdk.py
sdks/typescript/   zero-dependency JS client with TypeScript types
datasets/          labeled JSONL fixtures and the record format
docs/              getting-started.md, models.md, fine-tuning.md, results.md, how-it-works.md,
                   reference.md (settings, API, clients), measurements.md (the full record),
                   background.md (what Jev is, related work), adapter.md (the model card), development.md
assets/            figures and the scripts that redraw them from local evaluation runs (runs/, not committed)
tests/             unit (no model) and integration (real model, downloads Qwen3.5-0.8B on first run)
```

```bash
uv run pytest tests/unit
uv run pytest tests/integration                 # CI runs this with -m "not heavy": the training, PEFT and encoder modules need a GPU
```
