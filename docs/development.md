# Development

## Layout

```
ruling/            the package
  engine.py          MLX scoring: shared prefix cache, batched suffixes, the letter readout
  vision.py          images, through mlx-vlm
  hosted.py          openai:, openrouter: and vllm: engines
  cascade.py, failover.py, thinking.py
                     the wrappers: escalate doubtful questions, fail over, reason first
  questions.py       the request and answer models, limits, image blocks
  calibration.py     temperature scaling, confidence, calibration metrics
  evals.py, metrics.py
                     datasets, scoring, reports
  train.py, retrain.py, ledger.py
                     LoRA training, the retraining loop, the decision ledger
  registry.py        the models a server answers for
  server.py, usage.py
                     the HTTP API and its usage meter
  playground.html, history.html
                     the served pages
  cli.py, sdk.py     the command line and the Python client
sdks/typescript/   zero-dependency JS client with TypeScript types
datasets/          labeled JSONL fixtures and the record format
docs/              the documentation; docs/README.md is the index
assets/            figures and the scripts that redraw them from local evaluation runs (runs/, not committed)
tests/             unit (no model) and integration (real model, downloads Qwen3.5-0.8B on first run)
```

## Tests

```bash
uv sync --extra vision --extra wandb
uv run pytest tests/unit
uv run pytest tests/integration                 # CI runs this with -m "not heavy": the training, PEFT and encoder modules need a GPU
```

The integration tests drive a real server and a real model; six of them drive
the playground and the history page in a browser through Playwright
(`uv run playwright install chromium` once). Tests that need a vLLM server,
Node, or an OpenRouter account skip or fail with the reason when those are
absent.

## Conventions

- US English in code, comments, docs and commit messages.
- Comments say why, not what. Docs state a number only with the command that
  produced it; [Measurements](measurements.md) is the record.
- Every change ships with unit tests and, where it touches the server or a
  page, an integration test; a bug fix starts with a test that fails.

## Cutting a release

1. Commit "Release X.Y.Z": bump `version` in `pyproject.toml`, run `uv lock`,
   and commit both files. There is no changelog file; the GitHub release
   carries the notes.
2. Tag `vX.Y.Z` and push `main` and the tag.
3. Build from a fresh clone of the tag, not the working tree, and check it:

   ```bash
   git clone --branch vX.Y.Z --depth 1 https://github.com/bradAGI/ruling.git /tmp/ruling-release
   cd /tmp/ruling-release && uv build && uvx twine check --strict dist/*
   ```

4. Publish with `uv publish dist/*` and a PyPI token, then
   `gh release create vX.Y.Z --title "ruling X.Y.Z" --notes-file NOTES.md`.
