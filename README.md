![ruling: typed decisions, no text](assets/banner.png)

Typed decisions from a local model, no text generated. Not affiliated with TypeSafe AI.

ruling turns any open chat model into a decision engine like TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). You send
a state and typed questions; you get back a probability for every option you
declared. Nothing is generated, so there is nothing to parse and the model
cannot invent an answer you did not list.

![Jev against Qwen3.6-35B-A3B on TypeSafe's 102 published rows, Every's 154 author-labeled judgments, and the two pooled](assets/against-jev.png)

*On the 256 public judgments that carry Jev's own answers, a 19 GB open model
with no training gets 231 right to Jev's 238 — not a significant difference
(exact McNemar, p = 0.21). [Where Jev is ahead](docs/results.md#where-jev-leads).*

## Documentation

| Page | What's in it |
|---|---|
| [Getting started](docs/getting-started.md) | Install, the first request, the playground and clients, asking about images |
| [Choosing models](docs/models.md) | The primary model, serving several from one process, escalating doubtful questions to a secondary |
| [Fine-tuning](docs/fine-tuning.md) | Training an adapter on your own decisions, images included, and retraining from production outcomes |
| [Results](docs/results.md) | Accuracy and coverage against Jev and the open decision models, and where Jev leads |
| [How it works](docs/how-it-works.md) | One prefix, many suffixes; reading logits instead of sampling; confidence |
| [Reference](docs/reference.md) | Every setting, model prefix, API route, error code and client |
| [Measurements](docs/measurements.md) | The full record, with the command behind every number |
| [Adapter card](docs/adapter.md) | The adapter we trained, its data and how to reproduce it |
| [Background](docs/background.md) | What Jev is, the other open recreations, related work |
| [Development](docs/development.md) | Repository layout and running the tests |

## Installation

You need an Apple Silicon Mac, Python 3.12+, and [uv](https://docs.astral.sh/uv/).
On NVIDIA, see [vLLM](docs/reference.md#on-nvidia-vllm).

```bash
git clone https://github.com/bradAGI/ruling && cd ruling
uv sync                       # add --extra vision to ask about images
uv run ruling serve           # downloads a 2.3 GB model on first start, then serves http://127.0.0.1:8010
```

```bash
curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"message": "My card was charged twice for one order. Refund the duplicate."},
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": null, "technical": null, "sales": null}}}}'
```

Open `http://127.0.0.1:8010/` for the playground. The rest is in
[Getting started](docs/getting-started.md).

## Limits

- Probabilities are model scores. Calibration makes them honest on average on
  data like your calibration set; it does not make any single answer right.
- A local MLX model answers one request at a time; questions inside a request
  are batched. Up to `RULING_MAX_PENDING` requests wait their turn, and the
  next is refused with 429 and a `Retry-After`.
- The state is untrusted input: text in it can steer the answers (prompt
  injection), as with any model that reads it. Put independent checks in front
  of security-sensitive actions rather than acting on an answer alone.
- Images need a `vision:` model or an `openai:` vision host; the plain MLX
  path and `vllm:` read text only. No audio, no generated explanation.
- The native engine is MLX, so the fast path is Apple Silicon. On NVIDIA,
  ruling serves through `vllm:` (vLLM's own decision endpoint) or `openai:`
  against a stock vLLM or llama.cpp; see
  [docs/reference.md](docs/reference.md#on-nvidia-vllm).

## License

MIT. ruling is an independent project with no connection to TypeSafe AI; "Jev"
and "System One" are their names, used here to say what this reproduces. The
`authored144` and `perturbations108` datasets and the WANLI row selection are
converted from TheoLeeCJ/openjev under MIT, with their copyright notice in
[THIRD_PARTY.md](THIRD_PARTY.md). WANLI is CC-BY-4.0 and SST-5 keeps its own
terms; neither is committed here. Model weights keep their own licenses.
