# Choosing models

## Pick the primary model

`RULING_MODEL` is the **primary** model: it answers every question. It is the
only model you need. A **secondary** model is optional — it answers just the
questions the primary is unsure about, and is set separately with
`RULING_CASCADE_TO` ([below](#escalate-the-doubtful-questions-to-a-secondary-model)).

| you want as primary | set | notes |
|---|---|---|
| the default | nothing | `mlx-community/Qwen3.5-4B-4bit`, 2.3 GB, 3 questions in ~136 ms |
| the default, reading images too | `RULING_MODEL=vision:mlx-community/Qwen3.5-4B-4bit` | the same weights with their vision tower; needs `uv sync --extra vision` |
| Jev-level accuracy | `RULING_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit` | 19 GB, ~204 ms, the model in the [README](../README.md)'s chart |
| any other MLX checkpoint | `RULING_MODEL=<hub id or local path>` | tested with Qwen, Llama 3.2, Gemma 3 |
| a model you serve elsewhere | `RULING_MODEL=openai:<name>` + `RULING_OPENAI_BASE_URL` | vLLM, llama.cpp, `mlx_lm.server`, or a commercial API; needs `top_logprobs` |
| a model on OpenRouter | `RULING_MODEL=openrouter:<name>` | with a spend budget |
| a model or LoRA adapter on an NVIDIA box | `RULING_MODEL=vllm:<name>` + `RULING_VLLM_BASE_URL` | vLLM's own decision endpoint, batched on the GPU; export an adapter trained here with `ruling export-adapter` |
| Cloudflare's Clef-flash, locally | `RULING_MODEL=typesafe:clef-flash` + `TYPESAFE_BASE_URL=http://127.0.0.1:8000` | a trained 9B decision model; run the [MLX port](https://huggingface.co/mlx-community/clef-flash-4bit) with `python clef_mlx.py serve` and point ruling at it. 6.2 GB, the best-calibrated model we have measured |
| the most accurate open decision model we have measured | `RULING_MODEL=typesafe:pplx-decider-v1.1-27b` + `TYPESAFE_BASE_URL=http://127.0.0.1:8092` | Perplexity's pplx-decider, 206 of JevBench's 231 public items; start its MLX port with `assets/bench/serve.sh pplx 8092`. 15.3 GB, prompts up to 8,192 tokens |
| Jev itself, or any System One server | `RULING_MODEL=typesafe:<alias>` + `TYPESAFE_API_KEY` | `typesafe:jev-latest`; the host's own probabilities pass straight through. Mostly useful as the secondary, below |

Any of these can also be the secondary. The other settings — token limits, API
key, calibration, logprob caps — are in [docs/reference.md](reference.md#install-and-run).

## Serve several models from one process

One process can also serve several models by name — a plain base, a tuned
adapter per decision, a large model for escalation — and adapters on the same
base share one copy of its weights. Put them in a JSON file and set
`RULING_MODELS`; requests pick one with `model`, and `GET /v1/models` lists
them:

```bash
cat > models.json <<'EOF'
{"small": {"model": "mlx-community/Qwen3.5-0.8B-4bit"},
 "large": {"model": "mlx-community/Qwen3-4B-Instruct-2507-4bit"}}
EOF
RULING_MODELS=models.json uv run ruling serve
```

An entry can also name a decision server running elsewhere with `base_url`,
so pplx-decider, Clef-flash and your own tuned model can answer side by side
under one API. Details in [docs/reference.md](reference.md#serve-several-models).

## Escalate the doubtful questions to a secondary model

Once the primary model's confidence is honest, one threshold lets a bigger
model answer only what the primary is unsure about.

| role | set with | answers | example |
|---|---|---|---|
| **primary** | `RULING_MODEL` (+ `RULING_ADAPTER`) | every question | the 4B with your adapter, always loaded |
| **secondary** | `RULING_CASCADE_TO` | only questions whose top probability is under `RULING_CASCADE_THRESHOLD` | the local 35B, an `openai:` endpoint, or `typesafe:jev-latest` |

```bash
RULING_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit RULING_ADAPTER=runs/adapters/mine \
RULING_CASCADE_TO=mlx-community/Qwen3.6-35B-A3B-4bit RULING_CASCADE_THRESHOLD=0.84 \
uv run ruling serve
```

Measured with the threshold fitted on one dataset and applied to the other:
the secondary answered 13% of questions and the pair matched the secondary
alone (90 vs 91 on TypeSafe's rows, 145 vs 144 on Every's). Set the secondary
to `typesafe:jev-latest` and the same arithmetic applies to your Jev bill.

Both directions survive an outage. If the secondary is a host that returns
503 or times out, the primary's answers stand for the questions it would have
escalated. For the reverse, Jev first and a local model only when Jev is down,
set `RULING_MODEL=typesafe:jev-latest` and `RULING_FAILOVER_TO=<local model>`;
the response's `model` says which one answered.
