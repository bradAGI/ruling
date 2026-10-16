# Reference: settings, models, API, clients

Everything the front page summarizes, in full. Moved here to keep the README short; nothing was cut.

## Install and run

Requires an Apple Silicon Mac, Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruling serve          # http://127.0.0.1:8010, playground at /
```

The first start downloads the default model, `mlx-community/Qwen3.5-4B-4bit`
(about 2.3 GB). Settings come from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_MODEL` | `mlx-community/Qwen3.5-4B-4bit` | The primary model. Any MLX chat model, `openai:<model>` for a model behind an OpenAI-compatible server, `openrouter:<model>`, `vllm:<model or adapter>` for vLLM's own decision endpoint, `vision:<MLX model>` for a local vision-language model that also reads images in the state (needs `ruling[vision]`), or `typesafe:<alias>` for TypeSafe's API or any server speaking its protocol (`TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY`). Tested: Qwen3.5, Llama 3.2, Gemma 3 |
| `RULING_ROTATIONS` | `3` | Answer orderings averaged per question. `1` is fastest. See How it works |
| `RULING_PRIOR_DEBIAS` | `false` | Subtract a content-free letter prior. Measured, does not help; kept so you can re-check with another model |
| `RULING_CALIBRATION` | unset | Path to a `calibration.json` written by `ruling calibrate` |
| `RULING_API_KEY` | unset | When set, `/v1/*` requires `Authorization: Bearer <key>` |
| `RULING_MAX_INPUT_TOKENS` | `65536` | Whole-request budget for a local MLX model: state plus every question. Above this the request is rejected with 422 `context_too_long`. Remote models (`openai:`, `vllm:`, `typesafe:`) are held to their host's own limit instead |
| `RULING_MAX_BRANCH_TOKENS` | `32768` | Per-branch budget for a local MLX model: state plus one question. A single oversized question is rejected even when the request fits |
| `RULING_MAX_IMAGE_TOKENS` | `1024` | Tokens one image may become on a `vision:` model; a larger image is scaled down to fit. A 12-megapixel photo is about 15,000 tokens uncapped |
| `RULING_MEMORY_FRACTION`, `RULING_CACHE_LIMIT_GB` | `0.5`, `4` | MLX's ceiling as a share of physical memory, and how much freed buffer memory it may keep. A job that cannot fit raises instead of swapping. Ignored where the backend reports no memory size (Linux CPU) |
| `RULING_OPENAI_BASE_URL` | unset | Required for `openai:` models. The endpoint serving them, ending in `/v1` |
| `RULING_OPENAI_API_KEY` | unset | Sent as a bearer token when the host wants one. Local servers usually do not |
| `RULING_OPENAI_TOP_LOGPROBS` | `20` | How many top logprobs the host returns, and so the widest Choice read in one pass; wider ones run in two stages. At most 20. OpenAI allows 20, `mlx_lm.server` allows 11 |
| `RULING_OPENAI_CONCURRENCY` | `16` | Calls in flight to the host per request. Set `1` for single-threaded hosts such as `mlx_lm.server`, which reset extra connections |
| `RULING_OPENAI_EXTRA_BODY` | `{}` | JSON merged into every request, for host-specific fields such as `{"chat_template_kwargs": {"enable_thinking": false}}` |
| `RULING_OPENROUTER_PROVIDERS` | unset | Required for `openrouter:` models: comma-separated providers allowed to serve them, so every request reaches a host that returns logprobs. The key is `OPENROUTER_API_KEY` |
| `RULING_OPENROUTER_BUDGET_USD` | `1.0` | Dollars an `openrouter:` engine may spend per process; past it, further requests fail |
| `RULING_VLLM_BASE_URL` | unset | Required for `vllm:` models: the vLLM server, without a path. See [On NVIDIA: vLLM](#on-nvidia-vllm) |
| `RULING_VLLM_API_KEY` | unset | Sent as a bearer token when the vLLM server was started with one |
| `RULING_ADAPTER` | unset | A LoRA directory from `ruling train` or `ruling import-adapter`; see [docs/adapter.md](adapter.md) |
| `RULING_CASCADE_TO` | unset | The secondary model, in the same syntax as `RULING_MODEL`; it answers only the questions the primary is unsure about |
| `RULING_CASCADE_THRESHOLD` | unset | Top probability below which a question is escalated; required with `RULING_CASCADE_TO` |
| `RULING_FAILOVER_TO` | unset | A model that answers the whole request whenever the primary is unavailable (429, 502–504, or a timeout); the response's `model` names whichever answered |
| `RULING_MODELS` | unset | A JSON file of further models to serve by name beside the primary; see [Serve several models](#serve-several-models) |
| `RULING_LEDGER` | unset | A SQLite file that records every answered request not marked `exclude_from_training`, its state included, so outcomes can be reported and the decisions retrained on; see [Learn from your own traffic](#learn-from-your-own-traffic) |
| `RULING_MAX_PENDING` | `64` | Requests allowed in flight at once. The next one is refused with 429 `queue_full` and a `Retry-After` estimated from the queue and recent latency |
| `RULING_HOST`, `RULING_PORT` | `127.0.0.1`, `8010` | Bind address |

### Serve several models

One process can answer for more than the primary. `RULING_MODELS` names a JSON
file whose keys are model names a request may send as `model`, and whose
values say what answers under that name:

```json
{
  "routing/v1": {"model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                 "adapter": "runs/adapters/routing", "calibration": "runs/adapters/routing/calibration.json"},
  "large":      {"model": "mlx-community/Qwen3.6-35B-A3B-4bit"}
}
```

```bash
RULING_MODELS=models.json uv run ruling serve
```

`model` takes the same syntax as `RULING_MODEL`; `adapter` and `calibration`
are optional and resolve relative to the file. The primary is still what
`RULING_MODEL` names, and still answers `default`, `jev-latest`, and requests
with no `model`; the cascade and failover settings apply to the primary only.
Entries that fine-tune a base the server already holds share its weights: the
LoRA layers are swapped under the engine lock before each request, so a plain
base and five adapters on it cost one copy of the base. An adapter whose LoRA
layout (rank, depth) differs from the first one loaded on that base gets its
own copy. Each entry's calibration is its own and is listed in `GET /v1/models`,
so a client that applies its own temperature can see when the server already
has.

An entry for a `typesafe:`, `vllm:` or `openai:` model may add `base_url`, the
server it lives on, in place of the global `TYPESAFE_BASE_URL`,
`RULING_VLLM_BASE_URL` or `RULING_OPENAI_BASE_URL`. That is how one ruling
process puts the open decision models, each running under its publisher's own
code, behind one API beside your local model. `assets/bench/serve.sh` starts
each one as its publisher ships it, in its own virtualenv:

```bash
assets/bench/serve.sh pplx 8092 &         # Perplexity's pplx-decider-v1.1-27b, 4-bit MLX port, 15.3 GB
assets/bench/serve.sh decider-4b 8093 &   # Mapika's decider-4b, PyTorch on MPS
```

```json
{
  "pplx":       {"model": "typesafe:pplx-decider-v1.1-27b", "base_url": "http://127.0.0.1:8092"},
  "decider-4b": {"model": "typesafe:decider-4b",            "base_url": "http://127.0.0.1:8093"}
}
```

Requests then pick one with `"model": "pplx"`. pplx-decider scored highest of
every open model we measured, 206 of JevBench's 231 public items (see
[Measurements](measurements.md#other-open-decision-models-run-locally)); it
refuses prompts over 8,192 tokens, and that refusal reaches your client as an
error rather than an answer.

### Bring your own model

The readout needs one thing from a model: the probabilities of the next token,
restricted to your options. Three ways to give ruling that, all behind the same
API, all evaluated by the same harness:

| Your model | `RULING_MODEL` | Notes |
|---|---|---|
| An MLX checkpoint | `mlx-community/...` or a local path | Fastest: one shared prefix cache, every question in one batched forward pass |
| Anything you serve yourself | `openai:<model name>` | vLLM, SGLang, llama.cpp, LM Studio, `mlx_lm.server`, or a commercial API. Needs `/chat/completions` with `top_logprobs` |
| A model on OpenRouter | `openrouter:<model>` | Adds provider pinning and a spend budget |

```bash
python -m mlx_lm server --model mlx-community/Qwen3.5-4B-4bit --port 8080 &

export RULING_MODEL=openai:mlx-community/Qwen3.5-4B-4bit
export RULING_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export RULING_OPENAI_TOP_LOGPROBS=11
export RULING_OPENAI_CONCURRENCY=1
export RULING_OPENAI_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
uv run ruling eval datasets/authored144.jsonl
```

What a hosted model costs you: a Choice wider than the host's logprobs runs in
two stages, one call per option and then a final call, where the local engine
reads 62 options in one pass; there is no shared prefix cache, so the state is
re-read for every question unless the host caches prefixes; and a reasoning
model has to be told to stop reasoning, differently on each host. What it buys: your model, your hardware,
any platform. An integration test runs this path end to end against a real
server and checks the distributions match the local engine's.

One-shot, without a server:

```bash
uv run ruling ask "Since this morning every request with our API key returns 401." \
  --choice "team=Which team should handle this?|billing,technical,sales" \
  --score  "urgency=How urgent is this?|Can wait,This week,Today" \
  --noul   "blocked=The customer is blocked from using the product."
```

### On NVIDIA: vLLM

ruling's scoring runs on Apple Silicon through MLX. On NVIDIA (or AMD through
vLLM's ROCm build) the same decisions come from [vLLM](https://github.com/vllm-project/vllm),
whose `/v1/systemone` endpoint reads a question's label logprobs from the
reply's first token the way ruling does, batches requests across the GPU, and
serves LoRA adapters by name. It arrived on vLLM's main branch in October 2026
and is in releases after 0.31.

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-lora --lora-modules tickets=/adapters/tickets-peft \
  --logprobs-mode raw_logprobs
```

```bash
RULING_VLLM_BASE_URL=http://gpu-box:8000 RULING_MODEL=vllm:tickets uv run ruling serve
```

An adapter trained here goes across with `ruling export-adapter` (see
[Training](#training)). A `vllm:` model is any name vLLM lists at its
`/v1/models`: the base, or an adapter. It can be the primary, a registry
entry beside local models, or the secondary of a cascade; calibration files
fit on it with `ruling calibrate` apply as for any other engine. Two limits
are vLLM's: a choice offers at most 26 options, and a request at most 64
questions. vLLM also serves Cloudflare's Clef-flash, Clef-omni and Liquid's
d1 on the same endpoint with their own prompts, so `vllm:Cloudflare/clef-flash`
is the hosted equivalent of the MLX port used in the measurements.

Until that endpoint is in a release, a stock vLLM works through the `openai:`
path, with ruling itself on Linux using MLX's CPU wheel (it holds no weights
there, only the HTTP client):

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-logprobs 20 --enable-prefix-caching
pip install ruling 'mlx[cpu]'      # from a checkout: uv pip install . 'mlx[cpu]'
RULING_MODEL=openai:Qwen/Qwen3-4B-Instruct-2507 RULING_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  RULING_OPENAI_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}' ruling serve
```

Checked on 14 October 2026 on one NVIDIA L4 with vLLM 0.31.0: ruling's unit
tests pass on Linux, `ruling eval` scored authored144 at three orderings in
35 s (121 of 144, against 126 for the same model at 4-bit on a Mac), a
77-option choice was answered in two stages in 2.2 s, and three questions took
150 ms at the median. vLLM 0.31.0 crashed there while capturing CUDA graphs;
`--enforce-eager` avoids it, at some cost in speed.

## Training

`ruling train --data RECORDS... --output DIR` LoRA-fine-tunes the answer
readout; the README walks through a run and [docs/adapter.md](adapter.md)
describes the adapter it produces. Beyond the hyperparameters (`--steps`,
`--learning-rate`, `--lora-layers`, `--lora-rank`, `--lora-scale`, and the
rest of `ruling train --help`):

| Flag | Meaning |
|---|---|
| `--resume` | Continue from `DIR/checkpoint/`, written at every `--eval-every` step and at the end. Weights, optimizer state and step counter, the position in the batch stream, and the history all come back, so the run finishes with the same weights it would have had uninterrupted. Every setting must match the original run except `--steps`, which may be raised to extend a finished run |
| `--from-adapter DIR` | Start a fresh run, fresh optimizer and step 0, from an existing adapter's weights. Its `adapter_config.json` must match the requested LoRA layers, rank, scale and dropout |

The top-level adapter files in `DIR` are always the best validation loss so
far; `DIR/checkpoint/` is the latest step. Serve the former.

`ruling export-adapter DIR --output PEFT_DIR --base-model HF_ID` writes the
adapter in PEFT's layout, `adapter_model.safetensors` plus a `LoraConfig`,
for `PeftModel.from_pretrained` on the unquantized base `HF_ID`.
`ruling import-adapter PEFT_DIR --output DIR` is the inverse, for any PEFT
LoRA adapter whose adapted layers end the stack, which is the only shape
mlx-lm can load. A ruling adapter is trained against 4-bit MLX weights, so
under PEFT on the fp16 base it is an approximation of what ruling serves;
measure it there before trusting a threshold fitted here.

### Learn from your own traffic

With `RULING_LEDGER` set, the server keeps every decision it answers, and each
response carries an `id`. Whoever later learns what was right reports it:

```bash
curl -s localhost:8010/v1/decisions/$ID/outcome -d '{"labels": {"team": "billing"}}' \
  -H 'content-type: application/json'
```

Labels take every form a training record does, and are checked against the
logged question on arrival, so a bad one is refused with 422 rather than
breaking a training run later. A second report for the same question replaces
the first. `ruling export-decisions --ledger FILE --output RECORDS` writes every
decision with an outcome as a training record, unlabeled questions left
unlabeled, ready for `ruling train --data RECORDS`. Off by default: the ledger
holds your states verbatim.

A request with `"exclude_from_training": true`, the field Endor's clients send
for benchmark and evaluation traffic, is answered as usual but never logged,
and its response has no `id`.

`ruling retrain` closes the loop on a schedule, with nobody watching:

```bash
uv run ruling retrain --ledger ledger.sqlite --adapters runs/adapters/tickets --holdout my-holdout.jsonl
RULING_ADAPTER=runs/adapters/tickets/current uv run ruling serve
```

Each run exports the decisions with outcomes, drops any whose state appears in
a `--holdout` set, and trains the next version (`v1`, `v2`, …) from the
weights of the one being served. It then scores both on the held-out sets and
repoints `current` at the new version only if its log loss is lower and its
accuracy is no worse; restart the server to serve it. Every version
directory keeps its training records, its training log and `retrain.json`,
the comparison and the reason for the verdict, promoted or not. Until
`--min-records` decisions (100 by default) have outcomes, a run does nothing.
`--steps`, `--learning-rate`, `--batch-size`, `--warmup`, `--eval-every`,
`--valid-fraction` and `--seed` reach the trainer; the LoRA layout always
follows the version it starts from.

## API

`POST /v1/systemone`

```json
{
  "model": "optional; a name from GET /v1/models, or default for the primary",
  "exclude_from_training": "optional, default false; true keeps the decision out of the ledger",
  "state": "string, object, or array",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "description, JSON, or null"}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["lowest level", "...", "highest level"]},
    "<id>": {"type": "noul",   "instructions": "a statement to judge"},
    "<id>": {"type": "noul",   "criteria": {"true": "what a yes looks like", "false": "what a no looks like"}}
  }
}
```

A request carries 1 to 64 questions; a choice has 2 to 255 options and a score
2 to 10 levels. A choice or score needs instructions; a noul may describe what
true or false looks like in its criteria instead, with or without instructions.

Instructions, option descriptions, and levels may be strings or JSON
structure.

**Images.** Put `{"type": "image", "media_type": "image/png", "data":
"<base64>"}` anywhere in the state, the shape Endor uses: PNG, JPEG or WebP, at
most 8 per decision and 5 MB each. The model sees the state with each image
replaced by `<image 1>`, `<image 2>`, … in document order, and the images
themselves under those names, so a question can say "the receipt in
`<image 1>`". Three ways to serve them:

- **Locally on a Mac:** `vision:` loads a vision-language model through
  mlx-vlm, vision tower included. The default Qwen3.5 checkpoints are such
  models; the plain `RULING_MODEL` path loads only their text half.

  ```bash
  uv sync --extra vision          # or: pip install 'ruling[vision]'
  RULING_MODEL=vision:mlx-community/Qwen3.5-4B-4bit uv run ruling serve
  ```

  The state and its images are encoded once and every question is scored
  against them, as for text. Eight 12-megapixel photos with nine questions
  took 10 s and 7.2 GB at peak on the 4B model, every color question right.
  A `vision:` model takes no adapter: training reads text only.
- **Any vision model behind an OpenAI-compatible server** (vLLM, SGLang,
  `mlx_vlm.server`): `openai:` models get the images as content parts.
- **`typesafe:`** forwards the state unchanged; the remote service decides.

A model that reads only text answers 422 `unsupported_modality`, and
`GET /v1/models` lists each model's `modalities`.

Answers come back keyed by the same ids:

| Type | Fields |
|---|---|
| choice | `choice` (the argmax key), `probabilities` over keys, `confidence` |
| score | `score` (probability-weighted level index, so `1.74` sits between levels 1 and 2), `legend` index to level text, `probabilities` over indices, `confidence` |
| noul | `noul`, the probability the statement is true |

With a ledger configured, the response also carries `id`, the decision to
report an outcome for; without one the body is exactly the hosted API's.

`POST /v1/decisions/<id>/outcome` with `{"labels": {"<id>": label}}` records
what was right, and returns the labels as stored; it exists only with a ledger.

`usage.input_tokens` counts every token the model processed: the shared prefix
once, plus each question suffix for each ordering. `output_tokens` is always 0.

The response's `model` is the name the request selected; the primary reports
its engine's id, which a failover changes to say which side answered.

Errors come in the hosted API's shape, `{"error": {"code", "message",
"param"?}}`, where `code` is stable for programs and `message` is for people:

| Status | `code` | When |
|---|---|---|
| 422 | `invalid_input` | a malformed request: `param` names the field, such as a choice over 255 options, a score over 10 levels, or more than 64 questions |
| 422 | `invalid_options` | a well-formed question the model cannot answer: a choice with one option or more than the model reads, or nothing to ask; `param` is `questions.<id>` |
| 422 | `unsupported_modality` | the state holds images and the selected model reads only text; `param` is `state` |
| 422 | `context_too_long` | a request over a token limit; `param` is `questions.<id>` when one question is at fault |
| 422 | `invalid_input` | without `param`: a request the engine refuses, such as one a remote decision server rejects; on the outcome route, `param` is `labels` for a label that does not fit the logged question |
| 404 | `unknown_model` | a `model` the server does not answer for; the message lists the names it does |
| 404 | `unknown_decision` | an outcome for an id the ledger does not hold |
| 401 | `unauthorized` | an API key is configured and missing or wrong |
| 429 | `queue_full` | `RULING_MAX_PENDING` requests are already in flight; `Retry-After` says how many seconds to wait |
| 503 | `upstream_unavailable` | the selected model is served remotely and its host is down, with no failover |

TypeSafe's SDK reads `error.message` from this shape and retries 429 honoring
`Retry-After`, so both clients and
[rulesmith](https://github.com/bradAGI/rulesmith), which uses ruling as its
local judge, behave the same against ruling and the hosted services.

`GET /v1/models` returns `{"models": [...]}` with the hosted API's `name`,
`description` and `release_date`, plus `base` (the weights behind the name),
the Hugging Face revision, adapter digest, the calibration temperatures in
force, and limits that only a local engine knows. `GET /health` returns the
primary's name.

`GET /` is a playground page: a state box, a questions box, a model picker
filled from `GET /v1/models`, and each answer's probabilities drawn as bars
with the confidence beneath. A refused request shows its status, code and
message, so it is the quickest way to see what a server answers for and how.

![The playground after a request to a model named large](../assets/playground.png)

### TypeSafe's own SDK talks to ruling

The request shape, the response shape and both paths match the hosted API, so
TypeSafe's official client needs no patching, no shim, and no code change —
only two environment variables:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8010
export TYPESAFE_API_KEY=whatever-you-set-as-RULING_API_KEY
```

```python
from typesafe_sdk import Choice, TypeSafeClient   # the official package, unmodified

with TypeSafeClient() as client:                  # now answered locally
    response = client.system_one(state=..., questions={"team": Choice(...)})
```

`client.system_one`, `client.models.list()`, `response.choices`,
`response.nouls`, `response.scores` and `response.usage` all work. This is
covered by an integration test that installs the real
[`typesafe-sdk`](https://github.com/typesafe-ai/typesafe-sdk-python) package
and runs it against a live ruling server, so the compatibility claim fails
loudly if either side drifts.

Endor's client works the same way, by its base URL alone, so code written for
Endor can run against a local ruling for development, offline or in tests:

```python
import endor   # the official package, unmodified

client = endor.EndorClient(api_key="whatever-you-set-as-RULING_API_KEY", base_url="http://127.0.0.1:8010")
client.system_one(state, questions, model="tickets/v1")
```

Endor names models `<project>/<name>`; give a `RULING_MODELS` entry the same
name and the call needs no change. Decisions, `client.models.list()` and every
answer type are covered by an integration test against the real
[`endor`](https://pypi.org/project/endor/) package. Its projects, datasets,
training runs and billing are Endor's platform and have no counterpart here;
ruling trains with `ruling train`.

## Clients

Python, in this package:

```python
from ruling import Choice, Noul, Score, RulingClient

with RulingClient() as client:                          # RULING_BASE_URL or http://127.0.0.1:8010
    response = client.system_one(
        state={"message": "Since this morning every request with our API key returns 401."},
        questions={
            "team": Choice(instructions="Which team should handle this?",
                           criteria={"billing": "Charges and refunds", "technical": "Bugs and integrations"}),
            "urgency": Score(instructions="How urgent is this?", criteria=["Can wait", "This week", "Today"]),
            "blocked": Noul(instructions="The customer is blocked from using the product."),
        },
    )

team = response.answers["team"]
if team.confidence < 0.5:
    route_to_human()
elif team.choice == "technical" and response.answers["blocked"].noul > 0.9:
    page_on_call()
```

`RulingClient(api_key=...)` (or `RULING_API_KEY`, the server's own variable)
talks to a server that requires a key. With a ledger, `response.id` and
`client.outcome(response.id, {"team": "billing"})` report what was right, and
`system_one(..., exclude_from_training=True)` keeps a decision out of it.
`client.models()` lists the served models.

TypeScript and JavaScript, zero dependencies, in [`sdks/typescript`](../sdks/typescript):

```js
import { RulingClient, choice, noul, score } from "ruling";
const { answers } = await new RulingClient().systemOne({ state, questions: { team: choice("Which team?", { billing: null, technical: null }) } });
```

The thresholds belong in your code, versioned and tested. Fit them on labeled
data from your own traffic; see [calibration](measurements.md#evaluation-and-calibration).

## Evaluation

### Datasets

`ruling eval` and `ruling train` read a `.jsonl` file (one row per line) or a
`.json` array. A row is `{id?, state, questions, labels?, references?, family?,
weight?}`; the record format is described in
[datasets/README.md](../datasets/README.md). Rows written for hosted decision
platforms that share the System One question shape are accepted as they are. Their label
forms are normalized when the row is validated:

| Question | Hard label | Soft label |
|---|---|---|
| noul | `true`, `"true"`, `false`, `"false"` (also `"yes"`/`"no"`) | a probability of yes, or `{"true": p, "false": q}` |
| choice | the option key | `{option: p, ...}` over every option, or a list in option order |
| score | a level index `2`, its option id `"2"`, or the level text | `[p0, p1, ...]` from the lowest level up, or `{"0": p0, ...}` |

A soft label must cover every option, be non-negative, and sum to 1 within
0.02. It is stored as the record's `target` reference, with the hard label set
to its most likely option; that argmax is what accuracy scores.
A single-question row `{state, question, label}` (or `target`, or `expected`)
becomes one question named `decision`. `weight` (default 1.0, never negative)
is the row's share of a training update: a batch's loss is the weighted mean of
its rows, so weights only matter relative to each other.

### Output

`ruling eval` prints one JSON object. `question_types` holds the per-type
summaries (accuracy, balanced accuracy, calibration error, Brier, coverage at
a 5% error budget, option-reversal flips, reference agreement); those keys are
stable. Next to it sit three blocks laid out like a hosted platform's evaluation
results, each holding the same metrics dict over the labeled, scored
questions in its scope, computed on the calibrated probabilities:

- `overall`: every question pooled;
- `by_type`: keyed by `choice`, `score`, `noul`;
- `by_question`: keyed by question id.

The target `y` is the record's soft `target` reference when it has one, else
one-hot on the label. Each block is `{"n": 0}` when nothing is in scope, else:

| Key | Definition |
|---|---|
| `n` | Decisions in scope |
| `accuracy` | Share where argmax `p` equals argmax `y` |
| `nll` | Mean of −Σ `y` log `p`, with `p` floored at 1e-12 |
| `brier` | Mean of Σ (`p` − `y`)² |
| `ece` | Expected calibration error: top-1 confidence `c` is binned into 10 equal-width bins, bin `b` holding `b/10 < c ≤ (b+1)/10` (and `c = 0` in the first), and each bin adds its share of `n` times the gap between its mean confidence and its accuracy |
| `mean_confidence` | Mean top-1 probability |
| `selective` | Keyed `"0.5"`, `"0.7"`, `"0.9"`, `"0.95"`: `coverage` is the share of decisions with top-1 confidence ≥ the threshold, `accuracy` is accuracy over that share, or `null` when it is empty |

The keys and the four thresholds are fixed: other tools read them by name.

### Answer many rows at once

`ruling predict ROWS --output ANSWERS` answers every question of every row in
any dataset form above, labels ignored, without a server. It writes one JSON
line per row, in the rows' order: `{"id", "answers"}` with each answer exactly
as `/v1/systemone` returns it, or `{"id", "error": {"code", "message"}}` with
the API's code (`context_too_long`, `invalid_options`) for a row the model
cannot answer, so line *i* is always row *i*. Each line is written as it is
answered; run the same command again after an interruption and it carries on
after the last line. It takes the same `--model`, `--adapter`, `--calibration`
and `--rotations` as `ruling eval`.

To turn a strong model's answers into training targets for a small one, use
`ruling build-corpus answers --questions ROWS --writer MODEL` instead: it keeps
each answer's full distribution as a soft label.

## How it works

![Measured replay: ruling returns every typed answer at once while the same model streams JSON token by token](../assets/replay.gif)

*Same 4B model, same state, same three questions, measured on one Mac and
aligned at t = 0. ruling answered in 119 ms, median of five. Generating the
same answers as JSON took 755 ms and 105 tokens, median of three, and the
generated object invented a `refund` key inside `team` that no schema asked
for. Reproduce with `uv run --with pillow python assets/replay.py`.*

1. **One prefix, many suffixes, one forward pass.** The system prompt and the
   rendered state are pushed through the model once and the KV cache is kept.
   Each question becomes a short suffix (instructions, the options labeled
   `A.`, `B.`, `C.`, and "respond with only the letter"). All suffixes are
   right-padded into one batch and run in a single forward pass against
   replicas of the cached prefix, chunked so the replicated cache stays under a
   token budget. Adding a question adds a row to the batch, not a re-read of
   the state.
2. **Read, do not sample.** At each suffix's answer position the logits are
   restricted to the option-letter tokens and log-softmaxed. Nothing is
   decoded, so the answer is always one of your options.
3. **Ordering averaging.** Small models prefer some letters over others. With
   `RULING_ROTATIONS=n`, each Choice or Noul is scored under `n` cyclic shifts of
   its option order and each Score under both directions of its scale, and the
   log-probabilities are averaged. With `n` equal to the option count the
   result is exactly invariant to cyclic reordering of your options; there is a
   test for that.
4. **Two-stage Choice.** Above 62 options (the tokenizer's single-token codes),
   or above `RULING_OPENAI_TOP_LOGPROBS` on an `openai:` host, every option is first judged alone as a Noul, the strongest 8 go to a final
   Choice, and the rest get probability 0.
5. **Temperature scaling.** Log-probabilities are divided by a per-type
   temperature before the softmax. The temperature is 1 until you fit one.
6. **Confidence** follows the formulas in TypeSafe's MIT-licensed
   `system-one-adapter`, so a value here is comparable with one from the
   hosted API. Choice is `(max(p) - 1/n) / (1 - 1/n)`, how far the leading
   option stands above uniform. Score is one minus the probability-weighted
   distance from the modal level over the distance a uniform distribution
   would give, so mass spread across neighboring levels costs less than the
   same mass spread across distant ones. Both are 0 for a uniform
   distribution and 1 for a certain one. Noul answers carry no separate
   confidence; the probability is the whole answer.

![Prefill runs at 5,140 tokens per second and 41 TFLOP/s; decoding the same answer as JSON runs at 131 tokens per second and 1.0 TFLOP/s](../assets/readout.png)

*Why stopping after prefill is fast. Both bars are the same 4-bit 4B model on
the same Mac. Prefill processes every token of the prompt together and keeps
the GPU compute-bound; decoding reads the whole model out of memory for each
token it emits and leaves it memory-bound at about a fortieth of the
throughput. Regenerate with `uv run --with matplotlib python assets/charts.py`.*

