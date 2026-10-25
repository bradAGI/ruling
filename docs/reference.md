# Reference

The settings, models, command line, HTTP API, clients and dataset formats, in
full. The walkthroughs are elsewhere: [Getting started](getting-started.md) for
a first run, [Choosing models](models.md) for which model to serve, and
[Fine-tuning](fine-tuning.md) for training on your own decisions.

## Install and run

Requires an Apple Silicon Mac and Python 3.12+.

```bash
pip install ruling           # or: uv tool install ruling; from a checkout: uv sync, then uv run ruling
ruling serve                 # http://127.0.0.1:8010, playground at /
```

The first start downloads the default model, `mlx-community/Qwen3.5-4B-4bit`
(about 2.3 GB).

## Settings

Settings come from the environment. Model and scoring:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_MODEL` | `mlx-community/Qwen3.5-4B-4bit` | The primary model. Any MLX chat model, `openai:<model>` for a model behind an OpenAI-compatible server, `openrouter:<model>`, `vllm:<model or adapter>` for vLLM's own decision endpoint, `vision:<MLX model>` for a local vision-language model that also reads images in the state (needs `ruling[vision]`), or `typesafe:<alias>` for TypeSafe's API or any server speaking its protocol (`TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY`). Tested: Qwen3.5, Llama 3.2, Gemma 3 |
| `RULING_ROTATIONS` | `3` | Answer orderings averaged per question. `1` is fastest. See [How it works](how-it-works.md) |
| `RULING_PRIOR_DEBIAS` | `false` | Subtract a content-free letter prior. Measured, does not help; kept so you can re-check with another model |
| `RULING_CALIBRATION` | unset | Path to a `calibration.json` written by `ruling calibrate` |

Limits and memory:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_MAX_INPUT_TOKENS` | `65536` | Whole-request budget for a local MLX model: state plus every question. Above this the request is rejected with 422 `context_too_long`. Remote models (`openai:`, `vllm:`, `typesafe:`) are held to their host's own limit instead |
| `RULING_MAX_BRANCH_TOKENS` | `32768` | Per-branch budget for a local MLX model: state plus one question. A single oversized question is rejected even when the request fits |
| `RULING_LONG_CONTEXT_TOKENS` | `262144` | Both budgets for a request sent with `TypeSafe-Danger-Zone-Long-Context: true`, the hosted API's opt-in header. Slower, and a 4B model's reading degrades well before it; the request's memory grows with it, so keep `RULING_MEMORY_FRACTION` in mind |
| `RULING_MAX_IMAGE_TOKENS` | `1024` | Tokens one image may become on a `vision:` model; a larger image is scaled down to fit. A 12-megapixel photo is about 15,000 tokens uncapped |
| `RULING_MEMORY_FRACTION`, `RULING_CACHE_LIMIT_GB` | `0.5`, `4` | MLX's ceiling as a share of physical memory, and how much freed buffer memory it may keep. A job that cannot fit raises instead of swapping. Ignored where the backend reports no memory size (Linux CPU) |

Hosted models, for the `openai:`, `openrouter:` and `vllm:` prefixes:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_OPENAI_BASE_URL` | unset | Required for `openai:` models. The endpoint serving them, ending in `/v1` |
| `RULING_OPENAI_API_KEY` | unset | Sent as a bearer token when the host wants one. Local servers usually do not |
| `RULING_OPENAI_TOP_LOGPROBS` | `20` | How many top logprobs the host returns, and so the widest Choice read in one pass; wider ones run in two stages. At most 20. OpenAI allows 20, `mlx_lm.server` allows 11 |
| `RULING_OPENAI_CONCURRENCY` | `16` | Calls in flight to the host per request. Set `1` for single-threaded hosts such as `mlx_lm.server`, which reset extra connections |
| `RULING_OPENAI_EXTRA_BODY` | `{}` | JSON merged into every request, for host-specific fields such as `{"chat_template_kwargs": {"enable_thinking": false}}` |
| `RULING_OPENROUTER_PROVIDERS` | unset | Required for `openrouter:` models: comma-separated providers allowed to serve them, so every request reaches a host that returns logprobs. The key is `OPENROUTER_API_KEY` |
| `RULING_OPENROUTER_BUDGET_USD` | `1.0` | Dollars an `openrouter:` engine may spend per process; past it, further requests fail |
| `RULING_VLLM_BASE_URL` | unset | Required for `vllm:` models: the vLLM server, without a path. See [On NVIDIA: vLLM](#on-nvidia-vllm) |
| `RULING_VLLM_API_KEY` | unset | Sent as a bearer token when the vLLM server was started with one |

Adapters, cascade, failover and the models file:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_ADAPTER` | unset | A LoRA directory from `ruling train` or `ruling import-adapter`; see [docs/adapter.md](adapter.md) |
| `RULING_CASCADE_TO` | unset | The secondary model, in the same syntax as `RULING_MODEL`; it answers only the questions the primary is unsure about |
| `RULING_CASCADE_THRESHOLD` | unset | Top probability below which a question is escalated; required with `RULING_CASCADE_TO` |
| `RULING_FAILOVER_TO` | unset | A model that answers the whole request whenever the primary is unavailable (429, 502–504, or a timeout); the response's `model` names whichever answered |
| `RULING_MODELS` | unset | A JSON file of further models to serve by name beside the primary; see [Serve several models](#serve-several-models) |

The ledger:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_LEDGER` | unset | A SQLite file that records every answered request not marked `exclude_from_training`, its state included, so outcomes can be reported and the decisions retrained on; see [Learn from your own traffic](#learn-from-your-own-traffic) |
| `RULING_LEDGER_RETAIN_DAYS` | unset | With a ledger, how long a decision with no reported outcome is kept before it is dropped, state and all; decisions with outcomes are kept however old. Unset keeps everything |

The server:

| Variable | Default | Meaning |
|---|---|---|
| `RULING_API_KEY` | unset | When set, `/v1/*` requires `Authorization: Bearer <key>` |
| `RULING_MAX_PENDING` | `64` | Requests allowed in flight at once. The next one is refused with 429 `queue_full` and a `Retry-After` estimated from the queue and recent latency |
| `RULING_HOST`, `RULING_PORT` | `127.0.0.1`, `8010` | Bind address |

## Models

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
model has to be told to stop reasoning, differently on each host. What it buys:
your model, your hardware, any platform. An integration test runs this path end
to end against a real server and checks the distributions match the local
engine's.

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
[export-adapter and import-adapter](#export-adapter-and-import-adapter)). A
`vllm:` model is any name vLLM lists at its `/v1/models`: the base, or an
adapter. It can be the primary, a registry entry beside local models, or the
secondary of a cascade; calibration files fit on it with `ruling calibrate`
apply as for any other engine. vLLM labels options A to Z, so a choice wider
than 26 runs in two stages, as on an `openai:` host: every option is first
judged on its own as a yes/no question, 64 to a request, and the strongest 8
go to a final choice. vLLM also serves Cloudflare's Clef-flash, Clef-omni and
Liquid's d1 on the same endpoint with their own prompts, so
`vllm:Cloudflare/clef-flash` is the hosted equivalent of the MLX port used in
the measurements.

Until that endpoint is in a release, a stock vLLM works through the `openai:`
path, with ruling itself on Linux using MLX's CPU wheel (it holds no weights
there, only the HTTP client):

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-logprobs 20 --enable-prefix-caching
pip install ruling 'mlx[cpu]'      # from a checkout: uv pip install . 'mlx[cpu]'
RULING_MODEL=openai:Qwen/Qwen3-4B-Instruct-2507 RULING_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  RULING_OPENAI_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}' ruling serve
```

Checked on 16 October 2026 on one NVIDIA L4 with vLLM 0.31.0: ruling's unit
tests pass on Linux, `ruling eval` scored authored144 at three orderings in
32 s (129 of 144, one question apart from the same bf16 checkpoint through the
MLX engine on a Mac, which agrees with it on 143 of 144; the 4-bit MLX build
gets 126), a 77-option choice was answered in two stages in 2.2 s, and three
questions took 150 ms at the median. vLLM 0.31.0 crashed there while capturing
CUDA graphs; `--enforce-eager` avoids it, at some cost in speed.

## Command line

`ruling --help` lists thirteen commands:

| Command | Purpose |
|---|---|
| `serve` | run the HTTP API |
| `ask` | answer questions once, without a server |
| `eval` | score a labeled dataset and report metrics |
| `calibrate` | fit per-type temperatures on a labeled dataset |
| `predict` | answer every question of every row, labels ignored, as JSONL |
| `build-dataset` | download a pinned public dataset into ruling format |
| `build-corpus` | build training records from open data, program state, or synthetic worlds |
| `train` | LoRA fine-tune the answer readout on training records |
| `export-adapter` | write a ruling adapter in PEFT layout for transformers |
| `import-adapter` | write a PEFT LoRA adapter in the layout ruling serves |
| `export-decisions` | write the served decisions that have outcomes as training records |
| `retrain` | train on the ledger's resolved decisions and promote the result only if it beats the served adapter on held-out data |
| `train-encoder` | fully fine-tune a decision encoder on training records |

`serve` takes no flags; everything it needs is in [Settings](#settings).

### Engine flags

`ask`, `eval`, `calibrate` and `predict` build an engine the way `serve` does,
and each of these flags overrides the matching setting for that one run:

| Flag | Overrides |
|---|---|
| `--model MODEL` | `RULING_MODEL` |
| `--rotations N` | `RULING_ROTATIONS` |
| `--prior-debias {on,off}` | `RULING_PRIOR_DEBIAS` |
| `--calibration PATH` | `RULING_CALIBRATION` |
| `--adapter DIR` | `RULING_ADAPTER`, a LoRA directory from `ruling train` |

### ask

`ruling ask STATE` answers once, without a server, and prints the response as
JSON. `STATE` is the state text, or `@path` to read it from a file; text that
parses as JSON is sent as JSON. Questions come from `--choice ID=Q|a,b,c`,
`--score ID=Q|lo,mid,hi`, `--noul ID=STATEMENT` and `--box ID=WHAT` (on a
`vision:` model, with an image in the state), each repeatable, at least one
required:

```bash
uv run ruling ask "Since this morning every request with our API key returns 401." \
  --choice "team=Which team should handle this?|billing,technical,sales" \
  --score  "urgency=How urgent is this?|Can wait,This week,Today" \
  --noul   "blocked=The customer is blocked from using the product."
```

With no typed question, `ask` reads its text as a question in plain words:
`ruling ask "can penguins fly"`. It asks the model the nine routing questions
in `ruling/ask.py`, answers what they settle, has the served model write
options for a pick or a scale when one is needed, judges them, checks each
answer, and prints the same JSON `POST /v1/ask` returns. The writer is the
served model itself, so this works for a local text model; a `vision:`,
hosted or decision-server model answers the readings and says it cannot
write options.

### eval, calibrate and predict

`ruling eval DATASET` scores a labeled dataset (any shape under
[Datasets](#datasets)) and prints the report described under
[Evaluation output](#evaluation-output) as one JSON object; `--output FILE`
writes it there instead of stdout. `--predictions FILE` also writes each
question's distribution as JSONL. `--against typesafe` also sends every record
to the hosted TypeSafe API (`TYPESAFE_API_KEY`) and reports agreement;
`--typesafe-model ALIAS` picks the model there, the service default when
omitted.

`ruling calibrate DATASET` fits per-type temperatures on a labeled dataset and
prints them with the per-type summaries before and after. `--output FILE`
writes the `calibration.json` that `RULING_CALIBRATION` and `--calibration`
read.

Both take `--think-tokens N`, which lets a decoder model reason up to that many
tokens before each answer (slow; for teachers).

`ruling predict ROWS --output ANSWERS` answers every question of every row in
any dataset form, labels ignored, without a server. It writes one JSON line per
row, in the rows' order: `{"id", "answers"}` with each answer exactly as
`/v1/systemone` returns it, or `{"id", "error": {"code", "message"}}` with the
API's code (`context_too_long`, `invalid_options`, `unsupported_modality`) for
a row the model cannot answer, so line *i* is always row *i*. Each line is
written as it is answered; run the same command again after an interruption and
it carries on after the last line. Against a running server,
[`POST /v1/predict`](#post-v1predict-and-post-v1evaluations) answers the same
rows with the same per-row entries in one request, and `POST /v1/evaluations`
scores them; `client.predict(rows)` and `client.evaluate(rows)` in the
[Python client](#python) call them.

To turn a strong model's answers into training targets for a small one, use
`ruling build-corpus answers --questions ROWS --writer MODEL` instead: it keeps
each answer's full distribution as a soft label.

### build-dataset and build-corpus

`ruling build-dataset NAME --output FILE` downloads a pinned public dataset
into ruling format. `NAME` is one of `wanli`, `sst5`, `typesafe`, `every` or
`dev`; `--limit N` (default 300) is the rows to sample for `sst5`, or the rows
per source for `dev`.

`ruling build-corpus KIND --output FILE` builds training records from open
data, program state, or synthetic worlds. `KIND` is `public`, `states`,
`worlds`, `questions` or `answers`; every kind but `answers` needs `--count`
(records per public source, or total states, worlds, or question sets), and
`answers` needs `--questions`, the question records to answer. `--writer`
names the writer and judge model (defaulting to `RULING_MODEL`), `--teacher
{model,typesafe}` answers with the `--writer` model (local, or
`openrouter:<model>`) or the hosted TypeSafe API, and `--holdout` lists the
evaluation datasets whose states are never distilled, the bundled ones by
default. The rest of its flags (`--seed`, `--catalog`, `--batch-size`,
`--min-probability`, `--temperature`, `--states`, `--questions-per-state`,
`--max-options`, `--rotations`, `--think-tokens`) are in `ruling build-corpus
--help`. The record format is in [datasets/README.md](../datasets/README.md)
and the recipe that puts these together is in [docs/adapter.md](adapter.md).

### train

`ruling train --data RECORDS... --output DIR` LoRA-fine-tunes the answer
readout; [Fine-tuning](fine-tuning.md) walks through a run and [docs/adapter.md](adapter.md)
describes the adapter it produces. `--model` picks the base (defaulting to
`RULING_MODEL`) and `--holdout` the evaluation datasets whose states must not
appear in training data, the bundled ones by default. Beyond the
hyperparameters (`--steps`, `--learning-rate`, `--lora-layers`, `--lora-rank`,
`--lora-scale`, and the rest of `ruling train --help`):

| Flag | Meaning |
|---|---|
| `--resume` | Continue from `DIR/checkpoint/`, written at every `--eval-every` step and at the end. Weights, optimizer state and step counter, the position in the batch stream, and the history all come back, so the run finishes with the same weights it would have had uninterrupted. Every setting must match the original run except `--steps`, which may be raised to extend a finished run |
| `--from-adapter DIR` | Start a fresh run, fresh optimizer and step 0, from an existing adapter's weights. Its `adapter_config.json` must match the requested LoRA layers, rank, scale and dropout |
| `--max-image-tokens N` | Tokens one image may become when `--model` is a `vision:` model (default 1024); keep it equal to the server's `RULING_MAX_IMAGE_TOKENS` so the adapter sees images as they will be served |
| `--wandb [PROJECT]` | Mirror every progress line (step, losses, `progress`, `eta_seconds`, and the validation metrics) to a Weights & Biases run in `PROJECT`, or the account's default, opened on the run's config and finished when the run ends, marked failed if it raised. Needs the `wandb` extra: `uv sync --extra wandb` or `pip install 'ruling[wandb]'` |
| `--label-smoothing F` | Mix this share of a uniform distribution over the options into each target before the cross-entropy (default 0) |
| `--brier-weight F` | Add the Brier score (squared error between probabilities and target) to the loss at this weight (default 0) |
| `--ranking-weight F` | Add a pairwise ranking term at this weight (default 0): within a batch, a wrong answer more confident than a right one is penalized |
| `--ranking-margin F` | How much more confident a right answer must be than a wrong one before the ranking term is satisfied (default 0.05) |
| `--no-grad-checkpoint` | Keep all activations; faster but needs far more memory |

With all of `--label-smoothing`, `--brier-weight` and `--ranking-weight` at
their defaults the loss is exactly the soft-target cross-entropy.

**Records with images.** With `--model vision:<model>`, records whose states
hold images train as they are served: the image tokens sit in the prompt, the
vision tower stays frozen, and LoRA trains the language layers that read
them. Raise `--max-tokens` to fit them; at the default 1,024 tokens per image
the default `--max-tokens` of 1,024 leaves no room for the question. Examples
longer than `--max-tokens` are dropped and counted before the first step, and
a run they would empty refuses to start. A text model refuses records with
images rather than training on their text alone.
Measured on the 0.8B test model, teaching an arbitrary code for each of three
colors took 300 steps at a learning rate of 3e-5 to bring the validation loss
from 2.3 to 0.4; far fewer steps, or higher rates, left it at chance.

The top-level adapter files in `DIR` are always the best validation loss so
far; `DIR/checkpoint/` is the latest step. Serve the former.

### export-adapter and import-adapter

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
the first.

Whoever only knows whether the answer was right can say so instead of naming
the label:

```bash
curl -s localhost:8010/v1/decisions/$ID/outcome -H 'content-type: application/json' \
  -d '{"corrections": {"team": {"correct": true}, "refund": {"correct": false}}}'
```

Right labels the question with what was served: the choice, the likeliest
score level, or yes when the noul leaned yes. Wrong labels it with `answer`,
which a question of two options may leave out, since the other option is the
only one left; wrong among more options without an answer is refused, as is
an `answer` that is what was served. Box answers take no outcome.
`POST /v1/decisions/outcomes` reports up to 1,000 decisions at once, stored
all or none. `ruling export-decisions --ledger FILE --output RECORDS` writes every
decision with an outcome as a training record, unlabeled questions left
unlabeled, ready for `ruling train --data RECORDS`. Off by default: the ledger
holds your states verbatim. `RULING_LEDGER_RETAIN_DAYS=30` drops a decision
that has gone 30 days with no outcome, so the file holds one month of
unresolved traffic plus everything resolved, rather than growing forever.

A request with `"exclude_from_training": true`, meant for benchmark and
evaluation traffic, is answered as usual but never logged, and its response
has no `id`.

`ruling retrain` closes the loop on a schedule, with nobody watching:

```bash
uv run ruling retrain --ledger ledger.sqlite --adapters runs/adapters/tickets --holdout my-holdout.jsonl
RULING_ADAPTER=runs/adapters/tickets/current \
RULING_CALIBRATION=runs/adapters/tickets/current/calibration.json uv run ruling serve
```

Each run exports the decisions with outcomes, drops any whose state appears in
a `--holdout` set, and trains the next version (`v1`, `v2`, …) from the
weights of the one being served. It then scores both on the held-out sets and
repoints `current` at the new version only if its log loss is lower, its
accuracy is no worse, and no single question got worse; restart the server to
serve it. A question counts as worse when, on the held-out rows the two
versions answer differently, the new one loses more than chance allows (an
exact sign test at 5%), so a gain on a busy question cannot hide a collapse on
a rare one; the report lists them under `worse` and the verdict names them. Each version is scored
at its own fitted temperature, because training sharpens an adapter faster
than it corrects it and a version that is right more often must not lose for
being overconfident. The candidate's temperatures are saved beside its weights
as `calibration.json`, so `RULING_CALIBRATION=<adapters>/current/calibration.json`
serves the promoted version calibrated. Every version directory keeps its
training records, its training log and `retrain.json`, the comparison and the
reason for the verdict, promoted or not. The server's `/history` page lists
every version with those verdicts and, from the training log, its
validation-loss curve and final train and validation loss. Until
`--min-records` decisions (100 by default) have outcomes, a run does nothing.
`--model` picks the base, defaulting to `RULING_MODEL`. `--steps`,
`--learning-rate`, `--batch-size`, `--warmup`, `--eval-every`,
`--valid-fraction` and `--seed` reach the trainer; the LoRA layout always
follows the version it starts from.

### train-encoder

`ruling train-encoder --data RECORDS... --output DIR --base REPO --revision
COMMIT` fully fine-tunes a ModernBERT-architecture encoder on training records,
`--base` naming the encoder repository and `--revision` the commit to pin.
`--holdout` works as for `train`, and the hyperparameters (`--valid-fraction`,
`--steps`, `--batch-tokens`, `--learning-rate`, `--head-learning-rate`,
`--weight-decay`, `--warmup`, `--max-tokens`, `--eval-every`,
`--valid-records`, `--seed`, `--no-grad-checkpoint`) are in `ruling
train-encoder --help`. What the encoder is and how it measured is in
[design.md](design.md#first-decision-encoder).

## API

### POST /v1/systemone

```json
{
  "model": "optional; a name from GET /v1/models, or default for the primary",
  "exclude_from_training": "optional, default false; true keeps the decision out of the ledger",
  "state": "string, object, or array",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "description, JSON, or null"}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["lowest level", "...", "highest level"]},
    "<id>": {"type": "noul",   "instructions": "a statement to judge"},
    "<id>": {"type": "noul",   "criteria": {"true": "what a yes looks like", "false": "what a no looks like"}},
    "<id>": {"type": "box",    "instructions": "what to find, and in which image as <image N>"}
  },
  "assets": {"<name>": {"type": "image", "data_base64": "..."}}
}
```

A request carries 1 to 64 questions; a choice has 2 to 255 options and a score
2 to 10 levels. A choice or score needs instructions; a noul may describe what
true or false looks like in its criteria instead, with or without instructions.

Instructions, option descriptions, and levels may be strings or JSON
structure.

**Images.** Put `{"type": "image", "media_type": "image/png", "data":
"<base64>"}` anywhere in the state: PNG, JPEG or WebP, at
most 8 per decision and 5 MB each. The model sees the state with each image
replaced by `<image 1>`, `<image 2>`, … in document order, and the images
themselves under those names, so a question can say "the receipt in
`<image 1>`".

**Assets** are the hosted API's other way to send an image, and what its SDKs
send: an `assets` map at the top of the request, each image under a name the
text uses wherever it likes, in the state, in instructions, or in an option's
or level's description. The bytes go as `data_base64`, as the state's
`media_type` + `data` shape, or as a multipart part: send the JSON as a
`multipart/form-data` part named `request` and each asset's bytes as a part
named by its `part` field, so nothing is base64-encoded.

```json
{"state": "The customer returned <product>; the listing showed <listing>.",
 "questions": {"same": {"type": "noul", "instructions": "<product> is the item in <listing>."}},
 "assets": {"<product>": {"type": "image", "part": "p"}, "<listing>": {"type": "image", "data_base64": "..."}}}
```

The server puts each asset into the state after the images already there,
numbered in the order the text first mentions them, and rewrites every mention
to its `<image N>` name; the model never sees the asset names. An asset the
text never mentions, a part no asset names, or bytes that are not an image are
422 `invalid_input` on `assets`. The hosted API wants
`TypeSafe-Danger-Zone-Images: true` on such a request; ruling accepts the
header and does not need it.

**Box** asks where something is in an image, on a model whose
`question_types` include `box` (a `vision:` model, or `typesafe:` where the
service locates). The instructions say what to find and, with several images
in the state, which one as `<image N>`; with one image, nothing more. The
answer is always four edges, as fractions of that image from its top-left
corner, with a `confidence`; nothing is "not found", so pair a box with a noul
that asks whether the thing is there at all. A `vision:` model generates the
box, the one answer that is: it is asked for the subject's bounding box the way
its grounding training phrased it and names the edges on a 1000 × 1000 grid,
and `confidence` is the geometric mean probability of the leading digit of each
number it wrote. The tokens it generates are the response's `output_tokens`.
See [Measurements](measurements.md#locating-things-in-an-image) for how
close the boxes land.

Three ways to serve images:

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
  A `vision:` model serves an adapter like any other model, and trains on
  records whose states hold images; see [train](#train).
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
| box | `box` with `top`, `bottom`, `left`, `right` as fractions of the image, `confidence` |

With a ledger configured, the response also carries `id`, the decision to
report an outcome for; without one the body is exactly the hosted API's.

`usage.input_tokens` counts every token the model processed: the shared prefix
once, plus each question suffix for each ordering, plus each box question's
prompt. `output_tokens` is the tokens generated for box questions, and 0
without any.

**Long context.** A request sent with `TypeSafe-Danger-Zone-Long-Context:
true`, the hosted API's opt-in header, is read up to `RULING_LONG_CONTEXT_TOKENS`
(262,144 by default) for the whole request and for each branch instead of the
configured limits, on local models; a `typesafe:` engine forwards the header.
`GET /v1/models` reports each model's `long_context_tokens`.

The response's `model` is the name the request selected; the primary reports
its engine's id, which a failover changes to say which side answered.

### POST /v1/decisions/{id}/outcome

`POST /v1/decisions/<id>/outcome` with `{"labels": {"<id>": label}}` records
what was right, and returns the labels as stored; it exists only with a ledger.
`{"corrections": {"<id>": {"correct": false, "answer": "billing"}}}` marks a
served answer instead; both may come together for different questions. See
[Learn from your own traffic](#learn-from-your-own-traffic) for the label
forms it accepts and what happens to the decisions afterward.

### POST /v1/decisions/outcomes

`{"outcomes": [{"id": "<decision>", "labels"?: {…}, "corrections"?: {…}}, …]}`
reports up to 1,000 outcomes in one request and returns
`{"outcomes": [{"id", "labels"}, …]}` in the same order. They are stored
together or not at all: an unknown decision answers 404 `unknown_decision`,
and an outcome that does not fit answers 422 with `param` naming it, such as
`outcomes.3.labels`.

### POST /v1/ask

```json
{"model": "optional", "input": "is a hot dog a sandwich?", "images": "optional, image blocks the question is about"}
```

A question in plain words, with `images` as the blocks `/v1/systemone` takes
when it is about pictures; they are put in the state beside the input and
named `<image 1>`, `<image 2>`, … in every question, and a model that reads
only text refuses them with 422 `unsupported_modality`. The response carries
the model's reading of it and every answer with its work:

| Field | Meaning |
|---|---|
| `facts` | the nine routing questions' answers, keyed `answerable`, `kind`, `several`, `yes`, `degree`, `scale`, `chance`, `fit`, `whole` |
| `readings` | the kinds the input was read as: the likeliest of `yes_or_no`, `pick_one`, `how_much`, `how_likely`, and a runner-up at 0.3 or more |
| `refused` | true when the model judged the input is not a question its types answer; then `answers` is empty |
| `unfinished` | true when the input reads as cut off part way; it is answered as it stands |
| `fit` | the probability the input is fit to show strangers |
| `answers` | one per reading: `kind`, `source` (`fact` for an answer the first request held, `writer` for one the model wrote and then judged), the `question` asked, the `answer` as `/v1/systemone` returns it, `said` (the answer in words: `Yes.`, `No?`, `28%.`, `Vim.`), and `checked`, the probability that `said` is right, asked afterward |
| `wrote` | the writer's reply, byte for byte, when it was called |
| `dropped` | what the writer wrote that was not used, each with the reason, or why nothing could be written |
| `usage` | input tokens over every request made |

The rules, in order: an input the model cannot judge is refused; several
questions go whole to the writer with every tool; a yes-or-no is answered
from the `yes` fact, a how-likely from `chance`, a how-much from `degree`
unless `scale` says it needs its own units, in which case the writer drafts
the scale; a pick has the writer draft 2 to 8 options. The writer may draft
at most 4 questions and only of the kinds asked for; anything else it writes
is listed under `dropped`. The writer is the served model: a local text
model writes from the words, a `vision:` model writes with the pictures in
front of it, and a hosted or decision-server model cannot write, so it
answers the direct readings and reports the rest as dropped. The request
counts as one decision in `GET /v1/usage`.

### POST /v1/predict and POST /v1/evaluations

Both take `{"model"?, "rows": [...]}`, 1 to 50,000 rows, each in any shape
`ruling predict` and `ruling eval` read (see [Datasets](#datasets)): a record
with `state` and `questions`, or a single-question row with `question` and
`label`, with an optional `id`. A batch holds one of the `RULING_MAX_PENDING`
slots for its whole duration and is refused with 429 `queue_full` like any
other request when none is free. The `model` field resolves as it does for a
decision; an unknown name is 404 `unknown_model`.

`POST /v1/predict` answers every row, labels ignored:

```json
{
  "model": "mlx-community/Qwen3.5-4B-4bit",
  "answers": [
    {"id": "t-1", "answers": {"team": {"type": "choice", "choice": "billing", "...": "..."}}},
    {"id": "t-2", "error": {"code": "context_too_long", "message": "..."}}
  ],
  "usage": {"input_tokens": 1842, "output_tokens": 0}
}
```

`answers` holds one entry per row in the rows' order, each `{"id"?, "answers"}`
with every answer exactly as `/v1/systemone` returns it, or `{"id"?, "error":
{"code", "message"}}` for a row the model refuses on its own
(`context_too_long`, `invalid_options`, `unsupported_modality`, or
`invalid_input` for a row that is not a valid record), so one bad row never
fails the batch and entry *i* is always row *i*. `usage.input_tokens` is the
batch's total.

`POST /v1/evaluations` scores the rows against their labels and returns
`{"model", ...}` with exactly what `ruling eval` prints, described under
[Evaluation output](#evaluation-output): `question_types`, `overall`, `by_type`
and `by_question`, so a client gets accuracy, `nll`, `brier`, `ece`,
`mean_confidence` and `selective` per type and per question without a dataset
on disk. A row too long for the model counts as `unscored` in its type's
summary; a row the model cannot read at all (a one-option choice, images for a
text model) is a 422 with `param` `rows.<i>`, and a batch with no labels at all
is 422 `invalid_input` on `rows`.

### GET /v1/usage

```json
{"usage": [{"hour": "2026-10-16T14:00:00+00:00", "model": "large", "decisions": 312, "input_tokens": 91044}]}
```

One bucket per hour and model, sorted by hour then model, since the server
started: every answered `/v1/systemone` request and every row answered by
`/v1/predict` or `/v1/evaluations` is one decision, and refused ones count
nothing. The query parameters `starting_on` and `ending_before` (ISO 8601
datetimes; a naive one is read as UTC) keep the buckets whose hour starts
within the window. The counts are kept in memory for the server's lifetime: a
restart starts them over, and nothing is written to disk.

### GET /v1/models, GET /v1/retrain and GET /health

`GET /v1/models` returns `{"models": [...]}` with the hosted API's `name`,
`description` and `release_date`, plus `base` (the weights behind the name),
the Hugging Face revision, adapter digest, the calibration temperatures in
force, and limits that only a local engine knows.

`GET /v1/retrain` returns what the `/history` page shows: for each served
model whose adapter is a `ruling retrain` directory's `current` link,
`{"models": [{"name", "current", "versions": [...]}]}`, where `current` is the
version directory the link points at (`null` when there is none) and
`versions` holds, in version order, the `retrain.json` report of every
finished run, each with its `training` history from the training log when
that log exists. A version still training, or one whose run failed, has no
report and is left out. Like the other `/v1` routes it requires the API key
when one is set.

`GET /health` returns `{"status": "ok", "model", "revision"}` for the primary.

### Pages

`GET /` is a playground page: a state box, a questions box, a model picker
filled from `GET /v1/models`, and each answer's probabilities drawn as bars
with the confidence beneath. A refused request shows its status, code and
message, so it is the quickest way to see what a server answers for and how.

![The playground after a request to a model named large](../assets/playground.png)

`GET /history` lists every retrained version with its verdict and training
curves, read from `GET /v1/retrain`; see
[Learn from your own traffic](#learn-from-your-own-traffic).

### Compressed bodies

A POST body may be sent with `Content-Encoding: gzip` or `zstd`; the server
inflates it before reading it, and a body that does not inflate is 422
`invalid_input`. Any other encoding is 415 `unsupported_encoding`, and a body
over 32 MiB, declared or inflated, is 413 `body_too_large`. TypeSafe's SDKs
zstd-compress every body over 1 KiB on their own, so this is what lets them
talk to ruling at all; a batch of thousands of rows shrinks several times
over, and the Python client gzips any body of 64 KB or more.

### Errors

Errors come in the hosted API's shape, `{"error": {"code", "message",
"param"?}}`, where `code` is stable for programs and `message` is for people:

| Status | `code` | When |
|---|---|---|
| 422 | `invalid_input` | a malformed request: `param` names the field, such as a choice over 255 options, a score over 10 levels, or more than 64 questions |
| 422 | `invalid_options` | a well-formed question the model cannot answer: a choice with one option or more than the model reads, or nothing to ask; `param` is `questions.<id>` |
| 422 | `unsupported_modality` | the state holds images and the selected model reads only text (`param` is `state`), or a box question went to a model that does not locate (`param` is `questions`) |
| 422 | `context_too_long` | a request over a token limit; `param` is `questions.<id>` when one question is at fault |
| 422 | `invalid_input` | without `param`: a request the engine refuses, such as one a remote decision server rejects, or a compressed body that does not inflate; `param` is `assets` for an asset the text never mentions, a part no asset names, or bytes that are not an image; on the outcome routes, `param` is `labels` for a label that does not fit the logged question and `corrections` for a mark that does not say what was right, prefixed `outcomes.<i>.` in a batch; on the batch routes, `param` is `rows` for no rows, more than 50,000, or none with labels, and `rows.<i>` for a row an evaluation cannot score |
| 413 | `body_too_large` | a body over 32 MiB, as sent or once inflated |
| 415 | `unsupported_encoding` | a `Content-Encoding` other than `gzip` or `zstd` |
| 404 | `unknown_model` | a `model` the server does not answer for; the message lists the names it does |
| 404 | `unknown_decision` | an outcome for an id the ledger does not hold |
| 401 | `unauthorized` | an API key is configured and missing or wrong |
| 429 | `queue_full` | `RULING_MAX_PENDING` requests are already in flight; `Retry-After` says how many seconds to wait |
| 503 | `upstream_unavailable` | the selected model is served remotely and its host is down, with no failover |

TypeSafe's SDK reads `error.message` from this shape and retries 429 honoring
`Retry-After`, so both clients and
[rulesmith](https://github.com/bradAGI/rulesmith), which uses ruling as its
local judge, behave the same against ruling and the hosted services.

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
`response.nouls`, `response.scores` and `response.usage` all work, and so do
the client's images and `Box` questions against a `vision:` model: `Image`
objects in the state, in instructions or in criteria, named assets, and
`attach_assets`, which the client sends as a zstd-compressed multipart body.
This is covered by integration tests that install the real
[`typesafe-sdk`](https://github.com/typesafe-ai/typesafe-sdk-python) package
and run it against live ruling servers, so the compatibility claim fails
loudly if either side drifts.

## Clients

### Python

In this package:

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

The thresholds belong in your code, versioned and tested. Fit them on labeled
data from your own traffic; see [calibration](measurements.md#running-the-harness).

`RulingClient(api_key=...)` (or `RULING_API_KEY`, the server's own variable)
talks to a server that requires a key. With a ledger, `response.id` and
`client.outcome(response.id, {"team": "billing"})` report what was right
(`corrections={"team": {"correct": True}}` marks it instead, and
`client.outcomes([...])` sends up to 1,000), and
`system_one(..., exclude_from_training=True)` keeps a decision out of it.
`client.models()` lists the served models.

Batches go through the same client. `predict` takes rows in any
[dataset](#datasets) shape and returns one entry per row in order, each
answer parsed into the same `ChoiceAnswer`, `ScoreAnswer` and `NoulAnswer`
models a decision returns, and a refused row's `error` left as the server's
dict; `evaluate` returns the report `ruling eval` prints; `usage` the hourly
buckets:

```python
from datetime import datetime, timedelta

rows = [{"id": t.id, "state": t.body, "questions": questions, "labels": {"team": t.team}} for t in tickets]

for row in client.predict(rows):                        # labels ignored
    if "error" in row:
        print(row["id"], row["error"]["code"])
    else:
        print(row["id"], row["answers"]["team"].choice)

report = client.evaluate(rows)                           # labels scored
print(report["overall"]["accuracy"], report["by_question"]["team"]["selective"]["0.9"])

for bucket in client.usage(starting_on=datetime.now() - timedelta(days=1)):
    print(bucket["hour"], bucket["model"], bucket["decisions"], bucket["input_tokens"])
```

`client.ask("is water wet?")` posts a question in words to `/v1/ask` and
returns its reading as an `AskResponse`: `readings`, `answers` with what was
`said` and whether each was `checked`, the nine `facts`, and what the writer
`wrote`.

Images go into a state as the wire dict `/v1/systemone` reads; `Image` builds
one from a file (`.png`, `.jpg`, `.jpeg` or `.webp`; anything else raises
`ValueError`) or from bytes and a media type, checking the size limit the
server enforces:

```python
from ruling import Box, Image

response = client.system_one(
    state={"receipt": Image.from_path("receipt.jpg"), "claimed": 42.10},
    questions={"matches": Noul(instructions="The total in <image 1> is the claimed amount."),
               "where": Box(instructions="the total line on <image 1>")},
)
response.answers["where"].box.top          # fractions of the image, with .bottom, .left, .right

response = client.system_one(
    state="The customer returned <item>.",
    questions={"damaged": Noul(instructions="<item> is damaged.")},
    assets={"<item>": Image.from_path("returned.jpg")},   # named wherever the text says <item>
)
```

A refused request raises `RulingError`, carrying the server's `status`,
`code`, `message` and `param`. A full queue (429), an absent upstream (503)
and a connection failure are tried again first: `RetryPolicy(max_retries=2,
backoff_initial=0.5, backoff_max=5.0)` is the default, waiting the server's
`Retry-After` when it sends one and otherwise doubling from half a second;
`RulingClient(retry=RetryPolicy(max_retries=0))` turns retries off. Bodies of
64 KB or more are sent gzipped; `RulingClient(gzip=False)` sends them plain.

### TypeScript

TypeScript and JavaScript, zero dependencies, in [`sdks/typescript`](../sdks/typescript):

```js
import { RulingClient, box, choice, image, noul, score } from "ruling";
const { answers } = await new RulingClient().systemOne({ state, questions: { team: choice("Which team?", { billing: null, technical: null }) } });
const located = await new RulingClient().systemOne({
  state: "The customer returned <item>.", assets: { "<item>": image(bytes, "image/jpeg") },
  questions: { where: box("the damage on <item>") },
});
```

## Datasets and reports

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

### Evaluation output

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
