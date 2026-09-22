![ruling: typed decisions, no text](assets/banner.png)

Typed decisions from a local model, no text generated. Not affiliated with TypeSafe AI.

An open recreation of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model. You send a state and typed questions; you get
back probability distributions over answers you declared, with a confidence
score, in one request. No text is generated, so there is nothing to parse and
no way for the model to invent an option you did not list.

It runs on an Apple Silicon Mac with [MLX](https://github.com/ml-explore/mlx)
and any 4-bit chat checkpoint, or against any OpenAI-compatible endpoint that
returns logprobs. No training required. TypeSafe's own SDK works against it
unchanged.

![Jev against Qwen3.6-35B-A3B on TypeSafe's 102 published rows, Every's 154 author-labeled judgments, and the two pooled](assets/against-jev.png)

*Every judgment TypeSafe and Every have published with Jev's own answers
attached, re-scored by a 19 GB open model with no training: 231 of 256 against
Jev's 238, exact McNemar p = 0.21. Method, caveats and where Jev leads are
below; [docs/background.md](docs/background.md) has what Jev is and how the
other open recreations compare.*

curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"message": "My card was charged twice for one order. Refund the duplicate."},
  "questions": {
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today"]},
    "refund":  {"type": "noul",   "instructions": "The customer asks for a refund."}
  }}'
```

```json
{
  "model": "mlx-community/Qwen3.5-4B-4bit",
  "answers": {
    "team":    {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.997, "technical": 0.002, "sales": 0.001}, "confidence": 0.995},
    "urgency": {"type": "score", "score": 1.74, "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.066, "1": 0.131, "2": 0.803}, "confidence": 0.606},
    "refund":  {"type": "noul", "noul": 0.955}
  },
  "usage": {"input_tokens": 405, "output_tokens": 0}
}
```

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
| `RULING_MODEL` | `mlx-community/Qwen3.5-4B-4bit` | Any MLX chat model, `openai:<model>` for a model behind an OpenAI-compatible server, or `openrouter:<model>`. Tested: Qwen3.5, Llama 3.2, Gemma 3 |
| `RULING_ROTATIONS` | `3` | Answer orderings averaged per question. `1` is fastest. See How it works |
| `RULING_PRIOR_DEBIAS` | `false` | Subtract a content-free letter prior. Measured, does not help; kept so you can re-check with another model |
| `RULING_CALIBRATION` | unset | Path to a `calibration.json` written by `ruling calibrate` |
| `RULING_API_KEY` | unset | When set, `/v1/*` requires `Authorization: Bearer <key>` |
| `RULING_MAX_INPUT_TOKENS` | `65536` | Whole-request budget: state plus every question. Above this the request is rejected with 413 |
| `RULING_MAX_BRANCH_TOKENS` | `32768` | Per-branch budget: state plus one question. A single oversized question is rejected even when the request fits |
| `RULING_OPENAI_BASE_URL` | unset | Required for `openai:` models. The endpoint serving them, ending in `/v1` |
| `RULING_OPENAI_API_KEY` | unset | Sent as a bearer token when the host wants one. Local servers usually do not |
| `RULING_OPENAI_TOP_LOGPROBS` | `20` | How many top logprobs the host returns, which is also the option ceiling. OpenAI allows 20, `mlx_lm.server` allows 11 |
| `RULING_OPENAI_EXTRA_BODY` | `{}` | JSON merged into every request, for host-specific fields such as `{"chat_template_kwargs": {"enable_thinking": false}}` |
| `RULING_ADAPTER` | unset | A LoRA directory from `ruling train`; see [docs/adapter.md](docs/adapter.md) |
| `RULING_CASCADE_TO` | unset | A second model that answers the questions the first is unsure about |
| `RULING_CASCADE_THRESHOLD` | unset | Top probability below which a question is escalated; required with `RULING_CASCADE_TO` |
| `RULING_HOST`, `RULING_PORT` | `127.0.0.1`, `8010` | Bind address |

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
export RULING_OPENAI_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
uv run ruling eval datasets/authored144.jsonl
```

What a hosted model costs you: the option ceiling drops from 255 to however
many logprobs the host returns, there is no shared prefix cache so the state is
re-read for every question, and a reasoning model has to be told to stop
reasoning, differently on each host. What it buys: your model, your hardware,
any platform. An integration test runs this path end to end against a real
server and checks the distributions match the local engine's.

One-shot, without a server:

```bash
uv run ruling ask "Since this morning every request with our API key returns 401." \
  --choice "team=Which team should handle this?|billing,technical,sales" \
  --score  "urgency=How urgent is this?|Can wait,This week,Today" \
  --noul   "blocked=The customer is blocked from using the product."
```

## Escalate only the doubtful questions

`RULING_CASCADE_TO` names a second model that answers whichever questions the
first one is unsure about, judged by calibrated top probability against
`RULING_CASCADE_THRESHOLD`. The first model can be the 4B with its adapter; the
second can be the local 35B or any `openai:` endpoint.

```bash
RULING_ADAPTER=runs/adapters/v1 RULING_CASCADE_TO=mlx-community/Qwen3.6-35B-A3B-4bit \
RULING_CASCADE_THRESHOLD=0.88 uv run ruling serve
```

Measured on saved predictions, with the threshold fitted on one dataset and
applied to the other so it is never tuned on the rows it is scored on:

| | 4B + adapter alone | 35B alone | cascade | questions sent to the 35B |
|---|---:|---:|---:|---:|
| TypeSafe 102 | 86 | 91 | 90 | 13.7% |
| Every 154 | 141 | 144 | 145 | 13.0% |

The cascade does not beat the larger model; it matches it while that model
runs on one question in eight. It works because the adapter made the small
model's confidence trustworthy: see coverage below.

## API

`POST /v1/systemone`

```json
{
  "model": "optional; must match the loaded model",
  "state": "string, object, or array",
  "questions": {
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "description, JSON, or null"}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["lowest level", "...", "highest level"]},
    "<id>": {"type": "noul",   "instructions": "a statement to judge"}
  }
}
```

Instructions, option descriptions, and levels may be strings or JSON
structure. Answers come back keyed by the same ids:

| Type | Fields |
|---|---|
| choice | `choice` (the argmax key), `probabilities` over keys, `confidence` |
| score | `score` (probability-weighted level index, so `1.74` sits between levels 1 and 2), `legend` index to level text, `probabilities` over indices, `confidence` |
| noul | `noul`, the probability the statement is true |

`usage.input_tokens` counts every token the model processed: the shared prefix
once, plus each question suffix for each ordering. `output_tokens` is always 0.

Errors: 422 for a malformed request or a Choice over 255 options, 413 for a
request over a token limit, 404 for a `model` that is neither the loaded one
nor a default alias (`jev-latest`, `default`), 401 when an API key is
configured and missing.

`GET /v1/models` returns `{"models": [...]}` with the hosted API's `name`,
`description` and `release_date`, plus the Hugging Face revision, adapter
digest and limits that only a local engine knows. `GET /health` returns the
model id. `GET /` is a playground page.

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

TypeScript and JavaScript, zero dependencies, in [`sdks/typescript`](sdks/typescript):

```js
import { RulingClient, choice, noul, score } from "ruling";
const { answers } = await new RulingClient().systemOne({ state, questions: { team: choice("Which team?", { billing: null, technical: null }) } });
```

The thresholds belong in your code, versioned and tested. Fit them on labeled
data from your own traffic; see [calibration](docs/measurements.md#evaluation-and-calibration).

## Measurements, summarized

Every number here comes from one batch on 18 September 2026, one Mac, four
models against four held-out sets. The full record, with every setting and the
commands to reproduce each cell, is in [docs/measurements.md](docs/measurements.md).

### Every model on every held-out set

Accuracy, as argmax agreement with each set's own reference labels. `r` is the
number of option orderings averaged per question.

| model | typesafe102 (102) | authored144 (144) | perturbations108 (108) | Every (154) |
|---|---|---|---|---|
| | r=1 / r=3 / r=6 | r=1 / r=3 / r=6 | r=1 / r=3 / r=6 | r=1 / r=3 |
| Qwen3.5-4B, the default | 0.804 / 0.814 / 0.814 | 0.882 / 0.917 / 0.917 | 0.861 / 0.907 / 0.907 | 0.916 / 0.929 |
| Qwen3-4B-Instruct-2507 | 0.765 / 0.833 / 0.833 | 0.840 / 0.875 / 0.875 | 0.833 / 0.861 / 0.861 | 0.916 / 0.916 |
| Qwen3-4B-Instruct + our adapter | 0.794 / 0.843 / 0.843 | 0.861 / 0.889 / 0.889 | 0.861 / 0.861 / 0.861 | 0.903 / 0.916 |
| Qwen3.6-35B-A3B, 19 GB | 0.853 / 0.873 / **0.892** | 0.938 / 0.958 / 0.958 | 0.935 / 0.954 / 0.954 | **0.935** / 0.922 |
| Jev, published | 0.882 | — | — | 0.961 |

Jev has one number per set because rotations are ruling's setting, not
TypeSafe's, and it has no entry on the two authored fixtures because those are
[TheoLeeCJ's](https://github.com/TheoLeeCJ/openjev) and Jev was never run on
them. Every skips r=6: its questions carry at most three options, and the
engine caps orderings at the option count, so r=6 there is the same computation
as r=3.

![Agreement with TypeSafe's reference against latency for three questions, with Jev's published accuracy marked](assets/frontier.png)

*What each model costs for what it gets right. Jev's line is its published
accuracy; its latency is a vendor range, 70 to 500 ms, so it is drawn as a line
rather than a point.*

**Coverage at a 5% error budget** is the share of decisions you could automate,
accepting in confidence order, before the accepted set exceeds 5% errors. It is
what a threshold or a cascade actually depends on, and a temperature cannot
move it, because it measures the ordering. Three orderings, same batch:

| | TypeSafe 102 | Every 154 | authored144 | perturbations108 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.608 | 0.818 | 0.924 | 0.907 |
| Qwen3-4B-Instruct | 0.216 | 0.929 | 0.847 | 0.657 |
| ↳ + our adapter | 0.392 | 0.948 | 0.882 | 0.815 |
| Qwen3.6-35B-A3B | 0.716 | 0.948 | 1.000 | 1.000 |
| **Jev, published** | **0.784** | **1.000** | — | — |

The adapter roughly doubles the trainable 4B's coverage on TypeSafe's rows and
cuts its confident errors (wrong at p ≥ 0.9) from 13.7% to 9.8%, 9.0% to 0.7% on
authored144, 9.3% to 1.9% on perturbations108. Its accuracy barely moves; its
usefulness for automation does. The adapter's card is at
[docs/adapter.md](docs/adapter.md).

## Where Jev leads

The headline tie is real and so is this list. On the same rows, Jev is ahead on:

- **Every's 154 author-labeled judgments.** 148 against the 35B's 142 (p = 0.07
  on that set alone). This is the cleanest accuracy comparison we have, and Jev
  wins it.
- **Confidence ordering.** Pooled over all 256 replayable judgments, Jev's
  coverage at 5% error is 0.961 against the 35B's 0.863, and its confident-error
  rate is 0.4% against 3.5%. When Jev says 0.95 it is almost never wrong; our
  models are wrong at that confidence nine times as often. This is the gap that
  matters for automation, and post-training narrowed it without closing it.
- **Distributions, not just argmax.** Jev's probabilities sit closer to
  TypeSafe's reference than ours (0.130 against 0.142 mean total variation) even
  where the argmax agrees.
- **Long, knowledge-heavy inputs.** Other replications that run Jev on MMLU-style
  and multi-hop policy items ([Kev](https://github.com/jaredpalmer/kev),
  [decider](https://github.com/Mapika/decider)) report Jev 10 to 18 points ahead
  there; we have not measured that suite yet and expect the same.
- **No ordering averaging needed.** Jev reads the option list once and is
  order-stable enough not to need three passes. We buy the same stability with
  rotations, at the cost of extra suffix rows.

## How it works

![Measured replay: ruling returns every typed answer at once while the same model streams JSON token by token](assets/replay.gif)

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
4. **Two-stage Choice.** Above 62 options (the tokenizer's single-token codes)
   every option is first judged alone as a Noul, the strongest 8 go to a final
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

![Prefill runs at 5,140 tokens per second and 41 TFLOP/s; decoding the same answer as JSON runs at 131 tokens per second and 1.0 TFLOP/s](assets/readout.png)

*Why stopping after prefill is fast. Both bars are the same 4-bit 4B model on
the same Mac. Prefill processes every token of the prompt together and keeps
the GPU compute-bound; decoding reads the whole model out of memory for each
token it emits and leaves it memory-bound at about a fortieth of the
throughput. Regenerate with `uv run --with matplotlib python assets/charts.py`.*

## Limits

- Probabilities are model scores. Calibration makes them honest on average on
  data like your calibration set. It does not make any single answer right.
  A well-formed wrong answer is still wrong.
- One request at a time. Questions inside a request are batched; concurrent
  requests queue behind a lock. There is no cross-request batching and no
  per-request timeout, because an MLX forward pass cannot be interrupted.
- Text only. No images, no audio, no generated explanation.
- macOS on Apple Silicon only, because the inference path is MLX. A PyTorch
  path for Linux and CUDA would be a second inference engine to keep in sync
  with the batching and cache logic here, and has not been written.

## Layout

```
ruling/
  questions.py     request and answer types (pydantic)
  prompt.py        state and question rendering, orderings, the content-free prior prompt
  codes.py         single-token answer codes verified against the tokenizer
  engine.py        MLX scoring: prefix cache, batched suffixes, ordering averaging, two-stage Choice
  calibration.py   temperature scaling with provenance, confidence, ECE
  evals.py         labeled datasets, metrics, temperature fitting
  build.py         WANLI and SST-5 builders from pinned revisions
  server.py        FastAPI app, bearer auth, request log, playground
  playground.html  browser playground served at /
  sdk.py           Python client
  cascade.py       two engines behind one answer, the second for doubtful questions
  cli.py           ruling serve | ask | eval | calibrate | build-dataset
sdks/typescript/   zero-dependency JS client with TypeScript types
datasets/          labeled JSONL, the WANLI selection manifest, and the format
assets/replay.py   records the replay GIF above from real timings
assets/charts.py   redraws the figures from the evaluation JSON
docs/              the full measurement record, background on Jev and the other recreations, the adapter card
tests/unit         no model needed
tests/integration  real model (RULING_TEST_MODEL, default Qwen3.5-0.8B-4bit); node for the JS client test
.github/workflows  CI on GitHub's Apple Silicon macOS runners
```

```bash
uv run pytest tests/unit
uv run pytest tests/integration                                   # downloads the 0.8B test model on first run
RULING_SMOKE_MODELS=mlx-community/Llama-3.2-1B-Instruct-4bit uv run pytest tests/integration/test_models.py
```

## Other related work

- [snellingio/system-one](https://github.com/snellingio/system-one): the same
  API shape on MLX, one prompt per question with the state repeated, no
  calibration, no ordering averaging, no evaluation harness.
- Archer Hume, [Jev's Architecture
  Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/): the
  black-box probe this repository's architecture notes lean on, and the source
  of the listwise option-interaction experiment.

The methods here are standard, and the papers behind them are worth reading
directly:

- Zhao, Wallace, Feng, Klein and Singh, *Calibrate Before Use: Improving
  Few-Shot Performance of Language Models* (2021), for the content-free prior
  that `RULING_PRIOR_DEBIAS` implements.
- Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*
  (2017), for temperature scaling and expected calibration error.
- Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and
  Estimation* (2007), for why log loss and Brier score are the right training
  and reporting objectives here.
- The FIRST listwise ranking method, which reads a ranking from first-token
  logits rather than generating it, is the closest published relative of this
  readout; Hume's analysis above traces the connection.

## License

MIT. ruling is an independent project with no connection to TypeSafe AI; "Jev"
and "System One" are their names, used here to say what this reproduces. The
`authored144` and `perturbations108` datasets and the WANLI row selection are
converted from TheoLeeCJ/openjev under MIT, with their copyright notice in
[THIRD_PARTY.md](THIRD_PARTY.md). WANLI is CC-BY-4.0 and SST-5 keeps its own
terms; neither is committed here. Model weights keep their own licenses.
