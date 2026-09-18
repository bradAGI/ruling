![ruling: typed decisions, no text](assets/banner.png)

Typed decisions from a local model, no text generated. Not affiliated with TypeSafe AI.

An open-weight recreation of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model. You send a state and typed questions; you get
back probability distributions over answers you declared, with a confidence
score, in one request. No text is generated, so there is nothing to parse and
no way for the model to invent an option you did not list.

It runs on an Apple Silicon Mac with [MLX](https://github.com/ml-explore/mlx)
and a 4-bit Qwen3.5 checkpoint. No waitlist, no GPU server, no API key. Point
`RULING_MODEL` at a 19 GB mixture-of-experts model instead and it reaches
Jev's accuracy on TypeSafe's own evaluation rows; see Measurements.

![Measured replay: ruling returns every typed answer at once while the same model streams JSON token by token](assets/replay.gif)

*Same 4B model, same state, same three questions, measured on one Mac and
aligned at t = 0. ruling answered in 119 ms, median of five. Generating the
same answers as JSON took 755 ms and 105 tokens, median of three, and the
generated object invented a `refund` key inside `team` that no schema asked
for. Reproduce with `uv run --with pillow python assets/replay.py`.*

```
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

## What Jev is, and what this reproduces

Jev, launched September 15, 2026, is a closed hosted model that answers three
kinds of typed question against a shared state: **Choice** (one option from a
set), **Score** (a level on an ordered scale), and **Noul** (is a statement
true). Every answer carries probabilities. TypeSafe trains it with a method it
calls RLCD, Reinforcement Learning for Calibrated Decisions, and has published
neither the architecture, the weights, nor a paper.

Jev's architecture is undisclosed. It is not a fixed-label classifier, because
it accepts arbitrary instructions, option descriptions, and JSON state at
request time. A [black-box analysis by Archer
Hume](https://archerhume.com/posts/jevs-architecture-unmasked/) probes the
hosted API and concludes it is a causal decoder language model post-trained for
this contract: the state is shared across questions, each question is scored in
its own branch that cannot see its siblings, options inside one question
influence each other, and the answer is read as numbers rather than generated
as text. TypeSafe has not confirmed that, but it matches every behavior we can
measure from outside. The mechanism that buys the speed is the same either way:
encode the input once, read the model's scores at the answer position, restrict
them to the declared options, normalize. ruling implements that mechanism on a stock
instruction-tuned open model. On the two fixtures the other open recreation
published, ruling scores higher than it reported — 0.920 against 0.813 on
authored144, 0.684 against 0.637 on WANLI (see the comparison at the end). On
TypeSafe's own published evaluation rows, a 19 GB open model running through
ruling answers 91 of 102 questions the way TypeSafe's reference does, against
Jev's 90, with no training of any kind.

| | Jev (hosted) | ruling |
|---|---|---|
| Primitives | Choice, Score, Noul | Choice, Score, Noul, same request and response shape |
| Many questions per call, one state | Yes | Yes. The state is encoded once; every question suffix runs in batched forward passes against replicas of that cache |
| Probabilities and confidence | Yes | Yes. Confidence uses the formulas from TypeSafe's MIT-licensed client, so the number means the same thing: for Choice, how far the leading option stands above uniform; for Score, how tightly mass sits around the modal level |
| Calibration | RLCD training, undisclosed | Temperature scaling fit on your labeled data, bound to the model and settings it was fit for |
| Option-order robustness | Unknown | Averaging over option orderings, on by default |
| Max options per Choice | 255 | 255. Above the tokenizer's 62 single-token codes a Choice runs in two stages |
| Structured instructions and descriptions | JSON accepted | JSON accepted |
| TypeSafe's official SDK | Yes | Yes. Point `TYPESAFE_BASE_URL` at a ruling server and their unmodified client works; there is a test for it |
| Context | ~32k tokens per question branch, ~64k per request | Same two limits by default, both configurable |
| Weights | Closed | Any MLX chat model on Hugging Face |
| Latency, 3 questions, short state | 70 to 500 ms, vendor claim | 80 to 136 ms on a 4B, 141 to 204 ms on the 19 GB model, see Measurements |

Not reproduced: RLCD training, Jev's parallel sampler, and frontier-level
judgment. Read the Measurements section before trusting any number here.

## ruling versus TheoLeeCJ/openjev

[TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) is the other open
recreation, released the same day as Jev. It asks "can a home GPU do what Jev
does?" and answers with a benchmark. ruling is the thing you call from software.
Both read option logits from a Qwen3.5-4B model; everything above that differs.

| | TheoLeeCJ/openjev | ruling |
|---|---|---|
| Form | Research CLI: JSONL of questions in, results file out | Service: HTTP endpoint with Jev's request and response shape, Python and TypeScript clients, playground, one-shot CLI |
| Question types | One: pick from up to 16 lettered options | Three, matching Jev: Choice to 255 options, Score, Noul, each with probabilities and confidence |
| Questions per call | One per row | Any number against one state, one request |
| Shared-state reuse | Special mode; every row must carry the identical state | Always: state encoded once, all question suffixes in one batched forward pass |
| Option-order bias | Measured it, 10 of 36 flips, left it | Averaged over orderings; 4 of 144 flips |
| Calibration | None | Temperature scaling, bound to the model it was fit on |
| Hardware | CUDA GPU, BF16 | Apple Silicon Mac, 4-bit, no GPU box |
| authored144, their fixture | 0.813 | **0.920** at 4B, **0.964** at 35B |
| WANLI 256, their row selection | 0.637 | **0.684** |
| TypeSafe's public evals, agreement with reference | 0.85 | 0.814 at 4B, **0.892** at 35B |
| Still theirs | Committed raw predictions, a WebGPU demo | |

Same rows, same metrics, and on the 4B rows ruling runs a quantized version of
the same base model. Procedures and caveats are in the comparison section at
the end; the three fixtures in `datasets/` are theirs, converted and credited
in [THIRD_PARTY.md](THIRD_PARTY.md).

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
data from your own traffic; see Calibration.

## How it works

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

## Evaluation and calibration

Datasets are JSONL records with a state, questions in the API shape, and a
label per question. Four are bundled and two more build from pinned public
sources; see [datasets/README.md](datasets/README.md).

```bash
uv run ruling eval datasets/authored144.jsonl                   # metrics per question type
uv run ruling eval datasets/authored144.jsonl --rotations 1     # same, one ordering
uv run ruling build-dataset wanli --output datasets/wanli256.jsonl
uv run ruling calibrate datasets/wanli256.jsonl --output calibration.json
RULING_CALIBRATION=calibration.json uv run ruling serve
```

`eval` reports, per question type: accuracy, balanced accuracy, mean
per-family balanced accuracy when records carry a `family`, expected
calibration error, Brier score, negative log-likelihood, and mean confidence.
For Choice questions it also scores every record with the options reversed and
reports how many argmax answers flip. Reports record the model's Hugging Face
revision and the scoring settings.

`calibrate` fits one temperature per question type by minimizing negative
log-likelihood. The file records the model, revision, rotation count, and
debiasing setting it was fit under, and the server refuses to load it against
anything else.

## Measurements

Every number in this section comes from one batch, run on 18 September 2026 on
a single Apple Silicon Mac: one code version, one sitting, four models against
the same four held-out sets, with no calibration file loaded (temperature 1.0)
and prior debiasing off. Reproduce a cell with

```bash
uv run ruling eval <dataset> --model <model> --rotations <n>
```

**How the defaults were chosen, and what leaked.** `RULING_ROTATIONS=3` was
originally picked because it scored best on `authored144`, so that set is not
clean evidence for it. TypeSafe's 102 rows and Every's 154 were scored
afterwards and confirm it independently. Prior debiasing was tried the same way
and lost on every model and dataset, so it ships off.

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

### What ordering averaging buys

![Accuracy against the number of option orderings averaged, for four models on four held-out sets](assets/rotations.png)

Going from one ordering to three is worth 1 to 5 questions on every model and
set. Going from three to six changes nothing anywhere except TypeSafe's rows,
where the 35B gains two more — and those are the only questions in any of these
sets with five options. `Engine._rotations_for` caps orderings at the number of
options, so the flat cells are flat by construction, not by chance.

The dashed line is Jev's published figure on the two sets where it has one. The
flat right-hand halves are the cap at work, and the 35B crossing Jev on
TypeSafe's rows at six orderings is the only crossing anywhere in these panels.

What averaging reliably buys is order stability. Rescoring every row with its
options reversed, counting answers that flip:

| model | authored144 r=1 | r=3 | Every r=1 | r=3 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 15 | 6 | 26 | 5 |
| Qwen3-4B-Instruct | 15 | 5 | 29 | 12 |
| Qwen3-4B-Instruct + adapter | 12 | 3 | 24 | 6 |
| Qwen3.6-35B-A3B | 6 | 4 | 9 | 3 |

### What post-training bought

The adapter weights are not published; the corpus and the run that produced
them are. `ruling build-corpus public|states|worlds` rebuilds the training
records from pinned public datasets and the generators in this repository, and
the command below reproduces the adapter on a Mac in about 40 minutes:

```bash
uv run ruling train \
  --data runs/pool/public.jsonl runs/pool/states.jsonl runs/pool/worlds.jsonl \
  --holdout datasets/authored144.jsonl datasets/perturbations108.jsonl \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit --output runs/adapters/v1
```

`ruling train` fine-tuned a LoRA adapter on 27,021 constructed records —
reframed public datasets, generated program states, and synthetic worlds — with
every evaluation set above passed to `--holdout`. In-distribution validation
accuracy went from 0.607 to 0.844.

Held out, the adapter is worth 1 to 3 questions of accuracy at one ordering and
approximately nothing at three. What it does move is calibration, in all eleven
comparisons:

![Expected calibration error before and after the adapter, on four held-out sets](assets/calibration.png)

| set | expected calibration error, base → adapter |
|---|---|
| typesafe102 r=1 | 0.228 → 0.162 |
| typesafe102 r=3 | 0.150 → 0.122 |
| authored144 r=3 | 0.108 → **0.055** |
| perturbations108 r=3 | 0.124 → **0.045** |
| Every r=3 | 0.084 → 0.059 |

Training on a proper scoring rule made the probabilities honest without making
the judgments better, and the effect carried to distributions the adapter never
saw. The accuracy gain concentrates at r=1 and vanishes by r=3, which says
training and ordering averaging are correcting the same defect — positional
bias — and do not stack. With the adapter you can run one ordering and keep
three-ordering accuracy, at half the latency.

### Noul and Score, held out

Earlier batch, `Qwen3.5-4B-4bit`, kept because these two sets were not re-run.

| Dataset | Rows | Type | Metric | r=1 | r=3 |
|---|---:|---|---|---:|---:|
| WANLI | 256 | noul, "the evidence establishes this claim" | balanced accuracy | **0.853** | 0.827 |
| | | | ECE | 0.048 | 0.081 |
| SST-5 | 300 | score, five sentiment levels | accuracy | **0.530** | 0.487 |
| | | | balanced accuracy | 0.477 | 0.464 |
| | | | ECE | 0.086 | 0.121 |

SST-5 is a five-way ordinal task where fine-tuned encoders reach about 0.59.
Zero-shot, from a 4-bit 4B model, 0.53 with an ECE under 0.1 is usable.

### Latency and throughput

Median of five requests, short JSON state, engine warm, same batch as the
accuracy numbers above.

| Request | | Qwen3.5-4B | Qwen3-4B-Instruct | Qwen3.6-35B-A3B |
|---|---|---:|---:|---:|
| 3 questions (Choice + Score + Noul) | r=1 | 80 ms | 78 ms | 141 ms |
| | r=3 | 136 ms | 116 ms | 204 ms |
| 21 Noul questions, one state | r=1 | 216 ms, 97/s | 178 ms, 118/s | 263 ms, 80/s |
| | r=3 | 495 ms, 42/s | 399 ms, 53/s | 606 ms, 35/s |

Every setting lands inside Jev's claimed 70 to 500 ms window for the
three-question request, including the 35B that reaches Jev's accuracy on
TypeSafe's rows. Run-to-run variation is real: the 35B measured 228 ms at r=3 a
day earlier on the same machine, about 10% above this batch. Measured on an
idle machine; under concurrent load the same requests took two to three times
longer, because requests queue behind a lock rather than batching together.

## Against Jev itself

Two public sources carry Jev's own answers on inputs anyone can replay.
`ruling build-dataset typesafe` and `ruling build-dataset every` rebuild them
with Jev's answer attached to every question, and `ruling eval` reports how
often ruling lands on the same answer. Both were run with the default
settings; the one-ordering numbers are in parentheses.

![Jev against Qwen3.6-35B-A3B on TypeSafe's 102 published rows, Every's 154 author-labeled judgments, and the two pooled](assets/against-jev.png)

**TypeSafe's published evaluation cases, 102 questions.** TypeSafe's
reference is the mean of GPT-6 Astra and Claude Fable 5.1 at high reasoning,
so this measures agreement with large models on workflows TypeSafe built,
not accuracy against truth.

| System | Agrees with TypeSafe's reference | Agrees with Jev's argmax |
|---|---:|---:|
| ruling, 35B-A3B, r=6 | **0.892** (91/102) | 0.853 |
| Jev, published | 0.882 (90/102) | |
| ruling, 35B-A3B, r=3 | 0.873 (89/102) | 0.853 |
| ruling, 35B-A3B, r=1 | 0.853 (87/102) | 0.833 |
| TheoLeeCJ/openjev, 4B BF16 | 0.85 | |
| ruling, Qwen3-4B-Instruct + adapter, r=3 | 0.843 (86/102) | 0.824 |
| ruling, Qwen3.5-4B, r=3 | 0.814 (83/102) | 0.824 |

On the default 4B model Jev is better here by about seven points, and on the
35B the two are level: 91 of 102 against Jev's 90 of 102. One question is a tie,
not a lead.

Pooled with Every's 154 labeled judgments — every row anywhere that carries
Jev's own answer — the 35B at shipped defaults gets 231 of 256 and Jev gets
238. An exact McNemar test on the paired rows gives **p = 0.21**: no
significant difference, with Jev nominally ahead. Taking the best rotation
setting per dataset instead gives 235 of 256 and p = 0.66; that number is
recorded here rather than quoted, because choosing a setting per dataset is
tuning on the evaluation.

**[Every's parallel judgment lab](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/),
758 judgments on 208 documents.** Every recorded Jev's answers for nine
experiments in August 2026. Three of them
carry author labels (the support-reply judge grid and two retrieval grids,
154 judgments); the rest measure agreement only.

| | Jev | Qwen3.5-4B | 35B-A3B |
|---|---:|---:|---:|
| Accuracy on the 154 author-labeled judgments | **0.961** | 0.929 (0.916) | 0.922 (0.935) |
| Agreement with Jev, all 758 judgments | | 0.858 (0.832) | 0.922 (0.921) |

Jev leads on the author-labeled rows by three points against the 4B and four
against the 35B; these 154 rows are the one place in this README where Jev is
ahead of everything we run. The 35B tracks Jev's own answers much more closely
than the 4B does, 0.922 against 0.858 across all 758 judgments. Note that both
open models score *higher* than the 35B on the labeled subset at three
orderings while agreeing with Jev *less* — agreement with Jev and accuracy
against the authors are different measurements, and this set is small enough
that a few questions separate them.

**What the published data says about Jev's mechanism.** Every's file records
token usage per experiment. Jev reports almost exactly 16 output tokens per
judgment in every experiment (16.0 to 17.0 across six of them, from 5 to 21
questions per call), and roughly 350 to 400 input tokens per call beyond the
document and question text. A constant per-question output budget is
consistent with a fixed answer block computed in parallel per question rather
than a decoded string. Probabilities are published to two decimals. The
models endpoint lists `jev-latest`, released 2026-09-10, and `jev-preview`.

**Live comparison.** `ruling eval <dataset> --against typesafe` sends every
record to the hosted API with `TYPESAFE_API_KEY` and reports Jev's live
answers next to ruling's on the same rows, including the author-labeled
datasets above where Jev has never been measured. The client speaks
TypeSafe's protocol, which is the same as ruling's, and is tested end to end
against a ruling server. It needs API credits on the TypeSafe account; the
run for this README stopped at `402 Payment Required` before any call was
scored.

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
  cli.py           ruling serve | ask | eval | calibrate | build-dataset
sdks/typescript/   zero-dependency JS client with TypeScript types
datasets/          labeled JSONL, the WANLI selection manifest, and the format
assets/replay.py   records the replay GIF above from real timings
assets/charts.py   redraws the figures in Measurements from the evaluation JSON
tests/unit         no model needed
tests/integration  real model (RULING_TEST_MODEL, default Qwen3.5-0.8B-4bit); node for the JS client test
.github/workflows  CI on GitHub's Apple Silicon macOS runners
```

```bash
uv run pytest tests/unit
uv run pytest tests/integration                                   # downloads the 0.8B test model on first run
RULING_SMOKE_MODELS=mlx-community/Llama-3.2-1B-Instruct-4bit uv run pytest tests/integration/test_models.py
```

## Comparison with TheoLeeCJ/openjev, procedures and caveats

The summary table is near the top. This section shows where each number comes
from and what does not compare cleanly.

**Same rows, both systems**

| Dataset | Metric | TheoLeeCJ, 4B BF16 | ruling r=1, 4B 4-bit | ruling r=3 (default), 4B 4-bit | ruling r=3, 35B 4-bit |
|---|---|---:|---:|---:|---:|
| authored144 | mean family balanced accuracy | 0.813 | 0.884 | 0.920 | **0.964** |
| perturbations108 | mean family balanced accuracy | | 0.850 | 0.909 | **0.954** |
| WANLI 256 | balanced accuracy | 0.637 | **0.684** | 0.672 | |

Same rows, same metric definitions, their numbers read from their published
results. ruling runs a 4-bit quantization of the same base model, which if
anything favors them.

**Order stability, related but not identical procedures**

| System | Procedure | Flip rate |
|---|---|---:|
| TheoLeeCJ | 36 base cases, each compared with its reversed-option twin | 10 of 36, 28% |
| ruling r=1, 4B | every row of authored144 rescored with options reversed | 15 of 144, 10.4% |
| ruling r=3 (default), 4B | same | 6 of 144, 4.2% |
| ruling r=3 (default), 4B | every row of perturbations108 rescored with options reversed | 4 of 108, 3.7% |
| ruling r=3, 35B | every row of authored144 rescored with options reversed | 4 of 144, 2.8% |

Both procedures ask the same question, does the argmax survive reversing the
options, but on different row sets, so compare the rates loosely.

**Throughput, not a controlled comparison**

For 21 binary decisions against one state they report 1,023 ms with batched
suffixes on an RTX 3090 in BF16. ruling measures 216 ms at one ordering and 495 ms
at three on an Apple Silicon Mac in 4-bit, on a different state. Hardware,
precision, and inputs all differ; the numbers show both are in the same range,
nothing finer.

**Where they are still ahead**

- Committed raw per-row predictions with checksums. ruling reports are
  regenerated by command, not committed.
- A browser-only WebGPU demo.

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
