# Fine-tuning

The probabilities a stock model gives are usually right but often
overconfident. A short LoRA fine-tune on a few hundred of your own labeled
decisions fixes that, and it is the step that makes a confidence threshold
safe to act on. This page walks through a run; every flag is in the
[Reference](reference.md#train).

## Train on your own decisions

**1. Write your decisions as JSONL**, one record per line, in the same shape
as a request plus the right answers:

```json
{"state": "Order 4471 arrived damaged, customer wants a replacement.",
 "questions": {"team": {"type": "choice", "instructions": "Which team?",
                        "criteria": {"returns": null, "billing": null, "shipping": null}},
               "urgent": {"type": "noul", "instructions": "This needs a reply today."}},
 "labels": {"team": "returns", "urgent": false}}
```

Choice labels are the option key, score labels the level index, noul labels
`true`/`false`. Keep a separate file of records you will never train on.

**2. Train.** The trainer refuses to start if any training state also appears
in a holdout file, so leakage cannot happen by accident.

```bash
uv run ruling train --data my-decisions.jsonl --holdout my-holdout.jsonl \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit --output runs/adapters/mine
```

About 40 minutes for 25,000 records on a Mac; a few hundred records take
minutes. Use a pure-attention base like `Qwen3-4B-Instruct-2507`; Qwen3.5's
hybrid attention trains 15 to 30 times slower in MLX.

**3. Check it on the holdout, then fit a temperature on it.**

```bash
uv run ruling eval my-holdout.jsonl --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter runs/adapters/mine
uv run ruling calibrate my-holdout.jsonl --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter runs/adapters/mine --output calibration.json
```

`eval` prints accuracy, calibration error, and **coverage at a 5% error
budget**: the share of decisions you can automate, accepting in confidence
order, before the accepted set exceeds 5% errors. That number is your
threshold's safety margin. The calibration file is bound to the exact model
and adapter it was fit on and will not load against anything else.

**4. Serve with both.**

```bash
RULING_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit RULING_ADAPTER=runs/adapters/mine \
RULING_CALIBRATION=calibration.json uv run ruling serve
```

What to expect, from our own adapter on 27,000 constructed records, tested on
four sets it never saw: accuracy moved by a question or two, calibration error
roughly halved, and confident errors (wrong while reporting p ≥ 0.9) fell
three- to thirteen-fold. Full card and reproduction in the
[adapter card](adapter.md).

## Watch a run

Every `--eval-every` steps the trainer prints one JSON line with the step,
the train loss since the last line, tokens per second, `elapsed_seconds`,
`progress` (the share of `--steps` done) and `eta_seconds` (the steps left at
the run's pace so far; a resumed run counts from where its checkpoint left
off), then the validation loss, accuracy, calibration error and coverage. The
same lines are kept as the history in the adapter's `training.json`.

`--wandb [PROJECT]` mirrors each line to a Weights & Biases run opened on the
run's config, in `PROJECT` or your account's default. Install the extra first
with `pip install 'ruling[wandb]'` or `uv sync --extra wandb`.

The top-level adapter files in the output directory are always the best
validation loss so far; `checkpoint/` is the latest step, and `--resume`
continues from it. Serve the former.

## Fine-tune on images

Records can carry images in their state, in the same
`{"type": "image", "media_type", "data"}` objects a request takes. Train them
on a `vision:` base and serve the adapter on the same base:

```json
{"state": {"note": "Customer says order 4471 was charged twice.",
           "receipt": {"type": "image", "media_type": "image/png", "data": "<base64>"}},
 "questions": {"same_order": {"type": "noul", "instructions": "The order number on <image 1> is the one in the note."}},
 "labels": {"same_order": true}}
```

```bash
uv sync --extra vision
uv run ruling train --data my-image-decisions.jsonl --holdout my-holdout.jsonl \
  --model vision:mlx-community/Qwen3.5-4B-4bit --max-tokens 2048 --max-image-tokens 1024 \
  --output runs/adapters/receipts
RULING_MODEL=vision:mlx-community/Qwen3.5-4B-4bit RULING_ADAPTER=runs/adapters/receipts uv run ruling serve
```

Each image is read into the prompt exactly as the server reads it; the vision
tower stays frozen and LoRA trains the language layers that read its
features. Records without images can sit in the same file. Keep
`--max-image-tokens` equal to the server's `RULING_MAX_IMAGE_TOKENS`, so the
adapter sees images at the size it will be served them, and raise
`--max-tokens` to fit the image and the question. An example longer than it
is dropped: the run says how many before its first step and records the
count in `training.json`, and a run with no example left refuses to start.
A text model refuses records with images rather than training on their text
alone.

Expect it to be slow and to need steps. The vision models are Qwen3.5, whose
hybrid attention trains 15 to 30 times slower in MLX than a pure-attention
model, and every image adds up to `--max-image-tokens` tokens to its example.
On the 0.8B test model, teaching an arbitrary code for each of three colors,
which only the picture shows, took 300 steps at a learning rate of 3e-5 to
bring the validation loss from 2.3 to 0.4; fewer steps, or higher rates, left
it at chance. On real photos, 300 steps on 84 insect pictures took the 4B
from 0.886 to 0.924 on a held-out half (+5.7 points at one ordering), fixing
every fly it had mistaken for a bee or a bug, but left it about twice too
sharp: a temperature fit on photos it had not seen brought its log loss below
the base's. Fit one on a calibration set afterward, as for any adapter. The
numbers are in [Measurements](measurements.md#fine-tuning-a-vision-model-on-photos).

## Keep learning from production

Set `RULING_LEDGER=ledger.sqlite` and the server records every decision and
returns its `id` (a request can opt out with `exclude_from_training`); post
what turned out to be right to `/v1/decisions/<id>/outcome`. Then run, on a
schedule:

```bash
uv run ruling retrain --ledger ledger.sqlite --adapters runs/adapters/tickets --holdout my-holdout.jsonl
RULING_ADAPTER=runs/adapters/tickets/current \
RULING_CALIBRATION=runs/adapters/tickets/current/calibration.json uv run ruling serve
```

Each run trains a new version on the resolved decisions, starting from the one
being served, and promotes it to `current` only if it beats that one on your
holdout: lower log loss at each version's own fitted temperature, no lost
accuracy. The new version's temperatures are saved beside its weights, so the
promoted adapter is served calibrated. The server's `/history` page, linked
from the playground, lists every run: the held-out scores of both versions,
why each was or was not promoted, and its training curve. Details in
[Learn from your own traffic](reference.md#learn-from-your-own-traffic). A
`vision:` model retrains the same way, from decisions about images.
