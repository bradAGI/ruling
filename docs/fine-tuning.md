# Fine-tuning

## Make it yours: fine-tune on your own decisions

The probabilities a stock model gives are usually right but often
overconfident. A short LoRA fine-tune on a few hundred of your own labeled
decisions fixes that, and it is the step that makes a confidence threshold
safe to act on. Four commands.

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
three- to thirteen-fold. Full card and reproduction in
[docs/adapter.md](adapter.md).

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
`--max-tokens` to fit the image and the question: an example longer than it
is skipped. A text model refuses records with images rather than training
on their text alone.

Expect it to be slow and to need steps. The vision models are Qwen3.5, whose
hybrid attention trains 15 to 30 times slower in MLX than a pure-attention
model, and every image adds up to `--max-image-tokens` tokens to its example.
On the 0.8B test model, teaching an arbitrary code for each of three colors,
which only the picture shows, took 300 steps at a learning rate of 3e-5 to
bring the validation loss from 2.3 to 0.4; fewer steps, or higher rates, left
it at chance. We have not yet measured a gain on real photos.

## Keep learning from production

Set `RULING_LEDGER=ledger.sqlite` and
the server records every decision and returns its `id` (a request can opt out
with `exclude_from_training`); post what turned out to be right to
`/v1/decisions/<id>/outcome`. Then run, on a schedule:

```bash
uv run ruling retrain --ledger ledger.sqlite --adapters runs/adapters/tickets --holdout my-holdout.jsonl
RULING_ADAPTER=runs/adapters/tickets/current uv run ruling serve
```

Each run trains a new version on the resolved decisions, starting from the one
being served, and promotes it to `current` only if it beats that one on your
holdout: lower log loss, no lost accuracy. The server's `/history` page,
linked from the playground, lists every run: the held-out scores of both
versions, and why each was or was not promoted. See
[Learn from your own traffic](reference.md#learn-from-your-own-traffic). A
`vision:` model retrains the same way, from decisions about images.
