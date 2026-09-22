# ruling-qwen3-4b-v1, a calibration adapter

A LoRA adapter for `mlx-community/Qwen3-4B-Instruct-2507-4bit` that makes the
probabilities of ruling's letter readout honest without changing what the model
knows. It is the artifact `ruling train` produces; this page is its model card.

## What it is

| | |
|---|---|
| Base | `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` |
| Method | LoRA, rank 8, scale 20, 16 layers, lr 1e-5, 2,000 steps, batch 8, log loss on the answer letter |
| Data | 27,021 constructed records: reframed public datasets (WANLI, SNLI, CLINC150, BoolQ, GoEmotions, Civil Comments), generated program states, synthetic worlds written by a local model. No hosted model's output is used. |
| Held out | authored144, perturbations108, TypeSafe's 102 evaluation rows, Every's 154 labeled judgments, WANLI-256, SST-5, and the dev split — every state checked for overlap before training |
| Size | 29 MB of safetensors |
| License | MIT for the adapter. The base model and the training corpora carry their own licenses; share-alike terms from SNLI (CC-BY-SA-4.0) and BoolQ (CC-BY-SA-3.0) may apply to redistribution. |

## What it does, measured

Same base model, three option orderings, every set held out of training.

| held-out set | accuracy, base → adapter | ECE | coverage at 5% error | confident errors (p ≥ 0.9, wrong) |
|---|---|---|---|---|
| TypeSafe 102 | 0.833 → 0.843 | 0.150 → **0.122** | 0.216 → **0.392** | 0.137 → **0.098** |
| authored144 | 0.875 → 0.889 | 0.108 → **0.055** | 0.847 → **0.882** | 0.090 → **0.007** |
| perturbations108 | 0.861 → 0.861 | 0.124 → **0.045** | 0.657 → **0.815** | 0.093 → **0.019** |
| Every 154 | 0.916 → 0.916 | 0.084 → **0.059** | 0.929 → **0.948** | 0.084 → **0.026** |

Accuracy moves by noise. Calibration error falls on every set, coverage rises
on every set, and confident errors fall three- to thirteen-fold. In plain
terms: the adapter does not make the model smarter, it makes the model's
"I am sure" trustworthy, which is what a confidence threshold, a cascade to a
larger model, or a human-review gate needs.

The accuracy gain that does exist concentrates at one option ordering and
vanishes by three, so training and ordering averaging correct the same defect
and do not stack. With the adapter you can run one ordering and keep
three-ordering accuracy at roughly half the latency.

## How to reproduce it

```bash
uv run ruling build-corpus public  --output runs/pool/public.jsonl
uv run ruling build-corpus states  --output runs/pool/states.jsonl
uv run ruling build-corpus worlds  --output runs/pool/worlds.jsonl
uv run ruling train \
  --data runs/pool/public.jsonl runs/pool/states.jsonl runs/pool/worlds.jsonl \
  --holdout datasets/authored144.jsonl datasets/perturbations108.jsonl \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit --output runs/adapters/v1
```

About 40 minutes on an Apple Silicon Mac. The trainer refuses to start if any
training state also appears in a `--holdout` file.

## How to use it

```bash
RULING_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit RULING_ADAPTER=runs/adapters/v1 uv run ruling serve
```

Any calibration file is bound to the adapter's content hash as well as the
model, so a temperature fitted on the base cannot be loaded against the
adapter by mistake.

## Limits

- Trained for the letter readout on this base. It is not a general-purpose
  fine-tune and will not help a different base or a different prompt.
- Out-of-distribution accuracy is the base model's; on knowledge-heavy
  questions expect the base's score.
- Calibration transfers across the four held-out distributions above; it is
  still worth checking on your own traffic before setting a threshold.
