# ruling documentation

ruling turns any open chat model into a decision engine like TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): a state
and typed questions in, a probability for every declared option out, nothing
generated. Start with [Getting started](getting-started.md).

## Use it

| Page | What's in it |
|---|---|
| [Getting started](getting-started.md) | Install, the first request, the playground, images, the clients, limits |
| [Choosing models](models.md) | The primary model, serving several from one process, escalating doubtful questions to a secondary |
| [Reference](reference.md) | Every setting, model prefix, command, API route, error code and client |

## Improve it

| Page | What's in it |
|---|---|
| [Fine-tuning](fine-tuning.md) | Training an adapter on your own decisions, images included, and retraining from production outcomes |
| [Adapter card](adapter.md) | The adapter we trained, what it changed, its data, and how to reproduce it |

## Understand it

| Page | What's in it |
|---|---|
| [How it works](how-it-works.md) | One prefix, many suffixes; reading logits instead of sampling; confidence |
| [Results](results.md) | Accuracy and coverage against Jev and the open decision models, images, and where Jev leads |
| [Measurements](measurements.md) | The full record, with the command behind every number |
| [Design notes](design.md) | Why a decoder with a logit readout, what training taught us, the encoder that was parked |
| [Background](background.md) | What Jev is, the other open recreations, related work |

## Work on it

| Page | What's in it |
|---|---|
| [Development](development.md) | Repository layout, running the tests, cutting a release |
| [Datasets](../datasets/README.md) | The record format and the bundled and on-demand evaluation sets |
