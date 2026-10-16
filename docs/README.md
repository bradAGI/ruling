# ruling documentation

ruling turns any open chat model into a decision engine like TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): a state
and typed questions in, a probability for every declared option out, nothing
generated. Start with [Getting started](getting-started.md).

| Page | What's in it |
|---|---|
| [Getting started](getting-started.md) | Install, the first request, the playground and clients, asking about images, limits |
| [Choosing models](models.md) | The primary model, serving several from one process, escalating doubtful questions to a secondary |
| [Fine-tuning](fine-tuning.md) | Training an adapter on your own decisions, images included, and retraining from production outcomes |
| [Results](results.md) | Accuracy and coverage against Jev and the open decision models, and where Jev leads |
| [How it works](how-it-works.md) | One prefix, many suffixes; reading logits instead of sampling; confidence |
| [Reference](reference.md) | Every setting, model prefix, API route, error code and client |
| [Measurements](measurements.md) | The full record, with the command behind every number |
| [Adapter card](adapter.md) | The adapter we trained, its data and how to reproduce it |
| [Background](background.md) | What Jev is, the other open recreations, related work |
| [Development](development.md) | Repository layout and running the tests |
