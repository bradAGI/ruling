# Getting started

## Try it in two minutes

You need an Apple Silicon Mac, Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruling serve          # downloads a 2.3 GB model on first start, then serves http://127.0.0.1:8010
```

```bash
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

Three question types: **choice** picks one option, **score** places the state
on an ordered scale, **noul** judges whether a statement is true, or which of
two descriptions fits. Ask up to 64 in one request; the state is read once.

There is a playground at `http://127.0.0.1:8010/`, a one-shot CLI (`uv run
ruling ask --help`), a bulk one for files of rows (`ruling predict`), a Python client (`from ruling import RulingClient`), a
zero-dependency TypeScript client in [`sdks/typescript`](../sdks/typescript), and
TypeSafe's own official SDK works against it unchanged — point
`TYPESAFE_BASE_URL` at the server. Details for all of them are in
[docs/reference.md](reference.md).

![The playground: a state, three questions, a model picker, and each answer's probabilities as bars](../assets/playground.png)

On a server started with `RULING_API_KEY`, enter the key in the playground;
it is kept for the browser tab only. The playground's model picker lists
every name the server answers to; [serve
several models](models.md#serve-several-models-from-one-process) to compare them on the
same state.

## Ask about images

The default Qwen3.5 models can see. Serve one as a `vision:` model and put
images anywhere in the state as `{"type": "image", "media_type", "data"}`
objects; questions refer to them as `<image 1>`, `<image 2>`, … in order:

```bash
uv sync --extra vision
RULING_MODEL=vision:mlx-community/Qwen3.5-4B-4bit uv run ruling serve
```

```bash
curl -s localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"note": "Customer says the parcel arrived crushed.",
            "photo": {"type": "image", "media_type": "image/jpeg", "data": "'"$(base64 < parcel.jpg)"'"}},
  "questions": {"damaged": {"type": "noul", "instructions": "<image 1> shows a damaged parcel."}}}'
```

In the playground, pick image files under the state box; they are added to
the state as `images` and shown with the `<image N>` name to ask about them
by. The picker is off for a model that reads text only. **image example**
loads a support note with a receipt drawn in the page and three questions
that need both; a server whose default model reads images opens on it.

The state and its images are read once, however many questions you ask:
eight 12-megapixel photos and nine questions took 10 s and 7.2 GB on the 4B
model. Up to 8 images of 5 MB each, PNG, JPEG or WebP. Any vision model behind
an OpenAI-compatible server reads them too, through `openai:`; see
[docs/reference.md](reference.md#api). To teach a `vision:` model your own
decisions about images, see [Fine-tune on images](fine-tuning.md#fine-tune-on-images).

## Limits

- Probabilities are model scores. Calibration makes them honest on average on
  data like your calibration set; it does not make any single answer right.
- A local MLX model answers one request at a time; questions inside a request
  are batched. Up to `RULING_MAX_PENDING` requests wait their turn, and the
  next is refused with 429 and a `Retry-After`.
- The state is untrusted input: text in it can steer the answers (prompt
  injection), as with any model that reads it. Put independent checks in front
  of security-sensitive actions rather than acting on an answer alone.
- Images need a `vision:` model or an `openai:` vision host; the plain MLX
  path and `vllm:` read text only. No audio, no generated explanation.
- The native engine is MLX, so the fast path is Apple Silicon. On NVIDIA,
  ruling serves through `vllm:` (vLLM's own decision endpoint) or `openai:`
  against a stock vLLM or llama.cpp; see
  [docs/reference.md](reference.md#on-nvidia-vllm).
