# Datasets

Each file is JSONL. One record per line:

```json
{
  "family": "optional grouping used for balanced-accuracy reporting",
  "state": "string, object, or array",
  "questions": { "<id>": { "type": "choice|score|noul", ... } },
  "labels": { "<id>": "option key" | 2 | true }
}
```

Labels are an option key for Choice, a level index or level text for Score,
and a boolean for Noul. Labels are optional per question; accuracy metrics use
the labeled subset. An optional `references` map holds published answer
distributions per question and source name (for example Jev's), and `eval`
reports argmax agreement and total-variation distance against each. Extra
fields such as `variant` or `group` are kept for your own analysis and ignored.

```json
"references": {"<id>": {"jev": {"yes": 0.91, "no": 0.09}}}
```

## Bundled

| File | Records | Questions | Source |
|---|---:|---|---|
| `authored144.jsonl` | 144 | one Choice each, three families | Converted from the project-authored fixture in [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) (MIT, Copyright (c) 2026 TheoLeeCJ, notice in [THIRD_PARTY.md](../THIRD_PARTY.md)). Question text, options, and labels are unchanged. |
| `perturbations108.jsonl` | 108 | one Choice each | Same source. 36 base cases under three meaning-preserving edits: options reversed, the criterion wrapped in extra wording, and irrelevant context added. Used to measure robustness, not to pick defaults. |
| `support_tickets.jsonl` | 12 | Choice + Score + Noul per record | Written for this project. Small: it exists to exercise all three primitives, not to rank models. |
| `wanli256-selection.jsonl` | 256 | manifest only | The row selection TheoLeeCJ/openjev published for its WANLI evaluation, vendored so `ruling build-dataset wanli` rebuilds exactly those rows. |
| `typesafe102-selection.jsonl` | 102 | manifest only | TheoLeeCJ/openjev's alignment of TypeSafe's public evaluation cases, with a hash of each workflow payload. `ruling build-dataset typesafe` rebuilds exactly those rows and refuses if TypeSafe's payloads change. |

## Built on demand

These download a pinned revision from the Hugging Face hub and write ruling
records; nothing external is committed.

```bash
uv run ruling build-dataset wanli    --output datasets/wanli256.jsonl
uv run ruling build-dataset sst5     --output datasets/sst5.jsonl --limit 300
uv run ruling build-dataset typesafe --output datasets/typesafe102.jsonl
uv run ruling build-dataset every    --output datasets/every.jsonl
```

| Name | Records | Questions | Source and license |
|---|---:|---|---|
| `wanli` | 256 | a 3-way Choice (supported / insufficient / contradicted) and a Noul on entailment | [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) test split, CC-BY-4.0, pinned at `61c95318`. Held out from every default chosen in this repo. |
| `sst5` | up to 2,210, default 300 | a five-level Score from very negative to very positive | [SST-5](https://huggingface.co/datasets/SetFit/sst5) test split, pinned at `e51bdcd8`. Seeded sample. |
| `typesafe` | 102 | the original Choice or Noul from TypeSafe's workflow, with Jev's published answer and TypeSafe's reference distribution as `references` | [evals.typesafe.ai](https://evals.typesafe.ai/), four workflows, payloads pinned by hash. Labels are the reference argmax (mean of GPT-6 Astra and Fable 5.1), so they measure agreement with big models, not truth. |
| `every` | ~200 documents, ~760 judgments | each experiment's questions in TypeSafe's own format, with Jev's recorded answer per question as `references`; author labels for the judge grid and both retrieval grids (154 judgments) | [Every's parallel judgment lab](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/), `experiments.json` pinned by its generation timestamp and the source archive by SHA-256. The AI-writing checker is skipped because only excerpts are published. |

Fit calibration on data that looks like your production traffic, not on these.
