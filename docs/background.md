# Background: Jev, and the other open recreations

Moved here from the README so the front page leads with the result. Nothing below was cut.

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
in [THIRD_PARTY.md](../THIRD_PARTY.md).

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

