# Design notes

ruling reads option logits from a stock instruction-tuned decoder. This page
says why that design was kept over the alternatives that were tried, what the
evidence about Jev's shape is, what training has taught us, and what is not
reproduced. The mechanism itself is in [How it works](how-it-works.md).

## What TypeSafe has said

Two statements in TypeSafe's site source, neither prominent on the rendered
pages, constrain the design more than anything in the launch post.

> Jev is neither small nor an LLM, hence being off the intelligence Pareto curve.

> TypeSafe is primarily a data research lab, which is how the biggest results
> in AI get made. We make all the data ourselves.

Their "AI's bitterest lesson" post ranks "doing the right task > data >
compute > algorithms" and compares their data to self-play. Read together:
Jev is a large model trained for structured decisions from the start, on data
TypeSafe generated. A Mac cannot repeat the pretraining that gives a model its
language understanding, so ruling starts from a pretrained open model and
copies what can be copied: the decision objective and self-made data.

## Black-box evidence about Jev's shape

Archer Hume's [architecture
analysis](https://archerhume.com/posts/jevs-architecture-unmasked/) (17
September 2026) probes the hosted API with about 10,000 calls and reports
latency scaling, question-reordering and option-set experiments. The author is
explicit that the conclusions are inferred, not confirmed by TypeSafe, and that
the sparse backbone is the least certain part. What the evidence supports:

- **A causal transformer, post-trained.** "Jev is neither small nor an LLM"
  reads, on this account, as a claim about the product rather than the
  backbone: inference ends after prefill and a readout, so nothing is
  generated token by token, but the machinery underneath is a decoder.
  TypeSafe's own primer presents RLCD as a post-training path from pretrained
  language models.
- **Shared state, isolated questions.** The state is encoded once; each
  question builds its own branch from the state cache plus its own
  instructions and options, and branches cannot attend to each other.
- **Direct probability readout.** Distributions come from the representation
  at a decision position, not from generated text.
- **Options interact listwise.** Adding an irrelevant fifth option moved the
  log-odds between two existing options from +0.38 to +0.11, in all ten
  randomized blocks. Fixed per-option logits under one softmax could not do
  that, because the shared denominator cancels. Jev reads a question's options
  as an ordered list before scoring them.
- **Confidence is computed from the distribution**, not predicted separately;
  training targets the distribution with a proper scoring rule.

This is the architecture ruling's engine has: shared prefix cache, one suffix
per question holding its own option list, restricted-token readout at the
answer position, no decoding. Two design choices follow from it:

- **The whole option list stays in one suffix.** A per-option scoring head,
  one suffix per option with a reserved decision token, was sketched as the
  native design. It would remove positional bias by construction and lift the
  62-option ceiling, but scoring each option alone makes "none of the above"
  and set-dependent wording invisible to the model, which is exactly the
  interaction Jev shows. It was not built.
- **Option-order averaging stays.** Listwise reading is what makes a model
  order-sensitive; averaging over orderings costs rotations of compute and
  removes a bias Jev itself has.

## What training taught us

Every item below is a measurement, not a guess; the numbers are in
[Measurements](measurements.md) and the [adapter card](adapter.md).

- **Hybrid linear attention does not train on this hardware.** Qwen3.5 and
  the Qwen3.6 35B mixture-of-experts model switch mlx-lm's linear-attention
  layers to a step-by-step recurrence in training mode. One step on the 0.8B
  model took 27 to 51 seconds with LoRA on every layer. Pure-attention
  Qwen3-4B-Instruct-2507 takes about 2 seconds per 1,024 tokens. Serve Qwen3.5;
  train Qwen3, unless the records hold images, which only the Qwen3.5 vision
  models read.
- **Learning rate decides everything.** On held-out access-policy states,
  120 steps at 1e-5 took the untrained 4B model from 0.66 to 0.91 accuracy
  and its calibration error from 0.33 to 0.05. At 1e-4 accuracy fell to 0.32
  while loss still dropped: the model learned to hedge.
- **Models under about 1B do not learn these tasks.** Qwen3-0.6B got worse at
  every learning rate from 5e-5 to 3e-4.
- **Arbitrary associations are not a useful probe.** A code-word-to-bin toy
  task never left uniform answers once options were shuffled, under both the
  letter readout and yes/no candidate scoring, yet trained perfectly without
  shuffling. Real judgment reuses features the base model already has, so
  learnability is tested on realistic tasks instead.
- **Training moves calibration, not accuracy.** The adapter trained on 27,021
  constructed records changed held-out accuracy by a question or two and
  roughly halved calibration error on every set, with confident errors down
  three- to thirteen-fold. Its accuracy gain concentrates at one ordering and
  vanishes by three: training and ordering averaging correct the same
  positional bias and do not stack.
- **Training sharpens faster than it corrects.** The photo adapter was right
  more often than its base and twice too sharp, so its raw log loss rose. A
  temperature fit on photos it had not seen brought the log loss below the
  base's. This is why `ruling retrain` judges each version at its own fitted
  temperature.
- **Temperature does not transfer across distributions.** A temperature fit
  on 1,000 dev records (Choice 1.63, Noul 3.57) cut in-sample calibration
  error from 0.062 to 0.016 and 0.177 to 0.035, then made TypeSafe's rows
  worse, 0.133 to 0.188 and 0.100 to 0.156. The sign of the miscalibration
  flips between the two sets. Calibration is fitted per deployment, which is
  what the provenance check enforces, and no calibration file ships with
  ruling.
- **Sharper argmax, looser probabilities.** Going from one ordering to six
  raised Choice accuracy on TypeSafe's rows from 0.889 to 0.944 while
  expected calibration error rose from 0.088 to 0.133. Averaging orderings
  and calibrating the result are separate jobs.
- **Ordering averaging saturates at the option count.** Orderings are capped
  at the number of options, so six orderings equal three on binary questions
  and on three-option Choice. The gain on TypeSafe's rows comes from their
  five-option questions. Rotations are cheap: the state cache is shared, so
  only the question suffix is recomputed.
- **The untrained 35B already reaches Jev.** With six option orderings it
  agrees with TypeSafe's reference on 91 of their 102 questions; Jev agrees on
  90. Post-training is not about closing a gap on that set; it is about
  holding parity on rows where the base model is weak, and about the
  probabilities, where Jev sits closer to the reference than we do.

## Training data

`ruling build-corpus public|states|worlds` builds the training records, from
`ruling/corpus.py`, `ruling/states.py` and `ruling/worlds.py`:

1. **Public labeled datasets reframed as typed questions.** NLI (MNLI, ANLI,
   WANLI train split), intent (Banking77, CLINC150), sentiment and ordinal
   (SST-5 train, Amazon stars), topic (AG News), fact verification (FEVER),
   toxicity, code relevance (CodeSearchNet). Real labels give calibration
   ground truth. Nothing overlaps the evaluation sets in `datasets/`.
2. **Synthetic structured state with computed answers.** JSON states (orders,
   alerts, invoices, game state) plus rules evaluated in code. Unlimited and
   exact; this is what teaches reading program state.
3. **Synthetic worlds judged by a local model.** Diverse states and questions
   written and judged by an open-weight model through the same typed-question
   harness ruling exposes, giving soft distributions for the fuzzy middle.
   Teachers stay open-weight because OpenAI's and Anthropic's terms restrict
   training competing models on their outputs.

The loss is log loss on the answer letter, a proper scoring rule, so the
expected minimum is the true probability. Three optional terms, label
smoothing, a Brier term and a ranking term, are available and off; what they
bought is in [Measurements](measurements.md#calibration-terms-in-the-training-objective).

## First decision encoder

`ruling/decision.py` and `ruling train-encoder` answer every question in one
bidirectional pass over a ModernBERT-architecture encoder, with answers
independent of other questions and of option order by construction. Ettin-400M
(MIT) was fully fine-tuned for 2,400 steps on 15,783 records (9,000 public,
4,000 program states, 3,108 synthetic worlds); the checkpoint with the lowest
validation loss, step 1,200, was kept.

| Held-out set | Encoder, trained | 4B decoder, untrained | 35B, untrained | Jev |
|---|---:|---:|---:|---:|
| Validation split (same distribution as training), accuracy | 0.909 | | | |
| authored144, mean family balanced accuracy | 0.663 | 0.919 | 0.964 | |
| perturbations108, mean family balanced accuracy | 0.409 | 0.909 | | |
| WANLI 3-way, balanced accuracy | 0.650 | 0.672 | | |
| SST-5, accuracy | 0.427 | 0.487 | | |
| Every, 154 author labels, accuracy | 0.786 | 0.929 | 0.922 | 0.961 |
| TypeSafe, agreement with reference | 0.61 on 83 of 102 | 0.81 | 0.853 | 0.882 |
| Latency, 3 questions / 21 questions | 8 ms / 16 ms | 115 ms / 374 ms | | 70 to 500 ms |

The encoder is 14 to 23 times faster than the 4B decoder but does not
generalize: calibration error is 0.14 to 0.38 outside the training
distribution, wording changes cut accuracy to 0.41, and 19 TypeSafe rows
exceed its 7,999-position window. Sixteen thousand records from a handful of
sources is too narrow for a model that never saw instructions in
pretraining.

**The encoder path is parked.** It was a bet that a bidirectional model could
buy Jev's latency without Jev's backbone; the measurements above and the
black-box evidence both say no. The latency it wins is not the constraint
(the 4B decoder already answers 3 questions in 115 ms, inside Jev's published
70 to 500 ms), and the generalization it loses is the whole task. The code
and the command stay for anyone who wants to push it with a much wider corpus.

## Not reproduced

- **RLCD.** TypeSafe's reinforcement-learning stage is undisclosed. The
  closest reconstruction, a policy-gradient pass with reward equal to negative
  Brier on held-out groups, has the same expected gradient as training on a
  proper scoring rule, which is what `ruling train` does; the pass itself has
  not been run.
- **A native scoring head.** See the listwise finding above; a trained head
  is also what one-pass Choice above 62 options would need, as the
  [255-code experiment](measurements.md#one-pass-choice-above-62-options) shows.
- **Frontier-level judgment.** A 4B model has a judgment ceiling; Jev sits
  near a mid-tier frontier model on TypeSafe's own dashboard, and the 35B is
  what reaches it here.
