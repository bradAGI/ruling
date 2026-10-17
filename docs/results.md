# Results

![Jev against Qwen3.6-35B-A3B on TypeSafe's 102 published rows, Every's 154 author-labeled judgments, and the two pooled](../assets/against-jev.png)

## How good is it

One Mac, the same four held-out sets for every model. Our models at the
default three option orderings; Clef-flash is Cloudflare's trained 9B
decision model, run locally through its MLX port and scored by the same
harness, and the three rows below it are other open decision models from the
Hub (Perplexity's pplx-decider through its MLX port, Mapika's decider through
its own package), each run through its publisher's own inference code and
scored the same way. The [full record](measurements.md) has every
setting and the commands for each cell.

| model | TypeSafe 102 | authored144 | perturbations108 | Every 154 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B, the default | 0.814 | 0.917 | 0.907 | 0.929 |
| Qwen3-4B-Instruct + our adapter | 0.843 | 0.889 | 0.861 | 0.916 |
| Qwen3.6-35B-A3B | 0.873 | 0.958 | 0.954 | 0.922 |
| Clef-flash 9B, 4-bit MLX port | **0.892** | 0.903 | 0.954 | 0.955 |
| pplx-decider-v1.1-27b, 4-bit MLX port | 0.899 on 89 rows | **0.965** | **1.000** | 0.955 |
| decider-4b v2.1 | 0.814 | 0.910 | 0.880 | 0.942 |
| decider-2b v11 | 0.755 | 0.868 | 0.917 | 0.935 |
| Jev, published | 0.882 | — | — | **0.961** |

pplx-decider rejects any prompt over its 8,192-token input limit, as its
publisher's code does, so on TypeSafe's rows it is scored on the 89 it can
read (13 are longer); on those 89 it gets 80 right, and Clef-flash's 0.892
on all 102 stays the best full-set number. Mapika's two models run through
their own package on Apple Silicon in float16, its default there.

On [JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public
decisions, run through the harness's own adapter: ruling with the 35B 0.857,
Jev 0.866, and the two identical on the 111 hard items. Clef-flash gets
0.792, and 0.604 on the hard items, where a 9B model's reasoning runs out;
pplx-decider 0.892, with 0.775 on the hard items, the best of any row;
decider-4b 0.831 and decider-2b 0.758. Full table in
[docs/measurements.md](measurements.md#jevbench-the-public-items).

Coverage at a 5% error budget, the number a threshold depends on:

| | TypeSafe 102 | Every 154 | authored144 | perturbations108 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.608 | 0.818 | 0.924 | 0.907 |
| Qwen3-4B-Instruct + our adapter | 0.392 | 0.948 | 0.882 | 0.815 |
| Qwen3.6-35B-A3B | 0.716 | 0.948 | **1.000** | **1.000** |
| Clef-flash 9B, 4-bit MLX port | **0.843** | **1.000** | 0.764 | **1.000** |
| pplx-decider-v1.1-27b, 4-bit MLX port | 0.854 on 89 rows | **1.000** | **1.000** | **1.000** |
| decider-4b v2.1 | 0.588 | 0.987 | 0.896 | 0.815 |
| decider-2b v11 | 0.294 | 0.935 | 0.715 | 0.787 |
| Jev, published | 0.784 | **1.000** | — | — |

Pooled over the 256 judgments that carry Jev's own answers, Clef-flash and Jev
are level on everything we measure: 238 correct each, coverage at 5% error
0.961 each, and seven questions each that only one of them gets right
(McNemar p = 1.00). Clef-flash does it with no order averaging, at 217 ms
median per request on Every's rows on a laptop.

**Which model to run.** For short judgments of the kind TypeSafe and Every
publish — routing, triage, yes/no checks on a few hundred tokens — Clef-flash
alone: it matches Jev and its confidence is the most trustworthy we have
measured. For long policy texts and multi-step reasoning, the 35B: it gets 81
of JevBench's 111 hard items to Clef-flash's 67. pplx-decider, at 15 GB, is
ahead of both on every set here except TypeSafe's, where it cannot read the
13 rows over 8,192 tokens; it gets 86 of the 111 hard items.

## A question in plain words

The ask tab reads a free-text question as typed questions before answering
it. On the 25 routing cases adapted from lmjtfy, the 35B reads 23 the way
the cases say and the default 4B reads 18 at three orderings, against 25 for
Jev; the per-fact tally and the misses are in
[Measurements](measurements.md#reading-a-question-in-plain-words).

## Images

On [Glance](https://github.com/yoheinakajima/glance)'s three sets of photos
taken after the models were released, `vision:mlx-community/Qwen3.5-4B-4bit`
is level with Glance's Qwen3-VL-4B on Commons photos and iNaturalist organism
groups, and behind on insect orders, where the 4B confuses flies with bugs and
bees. Pooled, 0.915 on 541 yes/no questions and 0.911 on 270 pick-one, to
Glance's 0.939 and 0.933; the best hosted models in its table reach 0.961 and
0.937. The 35B as a `vision:` model fixes most of the insect errors and
pools to 0.930 and 0.933, level with Glance on pick-one, at 0.6 s a decision
against the 4B's 0.4 s. A yes/no about a full-size photo takes 0.9 s on the
4B. Fine-tuning the 4B on 84 insect photos lifts a held-out half from 0.886
to 0.924, at the cost of calibration until a temperature is fitted. Every number,
with intervals and the paired comparison, is in
[Measurements](measurements.md#glances-fresh-photo-sets). A box question, the
one answer that is generated, lands on a colored square with a mean overlap of
0.91 across four image shapes on the 4B
([measured](measurements.md#locating-things-in-an-image)).

## Where Jev leads

The tie in the chart above is real, and so is this. On the same rows Jev is ahead on:

- **Every's 154 labeled judgments**, 148 to the 35B's 142 and Clef-flash's
  147. The cleanest accuracy comparison we have, and Jev still wins it, by one
  question over Clef-flash.
- **Confidence ordering, against stock models.** Pooled over all 256
  judgments, Jev's coverage at 5% error is 0.961 to the stock 35B's 0.863, and
  it is confidently wrong 0.4% of the time to the 35B's 3.5%. Our adapter
  narrows that without closing it; Clef-flash, trained for it, closes it.
- **Probabilities, not just answers.** Jev's distributions sit closer to
  TypeSafe's reference than ours even where the top answer agrees.
- **Knowledge-heavy inputs.** Other replications ([Kev](https://github.com/jaredpalmer/kev),
  [decider](https://github.com/Mapika/decider)) report Jev 10 to 18 points ahead
  on MMLU-style items; we have not run that suite and expect the same.

Everything on this page, with the settings and the command behind each cell,
is in [Measurements](measurements.md); why the design is what it is, in
[Design notes](design.md).
