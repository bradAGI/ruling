"""Openly licensed labeled datasets, rewritten as typed-question records.

Only sources whose licenses permit training and redistribution of derived
weights are used: WANLI (CC-BY-4.0), SNLI (CC-BY-SA-4.0), CLINC150
(CC-BY-3.0), BoolQ (CC-BY-SA-3.0), GoEmotions (Apache-2.0), and Civil Comments
(CC0-1.0). Every download is pinned to a Hugging Face commit. Where a dataset
records how many raters chose each answer, the rater fraction becomes a soft
`target`, which is the only ground truth for calibration that does not come
from a model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ruling.evals import Record

NLI_OPTIONS = {
    "supported": "The evidence establishes the claim",
    "insufficient": "The evidence does not establish either",
    "contradicted": "The evidence establishes the opposite",
}
NLI_GOLD = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
SNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
NONE_OPTION = "none_of_these"
CLINC_OUT_OF_SCOPE = "oos"

SOURCES = {
    "wanli": ("alisawuffles/WANLI", "61c95318fd71c55b6ba355d76253254615f387ec", {"train": ["train.jsonl"]}),
    "snli": ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
             {"train": ["plain_text/train-00000-of-00001.parquet"], "dev": ["plain_text/validation-00000-of-00001.parquet"]}),
    "clinc": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec",
              {"train": ["plus/train-00000-of-00001.parquet"], "dev": ["plus/validation-00000-of-00001.parquet"]}),
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              {"train": ["data/train-00000-of-00001.parquet"], "dev": ["data/validation-00000-of-00001.parquet"]}),
    "emotions": ("google-research-datasets/go_emotions", "add492243ff905527e67aeb8b80c082af02207c3",
                 {"train": ["raw/train-00000-of-00001.parquet"]}),
    "civil": ("google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d",
              {"train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
               "dev": ["data/validation-00000-of-00001.parquet"]}),
}
"""Pinned source files per split. `dev` uses validation splits only, so test splits stay untouched and no
evaluation set in `datasets/` shares rows with it; WANLI's test split and GoEmotions have no suitable dev split."""


def nli_record(family: str, premise: str, hypothesis: str, gold: str) -> Record:
    verdict = NLI_GOLD[gold]
    return Record(
        family=family,
        state=premise,
        questions={
            "verdict": {"type": "choice", "instructions": "Assess the claim using only the supplied evidence: " + hypothesis,
                        "criteria": NLI_OPTIONS},
            "supported": {"type": "noul", "instructions": "The evidence establishes this claim: " + hypothesis},
        },
        labels={"verdict": verdict, "supported": verdict == "supported"},
    )


def clinc_record(text: str, gold: str | None, intents: list[str], rng: np.random.Generator) -> Record:
    """A routing question over a random subset of intents, with an explicit none option.

    A quarter of in-scope rows drop the gold intent from the offered options, so
    the right answer becomes `none_of_these`; that teaches abstention.
    """
    size = int(rng.integers(3, 13))
    pool = [i for i in intents if i != gold]
    offered = list(rng.choice(pool, size=size, replace=False))
    label = NONE_OPTION
    if gold is not None and rng.random() >= 0.25:
        offered[int(rng.integers(size))] = gold
        label = gold
    criteria = {str(i): str(i).replace("_", " ") for i in offered}
    criteria[NONE_OPTION] = "The request matches none of the other options"
    return Record(
        family="clinc",
        state=text,
        questions={"intent": {"type": "choice", "instructions": "Which request type does this user message belong to?",
                              "criteria": criteria}},
        labels={"intent": label},
    )


def boolq_record(passage: str, question: str, answer: bool) -> Record:
    return Record(
        family="boolq",
        state=passage,
        questions={"answer": {"type": "noul", "instructions": question[0].upper() + question[1:] + "?"}},
        labels={"answer": bool(answer)},
    )


def emotions_records(rows: list[dict], emotions: list[str], rng: np.random.Generator, questions_per_text: int) -> list[Record]:
    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_text[row["id"]].append(row)
    records = []
    for raters in by_text.values():
        fractions = {e: float(np.mean([r[e] for r in raters])) for e in emotions}
        present = [e for e in emotions if fractions[e] > 0]
        absent = [e for e in emotions if fractions[e] == 0]
        chosen = present[:questions_per_text]
        if len(chosen) < questions_per_text and absent:
            chosen += list(rng.choice(absent, size=min(questions_per_text - len(chosen), len(absent)), replace=False))
        questions = {e: {"type": "noul", "instructions": f"The text expresses {e.replace('_', ' ')}."} for e in chosen}
        records.append(Record(
            family="emotions",
            state=raters[0]["text"],
            questions=questions,
            labels={e: fractions[e] >= 0.5 for e in chosen},
            references={e: {"target": {"yes": fractions[e], "no": 1.0 - fractions[e]}} for e in chosen},
        ))
    return records


CIVIL_QUESTIONS = {
    "toxic": ("toxicity", "The comment is rude, disrespectful, or unreasonable."),
    "insult": ("insult", "The comment insults someone."),
    "threat": ("threat", "The comment threatens someone."),
}


def civil_record(row: dict) -> Record:
    return Record(
        family="civil",
        state=row["text"],
        questions={qid: {"type": "noul", "instructions": text} for qid, (_, text) in CIVIL_QUESTIONS.items()},
        labels={qid: float(row[column]) >= 0.5 for qid, (column, _) in CIVIL_QUESTIONS.items()},
        references={qid: {"target": {"yes": float(row[column]), "no": 1.0 - float(row[column])}}
                    for qid, (column, _) in CIVIL_QUESTIONS.items()},
    )


def _rows(name: str, split: str) -> list[dict]:
    repo, revision, splits = SOURCES[name]
    rows = []
    for file in splits[split]:
        path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
        if path.endswith(".jsonl"):
            rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
        else:
            rows.extend(pq.read_table(path).to_pylist())
    return rows


def build(per_source: int, seed: int, split: str = "train") -> list[Record]:
    """Up to `per_source` records from each source that has `split`, sampled with a fixed seed."""
    rng = np.random.default_rng(seed)
    available = {name for name, (_, _, splits) in SOURCES.items() if split in splits}

    def sample(rows):
        return [rows[i] for i in rng.permutation(len(rows))[:per_source]]

    records: list[Record] = []
    if "wanli" in available:
        records += [nli_record("wanli", r["premise"], r["hypothesis"], r["gold"]) for r in sample(_rows("wanli", split))]
    records += [nli_record("snli", r["premise"], r["hypothesis"], SNLI_LABELS[r["label"]])
                for r in sample([r for r in _rows("snli", split) if r["label"] in SNLI_LABELS])]

    names = _clinc_names()
    in_scope = [name for name in names.values() if name != CLINC_OUT_OF_SCOPE]
    records += [clinc_record(r["text"], None if names[r["intent"]] == CLINC_OUT_OF_SCOPE else names[r["intent"]], in_scope, rng)
                for r in sample(_rows("clinc", split))]

    records += [boolq_record(r["passage"], r["question"], r["answer"]) for r in sample(_rows("boolq", split))]

    if "emotions" in available:
        raw = _rows("emotions", split)
        emotions = [c for c in raw[0] if c not in {"text", "id", "author", "subreddit", "link_id", "parent_id",
                                                   "created_utc", "rater_id", "example_very_unclear", "neutral"}]
        wanted = set(sample(sorted({r["id"] for r in raw})))
        records += emotions_records([r for r in raw if r["id"] in wanted and not r["example_very_unclear"]], emotions, rng, 3)

    civil = _rows("civil", split)
    toxic = [r for r in civil if r["toxicity"] >= 0.2]
    calm = [r for r in civil if r["toxicity"] < 0.2]
    half = per_source // 2
    records += [civil_record(r) for r in [toxic[i] for i in rng.permutation(len(toxic))[:half]]
                + [calm[i] for i in rng.permutation(len(calm))[:per_source - half]]]
    return records


def _clinc_names() -> dict[int, str]:
    """Intent names for the `plus` configuration, read from the dataset card at the pinned revision."""
    repo, revision, _ = SOURCES["clinc"]
    lines = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().removeprefix("- ") == "config_name: plus")
    names: dict[int, str] = {}
    for line in lines[next(i for i in range(start, len(lines)) if lines[i].strip() == "names:") + 1:]:
        key, sep, value = line.strip().partition(":")
        if not (sep and key.strip("'\"").isdigit()):
            break
        names[int(key.strip("'\""))] = value.strip().strip("'\"")
    if CLINC_OUT_OF_SCOPE not in names.values():
        raise ValueError("CLINC150 card lacks the out-of-scope intent")
    return names
