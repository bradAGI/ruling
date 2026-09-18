"""Build labeled datasets from pinned public sources.

Nothing here is committed; the builders download a pinned revision and write
records in the ruling dataset format. WANLI rows follow the 256-row selection
published by TheoLeeCJ/openjev so results stay comparable. SST-5 gives an
ordinal five-level Score benchmark. The TypeSafe and Every builders carry
Jev's own published answers as references, so agreement with Jev can be
reported on identical inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

from ruling.evals import Distribution, answer_distribution

DATASETS = Path(__file__).resolve().parents[1] / "datasets"

WANLI = {"repo": "alisawuffles/WANLI", "revision": "61c95318fd71c55b6ba355d76253254615f387ec", "file": "test.jsonl"}
SST5 = {"repo": "SetFit/sst5", "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3", "file": "test.jsonl"}
WANLI_SELECTION = DATASETS / "wanli256-selection.jsonl"
TYPESAFE_SELECTION = DATASETS / "typesafe102-selection.jsonl"
TYPESAFE_CASES = "https://evals.typesafe.ai/{workflow}-cases.js"
EVERY_EXPERIMENTS = "https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json"
EVERY_ARCHIVE = "https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip"
EVERY_EXPERIMENTS_GENERATED_AT = "2026-08-28T19:55:23.191Z"
EVERY_ARCHIVE_SHA256 = "9fbf42e9d9e7cd3b072e0271a0959dbfdca85618e93fe4f0ca519d683c418bf0"

NLI_DESCRIPTIONS = {
    "supported": "The evidence establishes the claim",
    "insufficient": "The evidence does not establish either",
    "contradicted": "The evidence establishes the opposite",
}
NLI_LABELS = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _rows(source: dict) -> list[dict]:
    path = hf_hub_download(source["repo"], source["file"], repo_type="dataset", revision=source["revision"])
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fetch(url: str) -> bytes:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def build_wanli() -> list[dict]:
    """256 NLI rows as a 3-way Choice plus a Noul on entailment."""
    by_id = {str(row["id"]): row for row in _rows(WANLI)}
    selection = [json.loads(line) for line in WANLI_SELECTION.read_text().splitlines() if line.strip()]
    records = []
    for item in selection:
        row = by_id[str(item["upstream"]["source_id"])]
        gold = NLI_LABELS[row["gold"]]
        records.append({
            "family": item["family"],
            "source_id": row["id"],
            "state": row["premise"],
            "questions": {
                "verdict": {"type": "choice",
                            "instructions": "Assess the claim using only the supplied evidence: " + row["hypothesis"],
                            "criteria": {key: NLI_DESCRIPTIONS[key] for key in item["option_ids"]}},
                "supported": {"type": "noul",
                              "instructions": "The evidence establishes this claim: " + row["hypothesis"]},
            },
            "labels": {"verdict": gold, "supported": gold == "supported"},
        })
    if len(records) != 256:
        raise ValueError(f"expected 256 WANLI rows, built {len(records)}")
    return records


def build_sst5(limit: int, seed: int = 0) -> list[dict]:
    """A seeded sample of SST-5 test sentences as a five-level Score."""
    rows = _rows(SST5)
    sample = random.Random(seed).sample(rows, limit)
    return [{
        "family": "sentiment",
        "state": row["text"],
        "questions": {"sentiment": {"type": "score", "instructions": "What is the sentiment of the review sentence?",
                                    "criteria": SST5_LEVELS}},
        "labels": {"sentiment": int(row["label"])},
    } for row in sample]


def build_typesafe() -> list[dict]:
    """The 102 TypeSafe public-eval rows TheoLeeCJ aligned, with Jev's published answers.

    Labels are the argmax of TypeSafe's reference, the mean of GPT-6 Astra and
    Claude Fable 5.1 at high reasoning, so they measure agreement with big
    models, not ground truth. The selection manifest pins a hash of each
    workflow payload and the build refuses to run if a payload has changed.
    """
    selection = [json.loads(line) for line in TYPESAFE_SELECTION.read_text().splitlines() if line.strip()]
    payloads: dict[str, dict] = {}
    for workflow in sorted({item["upstream"]["workflow"] for item in selection}):
        text = _fetch(TYPESAFE_CASES.format(workflow=workflow)).decode()
        prefix = "__VIEWER_DATA__("
        if not text.startswith(prefix):
            raise ValueError(f"unexpected wrapper in TypeSafe {workflow} payload")
        payload, _ = json.JSONDecoder().raw_decode(text[len(prefix):])
        digest = hashlib.sha256(json.dumps(payload, indent=2).encode()).hexdigest()
        pinned = {item["upstream"]["snapshot_sha256"] for item in selection if item["upstream"]["workflow"] == workflow}
        if digest not in pinned:
            raise ValueError(f"TypeSafe {workflow} payload changed since the selection was frozen")
        payloads[workflow] = payload["eval"]

    records = []
    for item in selection:
        up = item["upstream"]
        evaluation = payloads[up["workflow"]]
        case = evaluation["cases"][up["case_id"]]
        source_question = evaluation["questions"][up["question_index"]]
        if source_question["type"] == "noul":
            criteria = source_question.get("criteria") or {}
            question = {"type": "noul", "instructions": source_question["instructions"],
                        "criteria": {"true": criteria.get("true"), "false": criteria.get("false")}}
            keys, source_keys = ["yes", "no"], ["true", "false"]
        elif source_question["type"] == "choice":
            question = {"type": "choice", "instructions": source_question["instructions"],
                        "criteria": source_question["criteria"]}
            keys = source_keys = list(question["criteria"])
        else:
            raise ValueError("the frozen selection excludes score questions")
        if source_keys != item["option_ids"]:
            raise ValueError(f"option order changed for {item['id']}")

        sets = case["reference_answers"][up["node"]][up["question_id"]]["sets"]
        reference = dict(zip(keys, _mean_reference(sets, source_keys)))
        top = max(reference, key=reference.get)
        if sum(1 for v in reference.values() if abs(v - reference[top]) < 1e-12) != 1:
            raise ValueError(f"reference argmax is not unique for {item['id']}")
        published = next(
            (node["answers"][up["question_id"]] for node in case["models"]["typesafe"]["nodes"]
             if node["node"] == up["node"] and node.get("doc") == up["document_index"]
             and node.get("questions", {}).get(up["question_id"]) == up["question_index"]
             and up["question_id"] in node.get("answers", {})),
            None,
        )
        if published is None:
            raise ValueError(f"no published Jev answer for {item['id']}")
        records.append({
            "family": item["family"],
            "source_id": item["id"],
            "state": evaluation["documents"][up["document_index"]],
            "questions": {up["question_id"]: question},
            "labels": {up["question_id"]: (top == "yes") if question["type"] == "noul" else top},
            "references": {up["question_id"]: {"jev": answer_distribution(published, keys), "reference": reference}},
        })
    if len(records) != 102:
        raise ValueError(f"expected 102 TypeSafe rows, built {len(records)}")
    return records


def _mean_reference(sets: list[dict], keys: list[str]) -> list[float]:
    total = [0.0] * len(keys)
    for answer in sets:
        if answer.get("probabilities"):
            raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
            norm = sum(raw.get(k, 0.0) for k in keys)
            for i, key in enumerate(keys):
                total[i] += raw.get(key, 0.0) / norm
        else:
            value = answer["value"]
            key = str(value).lower() if isinstance(value, bool) else str(value)
            total[keys.index(key)] += 1.0
    return [value / len(sets) for value in total]


def build_every() -> list[dict]:
    """Every's parallel judgment lab: Jev's recorded answers on identical inputs.

    One record per document with all of that experiment's questions. Gold
    labels exist for the judge grid and the two retrieval grids (author
    labels); the other experiments carry only Jev's answers, so they measure
    agreement. The AI-writing checker is skipped because only excerpts of its
    documents are published.
    """
    data = json.loads(_fetch(EVERY_EXPERIMENTS))
    if data["generated_at"] != EVERY_EXPERIMENTS_GENERATED_AT:
        raise ValueError(f"Every experiments file changed: generated_at is {data['generated_at']}")
    archive_bytes = _fetch(EVERY_ARCHIVE)
    if hashlib.sha256(archive_bytes).hexdigest() != EVERY_ARCHIVE_SHA256:
        raise ValueError("Every source archive changed")
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    script = archive.read("scripts/run-experiments.mjs").decode()
    policy = json.loads(re.search(r'const SUPPORT_POLICY = ("[^\n]+");', script).group(1))
    experiments = {e["id"]: e for e in data["experiments"]}
    records: list[dict] = []

    grid = experiments["judge-grid"]
    cells = {(c["output"], c["question"]): c for c in grid["typesafe"]["cells"]}
    for document in grid["documents"]:
        questions = {qid: {"type": "noul", "instructions": text} for qid, text in grid["questions"].items()}
        records.append({
            "family": "every_judge-grid",
            "state": {"policy": policy, "candidate_response": document["response"]},
            "questions": questions,
            "labels": {qid: bool(document["expected"][qid]) for qid in questions},
            "references": {qid: {"jev": _noul(cells[document["id"], qid]["value"])} for qid in questions},
        })

    for name, folder in (("code-rag", "code-repo"), ("company-brain", "company-brain")):
        by_document: dict[str, dict] = {}
        for query in experiments[name]["grid"]:
            for cell in query["cells"]:
                entry = by_document.setdefault(cell["document"], {"questions": {}, "labels": {}, "references": {}})
                entry["questions"][query["id"]] = {"type": "noul", "instructions": query["question"]}
                entry["labels"][query["id"]] = bool(cell["relevant"])
                entry["references"][query["id"]] = {"jev": _noul(cell["value"])}
        for document, entry in by_document.items():
            records.append({"family": f"every_{name}", "state": archive.read(f"fixtures/{folder}/{document}").decode(), **entry})

    for name in ("action-firewall", "vc-diligence"):
        experiment = experiments[name]
        questions = {qid: {"type": "noul", "instructions": text} for qid, text in experiment["questions"].items()}
        for row in experiment["rows"]:
            records.append({"family": f"every_{name}", "state": row["text"], "questions": questions,
                            "references": {qid: {"jev": _noul(row["signals"][qid])} for qid in questions}})

    for name in ("customer-voice", "ceo-radar", "inbox-urgency", "persona-panel"):
        experiment = experiments[name]
        questions = {qid: ({"type": "noul", "instructions": spec} if isinstance(spec, str) else spec)
                     for qid, spec in experiment["questions"].items()}
        for row in experiment["rows"]:
            references = {qid: {"jev": answer_distribution(answer, _keys(questions[qid]))}
                          for qid, answer in row["typed_answers"].items()}
            records.append({"family": f"every_{name}", "state": row["document"], "questions": questions,
                            "references": references})
    return records


def _noul(probability: float) -> Distribution:
    return {"yes": float(probability), "no": 1.0 - float(probability)}


def _keys(question: dict) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["yes", "no"]


def write(records: list[dict], output: Path) -> None:
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
