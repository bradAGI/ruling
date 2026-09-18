"""Send a dataset to the hosted TypeSafe API so Jev's live answers can sit next to ruling's.

TypeSafe's HTTP request and response shapes are the same as ruling's, so this
is a thin client: bearer token, one POST per record, answers converted to
distributions over the record's canonical keys. Nothing but the record's state
and questions is sent.
"""

from __future__ import annotations

import os

import httpx

from ruling.evals import Distribution, Record, answer_distribution
from ruling.questions import option_keys

API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
DEFAULT_BASE_URL = "https://api.typesafe.ai"


def live_answers(
    records: list[Record],
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
):
    """Yield (record index, answer distributions by question id) from the hosted API, one request per record."""
    base_url = base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
    api_key = api_key or os.environ.get(API_KEY_ENV)
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        for index, record in enumerate(records):
            body = {"state": record.state,
                    "questions": {qid: q.model_dump(exclude_none=True) for qid, q in record.questions.items()}}
            if model is not None:
                body["model"] = model
            response = client.post("/v1/systemone", json=body)
            response.raise_for_status()
            yield index, {qid: answer_distribution(answer, option_keys(record.questions[qid]))
                          for qid, answer in response.json()["answers"].items()}


def collect_live(records: list[Record], base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 30.0) -> dict[tuple[int, str], Distribution]:
    return {(index, qid): distribution
            for index, answers in live_answers(records, base_url, api_key, model, timeout)
            for qid, distribution in answers.items()}
