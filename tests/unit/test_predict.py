"""Bulk prediction keeps one output line per input row, whatever happens to the row."""

import numpy as np

from ruling.calibration import Calibration
from ruling.engine import InputTooLong, RawScore, answer
from ruling.evals import Record, predict
from ruling.questions import SystemOneResponse, Usage


class TooLongPast:
    """Answers every state no longer than `limit` characters, as an engine with a context limit does."""
    max_options = 3
    reads_images = False
    locates = False

    def __init__(self, limit: int):
        self.limit = limit
        self.calibration = Calibration()

    def evaluate(self, request):
        if len(request.state) > self.limit:
            raise InputTooLong("longer than the maximum context length")
        raw = RawScore(keys=["yes", "no"], logits=np.array([1.0, 0.0]))
        return SystemOneResponse(model="m", answers={qid: answer(q, raw, self.calibration)
                                                     for qid, q in request.questions.items()},
                                 usage=Usage(input_tokens=1))


def _record(state: str, **question) -> Record:
    return Record(id=state, state=state, questions={"q": {"instructions": "Is it?", **question}})


def test_every_row_gets_one_line_in_its_own_place():
    records = [_record("short", type="noul"), _record("far too long", type="noul"),
               _record("wide", type="choice", criteria={k: None for k in "abcd"}), _record("fine", type="noul")]
    lines = list(predict(TooLongPast(limit=5), records))
    assert [line["id"] for line in lines] == [r.id for r in records]
    assert [line["error"]["code"] if "error" in line else line["answers"]["q"]["type"] for line in lines] == \
        ["noul", "context_too_long", "invalid_options", "noul"]


def test_a_row_with_images_for_a_text_model_keeps_its_place():
    photo = {"type": "image", "media_type": "image/png", "data": "iVBORw0KGgo="}
    records = [Record(state={"photo": photo}, questions={"q": {"type": "noul", "instructions": "Red?"}}),
               _record("fine", type="noul")]
    lines = list(predict(TooLongPast(limit=5), records))
    assert lines[0]["error"]["code"] == "unsupported_modality" and lines[1]["answers"]["q"]["type"] == "noul"
