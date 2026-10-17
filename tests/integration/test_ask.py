"""A question in plain words through a real model: read, written, judged, checked, and routed as the cases say."""

import json
from pathlib import Path

import pytest

from ruling.ask import FACTS, LocalWriter, ask, plan, readings_of, writer_for
from ruling.cli import main
from ruling.questions import SystemOneRequest

from .conftest import TEST_MODEL

pytestmark = pytest.mark.integration
CASES = Path(__file__).resolve().parents[2] / "datasets" / "routing" / "ask25.jsonl"


def test_a_question_in_words_is_answered_with_its_work(single_engine):
    response = ask(single_engine, writer_for(single_engine), "is water wet?")
    assert set(response.facts) == set(FACTS) and response.usage.input_tokens > 0
    assert response.model == single_engine.model_id
    if not response.refused:
        assert response.answers, response.dropped
        for shown in response.answers:
            assert shown.said and shown.checked is not None


def test_the_served_model_writes_options_that_are_then_judged(single_engine):
    writer = LocalWriter(single_engine)
    reply = writer.write("what is the best text editor", ("choice",))
    assert reply.strip()
    # Whatever the small model wrote, only the tools the rules asked for are judged and shown.
    response = ask(single_engine, writer, "what is the best text editor")
    wanted = plan(response.facts).wanted
    for shown in response.answers:
        if shown.source == "writer":
            assert shown.question.type in wanted and shown.said.endswith(".")
    if wanted:
        assert response.wrote is not None


def test_the_routing_cases_are_read_without_error_and_the_tally_is_reported(single_engine, capsys):
    """Every case gets a plan; how many are read as the cases say is printed, not asserted, since a
    0.8B test model is not the model the wording was measured on. The numbers for the served models
    are in docs/measurements.md."""
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    right = 0
    for case in cases:
        answers = single_engine.evaluate(SystemOneRequest(state={"input": case["input"]}, questions=FACTS)).answers
        settled = plan(answers)
        wrong = []
        if (answers["answerable"].noul >= 0.5) != case["answerable"]:
            wrong.append("answerable")
        if (answers["fit"].noul >= 0.5) != case["fit"]:
            wrong.append("fit")
        if case["answerable"]:
            if case["several"] is not None and (answers["several"].noul >= 0.5) != case["several"]:
                wrong.append("several")
            if case["scale"] is not None and (answers["scale"].noul >= 0.5) != case["scale"]:
                wrong.append("scale")
            if not set(readings_of(answers)) <= set(case["reads"]):
                wrong.append(f"read as {readings_of(answers)}")
        assert settled.refused == (answers["answerable"].noul < 0.5)
        right += not wrong
        print(f"{'pass' if not wrong else 'FAIL'}  {case['input']}  {', '.join(wrong)}")
    print(f"routing cases: {right} of {len(cases)} read as the cases say on {TEST_MODEL}")
    assert len(cases) == 25


def test_the_command_line_takes_a_question_in_words(capsys):
    main(["ask", "can penguins fly", "--model", TEST_MODEL, "--rotations", "1"])
    body = json.loads(capsys.readouterr().out)
    assert set(body["facts"]) == set(FACTS) and body["input"] == "can penguins fly"
