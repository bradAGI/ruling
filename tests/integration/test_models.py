"""Smoke tests across tokenizer and template families.

Opt in with RULING_SMOKE_MODELS=repo1,repo2. Each model is loaded once and must
pass the prompt-boundary invariant, expose single-token answer codes, and get an
unambiguous question right.
"""

import os

import pytest

from ruling.calibration import Calibration
from ruling.engine import Engine
from ruling.prompt import render_question
from ruling.questions import Choice, Noul, SystemOneRequest

pytestmark = pytest.mark.integration

MODELS = [m for m in os.environ.get("RULING_SMOKE_MODELS", "").split(",") if m]


@pytest.fixture(scope="module", params=MODELS or [pytest.param(None, marks=pytest.mark.skip(reason="RULING_SMOKE_MODELS unset"))])
def smoke_engine(request):
    return Engine(request.param, Calibration(), max_input_tokens=4096, rotations=2, prior_debias=True)


def test_template_split_is_token_exact(smoke_engine):
    engine = smoke_engine
    state = {"weather": "clear and blue sky", "temp_c": 21}
    prefix, tail = engine.split_prompt(state)
    question = render_question(Choice(instructions="q", criteria={"a": None, "b": None}), engine.codes.codes)
    from ruling.prompt import SYSTEM_PROMPT, render_state
    user = render_state(state) + question.text
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
                if engine._system_role else [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}])
    whole = engine.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert engine._encode(prefix) + engine._encode(question.text + tail) == list(whole)


def test_codes_and_an_obvious_decision(smoke_engine):
    assert smoke_engine.codes.max_options >= 26
    response = smoke_engine.evaluate(SystemOneRequest(
        state="The forecast says the sky is clear and blue all day.",
        questions={"color": Choice(instructions="What color is the sky?", criteria={"red": None, "blue": None, "green": None}),
                   "blue": Noul(instructions="The sky is described as blue.")},
    ))
    assert response.answers["color"].choice == "blue"
    assert response.answers["blue"].noul > 0.5
