import math

import mlx.core as mx
import numpy as np
import pytest

from ruling.calibration import softmax
from ruling.engine import STAGE_TWO_CANDIDATES, InputTooLong
from ruling.prompt import SYSTEM_PROMPT, prior_question, render_question, render_state
from ruling.questions import Choice, Noul, Score, SystemOneRequest

pytestmark = pytest.mark.integration

SKY = "The forecast says the sky is clear and blue all day with no clouds."
COLOR = Choice(instructions="What color is the sky in the state?", criteria={"red": None, "blue": None, "green": None})


@pytest.mark.parametrize("state", [SKY, {"weather": SKY, "temp_c": 21}, ["a", {"b": 2}]])
def test_prefix_plus_suffix_tokens_equal_whole_template(engine, state):
    prefix, tail = engine.split_prompt(state)
    rendered = render_question(COLOR, engine.codes.codes)
    whole = engine.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": render_state(state) + rendered.text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert engine._encode(prefix) + engine._encode(rendered.text + tail) == list(whole)


def test_cached_prefix_matches_uncached_forward(single_engine, monkeypatch):
    engine = single_engine
    monkeypatch.setattr(engine, "prior_debias", False)
    scored = engine.score(SKY, {"color": COLOR})
    prefix, tail = engine.split_prompt(SKY)
    tokens = engine._encode(prefix) + engine._encode(render_question(COLOR, engine.codes.codes).text + tail)
    out = engine.model(mx.array(tokens)[None])
    ids = engine.codes.token_ids[:3]
    reference = softmax(np.array(out[0, -1, ids].astype(mx.float32)))
    assert np.allclose(softmax(scored.raw["color"].logits), reference, atol=0.03)


def test_batched_scoring_matches_one_question_at_a_time(engine):
    questions = {
        "color": COLOR,
        "clouds": Score(instructions="How cloudy is it?", criteria=["clear", "partly cloudy", "overcast"]),
        "blue": Noul(instructions="The sky is described as blue."),
        "rain": Noul(instructions="Rain is forecast."),
        "fruit": Choice(instructions="Which is a fruit?", criteria={"hammer": None, "apple": None, "granite": None}),
    }
    together = engine.score(SKY, questions).raw
    for qid, question in questions.items():
        alone = engine.score(SKY, {qid: question}).raw[qid]
        assert np.allclose(softmax(together[qid].logits), softmax(alone.logits), atol=0.05), qid


def test_content_free_prompt_is_uniform_after_debiasing(debiased_engine):
    engine = debiased_engine
    for n in (2, 3, 5):
        raw = engine.score("No information is available.", {"q": prior_question(n)}).raw["q"]
        assert np.allclose(softmax(raw.logits), np.full(n, 1 / n), atol=0.05)


def test_answers_are_typed_and_normalized(engine):
    request = SystemOneRequest(
        state=SKY,
        questions={
            "color": COLOR,
            "clouds": Score(instructions="How cloudy is it?", criteria=["clear", "partly cloudy", "overcast"]),
            "blue": Noul(instructions="The sky is described as blue."),
        },
    )
    response = engine.evaluate(request)
    color, clouds, blue = response.answers["color"], response.answers["clouds"], response.answers["blue"]
    assert color.choice == "blue"
    assert math.isclose(sum(color.probabilities.values()), 1.0, abs_tol=1e-6)
    assert 0.0 <= color.confidence <= 1.0
    assert 0.0 <= clouds.score <= 2.0 and clouds.legend["0"] == "clear"
    assert clouds.score < 1.0
    assert blue.noul > 0.5
    assert response.usage.output_tokens == 0 and response.usage.input_tokens > 0


def test_full_rotation_is_invariant_to_cyclic_option_order(rotating_engine):
    question = Choice(instructions="Which is a fruit?", criteria={"hammer": None, "apple": None, "granite": None, "spoon": None})
    shifted = Choice(instructions=question.instructions, criteria={k: None for k in ["apple", "granite", "spoon", "hammer"]})
    a = rotating_engine.score("Pick the fruit.", {"q": question}).raw["q"]
    b = rotating_engine.score("Pick the fruit.", {"q": shifted}).raw["q"]
    pa = dict(zip(a.keys, softmax(a.logits)))
    pb = dict(zip(b.keys, softmax(b.logits)))
    for key in pa:
        assert pa[key] == pytest.approx(pb[key], abs=0.02)
    assert max(pa, key=pa.get) == "apple"


def test_orderings_per_question_type(single_engine, rotating_engine):
    engine = single_engine
    score_q = {"s": Score(instructions="How cloudy?", criteria=["clear", "cloudy", "overcast"])}
    noul_q = {"n": Noul(instructions="It is sunny.")}
    prefix = len(engine._encode(engine.split_prompt(SKY)[0]))
    one_score = engine.score(SKY, score_q).input_tokens - prefix
    two_score = rotating_engine.score(SKY, score_q).input_tokens - prefix
    assert two_score == 2 * one_score
    one_noul = engine.score(SKY, noul_q).input_tokens - prefix
    assert rotating_engine.score(SKY, noul_q).input_tokens - prefix == 2 * one_noul


def test_large_choice_runs_in_two_stages(engine):
    keys = [f"tool{i}" for i in range(engine.codes.max_options + 10)]
    keys.insert(37, "apple")
    question = Choice(instructions="Which one is a fruit?", criteria={k: None for k in keys})
    scored = engine.score("Pick the fruit.", {"q": question})
    raw = scored.raw["q"]
    assert raw.keys == keys
    probs = softmax(raw.logits)
    assert raw.keys[int(probs.argmax())] == "apple"
    assert int((probs > 0).sum()) <= STAGE_TWO_CANDIDATES
    answer = engine.answer(question, raw)
    assert answer.choice == "apple" and set(answer.probabilities) == set(keys)


def test_input_too_long_is_rejected(engine):
    with pytest.raises(InputTooLong):
        engine.score("word " * 9000, {"q": Noul(instructions="q")})


def test_each_question_branch_has_its_own_token_limit(single_engine, monkeypatch):
    """Jev budgets one state-plus-question branch apart from the whole request; ruling does the same."""
    engine = single_engine
    short = Noul(instructions="The sky is blue.")
    branch = engine.score(SKY, {"q": short}).input_tokens

    monkeypatch.setattr(engine, "max_branch_tokens", branch)
    monkeypatch.setattr(engine, "max_input_tokens", 10 * branch)
    engine.score(SKY, {"q": short})
    with pytest.raises(InputTooLong, match="branch"):
        engine.score(SKY, {"q": Noul(instructions="The sky is blue. " + "Weigh every nuance. " * 200)})

    monkeypatch.setattr(engine, "max_branch_tokens", 10 * branch)
    monkeypatch.setattr(engine, "max_input_tokens", branch)
    with pytest.raises(InputTooLong, match="request"):
        engine.score(SKY, {"q1": short, "q2": short})
