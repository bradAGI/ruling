import numpy as np
import pytest

from ruling.decision import InputTooLong, Special, attention_mask, pack
from ruling.questions import Choice, Noul, Score

SPECIAL = Special(cls=1, sep=2, mask=3, pad=0)


def encode(text: str) -> list[int]:
    return [10 + (ord(c) % 90) for c in text]


def team():
    return Choice(instructions="Which team?", criteria={"billing": "Charges", "technical": "Bugs", "sales": None})


QUESTIONS = {
    "team": team(),
    "urgency": Score(instructions="How urgent?", criteria=["later", "soon", "now"]),
    "refund": Noul(instructions="Asks for a refund."),
}


def view(packed, qid):
    """Every token of one question, with its position and the tokens it can see."""
    mask = attention_mask([packed.segments])[0]
    rows = [i for i, (q, _) in enumerate(packed.segments) if q == packed.question_index[qid]]
    return [(packed.tokens[i], packed.positions[i], tuple(packed.tokens[j] for j in np.flatnonzero(mask[i]))) for i in rows]


def test_every_question_sees_the_same_thing_whatever_else_is_asked():
    alone = pack("my card was charged twice", {"refund": QUESTIONS["refund"]}, encode, SPECIAL, 4096)
    together = pack("my card was charged twice", QUESTIONS, encode, SPECIAL, 4096)
    reordered = pack("my card was charged twice", dict(reversed(list(QUESTIONS.items()))), encode, SPECIAL, 4096)
    assert view(alone, "refund") == view(together, "refund") == view(reordered, "refund")


def test_options_share_start_positions_and_cannot_see_each_other():
    packed = pack("state", {"team": team()}, encode, SPECIAL, 4096)
    shuffled = pack("state", {"team": Choice(instructions="Which team?", criteria={"sales": None, "technical": "Bugs", "billing": "Charges"})},
                    encode, SPECIAL, 4096)
    starts = {key: packed.positions[index] for key, index in packed.slots["team"]}
    assert len(set(starts.values())) == 1
    mask = attention_mask([packed.segments])[0]
    slots = dict(packed.slots["team"])
    assert not mask[slots["billing"], slots["technical"]] and mask[slots["billing"], slots["billing"]]
    by_key = lambda p: {k: (p.tokens[i], p.positions[i]) for k, i in p.slots["team"]}
    assert by_key(packed) == by_key(shuffled)


def test_state_never_sees_questions_and_question_text_never_sees_options():
    packed = pack("state", QUESTIONS, encode, SPECIAL, 4096)
    mask = attention_mask([packed.segments])[0]
    state = [i for i, s in enumerate(packed.segments) if s == (0, 0)]
    others = [i for i, s in enumerate(packed.segments) if s != (0, 0)]
    assert not mask[np.ix_(state, others)].any() and mask[np.ix_(others, state)].all()
    text = [i for i, s in enumerate(packed.segments) if s[0] == 1 and s[1] == 0]
    options = [i for i, s in enumerate(packed.segments) if s[0] == 1 and s[1] > 0]
    assert not mask[np.ix_(text, options)].any()


def test_slots_follow_canonical_keys_and_start_with_the_mask_token():
    packed = pack("state", QUESTIONS, encode, SPECIAL, 4096)
    assert [k for k, _ in packed.slots["team"]] == ["billing", "technical", "sales"]
    assert [k for k, _ in packed.slots["urgency"]] == ["0", "1", "2"]
    assert [k for k, _ in packed.slots["refund"]] == ["yes", "no"]
    assert all(packed.tokens[i] == SPECIAL.mask for slots in packed.slots.values() for _, i in slots)


def test_padding_rows_and_columns_are_masked_out():
    short = pack("a", {"refund": QUESTIONS["refund"]}, encode, SPECIAL, 4096)
    long = pack("a much longer state", QUESTIONS, encode, SPECIAL, 4096)
    mask = attention_mask([short.segments, long.segments])
    assert mask.shape == (2, len(long.tokens), len(long.tokens))
    assert not mask[0, :, len(short.tokens):].any()


def test_too_long_input_is_refused():
    with pytest.raises(InputTooLong):
        pack("x" * 5000, QUESTIONS, encode, SPECIAL, 4096)
