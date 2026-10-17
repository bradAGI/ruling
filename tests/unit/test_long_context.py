"""A request that asks for long context gets the extended limits, and only that request."""

from ruling.engine import LONG_CONTEXT_TOKENS, Engine, extended_context, long_context


def bare_engine(max_input_tokens: int, max_branch_tokens: int, long_context_tokens: int) -> Engine:
    engine = Engine.__new__(Engine)
    engine.max_input_tokens, engine.max_branch_tokens = max_input_tokens, max_branch_tokens
    engine.long_context_tokens = long_context_tokens
    return engine


def test_limits_rise_to_the_long_context_size_inside_the_switch_and_fall_back_after():
    engine = bare_engine(65_536, 32_768, LONG_CONTEXT_TOKENS)
    assert engine.limits() == (65_536, 32_768)
    with extended_context():
        assert long_context.get() and engine.limits() == (LONG_CONTEXT_TOKENS, LONG_CONTEXT_TOKENS)
    assert not long_context.get() and engine.limits() == (65_536, 32_768)


def test_long_context_never_lowers_a_limit_already_above_it():
    engine = bare_engine(500_000, 400_000, 262_144)
    with extended_context():
        assert engine.limits() == (500_000, 400_000)
