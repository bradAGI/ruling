import mlx.core as mx
import pytest

from ruling.modernbert import load, sequence_inputs

pytestmark = pytest.mark.integration

ETTIN_150M = ("jhu-clsp/ettin-encoder-150m", "57617ddb6eee7cdeb86dc7b3f76b8a5ac9b8f7b9")


@pytest.fixture(scope="module")
def ettin():
    return load(*ETTIN_150M)


def test_masked_language_model_fills_an_obvious_blank(ettin):
    model, tokenizer, _ = ettin
    ids = tokenizer("The capital of France is [MASK].")["input_ids"]
    tokens, positions, mask = sequence_inputs([ids], tokenizer.pad_token_id)
    logits = model.token_logits(model.encoder(tokens, positions, mask))
    slot = ids.index(tokenizer.mask_token_id)
    assert tokenizer.decode([int(mx.argmax(logits[0, slot]))]).strip() == "Paris"


def test_padding_does_not_change_real_tokens(ettin):
    model, tokenizer, _ = ettin
    short = tokenizer("A short sentence.")["input_ids"]
    long = tokenizer("A much longer sentence that forces the short one in the batch to be padded quite a bit.")["input_ids"]
    alone = model.encoder(*sequence_inputs([short], tokenizer.pad_token_id))
    batched = model.encoder(*sequence_inputs([short, long], tokenizer.pad_token_id))
    a, b = alone[0], batched[0, : len(short)]
    # The GPU attention kernel drifts slightly in a residual stream that reaches ~2e4, so compare directions;
    # a token that attends to padding moves far more than this.
    cosine = mx.sum(a * b, axis=-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1))
    assert mx.min(cosine).item() > 0.9999


def test_matches_the_hugging_face_pytorch_reference(ettin):
    import numpy as np
    import torch
    from transformers import ModernBertForMaskedLM

    model, tokenizer, _ = ettin
    reference = ModernBertForMaskedLM.from_pretrained(ETTIN_150M[0], revision=ETTIN_150M[1], attn_implementation="eager").eval()
    texts = ["The capital of France is [MASK].",
             "Ticket: " + "the customer was charged twice and wants a refund. " * 40]  # long enough to exercise sliding windows
    for text in texts:
        ids = tokenizer(text)["input_ids"]
        with torch.no_grad():
            expected = reference(input_ids=torch.tensor([ids])).logits[0].numpy()
        mx.set_default_device(mx.cpu)
        try:
            actual = np.array(model.token_logits(model.encoder(*sequence_inputs([ids], tokenizer.pad_token_id)))[0])
        finally:
            mx.set_default_device(mx.gpu)
        assert (actual.argmax(-1) == expected.argmax(-1)).mean() > 0.99
        assert np.abs(actual - expected).max() / np.abs(expected).max() < 1e-3
