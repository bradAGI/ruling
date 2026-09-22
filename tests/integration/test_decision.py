import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ruling import states
from ruling.build import write
from ruling.calibration import Calibration
from ruling.config import Settings
from ruling.decision import DecisionEngine, batch_inputs, load_model, pack
from ruling.decision_train import DecisionTrainConfig, DecisionTrainer
from ruling.questions import Choice, Noul, Score, SystemOneRequest
from ruling.server import build_engine, create_app

# Trains a model in a fixture: two minutes with a GPU, hours without one.
pytestmark = [pytest.mark.integration, pytest.mark.heavy]

BASE = ("jhu-clsp/ettin-encoder-150m", "57617ddb6eee7cdeb86dc7b3f76b8a5ac9b8f7b9")
QUESTIONS = {
    "team": Choice(instructions="Which team?", criteria={"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}),
    "urgency": Score(instructions="How urgent?", criteria=["can wait", "this week", "today"]),
    "refund": Noul(instructions="The customer asks for a refund."),
}


def test_answers_do_not_depend_on_other_questions_or_option_order():
    model, tokenizer, special = load_model(*BASE, None)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    state = "I was charged twice for order 4471 and want my money back today."

    def slot_vectors(questions, qid):
        """What the scoring head sees for each option, after the pretrained transform."""
        packed = pack(state, questions, encode, special, 4096)
        tokens, positions, mask = batch_inputs([packed], special.pad)
        hidden = model.lm.transform(model.lm.encoder(tokens, positions, mask))[0]
        return {key: np.array(hidden[col]) for key, col in packed.slots[qid]}

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    alone = slot_vectors({"team": QUESTIONS["team"]}, "team")
    reordered = slot_vectors(dict(reversed(list(QUESTIONS.items()))), "team")
    shuffled = slot_vectors({"refund": QUESTIONS["refund"],
                             "team": Choice(instructions="Which team?", criteria={"sales": "Pricing", "billing": "Charges and refunds", "technical": "Bugs"})}, "team")
    for key in alone:
        assert cosine(alone[key], reordered[key]) > 0.99999
        assert cosine(alone[key], shuffled[key]) > 0.99999
    # the check has teeth: different options are far apart
    assert cosine(alone["billing"], alone["sales"]) < 0.999


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision")
    data = root / "access.jsonl"
    write([r.model_dump(mode="json") for r in states.generate(6000, seed=5) if r.family == "state_access"], data)
    config = DecisionTrainConfig(base=BASE[0], revision=BASE[1], data=[data], holdout=[], output=root / "model", valid_fraction=0.15,
                                 steps=200, learning_rate=5e-5, warmup=20, eval_every=100, valid_records=200)
    history = DecisionTrainer(config).run(log=lambda _: None)
    return config.output, history


def test_training_stays_inside_the_configured_memory_ceiling(checkpoint):
    import mlx.core as mx

    ceiling = int(mx.device_info()["memory_size"] * Settings.from_env().memory_fraction)
    assert mx.get_peak_memory() < ceiling


def test_training_learns_a_policy_the_untrained_head_cannot_answer(checkpoint):
    """The claim is the move, not a peak.

    Two hundred steps on this fixture have landed anywhere from 0.79 to above
    0.95 across runs at a fixed seed, because MLX kernels are not bitwise
    deterministic. An absolute threshold tests the draw; the improvement over
    the untrained head, and a loss that falls several-fold, test the training.
    """
    _, history = checkpoint
    assert history[0]["accuracy"] < 0.6
    assert history[-1]["accuracy"] > history[0]["accuracy"] + 0.25
    assert history[-1]["accuracy"] > 0.75
    assert history[-1]["loss"] < history[0]["loss"] / 5


def test_checkpoint_serves_typed_answers_through_the_app(checkpoint, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setenv("RULING_MODEL", str(path))
    engine = build_engine(Settings.from_env())
    assert isinstance(engine, DecisionEngine) and engine.adapter
    held_out = [r for r in states.generate(400, seed=77) if r.family == "state_access"][:40]
    correct = sum(engine.evaluate(SystemOneRequest(state=r.state, questions=r.questions)).answers["decision"].choice == r.labels["decision"]
                  for r in held_out)
    assert correct >= 36
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/systemone", json={"state": "hello", "questions": {k: q.model_dump() for k, q in QUESTIONS.items()}})
        models = client.get("/v1/models").json()
    body = response.json()
    assert response.status_code == 200 and set(body["answers"]) == set(QUESTIONS)
    assert sum(body["answers"]["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert models["models"][0]["max_options"] == 255 and models["models"][0]["adapter"] == engine.adapter
