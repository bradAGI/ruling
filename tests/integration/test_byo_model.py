"""Bring your own model: any OpenAI-compatible server with top logprobs can answer typed questions.

The server here is `mlx_lm.server`, which speaks the same protocol as vLLM,
SGLang, llama.cpp and LM Studio, so this exercises the path a user takes when
their model does not run through ruling's own MLX engine.
"""

import socket
import subprocess
import time

import httpx
import numpy as np
import pytest

from ruling.calibration import Calibration, softmax
from ruling.hosted import HOSTED_PREFIX, HostedEngine
from ruling.questions import Choice, Noul, SystemOneRequest

from .conftest import TEST_MODEL

pytestmark = pytest.mark.integration

STATE = "The forecast says the sky is clear and blue all day with no clouds."
COLOR = Choice(instructions="What color is the sky in the state?",
               criteria={"red": None, "blue": None, "green": None})


@pytest.fixture(scope="module")
def openai_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "mlx_lm", "server", "--model", TEST_MODEL, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            pytest.skip("mlx_lm.server exited before it was ready")
        try:
            httpx.get(f"{url}/models", timeout=2)
            break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        process.terminate()
        pytest.skip("mlx_lm.server did not start in time")
    yield url
    process.terminate()
    process.wait(timeout=30)


def test_typed_answers_from_a_model_behind_an_openai_endpoint(openai_server, single_engine):
    # mlx_lm.server answers one connection at a time, returns at most 11 top logprobs,
    # and needs chat_template_kwargs to stop a thinking model from opening with <think>.
    hosted = HostedEngine(HOSTED_PREFIX + TEST_MODEL, Calibration(), rotations=1,
                          base_url=openai_server, concurrency=1, top_logprobs=11,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert hosted.max_options == 11
    request = SystemOneRequest(
        state=STATE,
        questions={"color": COLOR, "blue": Noul(instructions="The sky is described as blue.")},
    )
    response = hosted.evaluate(request)

    assert response.answers["color"].choice == "blue"
    assert response.answers["blue"].noul > 0.5
    assert response.usage.input_tokens > 0 and response.usage.output_tokens == 0

    # Same weights, same prompts, two readout paths: restricted logits locally,
    # top logprobs over HTTP. The distributions have to agree.
    local = single_engine.score(STATE, {"color": COLOR}).raw["color"]
    remote = hosted.score(STATE, {"color": COLOR}).raw["color"]
    assert local.keys == remote.keys
    assert np.allclose(softmax(local.logits), softmax(remote.logits), atol=0.1)
