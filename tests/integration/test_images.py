"""Images in the state reach a vision model behind an OpenAI-compatible server, and its answer reflects them.

The server is `mlx_vlm.server` running the small test model, which is a
vision-language model; vLLM and other hosts take the same content parts.
"""

import base64
import io
import socket
import subprocess
import time

import httpx
import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ruling.calibration import Calibration
from ruling.engine import suffix_positions
from ruling.hosted import HOSTED_PREFIX, HostedEngine
from ruling.prompt import render_question
from ruling.questions import Choice, Noul, SystemOneRequest

from .conftest import TEST_MODEL

pytestmark = pytest.mark.integration

COLOR = Choice(instructions="What color is the square in <image 1>?",
               criteria={"red": None, "blue": None, "green": None})


def square(color: str, side: int = 64) -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (side, side), color).save(buffer, format="PNG")
    return {"type": "image", "media_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()}


@pytest.fixture(scope="module")
def vision_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "mlx_vlm.server", "--model", TEST_MODEL, "--host", "127.0.0.1",
         "--port", str(port), "--top-logprobs-k", "20"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/v1"
    deadline = time.time() + 300
    while time.time() < deadline:
        assert process.poll() is None, "mlx_vlm.server exited before it was ready"
        try:
            httpx.get(f"{url}/models", timeout=2)
            break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        process.terminate()
        pytest.fail("mlx_vlm.server did not start in time")
    yield url
    process.terminate()
    process.wait(timeout=30)


@pytest.fixture(scope="module")
def sighted(vision_server):
    return HostedEngine(HOSTED_PREFIX + TEST_MODEL, Calibration(), rotations=3, base_url=vision_server,
                        top_logprobs=20, concurrency=1,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})


@pytest.mark.parametrize("color", ["red", "blue", "green"])
def test_the_answer_follows_the_image(sighted, color):
    request = SystemOneRequest(state={"photo": square(color)}, questions={"color": COLOR})
    assert sighted.evaluate(request).answers["color"].choice == color


def test_images_and_text_in_one_state_are_both_read(sighted):
    state = {"note": "The shopper says the parcel arrived damaged.", "photo": square("red")}
    questions = {"color": COLOR, "damaged": Noul(instructions="The shopper reports damage.")}
    answers = sighted.evaluate(SystemOneRequest(state=state, questions=questions)).answers
    assert answers["color"].choice == "red" and answers["damaged"].noul > 0.5


@pytest.fixture(scope="module")
def local_vision():
    from ruling.vision import VISION_PREFIX, VisionEngine
    return VisionEngine(VISION_PREFIX + TEST_MODEL, Calibration(), max_input_tokens=16_384, rotations=3,
                        prior_debias=False, max_image_tokens=1024)


@pytest.mark.parametrize("color", ["red", "blue", "green"])
def test_the_local_vision_engine_answers_from_the_image(local_vision, color):
    request = SystemOneRequest(state={"photo": square(color)}, questions={"color": COLOR})
    assert local_vision.evaluate(request).answers["color"].choice == color


def test_each_question_continues_the_positions_the_images_left_off_at(local_vision):
    """The state and its images are read once and every question is scored against that cache, so a
    question's tokens must sit at the multimodal positions they would have in one uncached sequence."""
    state = {"note": "Two shots of one parcel.", "front": square("red", 1024), "back": square("blue", 512)}
    prefix = local_vision._prefix(state)
    suffix = local_vision._encode(render_question(COLOR, local_vision.codes.codes).text + prefix.tail)
    whole, _ = local_vision.vlm.language_model.get_rope_index(
        mx.array([prefix.tokens + suffix]), prefix.vision_inputs["image_grid_thw"], None,
        mx.ones((1, len(prefix.tokens) + len(suffix)), dtype=mx.int32))
    expected = np.array(whole)[:, :, len(prefix.tokens):]
    assert np.array_equal(np.array(suffix_positions(prefix, 1, len(suffix))), expected)
    assert suffix_positions(local_vision._prefix("No pictures."), 1, len(suffix)) is None


def test_a_local_vision_model_lists_images_and_a_text_model_refuses_them(local_vision, engine):
    assert local_vision.reads_images and not engine.reads_images


def test_a_server_on_a_local_vision_model_answers_from_images(local_vision):
    from fastapi.testclient import TestClient

    from ruling.registry import Registry
    from ruling.server import create_app

    with TestClient(create_app(Registry.single(local_vision), None, 4)) as client:
        models = client.get("/v1/models").json()["models"]
        response = client.post("/v1/systemone", json={
            "state": {"photo": square("green")},
            "questions": {"color": COLOR.model_dump(), "photo": {"type": "noul", "instructions": "<image 1> is blank."}}})
    assert models[0]["modalities"] == ["text", "image"]
    assert response.status_code == 200 and response.json()["answers"]["color"]["choice"] == "green"
