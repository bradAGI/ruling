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
import pytest
from PIL import Image

from ruling.calibration import Calibration
from ruling.hosted import HOSTED_PREFIX, HostedEngine
from ruling.questions import Choice, Noul, SystemOneRequest

from .conftest import TEST_MODEL

pytestmark = pytest.mark.integration

COLOR = Choice(instructions="What color is the square in <image 1>?",
               criteria={"red": None, "blue": None, "green": None})


def square(color: str) -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
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
