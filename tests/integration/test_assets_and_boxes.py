"""Images named in the text reach the model as assets, a box says where something is in one, and a request
may ask for long context: the hosted API's latest surface, answered locally through real models."""

import base64
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image as Picture
from PIL import ImageDraw

from ruling.calibration import Calibration
from ruling.engine import Engine, InputTooLong, extended_context
from ruling.questions import Box, Choice, Noul, SystemOneRequest
from ruling.registry import Registry
from ruling.server import create_app
from ruling.vision import VISION_PREFIX, VisionEngine

from .conftest import TEST_MODEL, _serve

pytestmark = pytest.mark.integration

BOX_MODEL = os.environ.get("RULING_BOX_MODEL", "mlx-community/Qwen3.5-4B-4bit")
"""Grounding needs a model that learned it; the 0.8B test model writes numbers that mean nothing."""


def scene(size: tuple[int, int], box: tuple[int, int, int, int], color=(200, 20, 20)) -> bytes:
    image = Picture.new("RGB", size, (245, 245, 245))
    ImageDraw.Draw(image).rectangle(box, fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def block(data: bytes) -> dict:
    return {"type": "image", "media_type": "image/png", "data": base64.b64encode(data).decode()}


def iou(answer, size, truth) -> float:
    w, h = size
    left, top, right, bottom = answer.left * w, answer.top * h, answer.right * w, answer.bottom * h
    inter = max(0.0, min(right, truth[2]) - max(left, truth[0])) * max(0.0, min(bottom, truth[3]) - max(top, truth[1]))
    union = (right - left) * (bottom - top) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter
    return inter / union


@pytest.fixture(scope="module")
def small_vision():
    return VisionEngine(VISION_PREFIX + TEST_MODEL, Calibration(), max_input_tokens=16_384, rotations=1,
                        prior_debias=False, max_image_tokens=256)


@pytest.fixture(scope="module")
def vision_server(small_vision):
    server, thread, url = _serve(create_app(Registry.single(small_vision), "s3cret", 4))
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def test_the_official_sdk_sends_images_as_multipart_assets_and_boxes_and_they_are_answered(vision_server, monkeypatch):
    """TypeSafe's current client puts images in the state, in instructions and in criteria under names it makes
    up, sends them as multipart parts, and zstd-compresses the body; a box question rides along. All of it
    is read here as sent."""
    from typesafe_sdk import Box as HostedBox, Choice as HostedChoice, Image, Noul as HostedNoul, TypeSafeClient

    monkeypatch.setenv("TYPESAFE_BASE_URL", vision_server)
    monkeypatch.setenv("TYPESAFE_API_KEY", "s3cret")
    red, blue = scene((96, 96), (16, 16, 80, 80)), scene((96, 96), (16, 16, 80, 80), (20, 20, 220))
    padding = "The customer wrote a long note. " * 40  # over the SDK's 1 KiB compression threshold
    with TypeSafeClient(headers={"TypeSafe-Danger-Zone-Images": "true"}) as client:
        response = client.system_one(
            state=[padding, "The returned item: <returned>"],
            assets={"<returned>": Image(red)},
            questions={
                "color": HostedChoice(instructions="What color is the square in <returned>?",
                                      criteria={"red": None, "blue": None, "green": None}),
                "same": HostedNoul(instructions=["Is <returned> the item in this listing photo?", Image(blue)]),
                "where": HostedBox(instructions="the colored square in <returned>"),
            },
        )
        models = client.models.list()
    assert response.choices["color"].choice == "red"
    assert 0.0 <= response.nouls["same"].noul <= 1.0
    where = response.answers["where"]
    assert where.type == "box" and 0.0 <= where.box.top <= where.box.bottom <= 1.0
    assert 0.0 <= where.box.left <= where.box.right <= 1.0 and 0.0 <= where.confidence <= 1.0
    assert response.usage.input_tokens > 0
    assert models.models[0].name == VISION_PREFIX + TEST_MODEL


def test_named_assets_in_json_reach_the_state_as_images_the_question_can_name(vision_server):
    import httpx

    red = scene((96, 96), (16, 16, 80, 80))
    body = {"state": "The customer attached <photo>.", "assets": {"<photo>": {"type": "image", "data_base64": base64.b64encode(red).decode()}},
            "questions": {"color": {"type": "choice", "instructions": "What color is the square in <photo>?",
                                    "criteria": {"red": None, "blue": None, "green": None}}}}
    response = httpx.post(f"{vision_server}/v1/systemone", json=body, headers={"authorization": "Bearer s3cret"}, timeout=120)
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["color"]["choice"] == "red"


@pytest.fixture(scope="module")
def box_engine():
    return VisionEngine(VISION_PREFIX + BOX_MODEL, Calibration(), max_input_tokens=16_384, rotations=1,
                        prior_debias=False, max_image_tokens=1024)


SCENES = [((640, 480), (320, 120, 480, 280)), ((1200, 300), (900, 60, 1100, 240)), ((300, 900), (30, 600, 150, 850)),
          ((800, 800), (100, 100, 700, 700))]


def test_a_box_lands_on_the_square_whatever_the_images_shape(box_engine):
    """The model names positions on a 1000 × 1000 grid over the image; read back as fractions, the box
    overlaps the square it was asked for on every shape of image."""
    overlaps = []
    for size, truth in SCENES:
        request = SystemOneRequest(state={"photo": block(scene(size, truth))},
                                   questions={"where": Box(instructions="the red square")})
        response = box_engine.evaluate(request)
        answer = response.answers["where"]
        overlaps.append(iou(answer.box, size, truth))
        assert answer.confidence > 0.5, answer
        assert response.usage.output_tokens > 0 and response.usage.input_tokens > 0
    print(f"box IoU on {BOX_MODEL}: {[round(o, 3) for o in overlaps]} mean {sum(overlaps) / len(overlaps):.3f}")
    assert min(overlaps) > 0.7, overlaps


def test_a_box_searches_the_image_its_instructions_name_beside_scored_questions(box_engine):
    front, back = ((640, 480), (40, 40, 200, 200)), ((640, 480), (400, 300, 600, 440))
    request = SystemOneRequest(
        state={"front": block(scene(*front)), "back": block(scene(*back, color=(20, 20, 220)))},
        questions={"back_square": Box(instructions="the blue square in <image 2>"),
                   "front_color": Choice(instructions="What color is the square in <image 1>?",
                                         criteria={"red": None, "blue": None}),
                   "blank": Noul(instructions="<image 2> is blank.")})
    response = box_engine.evaluate(request)
    assert list(response.answers) == ["back_square", "front_color", "blank"]
    assert iou(response.answers["back_square"].box, back[0], back[1]) > 0.7
    assert response.answers["front_color"].choice == "red" and response.answers["blank"].noul < 0.5


def test_the_command_line_asks_for_a_box(tmp_path, capsys):
    from ruling.cli import main

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"photo": block(scene((640, 480), (320, 120, 480, 280)))}))
    main(["ask", f"@{state}", "--box", "where=the red square", "--model", VISION_PREFIX + TEST_MODEL, "--rotations", "1"])
    body = json.loads(capsys.readouterr().out)
    assert body["answers"]["where"]["type"] == "box" and set(body["answers"]["where"]["box"]) == {"top", "bottom", "left", "right"}


def test_a_request_asking_for_long_context_is_read_past_the_configured_limit(engine):
    """A real engine with a small limit refuses a long state, and reads it under the long-context switch."""
    short = Engine.__new__(Engine)
    short.__dict__.update(engine.__dict__)
    short.max_input_tokens = short.max_branch_tokens = 120
    short.long_context_tokens = 4096
    request = SystemOneRequest(state="The parcel arrived crushed. " * 40,
                               questions={"damaged": Noul(instructions="The parcel was damaged.")})
    with pytest.raises(InputTooLong):
        short.evaluate(request)
    with extended_context():
        assert short.evaluate(request).answers["damaged"].noul > 0.5
    server, thread, url = _serve(create_app(Registry.single(short), None, 4))
    try:
        import httpx

        body = request.model_dump(mode="json", exclude_none=True)
        refused = httpx.post(f"{url}/v1/systemone", json=body, timeout=120)
        allowed = httpx.post(f"{url}/v1/systemone", json=body, timeout=120,
                             headers={"TypeSafe-Danger-Zone-Long-Context": "true"})
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "context_too_long"
    assert allowed.status_code == 200 and allowed.json()["answers"]["damaged"]["noul"] > 0.5
