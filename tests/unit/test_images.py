"""Images ride in the state as typed objects; ruling checks them where the hosted services do."""

import base64
import struct
import zlib

import pytest
from pydantic import ValidationError

from ruling.prompt import hosted_user_message, render_question, render_state
from ruling.questions import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    Noul,
    SystemOneRequest,
    UnsupportedModality,
    check_readable,
    state_images,
)


def png(width: int = 1, height: int = 1) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def image(data: bytes | None = None, media_type: str = "image/png") -> dict:
    return {"type": "image", "media_type": media_type, "data": base64.b64encode(data or png()).decode()}


def ask(state) -> SystemOneRequest:
    return SystemOneRequest(state=state, questions={"q": Noul(instructions="The picture is red.")})


def test_images_are_found_anywhere_in_the_state_in_order():
    first, second = image(png(1, 1)), image(png(2, 2))
    request = ask({"note": "two shots", "shots": [first, {"zoomed": second}]})
    assert [i.data for i in state_images(request.state)] == [first["data"], second["data"]]
    assert state_images("plain text") == []


@pytest.mark.parametrize("state, reason", [
    ([image()] * (MAX_IMAGES + 1), "at most"),
    ({"photo": image(media_type="image/gif")}, "media_type"),
    ({"photo": {**image(), "data": "not base64!"}}, "base64"),
    ({"photo": image(b"\x00" * (MAX_IMAGE_BYTES + 1))}, "MB"),
    ({"photo": {"type": "image", "media_type": "image/png"}}, "data"),
])
def test_a_bad_image_is_refused_at_the_door(state, reason):
    with pytest.raises(ValidationError, match=reason):
        ask(state)


def test_an_engine_that_cannot_see_refuses_a_state_with_images():
    check_readable({"photo": image()}, reads_images=True)
    check_readable("no pictures here", reads_images=False)
    with pytest.raises(UnsupportedModality):
        check_readable({"photo": image()}, reads_images=False)


def test_the_prompt_text_never_carries_image_bytes():
    photo = image()
    assert photo["data"] not in render_state({"photo": photo})
    assert "<image 1>" in render_state({"photo": photo})


def test_a_hosted_message_attaches_each_image_after_the_state_and_is_unchanged_without_any():
    rendered = render_question(Noul(instructions="The picture is red."), ["A", "B"])
    assert isinstance(hosted_user_message("no pictures", rendered), str)

    first, second = image(png(1, 1)), image(png(2, 2), media_type="image/png")
    parts = hosted_user_message({"shots": [first, second]}, rendered)
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert urls == [f"data:image/png;base64,{first['data']}", f"data:image/png;base64,{second['data']}"]
    text = "".join(p["text"] for p in parts if p["type"] == "text")
    assert "<image 1>" in text and "<image 2>" in text and first["data"] not in text
    assert parts[-1]["type"] == "text" and parts[-1]["text"].strip() == rendered.closing
