"""Assets ride beside the request under names the text mentions; a box question says where in an image."""

import base64

import pytest
from pydantic import ValidationError

from ruling.questions import (
    IMAGE_REFERENCE,
    Box,
    Choice,
    InvalidOptions,
    Noul,
    SystemOneRequest,
    UnsupportedModality,
    check_locatable,
    resolve_assets,
    state_images,
    target_image,
    with_images,
)

from .test_images import image, png

PNG = png(2, 2)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x10\x00\x00\x00WEBP" + b"\x00" * 8


def asset(data: bytes = PNG, **fields) -> dict:
    return {"type": "image", "data_base64": base64.b64encode(data).decode(), **fields}


def test_assets_become_images_in_the_state_and_references_wherever_they_are_mentioned():
    request = SystemOneRequest(
        state="Inspect this: <product>",
        questions={"same": Noul(instructions="Is <product> the item in <listing>?"),
                   "kind": Choice(instructions="What is <product>?", criteria={"a": ["like", "<listing>"], "b": "other"})},
        assets={"<listing>": asset(JPEG), "<product>": asset(PNG)},
    )
    resolved = resolve_assets(request)
    # Numbered in the order the text first mentions them, not the order the assets map lists them.
    assert resolved.state == {"text": "Inspect this: <image 1>",
                              "images": [image(PNG), image(JPEG, media_type="image/jpeg")]}
    assert resolved.questions["same"].instructions == "Is <image 1> the item in <image 2>?"
    assert resolved.questions["kind"].instructions == "What is <image 1>?"
    assert resolved.questions["kind"].criteria == {"a": ["like", "<image 2>"], "b": "other"}
    assert resolved.assets is None
    assert [i.media_type for i in state_images(resolved.state)] == ["image/png", "image/jpeg"]


def test_assets_are_numbered_after_the_images_already_in_the_state():
    request = SystemOneRequest(state={"photo": image(PNG), "note": "compare with <ref>"},
                               questions={"q": Noul(instructions="Same as <ref>?")}, assets={"<ref>": asset(WEBP)})
    resolved = resolve_assets(request)
    assert resolved.state["note"] == "compare with <image 2>" and resolved.questions["q"].instructions == "Same as <image 2>?"
    assert [i.media_type for i in state_images(resolved.state)] == ["image/png", "image/webp"]


def test_a_multipart_part_supplies_the_bytes_and_an_unnamed_part_is_an_error():
    request = SystemOneRequest(state="<p>", questions={"q": Noul(instructions="Red?")},
                               assets={"<p>": {"type": "image", "part": "asset-0"}})
    resolved = resolve_assets(request, {"asset-0": PNG})
    assert state_images(resolved.state)[0].data == base64.b64encode(PNG).decode()
    with pytest.raises(ValueError, match="no multipart part named 'asset-0'"):
        resolve_assets(request, {})
    with pytest.raises(ValueError, match="asset-1.*named by no asset"):
        resolve_assets(request, {"asset-0": PNG, "asset-1": PNG})


@pytest.mark.parametrize("assets, reason", [
    ({"<p>": asset(b"not an image")}, "not a PNG"),
    ({"<p>": {"type": "image", "data_base64": "@@"}}, "base64"),
    ({"<p>": {"type": "image"}}, "exactly one"),
    ({"<p>": {"type": "image", "part": "a", "data_base64": "AA=="}}, "exactly one"),
    ({"<nowhere>": asset()}, "mentioned nowhere"),
])
def test_a_bad_asset_is_refused_by_name(assets, reason):
    with pytest.raises((ValueError, ValidationError), match=reason):
        resolve_assets(SystemOneRequest(state="see <p>", questions={"q": Noul(instructions="Red?")}, assets=assets))


def test_the_states_image_shape_is_an_asset_too():
    request = SystemOneRequest(state="see <p>", questions={"q": Noul(instructions="Red?")},
                               assets={"<p>": image(PNG)})
    assert state_images(resolve_assets(request).state)[0].media_type == "image/png"


def test_a_request_without_assets_passes_through_unchanged():
    request = SystemOneRequest(state="plain", questions={"q": Noul(instructions="Red?")})
    assert resolve_assets(request) is request


@pytest.mark.parametrize("state, expected", [
    ("text", {"text": "text", "images": [image()]}),
    (["a"], ["a", image()]),
    ({"k": "v"}, {"k": "v", "images": [image()]}),
    ({"images": [image(PNG)]}, {"images": [image(PNG), image()]}),
    ({"images": "taken"}, {"state": {"images": "taken"}, "images": [image()]}),
])
def test_images_are_added_after_everything_in_the_state(state, expected):
    assert with_images(state, state_images(image())) == expected


def test_a_box_question_needs_a_model_that_locates_and_an_image_to_search():
    questions = {"where": Box(instructions="the total line")}
    check_locatable(questions, {"photo": image()}, locates=True)
    check_locatable({"q": Noul(instructions="Red?")}, "no images", locates=False)
    with pytest.raises(UnsupportedModality):
        check_locatable(questions, {"photo": image()}, locates=False)
    with pytest.raises(InvalidOptions, match="questions.where: a box needs an image"):
        check_locatable(questions, "no images", locates=True)


@pytest.mark.parametrize("instructions, count, expected", [
    ("the total line", 1, 1),
    ("the signature on <image 2>", 3, 2),
    ({"find": "the stamp in <image 3>"}, 3, 3),
])
def test_a_box_searches_the_image_its_instructions_name_or_the_only_one(instructions, count, expected):
    assert target_image("where", Box(instructions=instructions), count) == expected


@pytest.mark.parametrize("instructions, count, reason", [
    ("the total line", 2, "name the one"),
    ("<image 1> and <image 2>", 2, "name several"),
    ("<image 3>", 2, "holds 2"),
])
def test_a_box_that_does_not_say_which_image_is_refused_by_name(instructions, count, reason):
    with pytest.raises(InvalidOptions, match=f"questions.where: .*{reason}"):
        target_image("where", Box(instructions=instructions), count)


def test_a_box_answer_is_typed_like_the_others():
    from ruling.questions import BoxAnswer, SystemOneResponse, Usage

    response = SystemOneResponse(model="m", usage=Usage(input_tokens=1), answers={
        "where": {"type": "box", "box": {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.9}, "confidence": 0.8}})
    assert isinstance(response.answers["where"], BoxAnswer) and response.answers["where"].box.right == 0.9
    with pytest.raises(ValidationError):
        BoxAnswer(box={"top": 0, "bottom": 1.5, "left": 0, "right": 1}, confidence=1)
