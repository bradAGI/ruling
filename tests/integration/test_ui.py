"""The served pages, driven in a real browser: images picked in the playground reach the model, and the
retraining history shows what `ruling retrain` decided."""

import json

import pytest
from PIL import Image
from playwright.sync_api import Page

from ruling.calibration import Calibration
from ruling.engine import Engine
from ruling.registry import Registry
from ruling.retrain import REPORT, promote
from ruling.train import TRAINING
from ruling.server import create_app
from ruling.vision import VISION_PREFIX, VisionEngine
from tests.integration.conftest import TEST_MODEL, _serve
from tests.integration.test_registry import write_adapter

pytestmark = pytest.mark.integration

COLOR = {"color": {"type": "choice", "instructions": "What color is the square in <image 1>?",
                   "criteria": {"red": None, "blue": None, "green": None}}}


@pytest.fixture(scope="module")
def vision():
    return VisionEngine(VISION_PREFIX + TEST_MODEL, Calibration(), max_input_tokens=8192, rotations=3,
                        prior_debias=False, max_image_tokens=256)


@pytest.fixture(scope="module")
def served(engine, vision):
    server, thread, url = _serve(create_app(Registry("text", {"text": engine, "vision": vision}), None, 4))
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def served_vision(vision):
    server, thread, url = _serve(create_app(Registry.single(vision), None, 4))
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.parametrize("color", ["red", "blue"])
def test_a_picked_image_reaches_the_model_and_decides_the_answer(page: Page, served, tmp_path, color):
    photo = tmp_path / f"{color}.png"
    Image.new("RGB", (64, 64), color).save(photo)
    page.goto(served)
    page.select_option("#model", "vision")
    page.fill("#state", '{"note": "A shot of the square."}')
    page.fill("#questions", json.dumps(COLOR))
    page.set_input_files("#images", str(photo))
    with page.expect_response("**/v1/systemone") as answered:
        page.click("#run")
    sent = json.loads(answered.value.request.post_data)
    assert sent["state"]["note"] == "A shot of the square." and len(sent["state"]["images"]) == 1
    assert answered.value.json()["answers"]["color"]["choice"] == color


def test_a_text_model_cannot_be_sent_images_from_the_page(page: Page, served):
    page.goto(served)
    page.select_option("#model", "text")
    assert page.is_disabled("#images") and page.is_disabled("#image-example")
    page.select_option("#model", "vision")
    assert page.is_enabled("#images") and page.is_enabled("#image-example")


def test_a_server_whose_default_model_sees_opens_on_an_image_example_it_answers(page: Page, served_vision):
    page.goto(served_vision)
    page.wait_for_selector(".thumb")
    with page.expect_response("**/v1/systemone") as answered:
        page.click("#run")
    assert len(json.loads(answered.value.request.post_data)["state"]["images"]) == 1
    answers = answered.value.json()["answers"]
    assert answers["kind"]["choice"] == "receipt" and answers["same_order"]["noul"] > 0.5
    assert max(answers["total"]["probabilities"], key=answers["total"]["probabilities"].get) == "1"


def test_a_server_with_a_key_answers_the_page_once_the_key_is_entered(page: Page, engine):
    server, thread, url = _serve(create_app(Registry.single(engine), "s3cret", 4))
    try:
        with page.expect_response("**/v1/models") as listed:
            page.goto(url)
        assert listed.value.status == 401 and page.locator("#model option").count() == 0
        with page.expect_response("**/v1/models") as relisted:
            page.fill("#key", "s3cret")
            page.dispatch_event("#key", "change")
        assert relisted.value.status == 200
        page.wait_for_selector("#model option", state="attached")
        with page.expect_response("**/v1/systemone") as answered:
            page.click("#run")
        assert answered.value.status == 200
        # The history page reads the key the playground kept for this tab.
        with page.expect_response("**/v1/retrain") as history:
            page.goto(f"{url}/history")
        assert history.value.status == 200
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_history_page_shows_every_finished_run_and_the_one_being_served(page: Page, tmp_path):
    adapters = tmp_path / "tickets"
    adapters.mkdir()
    for number, promoted in ((1, True), (2, False)):
        version = adapters / f"v{number}"
        write_adapter(version, num_layers=2, rank=4, seed=number)
        scores = {"n": 12, "accuracy": 0.5, "nll": 0.9}
        (version / REPORT).write_text(json.dumps({
            "version": version.name, "started_from": None if number == 1 else "v1", "records": 40,
            "current": scores, "candidate": {**scores, "nll": 0.8 if promoted else 1.0}, "promoted": promoted,
            "reason": "held-out log loss fell" if promoted else "held-out log loss did not fall"}))
    # v1 kept its training log; v2's is gone, as an imported adapter's would be.
    curve = [{"step": 0, "loss": 1.4}] + [{"step": step, "loss": 1.4 - step / 100, "train_loss": 1.5 - step / 100,
                                           "progress": step / 50, "eta_seconds": float(50 - step)} for step in (25, 50)]
    (adapters / "v1" / TRAINING).write_text(json.dumps({"config": {}, "history": curve}))
    promote(adapters, adapters / "v1")
    tuned = Engine(TEST_MODEL, Calibration(), 8192, 1, False, adapters / "current")
    server, thread, url = _serve(create_app(Registry.single(tuned), None, 4))
    try:
        page.goto(f"{url}/history")
        page.wait_for_selector("tr[data-version]")
        rows = page.locator("tr[data-version]")
        assert [rows.nth(i).get_attribute("data-version") for i in range(rows.count())] == ["v2", "v1"]
        assert page.locator("tr.serving").get_attribute("data-version") == "v1"
        assert page.locator("tr.promoted").get_attribute("data-version") == "v1"
        # The curve is drawn from the version's own training log: one point per logged step, none without a log.
        points = page.locator('tr[data-version="v1"] svg polyline').get_attribute("points").split()
        assert len(points) == len(curve)
        assert page.locator('tr[data-version="v2"] svg').count() == 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
