"""What a vision model writes for a box is read back as fractions of the image, edges in order."""

import math

import pytest

from ruling.questions import BoxCoordinates
from ruling.vision import WHOLE_IMAGE, read_box


def test_the_first_bbox_is_read_on_the_thousand_grid_at_the_numbers_mean_probability():
    written = '```json\n[\n\t{"bbox_2d": [495, 242, 750, 589], "label": "red square"},\n\t{"bbox_2d": [0, 0, 10, 10]}]'
    answer = read_box(written, [math.log(0.9), math.log(0.8)])
    assert answer.box == BoxCoordinates(top=0.242, bottom=0.589, left=0.495, right=0.75)
    assert answer.confidence == pytest.approx(math.sqrt(0.9 * 0.8))


def test_edges_are_ordered_and_clamped_whatever_the_model_wrote():
    answer = read_box('{"bbox_2d": [900, 1200, 100, 50]}', [0.0])
    assert answer.box == BoxCoordinates(top=0.05, bottom=1.0, left=0.1, right=0.9) and answer.confidence == 1.0


@pytest.mark.parametrize("written, digits", [("I cannot see a square.", [0.0]), ("", []), ('{"bbox_2d": [1, 2, 3]}', [0.0])])
def test_no_box_in_the_reply_is_the_whole_image_at_no_confidence(written, digits):
    assert read_box(written, digits).box == WHOLE_IMAGE and read_box(written, digits).confidence == 0.0
