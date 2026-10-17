"""Local vision-language models: the images in a state are read with its text, once per request.

`vision:<MLX model>` loads the checkpoint through mlx-vlm, vision tower
included, where the text engine's mlx-lm loads only the language half. The
state's text names each image `<image 1>`, `<image 2>`, …; the images follow it
in the prompt under those names. The state and its images are encoded once
into the prefix cache and every question is scored against it, as for text.
Images shorten the multimodal positions (a grid of tokens takes fewer
positions than tokens), so each question's positions continue from where the
prefix's left off rather than from its token count.
"""

from __future__ import annotations

import base64
import io
import math
import re
from importlib import import_module

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ruling.calibration import Calibration
from ruling.engine import LONG_CONTEXT_TOKENS, Engine, _Prefix
from ruling.prompt import SPLIT_MARKER, render_state
from ruling.questions import (
    IMAGE_REFERENCE,
    Box,
    BoxAnswer,
    BoxCoordinates,
    Image,
    State,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    render_text,
    state_images,
    target_image,
)

VISION_PREFIX = "vision:"

# A box is the one answer that is generated: the model is asked for the subject's bounding box the way
# its grounding training phrased it, and names the edges on a 1000 × 1000 grid over the image whatever
# the image's size, so the answer is read back as fractions.
BOX_PROMPT = "{instructions}\nOutline the position of what this asks for in the image and output its bbox_2d in JSON format."
BOX_GRID = 1000
BOX_TOKENS = 64
BOX_PATTERN = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
WHOLE_IMAGE = BoxCoordinates(top=0.0, bottom=1.0, left=0.0, right=1.0)


class VisionEngine(Engine):
    def __init__(self, model_id: str, calibration: Calibration, max_input_tokens: int, rotations: int,
                 prior_debias: bool, max_image_tokens: int, max_branch_tokens: int | None = None,
                 long_context_tokens: int = LONG_CONTEXT_TOKENS):
        self.max_image_tokens = max_image_tokens
        super().__init__(model_id.removeprefix(VISION_PREFIX), Calibration(), max_input_tokens, rotations,
                         prior_debias, None, max_branch_tokens, long_context_tokens)
        self.model_id = VISION_PREFIX + self.model_id
        self.reads_images = True
        self.locates = True
        calibration.check(self.provenance)
        self.calibration = calibration

    def _load(self, model_id: str, adapter: None):
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ImportError("vision: models need the vision extra: pip install 'ruling[vision]'") from exc
        self.vlm, self.processor = load(model_id)
        # A full-size photo would otherwise become tens of thousands of tokens; one token covers a
        # (patch × merge)² square of pixels, so this caps each image at max_image_tokens.
        images = self.processor.image_processor
        side = images.patch_size * images.merge_size
        images.max_pixels = self.max_image_tokens * side * side
        images.min_pixels = min(images.min_pixels, images.max_pixels)
        return self.vlm.language_model, self.processor.tokenizer

    def _user_content(self, state: State, rendered_state: str) -> str | list[dict]:
        images = state_images(state)
        if not images:
            return super()._user_content(state, rendered_state)
        parts = [{"type": "text", "text": rendered_state}]
        for number in range(1, len(images) + 1):
            parts += [{"type": "text", "text": f"{IMAGE_REFERENCE.format(number)}:"}, {"type": "image"}]
        return parts + [{"type": "text", "text": SPLIT_MARKER}]

    def _prefix(self, state: State) -> _Prefix:
        images = state_images(state)
        if not images:
            return super()._prefix(state)
        from mlx_vlm.utils import prepare_inputs

        prefix_text, tail = self.split_prompt(state)
        inputs = prepare_inputs(self.processor, images=[_picture(n, image) for n, image in enumerate(images, 1)],
                                prompts=prefix_text, image_token_index=self.vlm.config.image_token_index)
        ids, grid = inputs["input_ids"], inputs["image_grid_thw"]
        _, deltas = self.vlm.language_model.get_rope_index(ids, grid, None, inputs["attention_mask"])
        return _Prefix(tokens=ids[0].tolist(), tail=tail,
                       vision_inputs={"pixel_values": inputs["pixel_values"], "image_grid_thw": grid},
                       position_delta=int(deltas.reshape(-1)[0]))

    def _encode_prefix(self, prefix: _Prefix):
        if prefix.vision_inputs is None:
            return super()._encode_prefix(prefix)
        ids = mx.array([prefix.tokens])
        features = self.vlm.get_input_embeddings(ids, prefix.vision_inputs["pixel_values"],
                                                 image_grid_thw=prefix.vision_inputs["image_grid_thw"],
                                                 mask=mx.ones(ids.shape, dtype=mx.int32))
        cache = make_prompt_cache(self.model)
        # Hidden states only: the output head never runs over the prefix.
        self.model.model(ids, inputs_embeds=features.inputs_embeds, cache=cache, position_ids=features.position_ids)
        mx.eval([c.state for c in cache])
        return cache


    # ---- boxes -------------------------------------------------------------

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        """Scored questions against the state as for text; box questions located one by one after them."""
        scorable = {qid: q for qid, q in request.questions.items() if not isinstance(q, Box)}
        boxes = {qid: q for qid, q in request.questions.items() if isinstance(q, Box)}
        answers: dict = {}
        usage = Usage(input_tokens=0)
        if scorable:
            scored = self.score(request.state, scorable)
            answers.update({qid: self.answer(scorable[qid], scored.raw[qid]) for qid in scorable})
            usage.input_tokens += scored.input_tokens
        if boxes:
            located, used = self.locate(request.state, boxes)
            answers.update(located)
            usage.input_tokens += used.input_tokens
            usage.output_tokens += used.output_tokens
        return SystemOneResponse(model=self.model_id, answers={qid: answers[qid] for qid in request.questions},
                                 usage=usage)

    def locate(self, state: State, questions: dict[str, Box]) -> tuple[dict[str, BoxAnswer], Usage]:
        """Each box question's answer, generated from the image it names with the state's text in view.
        Confidence is how sure the model was of each edge: the geometric mean probability of the leading
        digit of every number it wrote, since doubt about a trailing digit is doubt at the thousandth.
        An answer with no box in it is the whole image at confidence zero."""
        from mlx_vlm.prompt_utils import apply_chat_template

        stream_generate = import_module("mlx_vlm.generate").stream_generate
        images = state_images(state)
        targets = {qid: target_image(qid, box, len(images)) for qid, box in questions.items()}
        answers, usage = {}, Usage(input_tokens=0)
        with self._lock:
            self._activate()
            for qid, box in questions.items():
                number = targets[qid]
                text = render_state(state) + BOX_PROMPT.format(instructions=render_text(box.instructions))
                prompt = apply_chat_template(self.processor, self.vlm.config, text, num_images=1)
                written, chosen, last = "", [], None
                for last in stream_generate(self.vlm, self.processor, prompt, image=[_picture(number, images[number - 1])],
                                            max_tokens=BOX_TOKENS, temperature=0.0):
                    # Text arrives in segments, so a token's own text is read back from its id below.
                    written += last.text
                    if last.token is not None and last.logprobs is not None:
                        chosen.append((last.token, float(last.logprobs[last.token])))
                    if BOX_PATTERN.search(written):
                        break
                usage.input_tokens += last.prompt_tokens if last else 0
                usage.output_tokens += last.generation_tokens if last else 0
                answers[qid] = read_box(written, self._leading_digits(chosen))
        return answers, usage


    def _leading_digits(self, chosen: list[tuple[int, float]]) -> list[float]:
        """The log probability of the first digit of each number among the generated tokens."""
        leading, in_number = [], False
        for token, log_prob in chosen:
            is_digit = self.tokenizer.decode([token]).strip().isdigit()
            if is_digit and not in_number:
                leading.append(log_prob)
            in_number = is_digit
        return leading


def read_box(written: str, leading_log_probs: list[float]) -> BoxAnswer:
    """The first bbox_2d in what the model wrote, as fractions of the image, with its edges in order,
    at the geometric mean of the probabilities given for its numbers."""
    found = BOX_PATTERN.search(written)
    if found is None or not leading_log_probs:
        return BoxAnswer(box=WHOLE_IMAGE, confidence=0.0)
    x1, y1, x2, y2 = (min(max(int(n) / BOX_GRID, 0.0), 1.0) for n in found.groups())
    box = BoxCoordinates(top=min(y1, y2), bottom=max(y1, y2), left=min(x1, x2), right=max(x1, x2))
    return BoxAnswer(box=box, confidence=math.exp(sum(leading_log_probs) / len(leading_log_probs)))


def _picture(number: int, image: Image):
    # Pillow comes with the vision extra, so it is imported only where an image is read.
    from PIL import Image as Picture
    from PIL import UnidentifiedImageError

    try:
        return Picture.open(io.BytesIO(base64.b64decode(image.data))).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{IMAGE_REFERENCE.format(number)} is not a readable {image.media_type}") from exc
