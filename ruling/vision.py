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

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ruling.calibration import Calibration
from ruling.engine import Engine, _Prefix
from ruling.prompt import SPLIT_MARKER
from ruling.questions import IMAGE_REFERENCE, Image, State, state_images

VISION_PREFIX = "vision:"


class VisionEngine(Engine):
    def __init__(self, model_id: str, calibration: Calibration, max_input_tokens: int, rotations: int,
                 prior_debias: bool, max_image_tokens: int, max_branch_tokens: int | None = None):
        self.max_image_tokens = max_image_tokens
        super().__init__(model_id.removeprefix(VISION_PREFIX), Calibration(), max_input_tokens, rotations,
                         prior_debias, None, max_branch_tokens)
        self.model_id = VISION_PREFIX + self.model_id
        self.reads_images = True
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

    def variant(self, adapter, calibration: Calibration) -> "VisionEngine":
        if adapter is not None:
            raise ValueError(f"{self.model_id} takes no adapter: LoRA training reads text only")
        return super().variant(adapter, calibration)

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


def _picture(number: int, image: Image):
    # Pillow comes with the vision extra, so it is imported only where an image is read.
    from PIL import Image as Picture
    from PIL import UnidentifiedImageError

    try:
        return Picture.open(io.BytesIO(base64.b64decode(image.data))).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{IMAGE_REFERENCE.format(number)} is not a readable {image.media_type}") from exc
