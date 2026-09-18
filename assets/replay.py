"""Record a measured replay: ruling's typed answers against the same model streaming JSON.

Both runs use the same model, state, and questions. ruling's time is the median of
five requests; the generative run is the median of three by completion time. ruling returns every answer in
one response; the generative baseline is asked for the same answers as a JSON
object and each token's arrival time is recorded. The GIF replays both timelines
aligned at t=0. Run with: uv run --with pillow python assets/replay.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mlx_lm import stream_generate
from PIL import Image, ImageDraw, ImageFont

from ruling.calibration import Calibration
from ruling.config import Settings
from ruling.engine import Engine
from ruling.questions import Choice, Noul, Score, SystemOneRequest

STATE = {"message": "My card was charged twice for one order. Refund the duplicate."}
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this?",
                   criteria={"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}),
    "urgency": Score(instructions="How urgent is this?", criteria=["Can wait", "This week", "Today"]),
    "refund": Noul(instructions="The customer asks for a refund."),
}
GENERATIVE_PROMPT = (
    "State:\n" + json.dumps(STATE) + "\n\nAnswer the questions as one JSON object with keys team, urgency, refund. "
    "team: one of billing (Charges and refunds), technical (Bugs), sales (Pricing), with a probability for each option. "
    "urgency: one of 0 (Can wait), 1 (This week), 2 (Today), with a probability for each level. "
    "refund: probability that the customer asks for a refund. Output only the JSON."
)
FPS = 20
W, H = 960, 420
BG, FG, MUTED, GREEN, ORANGE, LINE = (22, 22, 21), (236, 236, 230), (150, 150, 142), (95, 174, 133), (224, 119, 77), (51, 51, 47)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf", "/Library/Fonts/Courier New.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def measure(settings: Settings):
    engine = Engine(settings.model, Calibration(), settings.max_input_tokens, settings.rotations, settings.prior_debias)
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    engine.evaluate(request)  # warm-up
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        response = engine.evaluate(request)
        timings.append((time.perf_counter() - started) * 1000)
    ruling_ms = sorted(timings)[2]

    messages = [{"role": "user", "content": GENERATIVE_PROMPT}]
    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    runs = []
    for _ in range(3):
        tokens: list[tuple[float, str]] = []
        started = time.perf_counter()
        for chunk in stream_generate(engine.model, engine.tokenizer, prompt, max_tokens=200):
            tokens.append(((time.perf_counter() - started) * 1000, chunk.text))
        runs.append(tokens)
    tokens = sorted(runs, key=lambda r: r[-1][0])[1]  # median run by completion time
    return engine.model_id, response, ruling_ms, tokens


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.replace("\n", " \n ").split(" "):
        if word == "\n":
            lines.append(current); current = ""
            continue
        trial = (current + " " + word).strip()
        if draw.textlength(trial, font=fnt) > width and current:
            lines.append(current); current = word
        else:
            current = trial
    return lines + [current]


def frame(t_ms: float, model: str, response, ruling_ms: float, tokens, total_ms: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_big, f, f_small = font(22), font(15), font(13)
    d.text((24, 18), "ruling", font=f_big, fill=FG)
    d.text((110, 24), f"same {model.split('/')[-1]}, same state, same three questions", font=f_small, fill=MUTED)
    d.text((W - 150, 22), f"t = {t_ms / 1000:5.2f} s", font=f, fill=FG)
    d.line((24, 56, W - 24, 56), fill=LINE)
    d.line((W // 2, 56, W // 2, H - 40), fill=LINE)
    col = (W - 72) // 2
    d.text((24, 70), "typed decisions", font=f, fill=GREEN)
    d.text((W // 2 + 24, 70), "generated JSON, token by token", font=f, fill=ORANGE)
    y = 100
    if t_ms >= ruling_ms:
        d.text((24, y), f"all answers at {ruling_ms:.0f} ms", font=f_small, fill=MUTED); y += 26
        for qid, a in response.answers.items():
            d.text((24, y), f"{qid}", font=f, fill=FG); y += 22
            probs = {"yes": a.noul, "no": 1 - a.noul} if a.type == "noul" else a.probabilities
            for key, p in probs.items():
                label = a.legend[key] if a.type == "score" else key
                d.text((40, y), f"{label:<16}", font=f_small, fill=MUTED)
                d.rectangle((190, y + 4, 190 + int(220 * p), y + 14), fill=GREEN)
                d.text((420, y), f"{p:.3f}", font=f_small, fill=FG); y += 18
            y += 6
    else:
        d.text((24, y), "…", font=f, fill=MUTED)
    text = "".join(tok for at, tok in tokens if at <= t_ms)
    ty = 100
    for line in wrap(d, text, f_small, col - 10)[-14:]:
        d.text((W // 2 + 24, ty), line, font=f_small, fill=FG); ty += 18
    done = [at for at, _ in tokens if at <= t_ms]
    d.text((W // 2 + 24, H - 62), f"{len(done)} tokens" + (f", finished at {tokens[-1][0]:.0f} ms" if len(done) == len(tokens) else ""), font=f_small, fill=MUTED)
    d.text((24, H - 30), "measured on one Apple Silicon Mac; both timelines aligned at t = 0", font=f_small, fill=MUTED)
    return img


def main() -> None:
    settings = Settings.from_env()
    model, response, ruling_ms, tokens = measure(settings)
    total_ms = tokens[-1][0] + 1200
    frames = [frame(i * 1000 / FPS, model, response, ruling_ms, tokens, total_ms) for i in range(int(total_ms / 1000 * FPS))]
    out = Path(__file__).with_name("replay.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    summary = {"model": model, "ruling_ms": round(ruling_ms), "generated_tokens": len(tokens), "generation_ms": round(tokens[-1][0]),
               "generated_text": "".join(t for _, t in tokens)}
    Path(__file__).with_name("replay.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
