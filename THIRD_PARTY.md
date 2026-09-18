# Third-party material

ruling is not affiliated with, endorsed by, or connected to TypeSafe AI. "Jev"
and "System One" are TypeSafe's names for their product; they appear here only
to describe what ruling reproduces and is compatible with.

| Material | Where | License | Notes |
|---|---|---:|---|
| `datasets/authored144.jsonl`, `datasets/perturbations108.jsonl`, `datasets/wanli256-selection.jsonl`, `datasets/typesafe102-selection.jsonl` | converted from [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) | MIT, Copyright (c) 2026 TheoLeeCJ | Question text, options, labels, and row selections unchanged; only the container shape differs. Full license text below. |
| Confidence formulas in `ruling/calibration.py` | reimplemented from [typesafe-ai/system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python), `src/system_one_adapter/_utils/confidence_metrics.py` | MIT, Copyright (c) 2026 TypeSafe AI | Two short formulas, rewritten in NumPy so a ruling confidence value is comparable with a hosted one. No code is copied. |
| `typesafe-sdk` | dev dependency in `pyproject.toml`, used by one compatibility test | MIT, Copyright (c) 2026 TypeSafe AI | TypeSafe's official Python client, installed unmodified and run against a local ruling server. Not redistributed. |
| TypeSafe public evaluation cases | downloaded by `ruling build-dataset typesafe`, never committed | TypeSafe AI's published evaluation data at [evals.typesafe.ai](https://evals.typesafe.ai/) | Used only to compare answers on identical inputs. |
| Every parallel judgment lab data | downloaded by `ruling build-dataset every`, never committed | Every's published experiment data and source archive | Used only to compare answers on identical inputs. |
| WANLI | downloaded by `ruling build-dataset wanli`, never committed | CC-BY-4.0 | Liu et al., [alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI), pinned revision `61c95318`. |
| SST-5 | downloaded by `ruling build-dataset sst5`, never committed | Stanford Sentiment Treebank terms | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5), pinned revision `e51bdcd8`. |
| Training corpora: WANLI train (CC-BY-4.0), SNLI (CC-BY-SA-4.0), CLINC150 (CC-BY-3.0), BoolQ (CC-BY-SA-3.0), GoEmotions (Apache-2.0), Civil Comments (CC0-1.0) | downloaded by `ruling build-corpus public` at pinned revisions, never committed | as listed | Used to train adapters. Share-alike terms may apply to redistributed adapters trained on SNLI or BoolQ. |
| Synthetic worlds | written by `ruling build-corpus worlds` with a local Qwen model (Apache-2.0), never committed | Apache-2.0 model outputs | No TypeSafe API output is used for training. |
| Model weights | downloaded by the user at run time, never committed | each model's own license | Default `mlx-community/Qwen3.5-4B-4bit`, Apache-2.0. |
| MLX, mlx-lm, FastAPI, pydantic, httpx, uvicorn, numpy, huggingface_hub | dependencies in `pyproject.toml` | MIT / Apache-2.0 / BSD | Not redistributed. |

## MIT License for the converted TheoLeeCJ/openjev fixtures

Copyright (c) 2026 TheoLeeCJ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
