"""Live-mode metric implementations for the Comparison Matrix.

Default CI runs use deterministic proxies in ``metrics.score_entry``;
this module wires real implementations that the live runner consumes
when ``RUN_LIVE_COMPARISON_MATRIX=1``:

- ``compute_clip_score(prompt, image_uri)`` — OpenAI CLIP ViT-B/32
  cosine between the prompt's text embedding and the image's CLIP
  embedding, rescaled to [0, 1].
- ``compute_fid(images_a, images_b)`` — Fréchet Inception Distance via
  ``torchmetrics``. The reference set is the LoRA's onboarded artist's
  portfolio per the implementation plan; until #9 lands, no reference
  set exists and FID stays in proxy mode (``compute_fid`` returns
  ``None`` when the reference is empty).
- ``run_rubric_score(prompt, image_uri)`` — DeepEval ``GEval`` rubric
  for tattoo-specific quality (composition / linework / balance /
  originality), graded by an LLM judge.

Each function lazy-imports its heavy dependencies so the test suite's
CI path never pays for them.
"""

from __future__ import annotations

import io
import os
from urllib.parse import urlparse


def compute_clip_score(prompt: str, image_uri: str) -> float:
    """Compute CLIP score in [0, 1] for the (prompt, image) pair.

    Lazy-imports torch + open_clip + PIL. Real OpenAI CLIP ViT-B/32 by
    default; override the model via ``TATTD_CLIP_MODEL`` (e.g.
    ``ViT-L/14``).
    """
    import requests
    import torch
    from PIL import Image

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_torch is required for live CLIP scoring; "
            "install with `uv pip install open-clip-torch`"
        ) from exc

    model_id = os.environ.get("TATTD_CLIP_MODEL", "ViT-B-32")
    pretrained = os.environ.get("TATTD_CLIP_PRETRAINED", "openai")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_id, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_id)
    clip_model.training = False

    parsed = urlparse(image_uri)
    if parsed.scheme == "file":
        img = Image.open(parsed.path).convert("RGB")
    elif parsed.scheme in ("http", "https"):
        r = requests.get(image_uri, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(image_uri).convert("RGB")

    image = preprocess(img).unsqueeze(0)
    text = tokenizer([prompt])

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        similarity = (image_features @ text_features.T).item()

    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def compute_fid(image_uris_a: list[str], image_uris_b: list[str]) -> float | None:
    """FID between two image distributions, or ``None`` if either side is empty.

    Computes via ``torchmetrics.image.fid.FrechetInceptionDistance``.
    Per the implementation plan, the reference distribution should be
    the LoRA Artifact's onboarded artist's portfolio. Until #9 lands
    that portfolio doesn't exist, so callers pass an empty reference
    list and this function returns ``None``.
    """
    if not image_uris_a or not image_uris_b:
        return None

    import requests
    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance

    def _load(uri: str) -> torch.Tensor:
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            img = Image.open(parsed.path).convert("RGB")
        elif parsed.scheme in ("http", "https"):
            r = requests.get(uri, timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(uri).convert("RGB")
        img = img.resize((299, 299))
        tensor = (
            torch.tensor(list(img.getdata()), dtype=torch.uint8)
            .reshape(299, 299, 3)
            .permute(2, 0, 1)
        )
        return tensor.unsqueeze(0)

    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    for uri in image_uris_a:
        fid.update(_load(uri), real=False)
    for uri in image_uris_b:
        fid.update(_load(uri), real=True)
    return float(fid.compute().item())


def run_rubric_score(prompt: str, image_uri: str) -> float:
    """DeepEval ``GEval`` rubric for tattoo design quality.

    Returns the rubric's overall score in [0, 1]. The judge LLM is
    inherited from DeepEval's default (typically OpenAI gpt-4o); set
    ``DEEPEVAL_TELEMETRY_OPT_OUT=1`` to suppress telemetry.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="tattoo-design-quality",
        criteria=(
            "Score the Candidate Design's composition, linework, balance, "
            "and originality on a tattoo-specific rubric. Penalize generic "
            "stock-art aesthetics; reward designs that read clearly at "
            "the requested placement and respect the requested style "
            "vocabulary."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    case = LLMTestCase(input=prompt, actual_output=image_uri)
    metric.measure(case)
    return float(metric.score or 0.0)
