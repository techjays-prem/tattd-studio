"""Default ``generate_fn`` factory backed by Gemini Nano Banana 2 / Pro.

The Generation Client wraps this at runtime. Tests inject their own
``generate_fn``; production wires through ``build_gemini_generate_fn``.

Per IMPLEMENTATION_PLAN.md → Risks & Gotchas, Nano Banana 2 / Pro is not
in any framework's indexed docs, so this module hits ``google.genai``
directly with no third-party adapter.
"""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from collections.abc import Callable

from google import genai
from google.genai import types as genai_types


def build_gemini_generate_fn(
    source_model_id: str,
    *,
    api_key: str | None = None,
) -> Callable[[str, int], list[str]]:
    """Return a `generate_fn` that calls Gemini Nano Banana 2 / Pro.

    Returns a callable that takes ``(prompt, n)`` and returns ``n``
    ``file://`` URIs pointing at the decoded image bytes written under
    the system tempdir.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for the runtime generate_fn")
    client = genai.Client(api_key=key)

    def generate(prompt: str, n: int) -> list[str]:
        uris: list[str] = []
        for _ in range(n):
            response = client.models.generate_content(
                model=source_model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            uri = _extract_first_image_uri(response)
            if uri is None:
                raise RuntimeError(
                    "Gemini response contained no inline image data"
                )
            uris.append(uri)
        return uris

    return generate


def _extract_first_image_uri(response: object) -> str | None:
    """Pull the first inline image bytes from a genai response and persist them."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if data is None:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)
            path = os.path.join(
                tempfile.gettempdir(), f"tattd-cd-{uuid.uuid4().hex}.png"
            )
            with open(path, "wb") as f:
                f.write(data)
            return f"file://{path}"
    return None
