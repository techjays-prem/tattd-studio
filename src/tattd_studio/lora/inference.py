"""Inference path for LoRA-adapted FLUX.2 bases.

Two implementations:

- ``deterministic_stub_client`` — keyed on (prompt, base_model,
  artist_slug); returns a stable file:// URI for CI runs of the
  Comparison Matrix.
- ``build_lora_inference_client`` — production path through Replicate
  with the LoRA Artifact's URL as the model handle.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Protocol

from tattd_studio.lora.registry import LoRAArtifact


class LoRAInferenceClient(Protocol):
    """Generates an image from a prompt using a specific LoRA Artifact."""

    def generate(self, prompt: str, artifact: LoRAArtifact) -> str:
        """Returns a file:// URI of the generated image."""
        ...


def deterministic_stub_client() -> LoRAInferenceClient:
    """Return an inference client that produces stable, fake URIs.

    Used by the Comparison Matrix in CI when no real artifacts have
    been trained yet — verifies that the matrix runner end-to-end with
    no API spend.
    """

    class _Stub:
        def generate(self, prompt: str, artifact: LoRAArtifact) -> str:
            key = f"{prompt}|{artifact.base_model}|{artifact.artist_slug}"
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            return (
                f"file://{tempfile.gettempdir()}/"
                f"lora-{artifact.artist_slug}-{artifact.base_model}-{digest}.png"
            )

    return _Stub()  # type: ignore[return-value]


def build_lora_inference_client() -> LoRAInferenceClient:
    """Production LoRA inference via Replicate.

    Lazy-imports `replicate` so test environments without the SDK still
    import this module. Reads ``REPLICATE_API_TOKEN`` from env.
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError(
            "REPLICATE_API_TOKEN is required for the live LoRA inference path"
        )
    import replicate

    class _ReplicateClient:
        def generate(self, prompt: str, artifact: LoRAArtifact) -> str:
            full_prompt = f"{artifact.trigger_word} {prompt}".strip()
            output = replicate.run(
                artifact.replicate_url,
                input={"prompt": full_prompt, "num_inference_steps": 28},
            )
            uri = output[0] if isinstance(output, list) else output
            if not isinstance(uri, str) or not uri:
                raise RuntimeError(
                    f"Replicate returned no image URI for {artifact.replicate_url}"
                )
            return uri

    return _ReplicateClient()  # type: ignore[return-value]
