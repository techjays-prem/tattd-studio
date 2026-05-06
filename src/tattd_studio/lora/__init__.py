"""LoRA Artifact registry + inference path used by the Comparison Matrix.

The LoRA Artifact lives outside the runtime loop per CONTEXT.md; this
module is consumed only by the Comparison Matrix under the Eval Harness.

A real LoRA Artifact requires:

- An onboarded artist's signed permission (`data/lora_training/permission/`)
- A populated `data/lora_training/artifacts.toml` entry pointing at the
  Replicate URL produced by `infra/train_lora.yaml`

Until that lands, this module behaves in *deferred mode*: the registry
returns an empty list and the inference factory returns a deterministic
stub the Comparison Matrix can run end-to-end without paying for real
generations.
"""

from tattd_studio.lora.inference import (
    LoRAInferenceClient,
    build_lora_inference_client,
    deterministic_stub_client,
)
from tattd_studio.lora.registry import (
    LoRAArtifact,
    load_artifacts,
)

__all__ = [
    "LoRAArtifact",
    "LoRAInferenceClient",
    "build_lora_inference_client",
    "deterministic_stub_client",
    "load_artifacts",
]
