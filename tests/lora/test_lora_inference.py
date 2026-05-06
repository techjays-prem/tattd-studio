"""Inference smoke test for the LoRA Artifact path.

Default CI run uses the deterministic stub so the path is exercised
end-to-end without paying for real generations or requiring a real
training run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from tattd_studio.lora import (
    LoRAArtifact,
    deterministic_stub_client,
    load_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_TOML = REPO_ROOT / "data" / "lora_training" / "artifacts.toml"


def test_load_artifacts_returns_empty_list_when_no_real_runs() -> None:
    artifacts = load_artifacts(ARTIFACTS_TOML)
    assert artifacts == []


def test_deterministic_stub_returns_stable_uri_per_combination() -> None:
    client = deterministic_stub_client()
    artifact = LoRAArtifact(
        artist_slug="kira-fixture",
        base_model="FLUX.2-dev",
        replicate_url="https://example.com/replicate/kira",
        trigger_word="tattd_kira",
        trained_on=dt.date(2026, 5, 6),
        permission_record="permission/kira-fixture.toml",
    )
    uri_a = client.generate("fineline mountain", artifact)
    uri_b = client.generate("fineline mountain", artifact)
    uri_c = client.generate("traditional rose", artifact)

    assert uri_a.startswith("file://")
    assert uri_a == uri_b  # deterministic on (prompt, artifact)
    assert uri_a != uri_c  # different prompt → different URI


def test_deterministic_stub_distinguishes_base_models() -> None:
    client = deterministic_stub_client()
    base_dev = LoRAArtifact(
        artist_slug="x",
        base_model="FLUX.2-dev",
        replicate_url="u",
        trigger_word="t",
        trained_on=dt.date(2026, 5, 6),
        permission_record="p",
    )
    base_klein = LoRAArtifact(
        artist_slug="x",
        base_model="FLUX.2-klein",
        replicate_url="u",
        trigger_word="t",
        trained_on=dt.date(2026, 5, 6),
        permission_record="p",
    )
    assert client.generate("p", base_dev) != client.generate("p", base_klein)
