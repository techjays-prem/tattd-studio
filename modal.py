"""Modal deploy config for an auth-gated hosted Tattd Studio.

Deploy with:

    modal deploy modal.py

Requires:

- Modal CLI installed and authenticated (`modal token new`)
- A Modal secret named `tattd-gemini` containing `GEMINI_API_KEY`
- Optionally a Modal secret named `tattd-auth` for `BASIC_AUTH_USER` and
  `BASIC_AUTH_PASS` (auth gate; otherwise the URL is public)

The function exposes the Studio's Gradio UI under a Modal-issued web URL.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("tattd-studio")

REPO_ROOT = Path(__file__).resolve().parent

studio_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.11.9")
    .copy_local_file(REPO_ROOT / "pyproject.toml", "/app/pyproject.toml")
    .copy_local_file(REPO_ROOT / "uv.lock", "/app/uv.lock")
    .workdir("/app")
    .run_commands("uv sync --frozen --extra ui --no-dev")
    .copy_local_dir(REPO_ROOT / "src", "/app/src")
    .copy_local_dir(REPO_ROOT / "data", "/app/data")
    .copy_local_dir(REPO_ROOT / "evals", "/app/evals")
    .env({"PATH": "/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)


@app.function(
    image=studio_image,
    secrets=[
        modal.Secret.from_name("tattd-gemini"),
    ],
    timeout=600,
    keep_warm=0,
)
@modal.web_server(port=7860, startup_timeout=120)
def serve_studio() -> None:
    """Boot the Gradio Studio UI on Modal."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is required; configure the `tattd-gemini` Modal secret."
        )
    from tattd_studio.ui.gradio_app import launch

    launch(server_port=7860)
