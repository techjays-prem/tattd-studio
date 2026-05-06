"""`python -m tattd_studio.main` entry point.

Launches the Gradio shell. Requires the `ui` extra (gradio) and a populated
`GEMINI_API_KEY` so the Knowledge Retriever, Generation Client, and Anatomy
Critic all wire to live providers.
"""

from __future__ import annotations

from tattd_studio.ui.gradio_app import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
