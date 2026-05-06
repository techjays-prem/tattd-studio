"""Minimal Gradio shell for the Studio.

Renders chat input → Candidate Design grid with the Anatomy Critic verdict
overlaid per design. Slice #6 only wires the Anatomy Critic; the other
three Critics arrive in slice #7.

Imports of `gradio` and the Studio's runtime providers are deferred to
``launch`` so that ``python -m tattd_studio.main`` fails late and clearly
when an env var is missing, and the test suite can import this module
without paying the gradio install cost.
"""

from __future__ import annotations

import os
from typing import Any

from tattd_studio.graph.critics import PlacementContext
from tattd_studio.models import Intent


def _parse_placement(text: str) -> PlacementContext:
    """Best-effort placement extraction from a free-form chat line.

    Slice #6 keeps this trivial; the multi-turn Consultation in slice #7
    refines this into a proper structured Intent.
    """
    body_parts = (
        "inner forearm", "outer forearm", "upper arm", "shoulder blade",
        "chest", "ribs", "thigh", "calf", "ankle", "wrist", "neck",
    )
    lower = text.lower()
    for bp in body_parts:
        if bp in lower:
            return PlacementContext(body_part=bp, size_inches=3.0)
    return PlacementContext(body_part="inner forearm", size_inches=3.0)


def launch(server_port: int = 7860) -> None:
    """Boot the Studio and open the Gradio app on ``localhost:server_port``."""
    import gradio as gr

    from tattd_studio.generation import (
        GenerationClient,
        build_gemini_generate_fn,
    )
    from tattd_studio.graph.critics import (
        AnatomyCritic,
        build_gemini_anatomy_judge,
    )
    from tattd_studio.graph.studio import build_studio_graph
    from tattd_studio.knowledge import (
        KnowledgeRetriever,
        build_gemini_text_embedding_client,
        ingest_corpus,
    )
    from tattd_studio.knowledge.ingest import load_chunks_from_dir
    from tattd_studio.vectordb import VectorStore

    # Bootstrap the Vector Store with the Knowledge Corpus.
    store = VectorStore(location=":memory:")
    store.create_collection("knowledge_corpus")
    embedder = build_gemini_text_embedding_client()
    chunks = load_chunks_from_dir(_knowledge_dir())
    ingest_corpus(
        store=store, collection="knowledge_corpus", chunks=chunks, embedder=embedder
    )
    retriever = KnowledgeRetriever(
        store=store, collection="knowledge_corpus", embedder=embedder
    )

    source_model_id = os.environ.get(
        "TATTD_GENERATION_SOURCE_MODEL_ID", "gemini-nano-banana-2"
    )
    generation_client = GenerationClient(
        source_model_id=source_model_id,
        generate_fn=build_gemini_generate_fn(source_model_id),
    )
    critic = AnatomyCritic(judge_fn=build_gemini_anatomy_judge())

    graph = build_studio_graph(
        retriever=retriever,
        generation_client=generation_client,
        anatomy_critic=critic,
    )

    def run_session(message: str) -> tuple[list[dict[str, Any]], str]:
        intent = Intent(refined_description=message)
        placement = _parse_placement(message)
        state = graph.invoke(
            {
                "intent": intent,
                "placement_context": placement,
                "n_candidates": 4,
                "knowledge_chunks": [],
                "candidate_designs": [],
                "anatomy_checks": [],
                "metadata": {},
            }
        )
        gallery = []
        for cd, check in zip(state["candidate_designs"], state["anatomy_checks"], strict=False):
            label = (
                f"placement_valid={check.placement_valid} "
                f"({check.confidence:.2f})"
            )
            gallery.append({"image": cd.image_uri, "caption": label})
        latency = state["metadata"].get("latency_seconds", 0.0)
        summary = (
            f"Surfaced {len(gallery)} Candidate Designs in {latency:.2f}s "
            f"({len(state['knowledge_chunks'])} Knowledge Corpus chunks "
            f"used)."
        )
        return gallery, summary

    with gr.Blocks(title="Tattd Studio") as demo:
        gr.Markdown("# Tattd Studio")
        with gr.Row():
            chat_in = gr.Textbox(
                label="Describe your tattoo",
                placeholder=(
                    "fineline minimalist mountain on inner forearm, ~3 inches"
                ),
            )
            submit = gr.Button("Generate")
        gallery = gr.Gallery(label="Candidate Designs")
        summary = gr.Markdown()
        submit.click(run_session, inputs=[chat_in], outputs=[gallery, summary])
    demo.launch(server_port=server_port, server_name="127.0.0.1")


def _knowledge_dir() -> Any:
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / "data" / "knowledge"
