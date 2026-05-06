"""Multi-turn ConsultationSession tests."""

from __future__ import annotations

import textwrap

import pytest

from tattd_studio.consultation import (
    ConsultationSession,
    refine_intent,
)
from tattd_studio.knowledge import (
    DeterministicTextEmbeddingClient,
    KnowledgeRetriever,
    RetrievedChunk,
    ingest_corpus,
    parse_chunks_from_markdown,
)
from tattd_studio.models import Intent
from tattd_studio.vectordb import VectorStore

CORPUS = textwrap.dedent(
    """\
    <!-- CHUNK -->
    ---
    chunk_id: ts-fineline
    area: taxonomy
    title: Fineline
    source_url: https://example.com/styles/fineline
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Fineline tattooing emphasizes single-needle, minimal-weight linework.

    <!-- CHUNK -->
    ---
    chunk_id: ts-inner-forearm
    area: placement
    title: Inner forearm
    source_url: https://example.com/placement/inner-forearm
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Inner forearm holds detail well and is forgiving for fineline work.

    <!-- CHUNK -->
    ---
    chunk_id: ts-aftercare
    area: aftercare
    title: Saniderm
    source_url: https://example.com/aftercare/saniderm
    curator: tattd-studio-dev
    capture_date: 2026-05-06
    synthetic: true
    permission: synthetic-content-tattd-studio-poc
    ---

    Saniderm is a breathable second-skin bandage worn for 3-5 days.
    """
)


def _build_session() -> ConsultationSession:
    store = VectorStore(location=":memory:")
    store.create_collection("kc_session_test")
    embedder = DeterministicTextEmbeddingClient(dim=1024)
    ingest_corpus(
        store=store,
        collection="kc_session_test",
        chunks=parse_chunks_from_markdown(CORPUS),
        embedder=embedder,
    )
    retriever = KnowledgeRetriever(
        store=store, collection="kc_session_test", embedder=embedder
    )
    return ConsultationSession(retriever=retriever)


def test_session_advances_through_three_turns_and_accumulates_history() -> None:
    session = _build_session()

    turn_one = session.advance("I want a small fineline mountain")
    session.advance("on my inner forearm")
    turn_three = session.advance("about three inches")

    assert len(session.history) == 3
    assert session.history[-1] is turn_three
    assert "fineline" in turn_one.intent_after_turn.refined_description.lower()
    # Subsequent turns extend the Intent rather than replacing it.
    assert (
        "fineline mountain" in turn_three.intent_after_turn.refined_description.lower()
    )
    assert (
        "three inches" in turn_three.intent_after_turn.refined_description.lower()
    )


def test_session_records_grounding_chunks_per_turn() -> None:
    session = _build_session()
    turn = session.advance("I want a small fineline mountain on inner forearm")
    assert turn.grounding_chunks  # at least one chunk retrieved
    for chunk in turn.grounding_chunks:
        assert chunk.chunk_id
        assert chunk.source_url
        assert chunk.body


def test_session_rejects_empty_user_message() -> None:
    session = _build_session()
    with pytest.raises(ValueError, match="user_message"):
        session.advance("")


def test_session_synthesize_fn_is_injectable() -> None:
    session = _build_session()
    captured: list[str] = []

    def fake_synth(intent: Intent, msg: str, chunks: list[RetrievedChunk]) -> str:
        captured.append(intent.refined_description)
        return "fake response"

    session.synthesize_fn = fake_synth
    turn = session.advance("hello")
    assert turn.studio_response == "fake response"
    assert captured == [turn.intent_after_turn.refined_description]


def test_refine_intent_appends_message_to_running_description() -> None:
    intent = Intent(refined_description="fineline mountain")
    refined = refine_intent(intent, "on inner forearm", chunks=[])
    assert refined.refined_description == "fineline mountain; on inner forearm"


def test_session_exposes_retrieval_context_deduplicated() -> None:
    session = _build_session()
    session.advance("fineline")
    session.advance("inner forearm")
    contexts = session.retrieval_context
    # The same chunk may be retrieved across turns; dedup by chunk_id.
    assert len(contexts) == len(set(contexts))
