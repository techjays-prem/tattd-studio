"""Two-Stage Matcher tests.

Stage 1 recall + Stage 2 rerank exercised end-to-end against an
in-memory Vector Store seeded from the real ``data/artists/artists.jsonl``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from tattd_studio.embeddings import DeterministicVisualEmbeddingClient
from tattd_studio.knowledge.embedding import DeterministicTextEmbeddingClient
from tattd_studio.matching import (
    TwoStageMatcher,
    ingest_artist_portfolio_index,
    load_artist_records,
)
from tattd_studio.models.candidate_design import CandidateDesign
from tattd_studio.vectordb import VectorStore

ARTISTS_JSONL = Path(__file__).resolve().parents[2] / "data" / "artists" / "artists.jsonl"


def _ranked_results():
    artists = load_artist_records(ARTISTS_JSONL)
    store = VectorStore(location=":memory:")
    store.create_collection("artist_portfolio_index")
    text_embedder = DeterministicTextEmbeddingClient(dim=1024)
    visual_embedder = DeterministicVisualEmbeddingClient(dim=768)
    n = ingest_artist_portfolio_index(
        store=store,
        collection="artist_portfolio_index",
        artists=artists,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
    )
    assert n == len(artists)

    target = next(a for a in artists if a.artist_slug == "yuki-japanese-bk")
    chosen = CandidateDesign(
        image_uri=target.portfolio_url,
        prompt=target.style_text(),
        source_model_id="gemini-nano-banana-2",
        metadata={"prompt_template_version": "v1.0"},
        created_at=dt.datetime(2026, 5, 6, 12, 0, tzinfo=dt.UTC),
    )
    matcher = TwoStageMatcher(
        store=store,
        collection="artist_portfolio_index",
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        stage1_limit=10,
    )
    return matcher.find_artists(chosen, k=5), target


def test_two_stage_matcher_returns_top_k_with_full_metadata() -> None:
    ranked, target = _ranked_results()
    assert len(ranked) == 5
    for r in ranked:
        assert r.artist_slug
        assert r.display_name
        assert r.portfolio_url.startswith("https://")
        assert -1.0 <= r.stage2_score <= 1.0


def test_two_stage_matcher_ranks_target_artist_first_when_chosen_design_matches() -> None:
    ranked, target = _ranked_results()
    # The chosen Candidate Design's image_uri equals the target artist's
    # portfolio_url, so Stage 2 (visual rerank) should put the target on top.
    assert ranked[0].artist_slug == target.artist_slug
