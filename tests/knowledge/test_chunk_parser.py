"""Knowledge Corpus markdown parsing.

Each area's chunks live in one markdown file separated by `<!-- CHUNK -->`
markers. Each chunk has its own YAML frontmatter (Provenance) and body.
"""

from __future__ import annotations

import textwrap

from tattd_studio.knowledge import Chunk, parse_chunks_from_markdown


def test_parse_chunks_extracts_provenance_and_body() -> None:
    text = textwrap.dedent(
        """\
        <!-- CHUNK -->
        ---
        chunk_id: taxonomy-fineline
        area: taxonomy
        title: Fineline (style)
        source_url: https://example.com/styles/fineline
        curator: tattd-studio-dev
        capture_date: 2026-05-06
        synthetic: true
        permission: synthetic-content-tattd-studio-poc
        ---

        Fineline tattooing emphasizes single-needle, minimal-weight linework.

        <!-- CHUNK -->
        ---
        chunk_id: taxonomy-traditional
        area: taxonomy
        title: American Traditional
        source_url: https://example.com/styles/traditional
        curator: tattd-studio-dev
        capture_date: 2026-05-06
        synthetic: true
        permission: synthetic-content-tattd-studio-poc
        ---

        American Traditional uses bold black outlines and a limited palette.
        """
    )

    chunks = parse_chunks_from_markdown(text)

    assert len(chunks) == 2
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].chunk_id == "taxonomy-fineline"
    assert chunks[0].area == "taxonomy"
    assert chunks[0].source_url == "https://example.com/styles/fineline"
    assert chunks[0].synthetic is True
    assert "single-needle" in chunks[0].body
    assert chunks[1].chunk_id == "taxonomy-traditional"
    assert "bold black outlines" in chunks[1].body


def test_parse_chunks_ignores_leading_whitespace_before_first_marker() -> None:
    text = textwrap.dedent(
        """\



        <!-- CHUNK -->
        ---
        chunk_id: t1
        area: ip
        title: t
        source_url: x
        curator: y
        capture_date: 2026-05-06
        synthetic: true
        permission: p
        ---
        body
        """
    )
    chunks = parse_chunks_from_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "t1"
