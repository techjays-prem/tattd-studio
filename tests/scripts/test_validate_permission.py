"""Tests for the slice-#9 HITL permission validator."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra" / "scripts"))

# Module under test lives under infra/scripts/, not on PYTHONPATH normally.
import validate_permission as vp  # noqa: E402


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "marker.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_valid_record_passes(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        [artist]
        slug = "kira-ink-bk"
        display_name = "Kira (real onboarded artist)"
        public_portfolio_url = "https://www.instagram.com/kira-ink-bk/"
        contact = "kira@example.com"

        [permission]
        granted_on = 2026-05-06
        granted_by = "Kira Inks Brooklyn LLC"
        record_method = "signed_pdf"
        record_path = "permission/kira-ink-bk-signed.pdf"
        scope = "lora_training_only"
        revocation_terms = "Artist may revoke with 30 days written notice."

        [training_set]
        image_count = 28
        attribution_required = true
        credit_string = "Trained with permission of Kira Inks Brooklyn LLC."
        """,
    )
    assert vp.validate(path) == []


def test_missing_section_is_flagged(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        [artist]
        slug = "x"
        display_name = "x"
        public_portfolio_url = "x"
        contact = "x"
        """,
    )
    errors = vp.validate(path)
    assert any("[permission]" in e for e in errors)
    assert any("[training_set]" in e for e in errors)


def test_invalid_scope_is_flagged(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        [artist]
        slug = "x"
        display_name = "x"
        public_portfolio_url = "x"
        contact = "x"

        [permission]
        granted_on = 2026-05-06
        granted_by = "x"
        record_method = "signed_pdf"
        record_path = "x"
        scope = "world_domination"
        revocation_terms = "x"

        [training_set]
        image_count = 28
        attribution_required = true
        credit_string = "x"
        """,
    )
    errors = vp.validate(path)
    assert any("scope" in e for e in errors)


def test_image_count_outside_target_is_flagged(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
        [artist]
        slug = "x"
        display_name = "x"
        public_portfolio_url = "x"
        contact = "x"

        [permission]
        granted_on = 2026-05-06
        granted_by = "x"
        record_method = "signed_pdf"
        record_path = "x"
        scope = "lora_training_only"
        revocation_terms = "x"

        [training_set]
        image_count = 5
        attribution_required = true
        credit_string = "x"
        """,
    )
    errors = vp.validate(path)
    assert any("image_count" in e for e in errors)


def test_missing_file_returns_error(tmp_path: Path) -> None:
    errors = vp.validate(tmp_path / "does-not-exist.toml")
    assert any("does not exist" in e for e in errors)
