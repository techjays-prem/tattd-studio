"""Validate a per-artist permission marker against the schema.

Run this before opening a PR for slice #9 to catch missing fields:

    python infra/scripts/validate_permission.py data/lora_training/permission/<slug>.toml

Exits non-zero with a per-field reason on schema mismatch.
"""

from __future__ import annotations

import datetime as dt
import sys
import tomllib
from pathlib import Path

REQUIRED_SCHEMA: dict[str, dict[str, type]] = {
    "artist": {
        "slug": str,
        "display_name": str,
        "public_portfolio_url": str,
        "contact": str,
    },
    "permission": {
        "granted_on": dt.date,
        "granted_by": str,
        "record_method": str,
        "record_path": str,
        "scope": str,
        "revocation_terms": str,
    },
    "training_set": {
        "image_count": int,
        "attribution_required": bool,
        "credit_string": str,
    },
}

SCOPE_VALUES = {"lora_training_only", "full_use"}
RECORD_METHODS = {"signed_pdf", "signed_email", "recorded_video", "notarized_letter"}


def _check(table: dict, schema: dict[str, type], section: str) -> list[str]:
    errors: list[str] = []
    for field, expected_type in schema.items():
        if field not in table:
            errors.append(f"[{section}] missing required field: {field}")
            continue
        value = table[field]
        if expected_type is dt.date:
            if not isinstance(value, dt.date):
                errors.append(
                    f"[{section}] {field} must be a date (YYYY-MM-DD), got {type(value).__name__}"
                )
        elif not isinstance(value, expected_type):
            errors.append(
                f"[{section}] {field} must be {expected_type.__name__}, got {type(value).__name__}"
            )
        elif expected_type is str and not value.strip():
            errors.append(f"[{section}] {field} must be non-empty")
        elif expected_type is int and value <= 0:
            errors.append(f"[{section}] {field} must be positive (got {value})")
    return errors


def validate(path: Path) -> list[str]:
    if not path.exists():
        return [f"file does not exist: {path}"]
    with path.open("rb") as f:
        data = tomllib.load(f)

    errors: list[str] = []
    for section, schema in REQUIRED_SCHEMA.items():
        if section not in data:
            errors.append(f"missing required section: [{section}]")
            continue
        errors.extend(_check(data[section], schema, section))

    if "permission" in data:
        scope = data["permission"].get("scope")
        if scope and scope not in SCOPE_VALUES:
            errors.append(
                f"[permission] scope must be one of {sorted(SCOPE_VALUES)}, got {scope!r}"
            )
        method = data["permission"].get("record_method")
        if method and method not in RECORD_METHODS:
            errors.append(
                f"[permission] record_method must be one of "
                f"{sorted(RECORD_METHODS)}, got {method!r}"
            )

    if "training_set" in data:
        n = data["training_set"].get("image_count", 0)
        if isinstance(n, int) and not (20 <= n <= 35):
            errors.append(
                f"[training_set] image_count {n} is outside the spec target of 25–30 "
                f"(20–35 tolerated)"
            )

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python infra/scripts/validate_permission.py <path-to-toml>")
        return 2
    path = Path(argv[1])
    errors = validate(path)
    if errors:
        print(f"❌ {path} failed validation:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"✅ {path} passes the slice-#9 permission schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
