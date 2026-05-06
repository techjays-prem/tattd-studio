"""Smoke test: the `tattd_studio` package is importable and exposes a version.

This is the tracer bullet for slice #1: it proves the package is wired up,
the test runner discovers it, and CI can execute the suite end-to-end.
"""

import re

import tattd_studio


def test_package_exposes_semver_version() -> None:
    assert isinstance(tattd_studio.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", tattd_studio.__version__), tattd_studio.__version__
