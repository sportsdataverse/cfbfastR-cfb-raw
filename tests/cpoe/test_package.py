"""Phase 1 Task 1.1 — cpoe package scaffold smoke tests."""
from __future__ import annotations


def test_package_imports():
    import cpoe  # noqa: F401


def test_version_present():
    import cpoe
    assert hasattr(cpoe, "__version__")


def test_version_is_string():
    import cpoe
    assert isinstance(cpoe.__version__, str)
    assert cpoe.__version__
