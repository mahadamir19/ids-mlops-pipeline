"""Lifecycle exception types."""

from __future__ import annotations


class LifecycleError(RuntimeError):
    """Raised when a Phase 4 lifecycle operation cannot be completed."""
