"""Cooperative cancellation for long-running, Qt-free workflows."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

CancelCheck = Callable[[], bool]

_cancel_check: CancelCheck | None = None


class OperationCanceled(BaseException):
    """Raised at a cancellation checkpoint after the owner requested a stop.

    Derives from ``BaseException`` (like ``KeyboardInterrupt``) so the many
    ``except Exception`` fallbacks inside plotting and export code cannot
    swallow it, while ``finally`` blocks and transactions still roll back.
    """


@contextlib.contextmanager
def cancel_check_scope(check: CancelCheck | None) -> Iterator[None]:
    """Install ``check`` as the active cancellation predicate for this block."""
    global _cancel_check
    previous = _cancel_check
    _cancel_check = check
    try:
        yield
    finally:
        _cancel_check = previous


def cancel_requested() -> bool:
    check = _cancel_check
    return bool(check is not None and check())


def raise_if_canceled() -> None:
    """Raise ``OperationCanceled`` when the active predicate reports a stop request."""
    if cancel_requested():
        raise OperationCanceled("Operation canceled.")


__all__ = [
    "OperationCanceled",
    "cancel_check_scope",
    "cancel_requested",
    "raise_if_canceled",
]
