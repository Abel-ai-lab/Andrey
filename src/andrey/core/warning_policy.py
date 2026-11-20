"""Andrey's warning categories and the one frequency rule they share.

Every warning Andrey raises is an :class:`AndreyWarning`, so one filter silences or escalates all of
them; each category below it can be filtered on its own. A warning never changes a result.

Each distinct message warns once per process, pointing at the first line outside the package. A
message is recorded only after :func:`warnings.warn` returns, so an ``error`` filter fails every
call rather than only the first. A :class:`KnowledgeWarning` reports what one fit dropped, so it
warns for every fit that drops a claim (:func:`warn_each`).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import warnings
from collections.abc import Callable, Generator
from typing import TypeVar

_T = TypeVar("_T")

_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__)) + os.sep
_PACKAGE = __name__.partition(".")[0]

_warned: set[tuple[type[Warning], str]] = set()
#: The keys :func:`log_once` has logged in this process.
_logged: set[str] = set()
# Reentrant: a warning handler that warns again from the same thread must not deadlock.
_lock = threading.RLock()
#: Held for the whole of a :func:`held_records` block, which changes the one ``andrey`` logger.
_holding = threading.RLock()


class AndreyWarning(UserWarning):
    """Base class of every warning Andrey raises."""


class ExperimentalWarning(AndreyWarning):
    """A method without a published benchmark, whose API may change without deprecation."""


class PerformanceWarning(AndreyWarning):
    """A call that will run correctly but slowly.

    PC warns before a pass of ten million or more tests.
    """


class BackendFallbackWarning(AndreyWarning):
    """A requested compute backend is unavailable, so the call runs on another one."""


class SearchLimitWarning(AndreyWarning):
    """An iterative fit stopped at its iteration cap rather than on its own stopping rule, so a
    longer run might find a better graph.

    Every method with ``max_iter`` warns when it stops on the cap (:func:`warn_search_limit`), and
    a scikit-learn ``ConvergenceWarning`` from FastICA or LARS is raised again as this warning.
    """


class SampleSizeWarning(AndreyWarning):
    """A test runs below its minimum rows, so it may reject a true independence more often than
    ``alpha``.

    RCoT warns on a conditional query below 500 rows.
    """


class DataTypeWarning(AndreyWarning):
    """How a call reads its columns' types: inferred, missing, or possibly mistaken.

    A mixed-data test or score warns when it infers the categorical columns or finds none, and so
    runs as its linear-Gaussian form; every method that reads columns as numbers warns once when
    columns look like category codes.
    """


class BoundaryWarning(AndreyWarning):
    """A fit larger than the largest size the method finished within the caps of the published
    boundary lane, at the nearest measured setting.

    Each method warns once per process, whatever the size.
    """


class DeprecatedArgumentWarning(AndreyWarning):
    """A former argument name or value, which works for one more release as the current one, or a
    deprecated function or class, which also works for one more release.
    """


class KnowledgeWarning(AndreyWarning):
    """Background knowledge the graph a fit found cannot meet: a path claim the fit drops and lists
    in ``metadata["unmet_knowledge"]``. It warns once for each fit that drops a claim.
    """


def warn_once(message: str, category: type[AndreyWarning], *, key: str | None = None) -> None:
    """Warn with ``message`` the first time this process sees it, at the caller's line.

    With ``key``, the warning is seen once per ``key`` instead, whatever the message.
    """
    seen = (category, message if key is None else key)
    with _lock:
        if seen in _warned:
            return
        warnings.warn(message, category, stacklevel=_caller_stacklevel())
        _warned.add(seen)


def log_once(logger: logging.Logger, key: str, message: str) -> None:
    """Log ``message`` at warning level through ``logger`` the first time this process sees ``key``.

    For a notice about a default the fit chose, which changes no result and needs no filter: it
    goes through :mod:`logging`, so it reaches standard error unless the caller configures logging
    otherwise.
    """
    with _lock:
        if key in _logged:
            return
        _logged.add(key)
    logger.warning(message)


def mark_logged(key: str) -> None:
    """Record ``key`` as logged, so :func:`log_once` stays silent for it in this process.

    A worker process calls it for a notice that its parent logs, once for all of its workers.
    """
    with _lock:
        _logged.add(key)


class _Held(logging.Handler):
    """Holds the records it is handed, in order."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def held_records() -> Generator[list[logging.LogRecord]]:
    """Hold the warning-level records of the ``andrey`` loggers inside the block, in order.

    During the block the ``andrey`` logger holds only the handler that keeps them and does not
    propagate, so no record reaches standard error, through the root logger's handlers or an
    embedding application's either; its level is lowered to ``WARNING`` when it, or the logger it
    inherits from, is set above it, so the block keeps every warning-level record whatever the
    caller's threshold. The handlers, the propagation, and the level are restored after it. A
    block in another thread waits until this one ends, so each restores what it found; a block
    nested in this thread holds the records of its own span.
    """
    with _holding:
        handler = _Held()
        logger = logging.getLogger(_PACKAGE)
        handlers, propagate, level = logger.handlers[:], logger.propagate, logger.level
        held: list[logging.Handler] = [handler]
        logger.handlers, logger.propagate = held, False
        logger.setLevel(min(logger.getEffectiveLevel(), logging.WARNING))
        try:
            yield handler.records
        finally:
            logger.handlers, logger.propagate = handlers, propagate
            logger.setLevel(level)


def warn_each(message: str, category: type[AndreyWarning]) -> None:
    """Warn with ``message`` every time, at the caller's line, for a warning about one fit."""
    warnings.warn(message, category, stacklevel=_caller_stacklevel())


def warn_search_limit(method: str, parameter: str, value: int, unit: str) -> None:
    """Warn once that ``method`` stopped at ``parameter=value`` ``unit`` (its iteration cap)."""
    warn_once(
        f"{method} stopped at its limit of {parameter}={value} {unit}; a longer search may find a "
        f"better graph, so raise {parameter}",
        SearchLimitWarning,
    )


def convergence_as_search_limit(source: str, fit: Callable[[], _T]) -> _T:
    """Run ``fit``, one scikit-learn call, and return its result; raise each scikit-learn
    ``ConvergenceWarning`` it gives again as a :class:`SearchLimitWarning`, its text after
    ``source``. Other warnings pass through unchanged.

    One call only: :func:`warnings.catch_warnings` is not thread-safe.
    """
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = fit()
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            warn_once(f"{source}: {item.message}", SearchLimitWarning)
        else:
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return result


def _reset_after_fork() -> None:
    """Replace the inherited locks in a forked child.

    A child forked while another thread is warning, or holding records, inherits the lock held,
    with no thread left to release it. The warned messages are kept: the child does not repeat
    what the parent showed.
    """
    global _lock, _holding
    _lock = threading.RLock()
    _holding = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _caller_stacklevel() -> int:
    """The ``stacklevel``, counted from ``warn_once``, of the first frame outside the package.

    A frame is inside when its file is, or, for generated code such as a dataclass's ``__init__``,
    when its module is.
    """
    frame = sys._getframe(2)
    level = 2
    while frame is not None and (
        frame.f_code.co_filename.startswith(_PACKAGE_DIR)
        or frame.f_globals.get("__name__", "").partition(".")[0] == _PACKAGE
    ):
        frame = frame.f_back
        level += 1
    return level
