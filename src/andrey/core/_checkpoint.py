"""The private checkpoint sink of the identity harness (``qa/identity.py --checkpoints``)."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

#: The harness's recorder inside its block, called as ``take(stage, level, **pieces)``, or
#: ``None``. A search reads it once per stage and passes references to its own state; the recorder
#: serializes each piece at once, since the search rewrites its state in place.
TAKE: ContextVar[Callable[..., Any] | None] = ContextVar("andrey_checkpoint", default=None)
