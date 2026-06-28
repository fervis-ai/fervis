"""Host runtime output isolation."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from collections.abc import Iterator


@contextmanager
def suppress_host_output() -> Iterator[None]:
    """Keep host app import/startup output out of Fervis agent envelopes."""
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield
