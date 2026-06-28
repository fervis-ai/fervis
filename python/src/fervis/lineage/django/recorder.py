"""Django adapter for the framework-neutral lineage recorder."""

from __future__ import annotations

from fervis.lineage.recorder_core import LineageRecorder
from fervis.lineage.django.store import DjangoLineageRecorderStore


class DjangoLineageRecorder(LineageRecorder):
    def __init__(self) -> None:
        super().__init__(DjangoLineageRecorderStore())
