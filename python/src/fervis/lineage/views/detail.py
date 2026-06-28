"""Shared lineage view detail policy."""

from __future__ import annotations

from enum import StrEnum

from fervis.lineage.step_summary import StepSummaryDetail


class LineageRenderDetail(StrEnum):
    COMPACT = "compact"
    VERBOSE = "verbose"
    DEBUG = "debug"

    def includes_verbose(self) -> bool:
        return self in {self.VERBOSE, self.DEBUG}

    def includes_debug(self) -> bool:
        return self is self.DEBUG


def include_step_decision(
    decision_detail: StepSummaryDetail, requested_detail: LineageRenderDetail
) -> bool:
    if decision_detail is StepSummaryDetail.DEBUG:
        return requested_detail.includes_debug()
    if decision_detail is StepSummaryDetail.VERBOSE:
        return requested_detail.includes_verbose()
    return True
