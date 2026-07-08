from __future__ import annotations

from pathlib import Path
import re

from fervis.lookup import clarification


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fervis"
CLARIFICATION_MODULE = SRC / "lookup" / "clarification"
CLARIFICATION_CONSTRUCTOR = re.compile(r"(?<![A-Za-z0-9_])Clarification\(")


def test_clarification_construction_is_private_to_capability_module() -> None:
    offenders = tuple(
        relative
        for path in SRC.rglob("*.py")
        if CLARIFICATION_MODULE not in path.parents
        for relative in (_relative(path),)
        if CLARIFICATION_CONSTRUCTOR.search(path.read_text())
    )

    assert offenders == ()


def test_obsolete_outcomes_clarification_module_is_removed() -> None:
    assert not (SRC / "lookup" / "outcomes" / "clarifications.py").exists()


def test_clarification_causes_do_not_expose_old_ref_bags() -> None:
    cause_types = (
        clarification.TargetReferenceNotFound,
        clarification.TargetReferenceAmbiguous,
        clarification.TargetReferenceUnsupported,
        clarification.MissingAnswerMetric,
        clarification.MissingCatalogChoice,
        clarification.MissingCatalogRequiredValue,
        clarification.AmbiguousQuestionInterpretation,
    )

    forbidden_fields = {"candidate_refs", "evidence_refs"}
    offenders = {
        cause_type.__name__: sorted(
            forbidden_fields & set(cause_type.__dataclass_fields__)
        )
        for cause_type in cause_types
        if forbidden_fields & set(cause_type.__dataclass_fields__)
    }

    assert offenders == {}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
