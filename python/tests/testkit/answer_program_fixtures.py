from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from fervis.lookup.answer_program import AnswerProgram, BindingSet, decode_answer_program
from fervis.lookup.relation_catalog import RelationCatalog

from tests.testkit.answer_program_contracts import binding_set_from_payload
from tests.testkit.catalog import catalog_from_payload


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "fixtures"
    / "answer_program_v1.yaml"
)


@dataclass(frozen=True)
class AnswerProgramFixture:
    program: AnswerProgram
    bindings: BindingSet
    catalog: RelationCatalog


def load_answer_program_fixture(
    *,
    program: str,
    binding_set: str,
    catalog: str,
) -> AnswerProgramFixture:
    payload = _fixture_payload()
    return AnswerProgramFixture(
        program=decode_answer_program(payload["programs"][program]),
        bindings=binding_set_from_payload(
            {"bindings": payload["binding_sets"][binding_set]}
        ),
        catalog=catalog_from_payload(payload["catalogs"][catalog]),
    )


@lru_cache(maxsize=1)
def _fixture_payload() -> dict[str, Any]:
    payload = yaml.safe_load(_FIXTURE_PATH.read_text())
    if not isinstance(payload, dict) or payload.get("schema_revision") != 1:
        raise ValueError("answer-program fixture revision is not supported")
    return payload
