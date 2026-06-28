from __future__ import annotations

import pytest

from tests.testkit.case_loader import load_all_conformance_cases
from tests.testkit.runner import run_case


CASES = load_all_conformance_cases()


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_conformance(case):
    assert run_case(case) == []
