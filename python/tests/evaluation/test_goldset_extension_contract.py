from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite
from fervis.evaluation.goldsets.loader import load_goldset_suite


def test_goldset_runner_imports_without_loading_cli_dispatch() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import fervis.evaluation.goldsets.runner"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_path_suite_supports_postponed_annotation_dataclasses(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "fervis_goldset.py"
    suite_path.write_text(
        '''from __future__ import annotations
from dataclasses import dataclass
from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite

@dataclass
class SuiteMetadata:
    label: str

METADATA = SuiteMetadata(label="portable")

def load_suite():
    case = GoldsetCase(case_id="case_1", question="How many records?")
    return GoldsetSuite(
        name=METADATA.label,
        cases=(case,),
        match_answer=lambda case, result: GoldsetMatch(True),
    )
'''
    )

    suite = load_goldset_suite(suite_path)

    assert suite.name == "portable"
    assert suite.cases[0].case_id == "case_1"


def test_goldset_suite_rejects_duplicate_case_ids() -> None:
    cases = (
        GoldsetCase(case_id="same", question="Question one?"),
        GoldsetCase(case_id="same", question="Question two?"),
    )

    with pytest.raises(ValueError, match="case ids must be unique"):
        GoldsetSuite(
            name="duplicates",
            cases=cases,
            match_answer=lambda case, result: GoldsetMatch(True),
        )
