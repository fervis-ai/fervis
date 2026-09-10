from __future__ import annotations

import subprocess
import sys


def test_lookup_public_packages_cold_import_without_order_dependence() -> None:
    for imports in (
        (
            "import fervis.lookup.relation_catalog.row_sources\n"
            "import fervis.lookup.answer_program"
        ),
        (
            "import fervis.lookup.answer_program\n"
            "import fervis.lookup.relation_catalog.row_sources"
        ),
        (
            "import fervis.lookup.source_binding\n"
            "import fervis.lookup.answer_program\n"
            "import fervis.lookup.question_contract"
        ),
        (
            "import fervis.lookup.question_contract\n"
            "import fervis.lookup.source_binding\n"
            "import fervis.lookup.answer_program"
        ),
        (
            "import fervis.lookup.contract_codec\n"
            "import fervis.lookup.answer_program"
        ),
    ):
        completed = subprocess.run(
            [sys.executable, "-c", imports],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_deleted_fact_planning_package_has_no_import_path() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import fervis.lookup.fact_plan"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "No module named 'fervis.lookup.fact_plan'" in completed.stderr
