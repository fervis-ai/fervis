from __future__ import annotations

import subprocess
import sys


def test_lookup_public_packages_cold_import_without_order_dependence() -> None:
    for imports in (
        (
            "import fervis.lookup.fact_plan.row_sources\n"
            "import fervis.lookup.answer_program"
        ),
        (
            "import fervis.lookup.answer_program\n"
            "import fervis.lookup.fact_plan.row_sources"
        ),
        (
            "import fervis.lookup.source_binding\n"
            "import fervis.lookup.answer_program\n"
            "import fervis.lookup.fact_planning.required_inputs"
        ),
        (
            "import fervis.lookup.fact_planning.required_inputs\n"
            "import fervis.lookup.source_binding\n"
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
