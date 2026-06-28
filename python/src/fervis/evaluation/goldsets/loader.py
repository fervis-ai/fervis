"""Load host-owned goldset suites from a filesystem path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from .contracts import GoldsetSuite


SUITE_MODULE = "fervis_goldset.py"


def load_goldset_suite(suite_path: Path | str) -> GoldsetSuite:
    path = Path(suite_path).expanduser().resolve()
    module_path = path if path.is_file() else path / SUITE_MODULE
    if not module_path.exists():
        raise ValueError(f"goldset suite entrypoint not found: {module_path}")

    module_name = f"_fervis_goldset_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"goldset suite cannot be loaded: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass

    load_suite = getattr(module, "load_suite", None)
    if load_suite is None or not callable(load_suite):
        raise ValueError("goldset suite must define callable load_suite()")
    suite = load_suite()
    if not isinstance(suite, GoldsetSuite):
        raise ValueError("load_suite() must return GoldsetSuite")
    return suite
