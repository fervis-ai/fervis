"""Load host-owned goldset suites from a filesystem path."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
from pathlib import Path
import sys
from types import ModuleType

from .contracts import GoldsetSuite


SUITE_MODULE = "fervis_goldset.py"


def load_goldset_suite(suite_path: Path | str) -> GoldsetSuite:
    suite_ref = str(suite_path).strip()
    if ":" in suite_ref and not _looks_like_path(suite_ref):
        return _load_import_suite(suite_ref)
    return _load_path_suite(suite_ref)


def _load_path_suite(suite_ref: str) -> GoldsetSuite:
    path = Path(suite_ref).expanduser().resolve()
    module_path = path if path.is_file() else path / SUITE_MODULE
    if not module_path.exists():
        raise ValueError(f"goldset suite entrypoint not found: {module_path}")

    module_name = _path_module_name(module_path)
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return _suite_from_module(loaded)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"goldset suite cannot be loaded: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass

    return _suite_from_module(module)


def _load_import_suite(suite_ref: str) -> GoldsetSuite:
    module_name, object_name = suite_ref.split(":", 1)
    if not module_name or not object_name:
        raise ValueError(f"goldset suite import path is invalid: {suite_ref}")
    module = importlib.import_module(module_name)
    value = module
    for part in object_name.split("."):
        value = getattr(value, part)
    return _suite_from_loader(value)


def _suite_from_loader(load_suite: object) -> GoldsetSuite:
    if load_suite is None or not callable(load_suite):
        raise ValueError("goldset suite must define callable load_suite()")
    suite = load_suite()
    if not isinstance(suite, GoldsetSuite):
        raise ValueError("load_suite() must return GoldsetSuite")
    return suite


def _suite_from_module(module: ModuleType) -> GoldsetSuite:
    return _suite_from_loader(getattr(module, "load_suite", None))


def _path_module_name(module_path: Path) -> str:
    digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:20]
    return f"_fervis_goldset_{digest}"


def _looks_like_path(value: str) -> bool:
    path = Path(value).expanduser()
    return path.exists() or "/" in value or "\\" in value or value.endswith(".py")
