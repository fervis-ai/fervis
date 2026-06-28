from __future__ import annotations

from typing import Any

import yaml

from tests.testkit.case_loader import ROOT


FIXTURE_ROOT = ROOT / "conformance" / "fixtures"


def load_conformance_fixture(group: str, fixture_id: str) -> dict[str, Any]:
    path = FIXTURE_ROOT / group / f"{fixture_id}.yaml"
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    if str(payload.get("id") or "") != fixture_id:
        raise ValueError(f"{path} fixture id must be {fixture_id!r}")
    return payload
