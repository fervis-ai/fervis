from __future__ import annotations

import importlib.util


def test_private_goldsets_are_not_public_package_surface() -> None:
    assert importlib.util.find_spec("fervis.evaluation.goldsets") is not None
    assert importlib.util.find_spec("fervis.evaluation.goldsets.retail_ops") is None
    assert importlib.util.find_spec("fervis.evaluation.goldsets.catalog") is None
