from __future__ import annotations

from typing import Any


def subset_mismatches(
    *,
    actual: dict[str, Any],
    expected_subset: dict[str, Any],
    path: str = "",
) -> list[str]:
    errors: list[str] = []
    for key, expected_value in expected_subset.items():
        item_path = f"{path}.{key}" if path else str(key)
        if key not in actual:
            errors.append(f"{item_path}: missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                errors.append(
                    f"{item_path}: expected object, got {type(actual_value).__name__}"
                )
                continue
            errors.extend(
                subset_mismatches(
                    actual=actual_value,
                    expected_subset=expected_value,
                    path=item_path,
                )
            )
            continue
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list):
                errors.append(
                    f"{item_path}: expected array, got {type(actual_value).__name__}"
                )
                continue
            if not expected_value and actual_value:
                errors.append(f"{item_path}: expected empty array, got {actual_value!r}")
                continue
            if len(actual_value) < len(expected_value):
                errors.append(
                    f"{item_path}: expected at least {len(expected_value)} items, "
                    f"got {len(actual_value)}"
                )
                continue
            for index, expected_item in enumerate(expected_value):
                item_path_with_index = f"{item_path}[{index}]"
                actual_item = actual_value[index]
                if isinstance(expected_item, dict):
                    if not isinstance(actual_item, dict):
                        errors.append(
                            f"{item_path_with_index}: expected object, got "
                            f"{type(actual_item).__name__}"
                        )
                        continue
                    errors.extend(
                        subset_mismatches(
                            actual=actual_item,
                            expected_subset=expected_item,
                            path=item_path_with_index,
                        )
                    )
                    continue
                if actual_item != expected_item:
                    errors.append(
                        f"{item_path_with_index}: expected {expected_item!r}, "
                        f"got {actual_item!r}"
                    )
            continue
        if actual_value != expected_value:
            errors.append(
                f"{item_path}: expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def exact_mismatches(
    *,
    actual: Any,
    expected: Any,
    path: str = "",
) -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path or '<root>'}: expected object, got {type(actual).__name__}"]
        errors: list[str] = []
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(expected_keys - actual_keys):
            item_path = f"{path}.{key}" if path else str(key)
            errors.append(f"{item_path}: missing")
        for key in sorted(actual_keys - expected_keys):
            item_path = f"{path}.{key}" if path else str(key)
            errors.append(f"{item_path}: unexpected")
        for key in sorted(actual_keys & expected_keys):
            item_path = f"{path}.{key}" if path else str(key)
            errors.extend(
                exact_mismatches(
                    actual=actual[key],
                    expected=expected[key],
                    path=item_path,
                )
            )
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path or '<root>'}: expected array, got {type(actual).__name__}"]
        errors = []
        if len(actual) != len(expected):
            errors.append(
                f"{path or '<root>'}: expected {len(expected)} items, got {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            errors.extend(
                exact_mismatches(
                    actual=actual_item,
                    expected=expected_item,
                    path=f"{path}[{index}]",
                )
            )
        return errors
    if actual != expected:
        return [f"{path or '<root>'}: expected {expected!r}, got {actual!r}"]
    return []
