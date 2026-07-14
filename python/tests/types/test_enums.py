from enum import auto

from fervis.types.enums import StrEnum


class _State(StrEnum):
    READY = "ready"
    RUNNING = auto()


def test_string_enum_members_are_string_values() -> None:
    assert _State.READY == "ready"
    assert str(_State.READY) == "ready"
    assert _State.RUNNING.value == "running"
