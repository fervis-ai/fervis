from decimal import Decimal
import json

import pytest

from fervis.memory.addresses import (
    FactAddress,
    FactAddressKind,
    FactAddressValue,
    fact_address_from_payload,
)
from fervis.memory.artifacts import FactOutcome


def test_fact_address_rejects_incomplete_public_variant():
    with pytest.raises(ValueError, match="entity fact address requires"):
        FactAddress(
            address="entity.staff.missing_identity",
            kind=FactAddressKind.ENTITY,
            resource="staff",
        )


def test_fact_address_rejects_untyped_relation_source():
    with pytest.raises(ValueError, match="valid source.kind"):
        FactAddress.relation(
            address="relation.unknown",
            source={"kind": "untyped", "endpointName": "list_unknown"},
        )


def test_fact_address_rejects_non_terminal_outcome_address():
    with pytest.raises(ValueError, match="terminal outcome"):
        FactAddress.outcome(
            address="outcome.answered",
            terminal=FactOutcome.ANSWERED.value,
        )


def test_row_fact_address_round_trips_decimal_without_losing_type():
    address = FactAddress.row(
        address="row.sales.1",
        relation="relation.sales",
        values={"total": FactAddressValue(type="decimal", value=Decimal("10.50"))},
    )

    payload = json.loads(json.dumps(address.to_dict()))
    restored = fact_address_from_payload(payload)

    assert restored.values["total"].value == Decimal("10.50")


def test_scalar_fact_address_round_trips_decimal_without_losing_type():
    address = FactAddress.value(
        address="value.total",
        value={"type": "decimal", "value": Decimal("10.50")},
    )

    payload = json.loads(json.dumps(address.to_dict()))
    restored = fact_address_from_payload(payload)

    assert restored.scalar_value["value"] == Decimal("10.50")
