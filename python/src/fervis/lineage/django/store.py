"""Thin Django ORM store for lineage records."""

from __future__ import annotations

from contextlib import AbstractContextManager
from django.core.exceptions import ObjectDoesNotExist
from django.db import IntegrityError, models as django_models, transaction

from fervis.lineage import models
from fervis.lineage import records
from fervis.lineage.records import LineageRow

_MODEL_BY_RECORD_KEY: dict[str, type[django_models.Model]] = {}


class DjangoLineageRecorderStore:
    def transaction(self) -> AbstractContextManager[object]:
        return transaction.atomic()

    def get_or_insert_row(self, row: LineageRow) -> LineageRow:
        model = _model_for(row.key)
        try:
            record, _ = model._default_manager.get_or_create(
                **row.identity,
                defaults=row.defaults,
            )
        except IntegrityError as exc:
            record = self._get_existing_after_conflict(row, exc)
        return _stored_row(row, record)

    def insert_row(self, row: LineageRow) -> None:
        _model_for(row.key)._default_manager.create(**row.values)

    def find_row(
        self,
        *,
        key: str,
        lookup: dict[str, object],
        fields: tuple[str, ...],
    ) -> LineageRow | None:
        try:
            record = _model_for(key)._default_manager.get(**lookup)
        except ObjectDoesNotExist:
            return None
        return _stored_row_for_lookup(
            key=key, record=record, lookup=lookup, fields=fields
        )

    def _get_existing_after_conflict(
        self,
        row: LineageRow,
        original_error: IntegrityError,
    ) -> django_models.Model:
        model = _model_for(row.key)
        try:
            return model._default_manager.get(**row.identity)
        except ObjectDoesNotExist:
            pass
        if row.conflict_lookup:
            try:
                return model._default_manager.get(**row.conflict_lookup)
            except ObjectDoesNotExist:
                pass
        raise original_error


def _stored_row(row: LineageRow, record: django_models.Model) -> LineageRow:
    values = {field: getattr(record, field) for field in row.storage_fields}
    return LineageRow(
        key=row.key,
        identity={field: getattr(record, field) for field in row.identity},
        values=values,
        conflict_lookup=row.conflict_lookup,
        same_run_refs=row.same_run_refs,
    )


def _stored_row_for_lookup(
    *,
    key: str,
    record: django_models.Model,
    lookup: dict[str, object],
    fields: tuple[str, ...],
) -> LineageRow:
    values = {field: getattr(record, field) for field in fields}
    return LineageRow(
        key=key,
        identity={field: getattr(record, field) for field in lookup},
        values=values,
        conflict_lookup={},
    )


def _model_for(key: str) -> type[django_models.Model]:
    return lineage_model_by_record_key()[key]


def lineage_model_by_record_key() -> dict[str, type[django_models.Model]]:
    if not _MODEL_BY_RECORD_KEY:
        _MODEL_BY_RECORD_KEY.update(_discover_lineage_models())
    return dict(_MODEL_BY_RECORD_KEY)


def _discover_lineage_models() -> dict[str, type[django_models.Model]]:
    output: dict[str, type[django_models.Model]] = {}
    for model in models.__dict__.values():
        if not (
            isinstance(model, type)
            and issubclass(model, django_models.Model)
            and hasattr(model, "lineage_record_key")
        ):
            continue
        key = str(model.lineage_record_key)
        if key in output:
            raise RuntimeError(f"multiple Django models declare lineage record {key!r}")
        output[key] = model
    expected = set(records.RECORD_SPECS_BY_KEY)
    actual = set(output)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Django lineage model registry mismatch; missing={missing}, extra={extra}"
        )
    return output
