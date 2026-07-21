"""Concrete fact-planning schema registry."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.fact_planning.schema_helpers import (
    handle_schema as _handle_schema,
)
from fervis.lookup.fact_planning.fact_planning_family_schema import (
    SourceBoundPatternSchemaContext,
    source_bound_pattern_variant as _source_bound_pattern_variant,
)
from fervis.lookup.fact_planning.plan_shapes import SOURCE_BOUND_PLAN_SHAPES
from fervis.lookup.operation_families.computed_scalar.fact_planning import (
    COMPUTED_SCALAR_PATTERN_NAMES as _COMPUTED_SCALAR_PATTERN_NAMES,
    computed_scalar_pattern_answer_variants as _computed_scalar_pattern_answer_variants,
)
from fervis.lookup.operation_families.grouped_aggregate.fact_planning import (
    grouped_aggregate_pattern_answer_variants as _grouped_aggregate_pattern_answer_variants,
)
from fervis.lookup.operation_families.joined_rows.fact_planning import (
    JOINED_ROWS_PATTERN_NAMES as _JOINED_ROWS_PATTERN_NAMES,
    joined_rows_generic_pattern_answer_variants as _joined_rows_generic_pattern_answer_variants,
    joined_rows_pattern_answer_variants as _joined_rows_pattern_answer_variants,
)
from fervis.lookup.operation_families.list_rows.fact_planning import (
    SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS as _LIST_ROWS_PATTERN_SCHEMA_BUILDERS,
)
from fervis.lookup.operation_families.scalar_aggregate.fact_planning import (
    SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS as _SCALAR_AGGREGATE_PATTERN_SCHEMA_BUILDERS,
    scalar_aggregate_pattern_answer_variants as _scalar_aggregate_pattern_answer_variants,
)
from fervis.lookup.operation_families.scalar_value.fact_planning import (
    SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS as _SCALAR_VALUE_PATTERN_SCHEMA_BUILDERS,
)
from fervis.lookup.operation_families.set_difference.fact_planning import (
    SET_DIFFERENCE_PATTERN_NAMES as _SET_DIFFERENCE_PATTERN_NAMES,
    set_difference_generic_pattern_answer_variants as _set_difference_generic_pattern_answer_variants,
    set_difference_pattern_answer_variants as _set_difference_pattern_answer_variants,
)


def build_pattern_answer_schema(
    *,
    pattern_names: tuple[str, ...],
    require_pattern: bool,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None = None,
    identity_field_ids_by_source_binding_id: (
        Mapping[str, tuple[str, ...]] | None
    ) = None,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    grouped_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    ordering_required_by_requested_fact_id: Mapping[str, bool],
    value_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> dict[str, object] | None:
    variants = _selected_pattern_answer_variants(
        selected_plan_shapes_by_requested_fact_id=(
            selected_plan_shapes_by_requested_fact_id
        ),
        source_binding_ids_by_requested_fact_id=(
            source_binding_ids_by_requested_fact_id
        ),
        answer_output_ids_by_requested_fact_id=answer_output_ids_by_requested_fact_id,
        answer_output_ids_by_source_binding_id=answer_output_ids_by_source_binding_id,
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
        grouped_aggregate_choices_by_requested_fact_id=(
            grouped_aggregate_choices_by_requested_fact_id
        ),
        scalar_aggregate_choices_by_requested_fact_id=(
            scalar_aggregate_choices_by_requested_fact_id
        ),
        require_pattern=require_pattern,
        field_ids_by_source_binding_id=field_ids_by_source_binding_id or {},
        identity_field_ids_by_source_binding_id=(
            identity_field_ids_by_source_binding_id or {}
        ),
        ordering_required_by_requested_fact_id=(ordering_required_by_requested_fact_id),
        value_ids_by_requested_fact_id=value_ids_by_requested_fact_id,
    )
    if len(variants) == 1:
        return variants[0]
    if not variants:
        return None
    return {"oneOf": variants}


_SOURCE_BOUND_PATTERN_NAMES = frozenset(SOURCE_BOUND_PLAN_SHAPES)

_MULTI_RELATION_PATTERN_NAMES = frozenset(
    {*_JOINED_ROWS_PATTERN_NAMES, *_SET_DIFFERENCE_PATTERN_NAMES}
)
_COMPUTED_PATTERN_NAMES = _COMPUTED_SCALAR_PATTERN_NAMES

_SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS = {
    **_LIST_ROWS_PATTERN_SCHEMA_BUILDERS,
    **_SCALAR_VALUE_PATTERN_SCHEMA_BUILDERS,
    **_SCALAR_AGGREGATE_PATTERN_SCHEMA_BUILDERS,
}


def _multi_relation_pattern_answer_variants(
    *,
    plan_shape: str,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids_schema: dict[str, object] | None,
    source_binding_ids: tuple[str, ...],
    source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    require_pattern: bool,
) -> list[dict[str, object]]:
    if plan_shape == "joined_rows":
        return _joined_rows_pattern_answer_variants(
            requested_fact_id_schema=requested_fact_id_schema,
            answer_output_ids_schema=answer_output_ids_schema,
            source_binding_ids=source_binding_ids,
            source_binding_ids_by_requirement=source_binding_ids_by_requirement,
            field_ids_by_source_binding_id=field_ids_by_source_binding_id,
            require_pattern=require_pattern,
        )
    if plan_shape == "set_difference":
        return _set_difference_pattern_answer_variants(
            requested_fact_id_schema=requested_fact_id_schema,
            answer_output_ids_schema=answer_output_ids_schema,
            source_binding_ids=source_binding_ids,
            source_binding_ids_by_requirement=source_binding_ids_by_requirement,
            identity_field_ids_by_source_binding_id=(
                identity_field_ids_by_source_binding_id
            ),
            require_pattern=require_pattern,
        )
    raise ValueError(f"unsupported multi-relation plan shape: {plan_shape}")


def _pattern_answer_variants(
    *,
    pattern_names: tuple[str, ...],
    require_pattern: bool,
    requested_fact_id_schema: dict[str, object] | None = None,
    answer_output_ids_schema: dict[str, object] | None = None,
    source_binding_id_schema: dict[str, object],
    source_binding_id: str | None,
    field_ids: tuple[str, ...] | None,
    include_source_binding_id: bool = True,
    ordering_required: bool = False,
    value_ids: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    if not pattern_names:
        return []
    variants: list[dict[str, object]] = []
    source_bound_pattern_names = tuple(
        name for name in pattern_names if name in _SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS
    )
    if source_bound_pattern_names:
        context = SourceBoundPatternSchemaContext(
            require_pattern=require_pattern,
            requested_fact_id_schema=requested_fact_id_schema,
            answer_output_ids_schema=answer_output_ids_schema,
            source_binding_id_schema=source_binding_id_schema,
            source_binding_id=source_binding_id,
            include_source_binding_id=include_source_binding_id,
            field_ids=field_ids,
            ordering_required=ordering_required,
        )
        for name in source_bound_pattern_names:
            builder = _SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS.get(name)
            if builder is None:
                continue
            variants.append(_source_bound_pattern_variant(context, builder(context)))
    other_pattern_names = tuple(
        name
        for name in pattern_names
        if name not in _SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS
    )
    if not other_pattern_names:
        return variants
    if _COMPUTED_PATTERN_NAMES.intersection(other_pattern_names):
        variants.extend(
            _computed_scalar_pattern_answer_variants(
                requested_fact_id_schema=requested_fact_id_schema,
                require_pattern=require_pattern,
                value_ids=value_ids,
            )
        )
    if _SET_DIFFERENCE_PATTERN_NAMES.intersection(other_pattern_names):
        variants.extend(
            _set_difference_generic_pattern_answer_variants(
                requested_fact_id_schema=requested_fact_id_schema,
                require_pattern=require_pattern,
            )
        )
    if _JOINED_ROWS_PATTERN_NAMES.intersection(other_pattern_names):
        variants.extend(
            _joined_rows_generic_pattern_answer_variants(
                requested_fact_id_schema=requested_fact_id_schema,
                require_pattern=require_pattern,
            )
        )
    return variants


def _selected_pattern_answer_variants(
    *,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str,
        Mapping[str, tuple[str, ...]],
    ],
    grouped_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    require_pattern: bool,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    ordering_required_by_requested_fact_id: Mapping[str, bool],
    value_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for (
        requested_fact_id,
        plan_shapes,
    ) in selected_plan_shapes_by_requested_fact_id.items():
        requested_fact_id_schema: dict[str, object] = {"enum": [requested_fact_id]}
        answer_output_ids = answer_output_ids_by_requested_fact_id.get(
            requested_fact_id,
            (),
        )
        answer_output_ids_schema = _answer_output_ids_array_schema(answer_output_ids)
        fact_requires_pattern = require_pattern or len(plan_shapes) > 1
        for plan_shape in plan_shapes:
            variants.extend(
                _selected_plan_shape_answer_variants(
                    plan_shape=plan_shape,
                    requested_fact_id=requested_fact_id,
                    requested_fact_id_schema=requested_fact_id_schema,
                    answer_output_ids=answer_output_ids,
                    answer_output_ids_schema=answer_output_ids_schema,
                    answer_output_ids_by_source_binding_id=(
                        answer_output_ids_by_source_binding_id
                    ),
                    source_binding_ids_by_requested_fact_id=(
                        source_binding_ids_by_requested_fact_id
                    ),
                    source_binding_ids_by_requirement_by_requested_fact_id=(
                        source_binding_ids_by_requirement_by_requested_fact_id
                    ),
                    grouped_aggregate_choices_by_requested_fact_id=(
                        grouped_aggregate_choices_by_requested_fact_id
                    ),
                    scalar_aggregate_choices_by_requested_fact_id=(
                        scalar_aggregate_choices_by_requested_fact_id
                    ),
                    require_pattern=fact_requires_pattern,
                    field_ids_by_source_binding_id=field_ids_by_source_binding_id,
                    identity_field_ids_by_source_binding_id=(
                        identity_field_ids_by_source_binding_id
                    ),
                    ordering_required=ordering_required_by_requested_fact_id.get(
                        requested_fact_id, False
                    ),
                    value_ids=value_ids_by_requested_fact_id.get(requested_fact_id, ()),
                )
            )
    return variants


def _selected_plan_shape_answer_variants(
    *,
    plan_shape: str,
    requested_fact_id: str,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids: tuple[str, ...],
    answer_output_ids_schema: dict[str, object] | None,
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str,
        Mapping[str, tuple[str, ...]],
    ],
    grouped_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    require_pattern: bool,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    ordering_required: bool,
    value_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    if plan_shape == "aggregate_by_group":
        choices = tuple(
            {**choice, "ordering_required": ordering_required}
            for choice in grouped_aggregate_choices_by_requested_fact_id.get(
                requested_fact_id,
                (),
            )
        )
        variants.extend(
            _grouped_aggregate_pattern_answer_variants(
                plan_shape=plan_shape,
                requested_fact_id_schema=requested_fact_id_schema,
                choices=choices,
                require_pattern=require_pattern,
            )
        )
        return variants
    if plan_shape == "aggregate_scalar":
        variants.extend(
            _scalar_aggregate_pattern_answer_variants(
                requested_fact_id_schema=requested_fact_id_schema,
                choices=scalar_aggregate_choices_by_requested_fact_id.get(
                    requested_fact_id,
                    (),
                ),
                require_pattern=require_pattern,
            )
        )
        return variants
    if plan_shape in _MULTI_RELATION_PATTERN_NAMES:
        variants.extend(
            _multi_relation_pattern_answer_variants(
                plan_shape=plan_shape,
                requested_fact_id_schema=requested_fact_id_schema,
                answer_output_ids_schema=answer_output_ids_schema,
                source_binding_ids=source_binding_ids_by_requested_fact_id.get(
                    requested_fact_id,
                    (),
                ),
                source_binding_ids_by_requirement=(
                    source_binding_ids_by_requirement_by_requested_fact_id.get(
                        requested_fact_id,
                        {},
                    )
                ),
                field_ids_by_source_binding_id=field_ids_by_source_binding_id,
                identity_field_ids_by_source_binding_id=(
                    identity_field_ids_by_source_binding_id
                ),
                require_pattern=require_pattern,
            )
        )
        return variants
    if plan_shape in _SOURCE_BOUND_PATTERN_NAMES:
        source_binding_ids = source_binding_ids_by_requested_fact_id.get(
            requested_fact_id,
            (),
        )
        for source_binding_id in source_binding_ids:
            field_ids = field_ids_by_source_binding_id.get(source_binding_id)
            source_answer_output_ids = (
                answer_output_ids_by_source_binding_id.get(source_binding_id)
                or answer_output_ids
            )
            source_answer_output_ids_schema = _answer_output_ids_array_schema(
                source_answer_output_ids
            )
            variants.extend(
                _pattern_answer_variants(
                    pattern_names=(plan_shape,),
                    require_pattern=require_pattern,
                    requested_fact_id_schema=requested_fact_id_schema,
                    answer_output_ids_schema=source_answer_output_ids_schema,
                    source_binding_id_schema={"enum": [source_binding_id]},
                    source_binding_id=source_binding_id,
                    field_ids=field_ids,
                    ordering_required=ordering_required,
                )
            )
        return variants
    variants.extend(
        _pattern_answer_variants(
            pattern_names=(plan_shape,),
            require_pattern=require_pattern,
            requested_fact_id_schema=requested_fact_id_schema,
            answer_output_ids_schema=answer_output_ids_schema,
            source_binding_id_schema=_handle_schema(),
            source_binding_id=None,
            field_ids=None,
            ordering_required=ordering_required,
            value_ids=value_ids,
        )
    )
    return variants


def _answer_output_ids_array_schema(
    answer_output_ids: tuple[str, ...],
) -> dict[str, object] | None:
    if not answer_output_ids:
        return None
    return {
        "type": "array",
        "items": {
            "type": "string",
            "enum": list(answer_output_ids),
        },
        "minItems": len(answer_output_ids),
        "maxItems": len(answer_output_ids),
    }
