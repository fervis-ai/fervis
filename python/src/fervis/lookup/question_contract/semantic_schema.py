"""Strict provider schema for the semantic relational Question Contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import fervis.lookup.question_contract.clarification_provider_contract as clarification_output
import fervis.lookup.question_contract.semantic_provider_contract as output
from fervis.lookup.question_contract.semantic_model import InputDenotationKind
from fervis.lookup.semantic_types import (
    DECIMAL_OPERAND_PATTERN,
    INTEGER_OPERAND_PATTERN,
    CollectionType,
    TemporalScopeType,
)

if TYPE_CHECKING:
    from fervis.lookup.question_contract.semantic_parser import (
        ParsedSemanticQuestionMeaning,
    )


def build_semantic_question_frame_schema(
    *, conversation_input_refs: tuple[str, ...] = ()
) -> dict[str, object]:
    schema = output.SemanticQuestionFrameDecisionOutput.schema(
        {
            "decision_basis": {"type": "string", "minLength": 1},
            "outcome": {
                "oneOf": [
                    output.CompleteSemanticQuestionFrameOutput.schema(
                        {
                            "kind": {"enum": ["question_meaning"]},
                            "answer_requests": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": _answer_request_frame_schema(),
                            },
                            "supplied_values": {
                                "type": "array",
                                "items": _supplied_value_schema(),
                            },
                        }
                    ),
                    _missing_requested_fact_schema(),
                    _unresolved_prior_turn_references_schema(),
                ]
            },
        }
    )
    schema["$defs"] = {
        "frame_origin": _frame_origin_definition(
            conversation_input_refs=conversation_input_refs
        ),
        "source_origin": _meaning_origin_definition(),
        "supplied_input_value": _supplied_input_value_definition(
            conversation_input_refs=conversation_input_refs
        ),
        "identity_input_value": _supplied_input_value_definition(
            conversation_input_refs=conversation_input_refs,
            value_shapes=(
                {
                    "operand": {"type": "string", "minLength": 1},
                    "value_type": _closed_object(
                        {"kind": {"enum": ["identity_name_or_code"]}}
                    ),
                },
            ),
        ),
        "integer_input_value": _supplied_input_value_definition(
            conversation_input_refs=conversation_input_refs,
            value_shapes=(
                {
                    "operand": {"type": "string", "pattern": r"^[1-9][0-9]*$"},
                    "value_type": _closed_object({"kind": {"enum": ["integer"]}}),
                },
            ),
        ),
    }
    return schema


def build_semantic_question_contract_schema(
    *,
    answer_request_specs: tuple[
        tuple[str, str, int, int, int, str, str | None, str], ...
    ],
    input_refs: tuple[str, ...],
    identity_input_refs: tuple[str, ...] = (),
    collection_input_refs: tuple[str, ...] = (),
    identity_collection_input_refs: tuple[str, ...] = (),
    temporal_scope_input_refs: tuple[str, ...] = (),
    conversation_input_refs: tuple[str, ...] = (),
) -> dict[str, object]:
    if not answer_request_specs:
        raise ValueError("answer request specs are required")
    input_ref_set = frozenset(input_refs)
    for role_refs in (
        identity_input_refs,
        collection_input_refs,
        identity_collection_input_refs,
        temporal_scope_input_refs,
    ):
        if not frozenset(role_refs) <= input_ref_set:
            raise ValueError("typed input refs must belong to input_refs")
    if not frozenset(identity_collection_input_refs) <= frozenset(
        collection_input_refs
    ):
        raise ValueError("identity collection refs must be collection refs")
    specialized_input_refs = frozenset(
        (*identity_input_refs, *collection_input_refs, *temporal_scope_input_refs)
    )
    value_input_refs = tuple(
        input_ref for input_ref in input_refs if input_ref not in specialized_input_refs
    )
    schema = output.SemanticQuestionContractDecisionOutput.schema(
        {
            "decision_basis": {"type": "string", "minLength": 1},
            "outcome": {
                "oneOf": [
                    _complete_contract_schema(
                        answer_request_specs=answer_request_specs
                    ),
                    _missing_requested_fact_schema(),
                    _unresolved_prior_turn_references_schema(),
                ]
            },
        }
    )
    input_definitions: dict[str, object] = {}
    if value_input_refs:
        input_definitions["input_ref"] = _input_ref_definition(value_input_refs)
    if collection_input_refs:
        scalar_collection_refs = tuple(
            item
            for item in collection_input_refs
            if item not in frozenset(identity_collection_input_refs)
        )
        if scalar_collection_refs:
            input_definitions["collection_input_ref_value"] = (
                _input_ref_value_definition(scalar_collection_refs)
            )
    if identity_input_refs:
        input_definitions["identity_input_ref_value"] = _input_ref_value_definition(
            identity_input_refs
        )
    if identity_collection_input_refs:
        input_definitions["identity_collection_input_ref_value"] = (
            _input_ref_value_definition(identity_collection_input_refs)
        )
    if temporal_scope_input_refs:
        input_definitions["temporal_scope_input_ref_value"] = (
            _input_ref_value_definition(temporal_scope_input_refs)
        )
    schema["$defs"] = {
        "source_origin": _source_origin_definition(
            conversation_input_refs=conversation_input_refs
        ),
        **input_definitions,
        "fact_expression": _fact_expression_definition(_fact_type_schema()),
        "grouping_fact_expression": _fact_expression_definition(
            _property_fact_type_schema()
        ),
        "subject_fact_expression": _fact_expression_definition(
            _fact_type_schema(),
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        "identifier_fact_expression": _fact_expression_definition(
            _identifier_fact_type_schema()
        ),
        "subject_identifier_fact_expression": _fact_expression_definition(
            _identifier_fact_type_schema(),
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        "temporal_fact_expression": _fact_expression_definition(
            _temporal_fact_type_schema()
        ),
        "subject_temporal_fact_expression": _fact_expression_definition(
            _temporal_fact_type_schema(),
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        "boolean_fact_expression": _fact_expression_definition(
            _closed_object({"kind": {"enum": ["boolean"]}})
        ),
        "subject_boolean_fact_expression": _fact_expression_definition(
            _closed_object({"kind": {"enum": ["boolean"]}}),
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        **_finite_expression_definitions(allow_input_ref=bool(value_input_refs)),
        **_finite_condition_definitions(
            allow_scalar_input=bool(value_input_refs),
            allow_collection_input="collection_input_ref_value" in input_definitions,
            allow_identity_input=bool(identity_input_refs),
            allow_identity_collection_input=bool(identity_collection_input_refs),
            allow_temporal_scope_input=bool(temporal_scope_input_refs),
            allow_coverage=any(
                universal_shape == "every_required_member_has_observation"
                for _, _, _, _, _, _, _, universal_shape in answer_request_specs
            ),
        ),
        **{
            _answer_request_definition_name(requested_fact_ref): (
                _answer_request_schema(
                    requested_fact_ref=requested_fact_ref,
                    result_kind=result_kind,
                    grouping_count=grouping_count,
                    ordering_count=ordering_count,
                    output_count=output_count,
                    selection_kind=selection_kind,
                    selection_limit_input_ref=selection_limit_input_ref,
                )
            )
            for (
                requested_fact_ref,
                result_kind,
                grouping_count,
                ordering_count,
                output_count,
                selection_kind,
                selection_limit_input_ref,
                universal_shape,
            ) in answer_request_specs
        },
    }
    return schema


def build_semantic_question_contract_schema_for_meaning(
    meaning: ParsedSemanticQuestionMeaning,
    *,
    conversation_input_refs: tuple[str, ...] = (),
) -> dict[str, object]:
    denotation_by_input_ref = {
        item.input_ref: item for item in meaning.input_denotations
    }
    identity_refs = frozenset(
        input_ref
        for input_ref, denotation in denotation_by_input_ref.items()
        if denotation.kind is InputDenotationKind.IDENTITY_REFERENCE
    )
    collection_refs = tuple(
        item.id
        for item in meaning.inputs
        if isinstance(item.value_type, CollectionType)
    )
    return build_semantic_question_contract_schema(
        answer_request_specs=tuple(
            (
                item.requested_fact_id,
                item.result_kind,
                len(item.grouping_origins),
                len(item.ordering_origins),
                len(item.output_origins),
                item.selection_kind,
                item.selection_limit_input_ref,
                item.universal_shape,
            )
            for item in meaning.answer_requests
        ),
        input_refs=tuple(item.id for item in meaning.inputs),
        identity_input_refs=tuple(
            item.id
            for item in meaning.inputs
            if item.id in identity_refs
            and not isinstance(item.value_type, CollectionType)
        ),
        collection_input_refs=collection_refs,
        identity_collection_input_refs=tuple(
            item for item in collection_refs if item in identity_refs
        ),
        temporal_scope_input_refs=tuple(
            item.id
            for item in meaning.inputs
            if isinstance(item.value_type, TemporalScopeType)
        ),
        conversation_input_refs=conversation_input_refs,
    )


def _supplied_value_schema(
    *,
    input_definition: str = "supplied_input_value",
    allow_identity: bool = True,
) -> dict[str, object]:
    branches = [
        output.SuppliedValueOutput.schema(
            {
                "meaning": {"type": "string", "minLength": 1},
                "denotation": output.SuppliedValueDenotationOutput.schema(
                    {
                        "basis": {"type": "string", "minLength": 1},
                        "kind": {"enum": ["scalar"]},
                    }
                ),
                "value": {"$ref": f"#/$defs/{input_definition}"},
            }
        )
    ]
    if allow_identity:
        branches.insert(
            0,
            output.SuppliedValueOutput.schema(
                {
                    "meaning": {"type": "string", "minLength": 1},
                    "denotation": output.SuppliedValueDenotationOutput.schema(
                        {
                            "basis": {"type": "string", "minLength": 1},
                            "kind": {"enum": ["identity_reference"]},
                            "instance_kind": {
                                "type": "string",
                                "minLength": 1,
                            },
                        }
                    ),
                    "value": {"$ref": "#/$defs/identity_input_value"},
                }
            ),
        )
    return branches[0] if len(branches) == 1 else {"oneOf": branches}


def _supplied_input_value_definition(
    *,
    conversation_input_refs: tuple[str, ...],
    value_shapes: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    branches = [
        output.SuppliedInputValueOutput.schema(
            {
                "operands": {
                    "type": "array",
                    "minItems": 1,
                    "items": value_shape["operand"],
                },
                "value_type": value_shape["value_type"],
                "origin": {"$ref": "#/$defs/frame_origin"},
            }
        )
        for value_shape in (value_shapes or _scalar_input_value_shapes())
    ]
    return {"oneOf": branches}


def _answer_request_frame_schema() -> dict[str, object]:
    branches: list[dict[str, object]] = []
    for result_kind in ("scalar", "qualifying_instances", "grouped_results"):
        answer_values: dict[str, object] = {
            "type": "array",
            "maxItems": 4,
            "items": output.NonKeyFrameValueOutput.schema(
                {
                    "value_ref": {"enum": ["v1", "v2", "v3", "v4"]},
                    "meaning": {"type": "string", "minLength": 1},
                    "origin": {"$ref": "#/$defs/frame_origin"},
                }
            ),
        }
        if result_kind == "scalar":
            answer_values["minItems"] = 1
            answer_values["maxItems"] = 1
        grouping_meanings: dict[str, object] = {
            "type": "array",
            "items": _source_origin_schema(),
        }
        if result_kind == "grouped_results":
            grouping_meanings["minItems"] = 1
        else:
            grouping_meanings["maxItems"] = 0
        selection_schema = _selection_meaning_schema(
            allow_ordered_selection=result_kind != "scalar"
        )
        raw_selection_variants = selection_schema.get("oneOf")
        if raw_selection_variants is None:
            selection_variants = (selection_schema,)
        elif isinstance(raw_selection_variants, list) and all(
            isinstance(item, dict) for item in raw_selection_variants
        ):
            selection_variants = tuple(raw_selection_variants)
        else:
            raise TypeError("selection schema variants must be objects")
        for selection in selection_variants:
            selection_properties = selection.get("properties")
            if not isinstance(selection_properties, dict):
                raise TypeError("selection schema requires properties")
            kind_schema = selection_properties.get("kind")
            if not isinstance(kind_schema, dict):
                raise TypeError("selection schema requires a kind")
            kind_values = kind_schema.get("enum")
            if not isinstance(kind_values, list) or len(kind_values) != 1:
                raise TypeError("selection kind must contain one enum value")
            selection_kind = kind_values[0]
            if not isinstance(selection_kind, str):
                raise TypeError("selection kind must be text")
            for returned_result, returned_value_refs in _returned_result_variants(
                scalar=result_kind == "scalar"
            ):
                properties: dict[str, object] = {
                    "result_kind": {"enum": [result_kind]},
                    "qualifying_row_kind": _source_origin_schema(),
                    "grouping_meanings": grouping_meanings,
                }
                if result_kind == "qualifying_instances":
                    properties["returned_candidate_identity"] = (
                        _source_origin_schema()
                    )
                properties.update(
                    {
                        "return_request_basis": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "returned_result": returned_result,
                        "answer_values": answer_values,
                        "returned_value_refs": returned_value_refs,
                        "ordering_value_refs": {
                            "type": "array",
                            "items": {
                                "enum": ["v1", "v2", "v3", "v4"]
                            },
                            **(
                                {"maxItems": 0}
                                if result_kind == "scalar"
                                else {"minItems": 1}
                                if selection_kind
                                in {
                                    "first_rank_with_ties",
                                    "take_with_boundary_ties",
                                }
                                else {}
                            ),
                        },
                        "selection": selection,
                        "universal_shape": {
                            "enum": [
                                "none",
                                "every_related_row",
                                "every_required_member_has_observation",
                            ]
                        },
                    }
                )
                branches.append(output.AnswerRequestFrameOutput.schema(properties))
    return {"oneOf": branches}


def _returned_result_variants(
    *, scalar: bool
) -> tuple[tuple[dict[str, object], dict[str, object]], ...]:
    if scalar:
        return (
            (
                output.ReturnedResultOutput.schema(
                    {"kind": {"enum": ["values"]}}
                ),
                {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"enum": ["v1"]},
                },
            ),
        )
    refs = {"items": {"enum": ["v1", "v2", "v3", "v4"]}}
    return (
        (
            output.ReturnedResultOutput.schema(
                {"kind": {"enum": ["identities"]}}
            ),
            {"type": "array", "maxItems": 0, **refs},
        ),
        (
            output.ReturnedResultOutput.schema(
                {"kind": {"enum": ["identities_and_values"]}}
            ),
            {"type": "array", "minItems": 1, **refs},
        ),
    )


def _selection_meaning_schema(*, allow_ordered_selection: bool) -> dict[str, object]:
    all_results = _closed_object({"kind": {"enum": ["all_results"]}})
    if not allow_ordered_selection:
        return all_results
    return {
        "oneOf": [
            all_results,
            _closed_object({"kind": {"enum": ["first_rank_with_ties"]}}),
            _closed_object(
                {
                    "kind": {"enum": ["take_with_boundary_ties"]},
                    "limit": _supplied_value_schema(
                        input_definition="integer_input_value",
                        allow_identity=False,
                    ),
                }
            ),
        ]
    }


def _complete_contract_schema(
    *,
    answer_request_specs: tuple[
        tuple[str, str, int, int, int, str, str | None, str], ...
    ],
) -> dict[str, object]:
    return output.CompleteSemanticQuestionContractOutput.schema(
        {
            "kind": {"enum": ["question_contract"]},
            "answer_requests": {
                "type": "array",
                "minItems": len(answer_request_specs),
                "maxItems": len(answer_request_specs),
                "items": {
                    "oneOf": [
                        {
                            "$ref": "#/$defs/"
                            + _answer_request_definition_name(requested_fact_ref)
                        }
                        for requested_fact_ref, _, _, _, _, _, _, _ in answer_request_specs
                    ]
                },
            },
        }
    )


def _answer_request_definition_name(requested_fact_ref: str) -> str:
    return f"answer_request_{requested_fact_ref}"


def _answer_request_schema(
    *,
    requested_fact_ref: str,
    result_kind: str,
    grouping_count: int,
    ordering_count: int,
    output_count: int,
    selection_kind: str,
    selection_limit_input_ref: str | None,
) -> dict[str, object]:
    if result_kind not in {"scalar", "qualifying_instances", "grouped_results"}:
        raise ValueError("unknown result kind")
    if (result_kind == "grouped_results") != (grouping_count > 0):
        raise ValueError("grouping count must match grouped result kind")
    if output_count < 1:
        raise ValueError("answer request must have at least one output")
    expression = (
        _group_output_expression_schema()
        if result_kind == "grouped_results"
        else _output_expression_schema()
    )
    request = _semantic_request_schema(
        grouping_count=grouping_count,
        ordering_count=ordering_count,
        selection_kind=selection_kind,
        selection_limit_input_ref=selection_limit_input_ref,
    )
    request_properties = request["properties"]
    if not isinstance(request_properties, dict):
        raise TypeError("semantic request schema properties are invalid")
    if result_kind != "qualifying_instances":
        distinct_by_schema = request_properties["distinct_by"]
        if not isinstance(distinct_by_schema, dict):
            raise TypeError("distinct-by schema is invalid")
        distinct_by_schema["maxItems"] = 0
    return output.AnswerRequestOutput.schema(
        {
            "requested_fact_ref": {"enum": [requested_fact_ref]},
            **request_properties,
            "outputs": {
                "type": "array",
                "minItems": output_count,
                "maxItems": output_count,
                "items": output.RequestedOutputOutput.schema(
                    {"expression": expression}
                ),
            },
        }
    )


def _semantic_request_schema(
    *,
    grouping_count: int,
    ordering_count: int,
    selection_kind: str,
    selection_limit_input_ref: str | None,
) -> dict[str, object]:
    grouping_schema: dict[str, object] = {
        "type": "array",
        "items": _grouping_schema(),
    }
    if grouping_count < 0:
        raise ValueError("grouping count cannot be negative")
    grouping_schema["minItems"] = grouping_count
    grouping_schema["maxItems"] = grouping_count
    ordering_schema: dict[str, object] = {
        "type": "array",
        "minItems": ordering_count,
        "maxItems": ordering_count,
        "items": output.OrderingOutput.schema(
            {
                "expression": (
                    _group_ordering_expression_schema()
                    if grouping_count
                    else _output_expression_schema()
                ),
                "direction": {"enum": ["ascending", "descending"]},
            }
        ),
    }
    if ordering_count < 0:
        raise ValueError("ordering count cannot be negative")
    selection_schema = _fixed_selection_schema(
        kind=selection_kind,
        limit_input_ref=selection_limit_input_ref,
    )
    return _closed_object(
        {
            "origin": _source_origin_schema(),
            "candidate_set": output.CandidateSetOutput.schema(
                {
                    "instance_interpretation": {
                        "enum": [
                            "normal_business_instance",
                            "raw_data_record",
                        ]
                    },
                }
            ),
            "grouping": grouping_schema,
            "other_sets": {
                "type": "array",
                "items": output.SetTermOutput.schema(
                    {
                        "id": _additional_set_ref_schema(),
                        "origin": _source_origin_schema(),
                    }
                ),
            },
            "other_associations": {
                "type": "array",
                "items": output.AssociationTermOutput.schema(
                    {
                        "id": _local_ref_schema("a"),
                        "from_set_ref": _local_ref_schema("s"),
                        "to_set_ref": _local_ref_schema("s"),
                        "origin": _source_origin_schema(),
                    }
                ),
            },
            "qualification": {
                "oneOf": [_subject_condition_expression_schema(), {"type": "null"}]
            },
            "ordering": ordering_schema,
            "selection": selection_schema,
            "distinct_by": {
                "type": "array",
                "items": (
                    _group_output_expression_schema()
                    if grouping_count
                    else _output_expression_schema()
                ),
            },
        }
    )


def _grouping_schema() -> dict[str, object]:
    common = {
        "id": _local_ref_schema("g"),
        "grouping_basis": {"type": "string", "minLength": 1},
    }
    return {
        "oneOf": [
            output.CandidateIdentityGroupingOutput.schema(
                {
                    **common,
                    "kind": {"enum": ["candidate_instance_identity"]},
                }
            ),
            output.RelatedIdentityGroupingOutput.schema(
                {
                    **common,
                    "kind": {"enum": ["related_instance_identity"]},
                    "identified_set": output.GroupingSetOutput.schema(
                        {"id": _additional_set_ref_schema()}
                    ),
                    "association": output.GroupingAssociationOutput.schema(
                        {
                            "id": _local_ref_schema("a"),
                            "origin": _source_origin_schema(),
                        }
                    ),
                }
            ),
            output.ValueGroupingOutput.schema(
                {
                    **common,
                    "kind": {"enum": ["value"]},
                    "expression": _grouping_value_expression_schema(),
                }
            ),
        ]
    }


def _fixed_selection_schema(
    *, kind: str, limit_input_ref: str | None
) -> dict[str, object]:
    if kind == "all_results":
        if limit_input_ref is not None:
            raise ValueError("all-results selection cannot have a limit")
        return {"type": "null"}
    if kind == "first_rank_with_ties":
        if limit_input_ref is not None:
            raise ValueError("first-rank selection cannot have a limit")
        return _closed_object({"kind": {"enum": [kind]}})
    if kind == "take_with_boundary_ties":
        if limit_input_ref is None:
            raise ValueError("bounded selection requires a limit input")
        return _closed_object(
            {
                "kind": {"enum": [kind]},
                "limit": _closed_object(
                    {
                        "kind": {"enum": ["input_ref"]},
                        "input_ref": {"enum": [limit_input_ref]},
                    }
                ),
            }
        )
    raise ValueError("unknown result selection")


def _source_origin_schema() -> dict[str, object]:
    return {"$ref": "#/$defs/source_origin"}


def _meaning_origin_definition() -> dict[str, object]:
    return output.MeaningOriginOutput.schema(
        {
            "meaning": {"type": "string", "minLength": 1},
            "origin": {"$ref": "#/$defs/frame_origin"},
        }
    )


def _frame_origin_definition(
    *, conversation_input_refs: tuple[str, ...]
) -> dict[str, object]:
    variants = [output.FrameOriginOutput.schema({"kind": {"enum": ["question"]}})]
    if conversation_input_refs:
        variants.append(
            output.FrameOriginOutput.schema(
                {
                    "kind": {"enum": ["conversation_resolution"]},
                    "resolved_input_ref": {"enum": list(conversation_input_refs)},
                }
            )
        )
    return {"oneOf": variants}


def _source_origin_definition(
    *, conversation_input_refs: tuple[str, ...]
) -> dict[str, object]:
    common = {"meaning": {"type": "string", "minLength": 1}}
    variants = [
        output.SourceOriginOutput.schema(
            {
                "source": {"enum": ["question_context"]},
                **common,
                "resolved_input_ref": {"type": "null"},
            }
        )
    ]
    if conversation_input_refs:
        variants.append(
            output.SourceOriginOutput.schema(
                {
                    "source": {"enum": ["conversation_resolution"]},
                    **common,
                    "resolved_input_ref": {"enum": list(conversation_input_refs)},
                }
            )
        )
    return {"oneOf": variants}


def _scalar_input_value_shapes() -> tuple[dict[str, object], ...]:
    scalar_types = _schemas_by_kind(
        _scalar_type_schemas(
            include_boolean=False,
            include_identifier=False,
            include_temporal_scope=True,
            include_point_temporal=False,
        )
    )
    return (
        {
            "operand": {"enum": ["true", "false"]},
            "value_type": _closed_object({"kind": {"enum": ["boolean"]}}),
        },
        {
            "operand": {"type": "string", "pattern": INTEGER_OPERAND_PATTERN},
            "value_type": scalar_types["integer"],
        },
        {
            "operand": {"type": "string", "pattern": DECIMAL_OPERAND_PATTERN},
            "value_type": scalar_types["decimal"],
        },
        {
            "operand": {"type": "string", "minLength": 1},
            "value_type": {
                "oneOf": [
                    _closed_object({"kind": {"enum": ["property_value"]}}),
                    scalar_types["temporal_scope"],
                    scalar_types["duration"],
                ]
            },
        },
    )


def _schemas_by_kind(
    schemas: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for schema in schemas:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise TypeError("typed schema requires properties")
        kind = properties.get("kind")
        if not isinstance(kind, dict):
            raise TypeError("typed schema requires a kind property")
        values = kind.get("enum")
        if not isinstance(values, list) or len(values) != 1:
            raise TypeError("typed schema kind requires one enum value")
        output[str(values[0])] = schema
    return output


def _input_ref_value_definition(refs: tuple[str, ...]) -> dict[str, object]:
    if not refs:
        raise ValueError("input ref value definition requires at least one ref")
    return {"enum": list(refs)}


def _input_ref_definition(refs: tuple[str, ...]) -> dict[str, object]:
    return _closed_object(
        {
            "kind": {"enum": ["input_ref"]},
            "input_ref": _input_ref_value_definition(refs),
        }
    )


def _fact_type_schema() -> dict[str, object]:
    variants = list(
        _scalar_type_schemas(
            include_boolean=True,
            include_identifier=True,
            include_temporal_scope=False,
            include_point_temporal=True,
        )
    )
    return {"oneOf": variants}


def _property_fact_type_schema() -> dict[str, object]:
    return {
        "oneOf": list(
            _scalar_type_schemas(
                include_boolean=True,
                include_identifier=False,
                include_temporal_scope=False,
                include_point_temporal=True,
            )
        )
    }


def _identifier_fact_type_schema() -> dict[str, object]:
    return _closed_object(
        {"kind": {"enum": ["identifier"]}, "set_ref": _local_ref_schema("s")}
    )


def _temporal_fact_type_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _closed_object({"kind": {"enum": [kind]}}) for kind in ("date", "datetime")
        ]
    }


def _fact_expression_definition(
    value_type: dict[str, object],
    *,
    observed_for_ref: dict[str, object] | None = None,
) -> dict[str, object]:
    return output.FactTermOutput.schema(
        {
            "kind": {"enum": ["fact"]},
            "observed_for_ref": observed_for_ref or _semantic_owner_ref_schema(),
            "value_type": value_type,
            "origin": _source_origin_schema(),
        }
    )


def _scalar_type_schemas(
    *,
    include_boolean: bool,
    include_identifier: bool,
    include_temporal_scope: bool,
    include_point_temporal: bool,
) -> tuple[dict[str, object], ...]:
    simple_kinds = ["integer", "text"]
    if include_boolean:
        simple_kinds.insert(0, "boolean")
    if include_point_temporal:
        simple_kinds.extend(("date", "datetime"))
    if include_temporal_scope:
        simple_kinds.append("temporal_scope")
    simple = tuple(_closed_object({"kind": {"enum": [kind]}}) for kind in simple_kinds)
    schemas = (
        *simple,
        _closed_object({"kind": {"enum": ["decimal"]}, "measure": _measure_schema()}),
        _closed_object(
            {
                "kind": {"enum": ["duration"]},
                "unit": {
                    "enum": [
                        "second",
                        "minute",
                        "hour",
                        "day",
                        "week",
                        "month",
                        "quarter",
                        "year",
                    ]
                },
            }
        ),
    )
    if not include_identifier:
        return schemas
    return (
        *schemas,
        _closed_object(
            {
                "kind": {"enum": ["identifier"]},
                "set_ref": _local_ref_schema("s"),
            }
        ),
    )


def _collection_type_schema(
    *,
    element_schemas: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return _closed_object(
        {
            "kind": {"enum": ["collection"]},
            "element_type": {"oneOf": list(element_schemas)},
        }
    )


def _measure_schema() -> dict[str, object]:
    simple = tuple(
        _closed_object({"kind": {"enum": [kind]}})
        for kind in ("unitless", "count", "ratio", "percentage")
    )
    return {
        "oneOf": [
            *simple,
            _closed_object(
                {"kind": {"enum": ["money"]}, "currency": _currency_schema()}
            ),
            _closed_object(
                {"kind": {"enum": ["quantity"]}, "unit": _quantity_unit_schema()}
            ),
        ]
    }


def _currency_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _closed_object({"kind": {"enum": ["contextual"]}}),
            _closed_object(
                {"kind": {"enum": ["source_named"]}, "origin": _source_origin_schema()}
            ),
        ]
    }


def _quantity_unit_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _closed_object({"kind": {"enum": ["item"]}}),
            _closed_object(
                {"kind": {"enum": ["source_named"]}, "origin": _source_origin_schema()}
            ),
        ]
    }


def _output_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _value_expression_schema(),
            _condition_expression_schema(),
            _set_ref_expression_schema(),
        ]
    }


def _group_output_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _value_expression_schema(),
            _condition_expression_schema(),
            _group_ref_expression_schema(),
        ]
    }


def _group_ordering_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _group_result_value_expression_schema(),
            _group_ref_expression_schema(),
        ]
    }


def _condition_expression_definition(
    *,
    value: dict[str, object],
    child: dict[str, object] | None,
    quantified_child: dict[str, object] | None = None,
    allow_scalar_input: bool,
    allow_collection_input: bool,
    allow_identity_input: bool,
    allow_identity_collection_input: bool,
    allow_temporal_scope_input: bool,
    allow_coverage: bool,
    temporal_fact_definition: str = "temporal_fact_expression",
    boolean_fact_definition: str = "boolean_fact_expression",
    identifier_fact_definition: str = "identifier_fact_expression",
) -> dict[str, object]:
    quantified_condition = quantified_child or child
    variants = []
    if allow_scalar_input:
        variants.append(
            _closed_object(
                {
                    "kind": {"enum": ["input_comparison"]},
                    "input": _input_expression_schema(),
                    "operator": {
                        "enum": [
                            "equals",
                            "not_equals",
                            "lt",
                            "lte",
                            "gt",
                            "gte",
                            "contains",
                        ]
                    },
                    "fact": value,
                }
            )
        )
    variants.extend(
        [
            _closed_object(
                {
                    "kind": {"enum": ["value_comparison"]},
                    "operator": {
                        "enum": [
                            "equals",
                            "not_equals",
                            "lt",
                            "lte",
                            "gt",
                            "gte",
                            "contains",
                        ]
                    },
                    "left": value,
                    "right": value,
                }
            ),
            _closed_object(
                {
                    "kind": {"enum": ["null_check"]},
                    "operator": {"enum": ["is_null", "not_null"]},
                    "argument": value,
                }
            ),
            _fact_expression_schema(definition=boolean_fact_definition),
        ]
    )
    if allow_collection_input:
        variants.append(
            _closed_object(
                {
                    "kind": {"enum": ["input_comparison"]},
                    "input": _collection_input_expression_schema(),
                    "operator": {"enum": ["in"]},
                    "fact": value,
                }
            )
        )
    if allow_identity_input:
        variants.append(
            _closed_object(
                {
                    "kind": {"enum": ["input_comparison"]},
                    "input": _typed_input_expression_schema(
                        definition="identity_input_ref_value"
                    ),
                    "operator": {"enum": ["equals", "not_equals"]},
                    "fact": _fact_expression_schema(
                        definition=identifier_fact_definition
                    ),
                }
            )
        )
    if allow_identity_collection_input:
        variants.append(
            _closed_object(
                {
                    "kind": {"enum": ["input_comparison"]},
                    "input": _typed_input_expression_schema(
                        definition="identity_collection_input_ref_value"
                    ),
                    "operator": {"enum": ["in"]},
                    "fact": _fact_expression_schema(
                        definition=identifier_fact_definition
                    ),
                }
            )
        )
    if allow_temporal_scope_input:
        variants.append(
            _closed_object(
                {
                    "kind": {"enum": ["within"]},
                    "value": _fact_expression_schema(
                        definition=temporal_fact_definition
                    ),
                    "scope": _typed_input_expression_schema(
                        definition="temporal_scope_input_ref_value"
                    ),
                }
            )
        )
    if child is not None:
        variants.extend(
            [
                *(
                    _closed_object(
                        {
                            "kind": {"enum": [operator]},
                            "arguments": {
                                "type": "array",
                                "minItems": 2,
                                "items": child,
                            },
                        }
                    )
                    for operator in ("and", "or")
                ),
                _closed_object(
                    {
                        "kind": {"enum": ["not"]},
                        "argument": child,
                    }
                ),
                _closed_object(
                    {
                        "kind": {"enum": ["quantify"]},
                        "quantifier": {"enum": ["exists", "not_exists", "forall"]},
                        "over_set_ref": _local_ref_schema("s"),
                        "association_refs": _ref_array_schema(),
                        "condition": quantified_condition,
                    }
                ),
            ]
        )
        if allow_coverage:
            variants.append(
                _closed_object(
                    {
                        "kind": {"enum": ["coverage"]},
                        "required_member_set_ref": _local_ref_schema("s"),
                        "required_member_condition": {
                            "oneOf": [quantified_condition, {"type": "null"}]
                        },
                        "observation_set_ref": _local_ref_schema("s"),
                        "observation_condition": quantified_condition,
                    }
                )
            )
    return {"oneOf": variants}


def _generic_fact_expression_schema() -> dict[str, object]:
    return _fact_expression_schema(definition="fact_expression")


def _fact_expression_schema(*, definition: str) -> dict[str, object]:
    return {"$ref": f"#/$defs/{definition}"}


def _input_expression_schema() -> dict[str, object]:
    return {"$ref": "#/$defs/input_ref"}


def _typed_input_expression_schema(*, definition: str) -> dict[str, object]:
    return _closed_object(
        {
            "kind": {"enum": ["input_ref"]},
            "input_ref": {"$ref": f"#/$defs/{definition}"},
        }
    )


def _collection_input_expression_schema() -> dict[str, object]:
    return _typed_input_expression_schema(definition="collection_input_ref_value")


def _set_ref_expression_schema() -> dict[str, object]:
    return _closed_object(
        {"kind": {"enum": ["set_ref"]}, "set_ref": _local_ref_schema("s")}
    )


def _group_ref_expression_schema() -> dict[str, object]:
    return _closed_object(
        {"kind": {"enum": ["group_ref"]}, "ref": _local_ref_schema("g")}
    )


def _finite_condition_definitions(
    *,
    allow_scalar_input: bool,
    allow_collection_input: bool,
    allow_identity_input: bool,
    allow_identity_collection_input: bool,
    allow_temporal_scope_input: bool,
    allow_coverage: bool,
) -> dict[str, object]:
    definitions: dict[str, object] = {}
    for depth in range(4):
        row_child = (
            None if depth == 0 else _row_condition_expression_schema(depth=depth - 1)
        )
        definitions[f"row_condition_expression_{depth}"] = (
            _condition_expression_definition(
                value=_row_value_expression_schema(),
                child=row_child,
                allow_scalar_input=allow_scalar_input,
                allow_collection_input=allow_collection_input,
                allow_identity_input=allow_identity_input,
                allow_identity_collection_input=allow_identity_collection_input,
                allow_temporal_scope_input=allow_temporal_scope_input,
                allow_coverage=allow_coverage,
            )
        )
        subject_child = (
            None
            if depth == 0
            else _subject_condition_expression_schema(depth=depth - 1)
        )
        definitions[f"subject_condition_expression_{depth}"] = (
            _condition_expression_definition(
                value=_subject_row_value_expression_schema(),
                child=subject_child,
                quantified_child=_row_condition_expression_schema(
                    depth=max(0, depth - 1)
                ),
                allow_scalar_input=allow_scalar_input,
                allow_collection_input=allow_collection_input,
                allow_identity_input=allow_identity_input,
                allow_identity_collection_input=allow_identity_collection_input,
                allow_temporal_scope_input=allow_temporal_scope_input,
                allow_coverage=allow_coverage,
                temporal_fact_definition="subject_temporal_fact_expression",
                boolean_fact_definition="subject_boolean_fact_expression",
                identifier_fact_definition="subject_identifier_fact_expression",
            )
        )
        result_child = (
            None if depth == 0 else _condition_expression_schema(depth=depth - 1)
        )
        definitions[f"condition_expression_{depth}"] = _condition_expression_definition(
            value=_value_expression_schema(),
            child=result_child,
            allow_scalar_input=allow_scalar_input,
            allow_collection_input=allow_collection_input,
            allow_identity_input=allow_identity_input,
            allow_identity_collection_input=allow_identity_collection_input,
            allow_temporal_scope_input=allow_temporal_scope_input,
            allow_coverage=allow_coverage,
        )
    return definitions


def _condition_expression_schema(*, depth: int = 3) -> dict[str, object]:
    return {"$ref": f"#/$defs/condition_expression_{depth}"}


def _row_condition_expression_schema(*, depth: int = 3) -> dict[str, object]:
    return {"$ref": f"#/$defs/row_condition_expression_{depth}"}


def _subject_condition_expression_schema(*, depth: int = 3) -> dict[str, object]:
    return {"$ref": f"#/$defs/subject_condition_expression_{depth}"}


def _finite_expression_definitions(*, allow_input_ref: bool) -> dict[str, object]:
    definitions: dict[str, object] = {}
    for depth in range(3):
        definitions[f"row_value_expression_{depth}"] = _row_value_expression_definition(
            depth,
            allow_input_ref=allow_input_ref,
            definition_prefix="row_value_expression",
            fact_definition="fact_expression",
        )
        definitions[f"subject_row_value_expression_{depth}"] = (
            _row_value_expression_definition(
                depth,
                allow_input_ref=allow_input_ref,
                definition_prefix="subject_row_value_expression",
                fact_definition="subject_fact_expression",
            )
        )
        definitions[f"grouping_value_expression_{depth}"] = (
            _row_value_expression_definition(
                depth,
                allow_input_ref=allow_input_ref,
                definition_prefix="grouping_value_expression",
                fact_definition="grouping_fact_expression",
            )
        )
    for depth in range(4):
        definitions[f"value_expression_{depth}"] = _value_expression_definition(
            depth,
            allow_input_ref=allow_input_ref,
        )
        definitions[f"group_result_value_expression_{depth}"] = (
            _group_result_value_expression_definition(
                depth,
                allow_input_ref=allow_input_ref,
            )
        )
    return definitions


def _value_expression_schema(*, depth: int = 3) -> dict[str, object]:
    return {"$ref": f"#/$defs/value_expression_{depth}"}


def _group_result_value_expression_schema(*, depth: int = 3) -> dict[str, object]:
    return {"$ref": f"#/$defs/group_result_value_expression_{depth}"}


def _row_value_expression_schema(*, depth: int = 2) -> dict[str, object]:
    return {"$ref": f"#/$defs/row_value_expression_{depth}"}


def _subject_row_value_expression_schema(*, depth: int = 2) -> dict[str, object]:
    return {"$ref": f"#/$defs/subject_row_value_expression_{depth}"}


def _grouping_value_expression_schema(*, depth: int = 2) -> dict[str, object]:
    return {"$ref": f"#/$defs/grouping_value_expression_{depth}"}


def _row_value_expression_definition(
    depth: int,
    *,
    allow_input_ref: bool,
    definition_prefix: str,
    fact_definition: str,
) -> dict[str, object]:
    variants = [_fact_expression_schema(definition=fact_definition)]
    if allow_input_ref:
        variants.append(_input_expression_schema())
    if depth > 0:
        child = {"$ref": f"#/$defs/{definition_prefix}_{depth - 1}"}
        variants.extend(_derived_value_schemas(child))
    return {"oneOf": variants}


def _value_expression_definition(
    depth: int,
    *,
    allow_input_ref: bool,
) -> dict[str, object]:
    variants = [_generic_fact_expression_schema()]
    if allow_input_ref:
        variants.append(_input_expression_schema())
    if depth > 0:
        child = _value_expression_schema(depth=depth - 1)
        variants.extend(_derived_value_schemas(child))
        variants.extend(
            (
                _aggregate_expression_schema(functions=("count",), allow_set=True),
                _aggregate_expression_schema(
                    functions=("sum", "average", "minimum", "maximum"),
                    allow_set=False,
                ),
            )
        )
    return {"oneOf": variants}


def _group_result_value_expression_definition(
    depth: int,
    *,
    allow_input_ref: bool,
) -> dict[str, object]:
    variants: list[dict[str, object]] = [
        _aggregate_expression_schema(functions=("count",), allow_set=True),
        _aggregate_expression_schema(
            functions=("sum", "average", "minimum", "maximum"),
            allow_set=False,
        ),
    ]
    if allow_input_ref:
        variants.append(_input_expression_schema())
    if depth > 0:
        variants.extend(
            _derived_value_schemas(
                _group_result_value_expression_schema(depth=depth - 1)
            )
        )
    return {"oneOf": variants}


def _derived_value_schemas(
    child: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    return (
        _closed_object(
            {
                "kind": {"enum": ["add", "subtract", "multiply", "divide"]},
                "left": child,
                "right": child,
            }
        ),
        _closed_object(
            {
                "kind": {"enum": ["negate"]},
                "argument": child,
            }
        ),
        _closed_object(
            {
                "kind": {"enum": ["temporal_bucket"]},
                "value": child,
                "grain": {"enum": ["day", "week", "month", "quarter", "year"]},
            }
        ),
    )


def _aggregate_expression_schema(
    *, functions: tuple[str, ...], allow_set: bool
) -> dict[str, object]:
    argument: dict[str, object] = _row_value_expression_schema()
    if allow_set:
        argument = {"oneOf": [argument, _set_ref_expression_schema()]}
    common = {
        "function": {"enum": list(functions)},
        "argument": argument,
    }
    return {
        "oneOf": [
            _closed_object(
                {
                    "kind": {"enum": ["aggregate"]},
                    **common,
                    "distinct_argument": {"type": "boolean"},
                }
            ),
            _closed_object(
                {
                    "kind": {"enum": ["filtered_aggregate"]},
                    **common,
                    "filter": _row_condition_expression_schema(),
                    "distinct_argument": {"type": "boolean"},
                }
            ),
        ]
    }


def _missing_requested_fact_schema() -> dict[str, object]:
    return clarification_output.MissingRequestedFactOutput.schema(
        {
            "kind": {"enum": ["missing_requested_fact"]},
            "source_text": {"type": "string", "minLength": 1},
            "why_question_is_incomplete": {"type": "string", "minLength": 1},
        }
    )


def _unresolved_prior_turn_references_schema() -> dict[str, object]:
    return clarification_output.UnresolvedPriorTurnReferencesOutput.schema(
        {
            "kind": {"enum": ["unresolved_prior_turn_references"]},
            "references": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": clarification_output.UnresolvedPriorTurnReferenceOutput.schema(
                    {
                        "source_text": {"type": "string", "minLength": 1},
                        "target_label": {"type": "string", "minLength": 1},
                        "why_question_is_incomplete": {
                            "type": "string",
                            "minLength": 1,
                        },
                    }
                ),
            },
        }
    )


def _closed_object(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _text_ref_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def _local_ref_schema(prefix: str) -> dict[str, object]:
    return {"type": "string", "pattern": rf"^{prefix}[1-9][0-9]*$"}


def _additional_set_ref_schema() -> dict[str, object]:
    return {"type": "string", "pattern": r"^s(?:[2-9]|[1-9][0-9]+)$"}


def _semantic_owner_ref_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _local_ref_schema("s"),
            _local_ref_schema("a"),
        ]
    }


def _subject_owner_ref_schema() -> dict[str, object]:
    return {
        "oneOf": [
            {"enum": ["s1"]},
            _local_ref_schema("a"),
        ]
    }


def _ref_array_schema(*, min_items: int = 0) -> dict[str, object]:
    return {"type": "array", "minItems": min_items, "items": _text_ref_schema()}


__all__ = [
    "build_semantic_question_contract_schema",
    "build_semantic_question_frame_schema",
]
