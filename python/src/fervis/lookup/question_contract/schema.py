"""Strict provider schema for the relational Question Contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import fervis.lookup.question_contract.clarification_provider_contract as clarification_output
import fervis.lookup.question_contract.provider_contract as output
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.model_io.structured_output.schema import (
    without_unreferenced_definitions,
)
from fervis.lookup.semantic_types import (
    DECIMAL_OPERAND_PATTERN,
    INTEGER_OPERAND_PATTERN,
    CollectionType,
    TemporalScopeType,
)

if TYPE_CHECKING:
    from fervis.lookup.question_contract.parser import (
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
                            "supplied_values": _supplied_values_ledger_schema(),
                            "question_input_inventory_check": (
                                output.QuestionInputInventoryCheckOutput.schema(
                                    {
                                        "all_input_like_phrases_declared": {
                                            "enum": [True]
                                        }
                                    }
                                )
                            ),
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
        "grouping_origin": _grouping_origin_definition(),
        "supplied_input_value": _supplied_input_value_definition(
            conversation_input_refs=conversation_input_refs
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
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str | None, ...],
            int,
            tuple[tuple[str, str], ...],
            str,
            str | None,
            str,
        ],
        ...,
    ],
    input_refs: tuple[str, ...],
    identity_input_refs: tuple[str, ...] = (),
    collection_input_refs: tuple[str, ...] = (),
    identity_collection_input_refs: tuple[str, ...] = (),
    temporal_scope_input_refs: tuple[str, ...] = (),
    conversation_input_refs: tuple[str, ...] = (),
    grouping_value_shapes_by_request: Mapping[
        str, tuple[tuple[str, str | None] | None, ...]
    ]
    | None = None,
) -> dict[str, object]:
    if not answer_request_specs:
        raise ValueError("answer request specs are required")
    input_ref_set = frozenset(input_refs)
    grouping_value_shapes_by_request = dict(grouping_value_shapes_by_request or {})
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
    all_identity_input_refs = tuple(
        dict.fromkeys((*identity_input_refs, *identity_collection_input_refs))
    )
    coverage_enabled = any(
        relational_shape == "every_required_member_has_observation"
        for _, _, _, _, _, _, _, _, relational_shape in answer_request_specs
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
                        answer_request_specs=answer_request_specs,
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
        "fact_expression": _fact_expression_definition(),
        "grouping_fact_expression": _fact_expression_definition(),
        "subject_fact_expression": _fact_expression_definition(
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        "identifier_fact_expression": _identifier_fact_expression_definition(),
        "subject_identifier_fact_expression": (
            _identifier_fact_expression_definition()
        ),
        "temporal_fact_expression": _fact_expression_definition(),
        "subject_temporal_fact_expression": _fact_expression_definition(
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        "boolean_fact_expression": _fact_expression_definition(),
        "subject_boolean_fact_expression": _fact_expression_definition(
            observed_for_ref=_subject_owner_ref_schema(),
        ),
        **_finite_expression_definitions(allow_input_ref=bool(value_input_refs)),
        **_finite_condition_definitions(
            allow_scalar_input=bool(value_input_refs),
            allow_collection_input="collection_input_ref_value" in input_definitions,
            allow_identity_input=bool(identity_input_refs),
            allow_identity_collection_input=bool(identity_collection_input_refs),
            allow_temporal_scope_input=bool(temporal_scope_input_refs),
            allow_coverage=False,
        ),
        **(
            _finite_scoped_condition_definitions(
                allow_scalar_input=bool(value_input_refs),
                allow_collection_input=(
                    "collection_input_ref_value" in input_definitions
                ),
                allow_identity_input=bool(identity_input_refs),
                allow_identity_collection_input=bool(identity_collection_input_refs),
                allow_temporal_scope_input=bool(temporal_scope_input_refs),
            )
            if coverage_enabled
            else {}
        ),
        **{
            _association_graph_definition_name(requested_fact_ref): (
                _association_graph_related_set_schema(
                    definition_name=_association_graph_definition_name(
                        requested_fact_ref
                    ),
                )
            )
            for (
                requested_fact_ref,
                _,
                grouping_kinds,
                _,
                _,
                requested_value_specs,
                _,
                _,
                _,
            ) in answer_request_specs
        },
        **{
            _answer_request_definition_name(requested_fact_ref): (
                _answer_request_schema(
                    requested_fact_ref=requested_fact_ref,
                    result_kind=result_kind,
                    grouping_kinds=grouping_kinds,
                    grouping_value_shapes=grouping_value_shapes_by_request.get(
                        requested_fact_ref,
                        tuple(None for _ in grouping_kinds),
                    ),
                    ordering_group_refs=ordering_group_refs,
                    result_key_count=result_key_count,
                    requested_value_specs=requested_value_specs,
                    selection_kind=selection_kind,
                    selection_limit_input_ref=selection_limit_input_ref,
                    relational_shape=relational_shape,
                    identity_input_refs=all_identity_input_refs,
                )
            )
            for (
                requested_fact_ref,
                result_kind,
                grouping_kinds,
                ordering_group_refs,
                result_key_count,
                requested_value_specs,
                selection_kind,
                selection_limit_input_ref,
                relational_shape,
            ) in answer_request_specs
        },
    }
    return without_unreferenced_definitions(schema)


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
    schema = build_semantic_question_contract_schema(
        answer_request_specs=tuple(
            (
                item.requested_fact_id,
                item.result_kind,
                item.grouping_kinds,
                item.ordering_group_refs,
                item.result_key_count,
                tuple(
                    zip(
                        item.requested_value_refs,
                        item.output_kinds[item.result_key_count :],
                        strict=True,
                    )
                ),
                item.selection_kind,
                item.selection_limit_input_ref,
                item.relational_shape,
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
        grouping_value_shapes_by_request={
            item.requested_fact_id: item.grouping_value_shapes
            for item in meaning.answer_requests
        },
    )
    return _apply_frame_commitments(schema, meaning=meaning)


def _apply_frame_commitments(
    schema: dict[str, object],
    *,
    meaning: ParsedSemanticQuestionMeaning,
) -> dict[str, object]:
    """Bind provider choices to the immutable commitments made by the frame."""

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise TypeError("question contract schema definitions are missing")
    inputs = {item.id: item for item in meaning.inputs}
    identity_kinds = {
        item.input_ref: item.denoted_instance_kind
        for item in meaning.input_denotations
        if item.kind is InputDenotationKind.IDENTITY_REFERENCE
    }
    if any(not kind for kind in identity_kinds.values()):
        raise ValueError("identity input requires a nominal instance kind")
    identity_inputs = {
        input_ref: {
            "operand_meaning": next(
                item.operand_meaning
                for item in meaning.input_denotations
                if item.input_ref == input_ref
            ),
            "instance_kind": str(instance_kind),
            "collection": isinstance(inputs[input_ref].value_type, CollectionType),
        }
        for input_ref, instance_kind in identity_kinds.items()
    }
    for name, definition in tuple(definitions.items()):
        if name.startswith(("row_condition_expression_", "condition_expression_")):
            _expand_identity_comparison_branches(
                definition,
                definitions=definitions,
                identity_inputs=identity_inputs,
                candidate_instance_kind=None,
            )
    for request in meaning.answer_requests:
        candidate_kind = request.candidate_set_origin.meaning
        related_output_kinds = {
            output_ref: origin.meaning
            for output_ref, output_kind, origin in zip(
                request.requested_value_refs,
                request.output_kinds[request.result_key_count :],
                request.output_origins[request.result_key_count :],
                strict=True,
            )
            if output_kind == "related_entity"
        }
        answer_definition = definitions[
            _answer_request_definition_name(request.requested_fact_id)
        ]
        answer_properties = answer_definition["properties"]
        candidate_schema = answer_properties["candidate_set"]
        candidate_properties = candidate_schema["properties"]
        candidate_schema["properties"] = {
            **candidate_properties,
            "instance_kind": {"enum": [candidate_kind]},
        }
        candidate_schema["required"] = list(candidate_schema["properties"])

        if request.grouping_refs:
            branches = []
            for ref, kind, shape in zip(
                request.grouping_refs,
                request.grouping_kinds,
                request.grouping_value_shapes,
                strict=True,
            ):
                branch = cast(
                    dict[str, Any],
                    _grouping_schema(
                        grouping_kinds=(kind,),
                        grouping_value_shapes=(shape,),
                    ),
                )["oneOf"][0]
                branch["properties"]["id"] = {"enum": [ref]}
                branches.append(branch)
            answer_properties["grouping"]["items"] = {"oneOf": branches}
        if request.ordering_origins:
            previous = answer_properties["ordering"]["items"]
            previous_branches = previous.get("oneOf", [previous])
            ordering_branches = []
            for index, value_ref in enumerate(request.ordering_value_refs):
                branch = deepcopy(
                    previous_branches[index]
                    if len(previous_branches) > 1
                    else previous_branches[0]
                )
                if value_ref is not None:
                    branch["properties"]["expression"] = {
                        "type": "object",
                        "properties": {
                            "kind": {"enum": ["requested_value_ref"]},
                            "value_ref": {"enum": [value_ref]},
                        },
                        "required": ["kind", "value_ref"],
                        "additionalProperties": False,
                    }
                ordering_branches.append(branch)
            unique = list(
                {repr(branch): branch for branch in ordering_branches}.values()
            )
            answer_properties["ordering"]["items"] = (
                unique[0] if len(unique) == 1 else {"anyOf": unique}
            )

        graph_properties = answer_properties["set_graph"]["properties"]
        input_relation_properties = graph_properties["identity_input_relations"][
            "properties"
        ]
        for input_ref, metadata in identity_inputs.items():
            relation = input_relation_properties[input_ref]["oneOf"][1]
            relation["properties"]["set"]["properties"]["instance_kind"] = {
                "enum": [str(metadata["instance_kind"])]
            }
        output_relation_properties = graph_properties["requested_output_relations"][
            "properties"
        ]
        for output_ref, instance_kind in related_output_kinds.items():
            relation = output_relation_properties[output_ref]
            relation["properties"]["set"]["properties"]["instance_kind"] = {
                "enum": [instance_kind]
            }

        output_branches = answer_properties["outputs"]["properties"][
            "requested_value_outputs"
        ]["items"].get("oneOf", ())
        for branch in output_branches:
            properties = branch.get("properties", {})
            output_refs = properties.get("output_ref", {}).get("enum", ())
            if len(output_refs) != 1 or output_refs[0] not in related_output_kinds:
                continue
            branch["properties"] = {
                "output_ref": properties["output_ref"],
                "instance_kind": {"enum": [related_output_kinds[output_refs[0]]]},
            }
            branch["required"] = list(branch["properties"])

        suffix = request.requested_fact_id
        subject_names = tuple(
            name
            for name in definitions
            if name.startswith("subject_condition_expression_")
        )
        for name in subject_names:
            clone = deepcopy(definitions[name])
            _rewrite_definition_refs(
                clone,
                names=frozenset(subject_names),
                suffix=suffix,
            )
            _expand_identity_comparison_branches(
                clone,
                definitions=definitions,
                identity_inputs=identity_inputs,
                candidate_instance_kind=candidate_kind,
            )
            definitions[f"{name}_{suffix}"] = clone
        _rewrite_definition_refs(
            answer_properties["qualification"],
            names=frozenset(subject_names),
            suffix=suffix,
        )
    return _order_comparison_operands(
        without_unreferenced_definitions(schema), input_first=bool(identity_inputs)
    )


def _order_comparison_operands(
    schema: dict[str, object], *, input_first: bool
) -> dict[str, object]:
    def visit(node: object) -> None:
        if isinstance(node, dict):
            properties = node.get("properties", {})
            if isinstance(properties, dict) and properties.get("kind", {}).get(
                "enum"
            ) == ["input_comparison"]:
                order = (
                    ("kind", "input", "operator", "fact")
                    if input_first
                    else ("kind", "fact", "operator", "input")
                )
                node["properties"] = {key: properties[key] for key in order}
                node["required"] = list(node["properties"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return schema


def _expand_identity_comparison_branches(
    definition: object,
    *,
    definitions: Mapping[str, object],
    identity_inputs: Mapping[str, Mapping[str, object]],
    candidate_instance_kind: str | None,
) -> None:
    if not isinstance(definition, dict):
        return
    branches = definition.get("oneOf")
    if not isinstance(branches, list):
        return
    expanded: list[object] = []
    for branch in branches:
        if not isinstance(branch, dict):
            expanded.append(branch)
            continue
        properties = branch.get("properties")
        input_schema = properties.get("input") if isinstance(properties, dict) else None
        input_ref = (
            input_schema.get("properties", {}).get("input_ref", {}).get("$ref")
            if isinstance(input_schema, dict)
            else None
        )
        if input_ref not in {
            "#/$defs/identity_input_ref_value",
            "#/$defs/identity_collection_input_ref_value",
        }:
            expanded.append(branch)
            continue
        collection = input_ref.endswith("identity_collection_input_ref_value")
        assert isinstance(properties, dict)
        fact_ref = properties["fact"]["$ref"]
        fact_definition = definitions[fact_ref.rsplit("/", 1)[-1]]
        for ref, metadata in identity_inputs.items():
            if bool(metadata["collection"]) != collection:
                continue
            typed_branch = deepcopy(branch)
            typed_properties = typed_branch["properties"]
            typed_properties["input"] = _closed_object(
                {
                    "kind": {"enum": ["input_ref"]},
                    "input_ref": {"enum": [ref]},
                    "operand_meaning": {"enum": [str(metadata["operand_meaning"])]},
                    "instance_kind": {"enum": [str(metadata["instance_kind"])]},
                }
            )
            typed_fact = deepcopy(cast(dict[str, Any], fact_definition))
            if (
                candidate_instance_kind is not None
                and metadata["instance_kind"] != candidate_instance_kind
            ):
                path = typed_fact["properties"]["identity_path"]
                path["oneOf"] = [
                    item
                    for item in path["oneOf"]
                    if item["properties"]["kind"]["enum"] != ["candidate_instance"]
                ]
            typed_properties["fact"] = typed_fact
            expanded.append(typed_branch)
    definition["oneOf"] = expanded


def _rewrite_definition_refs(
    value: object,
    *,
    names: frozenset[str],
    suffix: str,
) -> None:
    if isinstance(value, list):
        for item in value:
            _rewrite_definition_refs(item, names=names, suffix=suffix)
        return
    if not isinstance(value, dict):
        return
    ref = value.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.rsplit("/", 1)[-1]
        if name in names:
            value["$ref"] = f"#/$defs/{name}_{suffix}"
    for child in value.values():
        _rewrite_definition_refs(child, names=names, suffix=suffix)


def _identity_operand_schema():
    return output.IdentityOperandOutput.schema({
        "kind": {"enum": ["literal", "description"]},
        "value": {"type": "string", "minLength": 1,
                  "description": "For literal, copy only the supplied name or code, excluding surrounding naming words. For description, copy the complete role or relational description whose entity must be established from data."},
    })


def _supplied_values_ledger_schema() -> dict[str, object]:
    common: dict[str, object] = {
        "meaning": {"type": "string", "minLength": 1},
        "denotation_basis": {"type": "string", "minLength": 1},
        "answer_request_numbers": {"type":"array", "minItems":1, "maxItems":4,
            "items":{"type":"integer", "minimum":1, "maximum":4}},
    }
    entity_item = output.SuppliedEntityReferenceOutput.schema(
        {
            **common,
            "entity_reference": output.EntityReferenceOutput.schema(
                {
                    "instance_kind": {"type": "string", "minLength": 1},
                    "value": {
                        "oneOf": [
                            output.SingleIdentityValueOutput.schema(
                                {
                                    "kind": {"enum": ["single_identity"]},
                                    "identity_value": _identity_operand_schema(),
                                    "origin": {"$ref": "#/$defs/frame_origin"},
                                }
                            ),
                            output.IdentityAlternativesValueOutput.schema(
                                {
                                    "kind": {"enum": ["identity_alternatives"]},
                                    "identity_values": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": _identity_operand_schema(),
                                    },
                                    "origin": {"$ref": "#/$defs/frame_origin"},
                                }
                            ),
                        ]
                    },
                }
            ),
        }
    )
    operand_item = {"oneOf": [entity_item, *_non_entity_ledger_item_schemas(common)]}
    selection_limit_properties = _selection_limit_properties()
    return output.SuppliedValuesLedgerOutput.schema(
        {
            "operands": {
                "type": "array",
                "items": operand_item,
            },
            "selection_limits": {
                "type": "array",
                "items": output.SelectionLimitOutput.schema(
                    {
                        "answer_request_number": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 4,
                        },
                        **selection_limit_properties,
                    }
                ),
            },
        }
    ) | {
        "description": (
            "Entity references and literal values given by the question. References may "
            "be names, codes, or definite descriptions whose identities are resolved "
            "later from data. Candidate "
            "kinds, requested unknowns, grouping meanings, and observed or computed "
            "ordering meanings remain in answer_requests."
        )
    }


def _non_entity_ledger_item_schemas(
    common: dict[str, object],
) -> tuple[dict[str, object], ...]:
    def item(
        kind: str,
        *,
        operands: dict[str, object],
        unit: dict[str, object] | None = None,
    ) -> dict[str, object]:
        nested_properties = {
            "kind": {"enum": [kind]},
            **({"unit": unit} if unit is not None else {}),
            "value": output.NonEntityLedgerValueOutput.schema(
                {
                    "operands": {
                        "type": "array",
                        "minItems": 1,
                        "items": operands,
                        **({"maxItems": 1} if kind == "temporal_scope" else {}),
                    },
                    "origin": {"$ref": "#/$defs/frame_origin"},
                }
            ),
        }
        return output.SuppliedNonEntityValueOutput.schema(
            {
                **common,
                "non_entity_value": (
                    output.DurationOperandOutput.schema(nested_properties)
                    if unit is not None
                    else output.NonEntityOperandOutput.schema(nested_properties)
                ),
            }
        )

    text = {"type": "string", "minLength": 1}
    return (
        item("categorical_value", operands=text),
        item("temporal_scope", operands=text),
        item("number", operands={"type": "string", "pattern": DECIMAL_OPERAND_PATTERN}),
        item("boolean", operands={"enum": ["true", "false"]}),
        item(
            "duration",
            operands=text,
            unit={
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
        ),
    )


def _selection_limit_properties() -> dict[str, object]:
    return {
        "meaning": {"type": "string", "minLength": 1},
        "denotation_basis": {"type": "string", "minLength": 1},
        "non_entity_value": output.NonEntityValueOutput.schema(
            {"value": {"$ref": "#/$defs/integer_input_value"}}
        ),
    }


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
    standard_results = _frame_result_schemas(
        candidate_field_by_kind={
            "one_value_for_population": "population_rows",
            "one_result_per_qualifying_row": "result_candidates",
            "one_result_per_group": "grouped_observation_rows",
        }
    )
    coverage_results = _frame_result_schemas(
        candidate_field_by_kind={
            "one_value_for_population": "coverage_candidates",
            "one_result_per_qualifying_row": "coverage_candidates",
        }
    )
    return output.AnswerRequestFrameOutput.schema(
        {
            "return_request_basis": {"type": "string", "minLength": 1},
            "relational_shape_basis": {"type": "string", "minLength": 1},
            "request": {
                "oneOf": [
                    _frame_request_schema(
                        relational_shapes=("every_required_member_has_observation",),
                        result_schemas=coverage_results,
                    ),
                    _frame_request_schema(
                        relational_shapes=(
                            "ordinary",
                            "every_related_row",
                            "same_related_row",
                        ),
                        result_schemas=standard_results,
                    ),
                ]
            },
        }
    )


def _frame_request_schema(
    *,
    relational_shapes: tuple[str, ...],
    result_schemas: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return output.QuestionFrameRequestOutput.schema(
        {
            "relational_shape": {"enum": list(relational_shapes)},
            "result_grain_basis": {"type": "string", "minLength": 1},
            "result": {"oneOf": list(result_schemas)},
        }
    )


def _frame_result_schemas(
    *,
    candidate_field_by_kind: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    return tuple(
        _frame_result_schema(kind=kind, candidate_field=candidate_field)
        for kind, candidate_field in candidate_field_by_kind.items()
    )


def _frame_result_schema(*, kind: str, candidate_field: str) -> dict[str, object]:
    properties: dict[str, object] = {"kind": {"enum": [kind]}}
    if kind == "one_result_per_group":
        properties["grouping_meanings"] = {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/grouping_origin"},
        }
    properties[candidate_field] = _frame_row_source_schema()
    if kind == "one_value_for_population":
        properties["returned_meanings"] = _scalar_returned_meanings_schema()
    else:
        properties["projection"] = _row_projection_schema(result_kind=kind)
    properties["result_order"] = _frame_result_order_schema(
        result_kind=kind,
    )
    return _closed_object(properties)


def _frame_row_source_schema() -> dict[str, object]:
    return output.FrameRowSourceOutput.schema(
        {
            "instance_kind": {"type": "string", "minLength": 1},
            "origin": {"$ref": "#/$defs/frame_origin"},
        }
    )


def _frame_result_order_schema(*, result_kind: str) -> dict[str, object]:
    allow_ordering = result_kind != "one_value_for_population"
    ordering = (
        {
            "oneOf": [
                _closed_object({"kind": {"enum": ["no_ordering_requested"]}}),
                output.OrderedByOutput.schema(
                    {
                        "kind": {"enum": ["ordered_by"]},
                        "values": _ordering_meanings_schema(
                            allow_group_ref=result_kind == "one_result_per_group"
                        ),
                    }
                ),
            ]
        }
        if allow_ordering
        else _closed_object({"kind": {"enum": ["no_ordering_requested"]}})
    )
    return output.FrameResultOrderOutput.schema(
        {
            "ordering_request_basis": {"type": "string", "minLength": 1},
            "ordering": ordering,
            "selection": _selection_meaning_schema(
                allow_ordered_selection=allow_ordering
            ),
        }
    )


def _ordering_meanings_schema(*, allow_group_ref: bool) -> dict[str, object]:
    branches = [
        output.RequestedValueOrderingOutput.schema(
            {
                "ownership_basis": {"type": "string", "minLength": 1},
                "kind": {"enum": ["requested_value_ref"]},
                "value_ref": {"enum": ["v1", "v2", "v3", "v4"]},
            }
        )
    ]
    if allow_group_ref:
        branches.append(
            output.GroupOrderingReferenceOutput.schema(
                {
                    "ownership_basis": {"type": "string", "minLength": 1},
                    "kind": {"enum": ["group_ref"]},
                    "group_ref": {"enum": ["g1", "g2", "g3", "g4"]},
                }
            )
        )
    branches.append(
        output.UnreturnedOrderingMeaningOutput.schema(
            {
                "ownership_basis": {"type": "string", "minLength": 1},
                "kind": {"enum": ["unreturned_ordering_meaning"]},
                "meaning": {"type": "string", "minLength": 1},
                "origin": {"$ref": "#/$defs/frame_origin"},
            }
        )
    )
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "oneOf": branches,
        },
    }


def _scalar_returned_meanings_schema() -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": output.ReturnedMeaningOutput.schema(
            {
                "meaning": {"type": "string", "minLength": 1},
                "origin": {"$ref": "#/$defs/frame_origin"},
                "meaning_ref": {"enum": ["r1"]},
            }
        ),
    }


def _row_projection_schema(*, result_kind: str) -> dict[str, object]:
    values = {
        "type": "array",
        "minItems": 0,
        "maxItems": 4,
        "description": "Requested values stated by the answer.",
        "items": output.ReturnedProjectionValueOutput.schema(
            {
                "value_ref": {"enum": ["v1", "v2", "v3", "v4"]},
                "value_kind_basis": {"type": "string", "minLength": 1},
                "value_kind": {
                    "enum": ["value"]
                    if result_kind == "one_result_per_group"
                    else ["related_entity", "value"]
                },
                "meaning": {"type": "string", "minLength": 1},
                "origin": {"$ref": "#/$defs/frame_origin"},
            }
        ),
    }
    common: dict[str, object] = {
        "projection_basis": {"type": "string", "minLength": 1},
        "explicitly_requested_values": values,
    }
    if result_kind == "one_result_per_group":
        return output.GroupProjectionOutput.schema(
            {
                "projection_basis": common["projection_basis"],
                "returned_grouping_keys": {"enum": ["all"]},
                "explicitly_requested_values": values,
            }
        )
    return output.CandidateProjectionOutput.schema(
        {
            "projection_basis": common["projection_basis"],
            "candidate_identity": {"enum": ["returned", "omitted"]},
            "explicitly_requested_values": values,
        }
    )


def _selection_meaning_schema(*, allow_ordered_selection: bool) -> dict[str, object]:
    all_results = _closed_object({"kind": {"enum": ["all_results"]}})
    if not allow_ordered_selection:
        return all_results
    return {
        "oneOf": [
            all_results,
            _closed_object({"kind": {"enum": ["first_rank_with_ties"]}}),
            _closed_object({"kind": {"enum": ["take_with_boundary_ties"]}}),
            _closed_object({"kind": {"enum": ["position_with_ties"]}}),
        ]
    }


def _complete_contract_schema(
    *,
    answer_request_specs: tuple[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str | None, ...],
            int,
            tuple[tuple[str, str], ...],
            str,
            str | None,
            str,
        ],
        ...,
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
                        for requested_fact_ref, _, _, _, _, _, _, _, _ in answer_request_specs
                    ]
                },
            },
        }
    )


def _answer_request_definition_name(requested_fact_ref: str) -> str:
    return f"answer_request_{requested_fact_ref}"


def _association_graph_definition_name(requested_fact_ref: str) -> str:
    return f"association_graph_related_set_{requested_fact_ref}"


def _answer_request_schema(
    *,
    requested_fact_ref: str,
    result_kind: str,
    grouping_kinds: tuple[str, ...],
    grouping_value_shapes: tuple[tuple[str, str | None] | None, ...],
    ordering_group_refs: tuple[str | None, ...],
    result_key_count: int,
    requested_value_specs: tuple[tuple[str, str], ...],
    selection_kind: str,
    selection_limit_input_ref: str | None,
    relational_shape: str,
    identity_input_refs: tuple[str, ...],
) -> dict[str, object]:
    if result_kind not in {"scalar", "qualifying_instances", "grouped_results"}:
        raise ValueError("unknown result kind")
    grouping_count = len(grouping_kinds)
    if (result_kind == "grouped_results") != (grouping_count > 0):
        raise ValueError("grouping count must match grouped result kind")
    output_count = result_key_count + len(requested_value_specs)
    if output_count < 1:
        raise ValueError("answer request must have at least one output")
    if not 0 <= result_key_count <= output_count:
        raise ValueError("result-key count must fit requested outputs")
    result_key_expression = (
        _group_identity_output_expression_schema()
        if result_kind == "grouped_results"
        else _identity_output_expression_schema()
    )
    request = _semantic_request_schema(
        grouping_count=grouping_count,
        grouping_kinds=grouping_kinds,
        grouping_value_shapes=grouping_value_shapes,
        ordering_group_refs=ordering_group_refs,
        selection_kind=selection_kind,
        selection_limit_input_ref=selection_limit_input_ref,
        relational_shape=relational_shape,
        graph_definition_name=_association_graph_definition_name(requested_fact_ref),
        identity_input_refs=identity_input_refs,
        requested_value_specs=requested_value_specs,
    )
    request_properties = request["properties"]
    if not isinstance(request_properties, dict):
        raise TypeError("semantic request schema properties are invalid")
    if result_kind != "qualifying_instances":
        distinct_by_schema = request_properties["distinct_by"]
        if not isinstance(distinct_by_schema, dict):
            raise TypeError("distinct-by schema is invalid")
        distinct_by_schema["maxItems"] = 0
    if relational_shape == "every_required_member_has_observation":
        request_properties["qualification"] = _coverage_expression_schema()
    elif relational_shape == "same_related_row":
        request_properties["qualification"] = _related_row_expression_schema()
    return output.AnswerRequestOutput.schema(
        {
            "requested_fact_ref": {"enum": [requested_fact_ref]},
            **request_properties,
            "outputs": output.RequestedOutputsOutput.schema(
                {
                    "result_key_outputs": _fixed_output_array_schema(
                        count=result_key_count,
                        expression=result_key_expression,
                    ),
                    "requested_value_outputs": _requested_value_outputs_schema(
                        grouping_kinds=grouping_kinds,
                        requested_value_specs=requested_value_specs,
                        grouped=result_kind == "grouped_results",
                    ),
                }
            ),
        }
    )


def _fixed_output_array_schema(
    *, count: int, expression: dict[str, object]
) -> dict[str, object]:
    return {
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": output.RequestedOutputOutput.schema({"expression": expression}),
    }


def _requested_value_outputs_schema(
    *,
    grouping_kinds: tuple[str, ...],
    requested_value_specs: tuple[tuple[str, str], ...],
    grouped: bool,
) -> dict[str, object]:
    branches = []
    for output_ref, output_kind in requested_value_specs:
        if output_kind == "related_entity":
            branches.append(
                output.RequestedRelatedEntityOutputOutput.schema(
                    {
                        "output_ref": {"enum": [output_ref]},
                        "instance_kind": {"type": "string", "minLength": 1},
                    }
                )
            )
            continue
        elif output_kind == "value":
            expression = (
                _group_result_value_expression_schema()
                if grouped
                else _value_output_expression_schema()
            )
        else:
            raise ValueError(f"unknown requested output kind: {output_kind}")
        branches.append(
            output.RequestedValueOutputOutput.schema(
                {
                    "output_ref": {"enum": [output_ref]},
                    "expression": expression,
                }
            )
        )
    return {
        "type": "array",
        "minItems": len(branches),
        "maxItems": len(branches),
        "items": {"oneOf": branches} if branches else {"type": "null"},
    }


def _semantic_request_schema(
    *,
    grouping_count: int,
    grouping_kinds: tuple[str, ...],
    grouping_value_shapes: tuple[tuple[str, str | None] | None, ...],
    ordering_group_refs: tuple[str | None, ...],
    selection_kind: str,
    selection_limit_input_ref: str | None,
    relational_shape: str,
    graph_definition_name: str,
    identity_input_refs: tuple[str, ...],
    requested_value_specs: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    grouping_schema: dict[str, object] = {
        "type": "array",
        "items": (
            _grouping_schema(
                grouping_kinds=grouping_kinds,
                grouping_value_shapes=grouping_value_shapes,
            )
            if grouping_kinds
            else {"type": "null"}
        ),
    }
    if grouping_count < 0:
        raise ValueError("grouping count cannot be negative")
    grouping_schema["minItems"] = grouping_count
    grouping_schema["maxItems"] = grouping_count
    ordering_items = [
        output.OrderingOutput.schema(
            {
                "ordering_basis": {"type": "string", "minLength": 1},
                "expression": (
                    _closed_object(
                        {
                            "kind": {"enum": ["group_ref"]},
                            "ref": {"enum": [group_ref]},
                        }
                    )
                    if group_ref is not None
                    else (
                        _group_ordering_expression_schema()
                        if grouping_count
                        else _value_expression_schema()
                    )
                ),
                "direction": {"enum": ["ascending", "descending"]},
            }
        )
        for group_ref in ordering_group_refs
    ]
    ordering_schema: dict[str, object] = {
        "type": "array",
        "minItems": len(ordering_items),
        "maxItems": len(ordering_items),
        "items": (
            ordering_items[0]
            if len(ordering_items) == 1
            else {"oneOf": ordering_items}
            if ordering_items
            else {"type": "null"}
        ),
    }
    selection_schema = _fixed_selection_schema(
        kind=selection_kind,
        limit_input_ref=selection_limit_input_ref,
    )
    return _closed_object(
        {
            "origin": _source_origin_schema(),
            "candidate_set": output.CandidateSetOutput.schema(
                {
                    "instance_kind": {"type": "string", "minLength": 1},
                    "instance_interpretation": {
                        "enum": [
                            "resource_population",
                            "raw_data_record",
                        ]
                    },
                }
            ),
            "set_graph": _association_graph_schema(
                relational_shape=relational_shape,
                graph_definition_name=graph_definition_name,
                identity_input_refs=identity_input_refs,
                requested_value_specs=requested_value_specs,
            ),
            "grouping": grouping_schema,
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


def _association_declaration_schema() -> dict[str, object]:
    return _closed_object(
        {
            "id": _local_ref_schema("a"),
            "origin": _source_origin_schema(),
        }
    )


def _set_declaration_schema(
    *,
    set_ref: dict[str, object] | None = None,
) -> dict[str, object]:
    return output.SetTermOutput.schema(
        {
            "id": set_ref or _additional_set_ref_schema(),
            "instance_kind": {"type": "string", "minLength": 1},
            "origin": _source_origin_schema(),
        }
    )


def _association_graph_related_set_schema(
    *,
    definition_name: str,
) -> dict[str, object]:
    return _closed_object(
        {
            "association": _association_declaration_schema(),
            "set": _set_declaration_schema(),
            "related_sets": {
                "type": "array",
                "items": {"$ref": f"#/$defs/{definition_name}"},
            },
        }
    )


def _association_graph_schema(
    *,
    relational_shape: str,
    graph_definition_name: str,
    identity_input_refs: tuple[str, ...],
    requested_value_specs: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    related_output_refs = tuple(
        output_ref
        for output_ref, output_kind in requested_value_specs
        if output_kind == "related_entity"
    )
    role_index_by_ref = {
        ref: index
        for index, ref in enumerate(
            (*identity_input_refs, *related_output_refs),
            start=2,
        )
    }
    other_graph = _other_association_graph_schema(
        relational_shape=relational_shape,
        graph_definition_name=graph_definition_name,
    )
    return output.SetGraphOutput.schema(
        {
            "identity_input_relations": _closed_object(
                {
                    input_ref: {
                        "oneOf": [
                            {"type": "null"},
                            _role_relation_schema(
                                set_index=role_index_by_ref[input_ref],
                                graph_definition_name=graph_definition_name,
                            ),
                        ]
                    }
                    for input_ref in identity_input_refs
                }
            ),
            "requested_output_relations": _closed_object(
                {
                    output_ref: _role_relation_schema(
                        set_index=role_index_by_ref[output_ref],
                        graph_definition_name=graph_definition_name,
                    )
                    for output_ref in related_output_refs
                }
            ),
            "other_related_sets": cast(dict[str, Any], other_graph)["properties"][
                "related_sets"
            ],
        }
    )


def _role_relation_schema(
    *,
    set_index: int,
    graph_definition_name: str,
) -> dict[str, object]:
    return output.RoleRelationOutput.schema(
        {
            "association": _closed_object(
                {
                    "id": {"enum": [f"a{set_index - 1}"]},
                    "origin": _source_origin_schema(),
                }
            ),
            "set": _set_declaration_schema(set_ref={"enum": [f"s{set_index}"]}),
            "related_sets": {
                "type": "array",
                "items": {"$ref": f"#/$defs/{graph_definition_name}"},
            },
        }
    )


def _other_association_graph_schema(
    *,
    relational_shape: str,
    graph_definition_name: str,
) -> dict[str, object]:
    if relational_shape == "same_related_row":
        related_set = _closed_object(
            {
                "associations": {
                    "type": "array",
                    "minItems": 2,
                    "items": _association_declaration_schema(),
                },
                "set": _set_declaration_schema(set_ref=_local_ref_schema("s")),
                "related_sets": {
                    "type": "array",
                    "items": {"$ref": f"#/$defs/{graph_definition_name}"},
                },
            }
        )
        return _closed_object(
            {
                "related_sets": {
                    "type": "array",
                    "minItems": 1,
                    "items": related_set,
                }
            }
        )
    if relational_shape == "every_required_member_has_observation":
        required_member = _closed_object(
            {
                "association": _closed_object(
                    {
                        "id": {"enum": ["a2"]},
                        "origin": _source_origin_schema(),
                    }
                ),
                "set": _set_declaration_schema(set_ref={"enum": ["s2"]}),
                "related_sets": {
                    "type": "array",
                    "maxItems": 0,
                    "items": _closed_object({}),
                },
            }
        )
        observation = _closed_object(
            {
                "association": _closed_object(
                    {
                        "id": {"enum": ["a1"]},
                        "origin": _source_origin_schema(),
                    }
                ),
                "set": _set_declaration_schema(set_ref={"enum": ["s3"]}),
                "related_sets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": required_member,
                },
            }
        )
        return _closed_object(
            {
                "related_sets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": observation,
                }
            }
        )
    if relational_shape not in {"ordinary", "every_related_row"}:
        raise ValueError("unknown relational shape")
    related_sets: dict[str, object] = {
        "type": "array",
        "items": {"$ref": f"#/$defs/{graph_definition_name}"},
    }
    return _closed_object(
        {
            "related_sets": related_sets,
        }
    )


def _grouping_schema(
    *,
    grouping_kinds: tuple[str, ...],
    grouping_value_shapes: tuple[tuple[str, str | None] | None, ...],
) -> dict[str, object]:
    if len(grouping_kinds) != len(grouping_value_shapes):
        raise ValueError("grouping kinds and value shapes must align")
    common = {
        "id": _local_ref_schema("g"),
        "grouping_basis": {"type": "string", "minLength": 1},
    }
    branches: list[dict[str, object]] = []
    for grouping_kind, value_shape in zip(
        grouping_kinds,
        grouping_value_shapes,
        strict=True,
    ):
        if grouping_kind == "qualifying_row_identity":
            branches.append(
                output.CandidateIdentityGroupingOutput.schema(
                    {
                        **common,
                        "kind": {"enum": ["candidate_instance_identity"]},
                    }
                )
            )
            continue
        if grouping_kind == "related_entity_identity":
            branches.append(
                output.RelatedIdentityGroupingOutput.schema(
                    {
                        **common,
                        "kind": {"enum": ["related_instance_identity"]},
                        "set_ref": _additional_set_ref_schema(),
                    }
                )
            )
            continue
        if grouping_kind != "non_identity_value":
            raise ValueError(f"unknown grouping kind: {grouping_kind}")
        branches.append(
            output.ValueGroupingOutput.schema(
                {
                    **common,
                    "kind": {"enum": ["value"]},
                    "expression": _fixed_grouping_value_expression_schema(value_shape),
                }
            )
        )
    unique = {_schema_key(branch): branch for branch in branches}
    return {"oneOf": list(unique.values())}


def _fixed_grouping_value_expression_schema(
    shape: tuple[str, str | None] | None,
) -> dict[str, object]:
    if shape is None:
        return _grouping_value_expression_schema()
    kind, grain = shape
    if kind == "observed_value" and grain is None:
        return _fact_expression_schema(definition="grouping_fact_expression")
    if kind == "computed_value" and grain is None:
        return {
            "oneOf": list(
                _derived_value_schemas(_grouping_value_expression_schema(depth=1))[:2]
            )
        }
    if kind == "condition" and grain is None:
        return _row_condition_expression_schema()
    if kind == "temporal_bucket" and grain is not None:
        return _closed_object(
            {
                "kind": {"enum": ["temporal_bucket"]},
                "value": _fact_expression_schema(definition="grouping_fact_expression"),
                "grain": {"enum": [grain]},
            }
        )
    raise ValueError("invalid grouping value shape")


def _schema_key(schema: dict[str, object]) -> str:
    return repr(schema)


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
    if kind in {"take_with_boundary_ties", "position_with_ties"}:
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


def _grouping_origin_definition() -> dict[str, object]:
    common = {
        "group_ref": {"enum": ["g1", "g2", "g3", "g4"]},
        "grouping_basis": {"type": "string", "minLength": 1},
        "meaning": {"type": "string", "minLength": 1},
        "origin": {"$ref": "#/$defs/frame_origin"},
    }
    observed_value = output.NonTemporalGroupingValueOutput.schema(
        {"kind": {"enum": ["observed_value", "computed_value", "condition"]}}
    )
    temporal_bucket = output.TemporalBucketGroupingValueOutput.schema(
        {
            "kind": {"enum": ["temporal_bucket"]},
            "grain": {"enum": ["day", "week", "month", "quarter", "year"]},
        }
    )
    return {
        "oneOf": [
            output.GroupingMeaningOutput.schema(
                {
                    **common,
                    "grouping_kind": {"enum": ["qualifying_row_identity"]},
                }
            ),
            output.GroupingMeaningOutput.schema(
                {
                    **common,
                    "grouping_kind": {"enum": ["related_entity_identity"]},
                }
            ),
            output.GroupingMeaningOutput.schema(
                {
                    **common,
                    "grouping_kind": {"enum": ["non_identity_value"]},
                    "grouping_value": {"oneOf": [observed_value, temporal_bucket]},
                }
            ),
        ]
    }


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
                    _closed_object({"kind": {"enum": ["categorical_value"]}}),
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


def _fact_expression_definition(
    *,
    observed_for_ref: dict[str, object] | None = None,
    identified_set_ref: dict[str, object] | None = None,
) -> dict[str, object]:
    properties = {
        "kind": {"enum": ["fact"]},
        "observed_for_ref": observed_for_ref or _semantic_owner_ref_schema(),
        "origin": _source_origin_schema(),
    }
    if identified_set_ref is not None:
        properties["identified_set_ref"] = identified_set_ref
    return output.FactTermOutput.schema(properties)


def _identifier_fact_expression_definition() -> dict[str, object]:
    identity_paths = [
        _closed_object(
            {
                "kind": {"enum": ["candidate_instance"]},
            }
        )
    ]
    identity_paths.append(
        _closed_object(
            {
                "kind": {"enum": ["related_instance"]},
                "association_ref": {"type": "string", "minLength": 1},
            }
        )
    )
    return _closed_object(
        {
            "kind": {"enum": ["fact"]},
            "identity_path": {"oneOf": identity_paths},
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
        "anyOf": [
            _value_expression_schema(),
            _condition_expression_schema(),
            _fact_expression_schema(definition="identifier_fact_expression"),
            _set_ref_expression_schema(),
        ]
    }


def _value_output_expression_schema() -> dict[str, object]:
    return {
        "anyOf": [
            _value_expression_schema(),
            _condition_expression_schema(),
        ]
    }


def _identity_output_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _closed_object(
                {
                    "kind": {"enum": ["fact"]},
                    "identity_path": _closed_object(
                        {"kind": {"enum": ["candidate_instance"]}}
                    ),
                    "origin": _source_origin_schema(),
                }
            ),
            _closed_object(
                {"kind": {"enum": ["set_ref"]}, "set_ref": {"enum": ["s1"]}}
            ),
        ]
    }


def _group_output_expression_schema() -> dict[str, object]:
    return {
        "anyOf": [
            _value_expression_schema(),
            _condition_expression_schema(),
            _group_ref_expression_schema(),
        ]
    }


def _group_identity_output_expression_schema() -> dict[str, object]:
    return _group_ref_expression_schema()


def _group_ordering_expression_schema() -> dict[str, object]:
    return _group_result_value_expression_schema()


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
    allow_global_quantification: bool = False,
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
                    "input": _typed_identity_input_expression_schema(
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
                    "input": _typed_identity_input_expression_schema(
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
                        "association_refs": {**_ref_array_schema(), "minItems": 0 if allow_global_quantification else 1},
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


def _typed_identity_input_expression_schema(*, definition: str) -> dict[str, object]:
    return _closed_object(
        {
            "kind": {"enum": ["input_ref"]},
            "input_ref": {"$ref": f"#/$defs/{definition}"},
            "operand_meaning": {"type": "string", "minLength": 1},
            "instance_kind": {"type": "string", "minLength": 1},
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
                identifier_fact_definition="identifier_fact_expression",
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
            quantified_child=_row_condition_expression_schema(depth=max(0, depth - 1)),
            allow_global_quantification=True,
            allow_scalar_input=allow_scalar_input,
            allow_collection_input=allow_collection_input,
            allow_identity_input=allow_identity_input,
            allow_identity_collection_input=allow_identity_collection_input,
            allow_temporal_scope_input=allow_temporal_scope_input,
            allow_coverage=allow_coverage,
        )
    return definitions


def _finite_scoped_condition_definitions(
    *,
    allow_scalar_input: bool,
    allow_collection_input: bool,
    allow_identity_input: bool,
    allow_identity_collection_input: bool,
    allow_temporal_scope_input: bool,
) -> dict[str, object]:
    definitions: dict[str, object] = {
        "scoped_fact_expression": _closed_object(
            {
                "kind": {"enum": ["fact"]},
                "origin": _source_origin_schema(),
            }
        )
    }
    for depth in range(3):
        value_name = f"scoped_row_value_expression_{depth}"
        value_variants: list[dict[str, object]] = [
            _fact_expression_schema(definition="scoped_fact_expression")
        ]
        if allow_scalar_input:
            value_variants.append(_input_expression_schema())
        if depth:
            value_variants.extend(
                _derived_value_schemas(
                    {"$ref": (f"#/$defs/scoped_row_value_expression_{depth - 1}")}
                )
            )
        definitions[value_name] = {"oneOf": value_variants}
        child: dict[str, object] | None = (
            None
            if depth == 0
            else {"$ref": f"#/$defs/scoped_condition_expression_{depth - 1}"}
        )
        definitions[f"scoped_condition_expression_{depth}"] = (
            _condition_expression_definition(
                value={"$ref": f"#/$defs/{value_name}"},
                child=child,
                quantified_child=child,
                allow_scalar_input=allow_scalar_input,
                allow_collection_input=allow_collection_input,
                allow_identity_input=allow_identity_input,
                allow_identity_collection_input=allow_identity_collection_input,
                allow_temporal_scope_input=allow_temporal_scope_input,
                allow_coverage=False,
                temporal_fact_definition="scoped_fact_expression",
                boolean_fact_definition="scoped_fact_expression",
                identifier_fact_definition="scoped_fact_expression",
            )
        )
    return definitions


def _scoped_condition_expression_schema() -> dict[str, object]:
    return {"$ref": "#/$defs/scoped_condition_expression_2"}


def _coverage_expression_schema() -> dict[str, object]:
    condition = _scoped_condition_expression_schema()
    return _closed_object(
        {
            "kind": {"enum": ["coverage"]},
            "candidate_condition": {"oneOf": [condition, {"type": "null"}]},
            "required_member_set_ref": {"enum": ["s2"]},
            "required_member_condition": {"oneOf": [condition, {"type": "null"}]},
            "observation": _closed_object(
                {
                    "set_ref": {"enum": ["s3"]},
                    "condition": condition,
                }
            ),
        }
    )


def _related_row_expression_schema() -> dict[str, object]:
    return _closed_object(
        {
            "kind": {"enum": ["related_row"]},
            "set_ref": _local_ref_schema("s"),
            "condition": {
                "oneOf": [
                    {"type": "null"},
                    {"$ref": "#/$defs/row_condition_expression_3"},
                ]
            },
        }
    )


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


def _additional_set_ref_schema(*, minimum_index: int = 2) -> dict[str, object]:
    if minimum_index < 2 or minimum_index > 9:
        raise ValueError("additional set reference minimum must be between 2 and 9")
    return {
        "type": "string",
        "pattern": rf"^s(?:[{minimum_index}-9]|[1-9][0-9]+)$",
    }


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
