"""Temporary isolated comparison of Grounding prompt clarifications."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI

from fervis.model_io.providers.chat_runtime import (
    chat_tool_system_prompt,
    provider_max_output_tokens,
    provider_max_retries,
    provider_timeout_seconds,
)
from fervis.model_io.providers.openai_compatible_adapter.loop_adapter import (
    _openai_strict_schema,
)


CAPTURES = (Path("/tmp/single-inspect/index.json"), Path("/tmp/pair-inspect/index.json"))
RESOURCE_TYPE_X = "staff member"
CLARIFICATIONS = {
    "baseline": "",
    "agent_1": (
        "Classify lookup_text once from its supplied field label and meaning; keep "
        "the same resource type X and primary-key-versus-descriptive classification "
        "across every read. A parameter merely accepting the value's data type does "
        "not reclassify it."
    ),
    "agent_2": (
        "For each known input, determine resource type X and whether lookup_text is "
        "X's primary key once from question_text, field_label_text, and "
        "value_meaning_hint; keep both conclusions fixed across all binding options. "
        "An option's returned type, parameter name, or matching field must not "
        "redefine X or turn a primary-key lookup into a descriptive lookup."
    ),
    "agent_3": (
        "For each known input, determine the lookup's entity kind and "
        "primary-key-versus-descriptive status once from field_label_text and "
        "value_meaning_hint, then keep that classification fixed across every option."
    ),
    "derived_resource_type": "",
    "lookup_request_params": "",
    "lookup_request_closed_union": "",
    "lookup_request_reasoned_union": "",
}
OLD_QUESTION = re.compile(
    r"^Can read (?P<read>\S+) resolve (?P<lookup>.+?) as the returned resource "
    r"and produce .+? for target meaning .+?\?$"
)
DELETE = (
    "Do not choose one resolver.",
    "Do not execute a resolver read during this turn.",
    "Do not decide which answer read will consume the result.",
    "Review every binding option independently. More than one option may be positive.",
    "A positive review means only that the read can validate or match the supplied "
    "lookup text and produce its declared canonical result through the selected "
    "request values and exact-match fields.",
    "A route can resolve lookup_text only when both mechanics work: its selected "
    "declared request parameters can perform this lookup, and its selected "
    "returned-resource fields can exact-match lookup_text on the returned resource.",
    "Use CAN_RESOLVE_LOOKUP_TEXT when both mechanics work. Use "
    "CANNOT_RESOLVE_LOOKUP_TEXT when either mechanic fails.",
    "A positive option must use the lookup text to identify the returned resource itself.",
    "Select every returned-resource field that may exactly equal lookup_text.",
    "response_match_alternatives has OR semantics: an exact match in any selected "
    "field verifies the returned resource. Include each selected field exactly once. "
    "Do not select fields that describe another entity, category, or surrounding context.",
    'For CAN_RESOLVE_LOOKUP_TEXT, write the because field as: "The route can look '
    "up {lookup_text} using {selected request parameters} because {what those "
    "parameters accept or search}. If returned, {selected response fields} can "
    "exact-match {lookup_text} on the returned {resource}. The route returns "
    '{canonical result}."',
    'For CANNOT_RESOLVE_LOOKUP_TEXT, write the because field as: "The route cannot '
    "resolve {lookup_text}. Its shown request parameters {can/cannot} perform this "
    "lookup because {what those parameters accept or search}. If returned, {shown "
    "response fields} {can/cannot} exact-match {lookup_text} on the returned "
    '{resource}. The route returns {canonical result}." At least one stated mechanic '
    "must be cannot.",
    "Replace every template term with concrete text from the option. Write decision after because.",
    "Judge business meaning, not word equality between the input hint and the returned entity kind.",
    "Do not reject an option because its result might not fit the final answer source. Later stages own that decision.",
)
OLD_MATCH_RULE = (
    "Match fields establish which resource was named, but they never become computation values."
)
OLD_TASK_DESCRIPTION = (
    "Your task is to resolve time inputs and review named-reference resolver options."
)
NEUTRAL_TASK_DESCRIPTION = (
    "Your task is to resolve time inputs and review identity resolver options."
)


def resolver_question(
    read: str,
    lookup: str,
    resource_type: str,
    *,
    derive_resource_type: bool,
) -> str:
    resource_type_x = (
        "the input-wide resource_type_x"
        if derive_resource_type
        else f"resource_type_x {RESOURCE_TYPE_X}"
    )
    return (
        f"Does read {read}, which returns resource_type {resource_type}, return the "
        f"resource type described by {resource_type_x}? If not, it "
        f"cannot resolve {lookup}. If it does, can this route resolve {lookup} under "
        "the input-wide identifier_kind?"
    )


OPTION_CARD = re.compile(
    r'<binding_option id="(?P<option_id>[^"]+)">\s*'
    r'<resolver_fit_question>(?P<question>[^<]+)</resolver_fit_question>'
    r'(?P<body>.*?)'
    r'<canonical_result entity_kind="(?P<resource_type>[^"]+)"',
    re.DOTALL,
)


def option_contracts(
    prompt: str,
    *,
    derive_resource_type: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    question_mapping: dict[str, str] = {}
    option_resource_types: dict[str, str] = {}
    for match in OPTION_CARD.finditer(prompt):
        question = match.group("question")
        parsed = OLD_QUESTION.fullmatch(question)
        if parsed is None:
            raise ValueError(f"unexpected resolver question: {question}")
        resource_type = match.group("resource_type")
        question_mapping[question] = resolver_question(
            parsed.group("read"),
            parsed.group("lookup"),
            resource_type,
            derive_resource_type=derive_resource_type,
        )
        option_resource_types[match.group("option_id")] = resource_type
    if not question_mapping:
        raise ValueError("grounding prompt has no resolver option cards")
    return question_mapping, option_resource_types


def add_resource_type_evidence(
    prompt: str,
    *,
    option_resource_types: dict[str, str],
) -> str:
    prompt = re.sub(
        r'(<known_input id="[^"]+")',
        rf'\1 resource_type_x="{RESOURCE_TYPE_X}"',
        prompt,
    )
    for option_id, resource_type in option_resource_types.items():
        opening = f'<binding_option id="{option_id}">'
        replacement = f'{opening}\n      <resource_type>{resource_type}</resource_type>'
        if opening not in prompt:
            raise ValueError(f"missing option card: {option_id}")
        prompt = prompt.replace(opening, replacement)
    return prompt


def add_resource_type_output_contract(
    schema: dict[str, Any],
    *,
    option_resource_types: dict[str, str],
    derive_resource_type: bool,
) -> None:
    reviews = schema["properties"]["known_input_binding_reviews"]["properties"]
    for review in reviews.values():
        old_properties = review["properties"]
        option_reviews = old_properties["option_reviews"]
        shown_resource_types = sorted(
            {
                option_resource_types[option_id]
                for option_id in option_reviews["properties"]
            }
        )
        review["properties"] = {
            **(
                {"resource_type_basis": {"type": "string", "minLength": 1}}
                if derive_resource_type
                else {}
            ),
            "resource_type_x": (
                {
                    "type": "string",
                    "enum": [*shown_resource_types, "NO_SHOWN_RESOURCE_TYPE"],
                }
                if derive_resource_type
                else {"type": "string", "enum": [RESOURCE_TYPE_X]}
            ),
            "identifier_kind_basis": {"type": "string", "minLength": 1},
            "identifier_kind": {
                "type": "string",
                "enum": ["PRIMARY_KEY", "DESCRIPTIVE"],
            },
            "option_reviews": option_reviews,
        }
        review["required"] = [
            *(["resource_type_basis"] if derive_resource_type else []),
            "resource_type_x",
            "identifier_kind_basis",
            "identifier_kind",
            "option_reviews",
        ]
        for option_id, option_schema in option_reviews["properties"].items():
            resource_type = option_resource_types[option_id]
            variants = option_schema.pop("oneOf")
            negative = next(
                variant
                for variant in variants
                if variant["properties"]["decision"]["enum"]
                == ["CANNOT_RESOLVE_LOOKUP_TEXT"]
            )
            positive = next(
                (
                    variant
                    for variant in variants
                    if variant["properties"]["decision"]["enum"]
                    == ["CAN_RESOLVE_LOOKUP_TEXT"]
                ),
                None,
            )
            common = negative["properties"]
            decision_values = ["CANNOT_RESOLVE_LOOKUP_TEXT"]
            request_value_variants = [common["request_values"]]
            response_match_schema = common["response_match_alternatives"]
            if positive is not None:
                positive_properties = positive["properties"]
                decision_values.insert(0, "CAN_RESOLVE_LOOKUP_TEXT")
                request_value_variants.insert(
                    0,
                    positive_properties["request_values"],
                )
                response_match_schema = copy.deepcopy(
                    positive_properties["response_match_alternatives"]
                )
                response_match_schema.pop("minItems", None)
            option_schema.update(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "resource_type": {
                            "type": "string",
                            "enum": [resource_type],
                        },
                        "resource_type_match": {
                            "type": "string",
                            "enum": [
                                "SAME_RESOURCE_TYPE",
                                "DIFFERENT_RESOURCE_TYPE",
                            ],
                        },
                        "resolver_fit_question": common[
                            "resolver_fit_question"
                        ],
                        "because": common["because"],
                        "decision": {
                            "type": "string",
                            "enum": decision_values,
                        },
                        "request_values": {
                            "anyOf": request_value_variants,
                        },
                        "response_match_alternatives": response_match_schema,
                    },
                }
            )
            option_schema["required"] = list(option_schema["properties"])


def add_lookup_request_output_contract(schema: dict[str, Any]) -> None:
    reviews = schema["properties"]["known_input_binding_reviews"]["properties"]
    for review in reviews.values():
        option_reviews = review["properties"]["option_reviews"]["properties"]
        for option_schema in option_reviews.values():
            properties = option_schema["properties"]
            option_schema["properties"] = {
                "resource_type": properties["resource_type"],
                "resource_type_match": properties["resource_type_match"],
                "resolver_fit_question": properties["resolver_fit_question"],
                "lookup_request_params": properties["request_values"],
                "returned_identity_verification_fields": properties[
                    "response_match_alternatives"
                ],
                "because": properties["because"],
                "decision": properties["decision"],
            }
            option_schema["required"] = list(option_schema["properties"])


def add_lookup_request_closed_union(schema: dict[str, Any]) -> None:
    reviews = schema["properties"]["known_input_binding_reviews"]["properties"]
    for review in reviews.values():
        option_reviews = review["properties"]["option_reviews"]["properties"]
        for option_schema in option_reviews.values():
            properties = option_schema["properties"]
            request_variants = properties["request_values"]["anyOf"]
            positive_request = next(
                (
                    variant
                    for variant in request_variants
                    if variant.get("properties")
                ),
                None,
            )
            response_fields = properties["response_match_alternatives"]
            positive_allowed = (
                "CAN_RESOLVE_LOOKUP_TEXT"
                in properties["decision"]["enum"]
                and positive_request is not None
                and bool(response_fields["items"].get("enum"))
            )
            negative = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "resource_type": properties["resource_type"],
                    "resource_type_match": properties["resource_type_match"],
                    "resolver_fit_question": properties["resolver_fit_question"],
                    "lookup_request_params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {},
                            "required": [],
                        },
                        "maxItems": 0,
                    },
                    "returned_identity_verification_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 0,
                    },
                    "because": properties["because"],
                    "decision": {
                        "type": "string",
                        "enum": ["CANNOT_RESOLVE_LOOKUP_TEXT"],
                    },
                },
            }
            negative["required"] = list(negative["properties"])
            if not positive_allowed:
                option_schema.clear()
                option_schema.update(negative)
                continue
            param_variants = [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "param_ref": {"type": "string", "enum": [param_ref]},
                        "value": value_schema,
                    },
                    "required": ["param_ref", "value"],
                }
                for param_ref, value_schema in positive_request["properties"].items()
            ]
            positive = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "resource_type": properties["resource_type"],
                    "resource_type_match": properties["resource_type_match"],
                    "resolver_fit_question": properties["resolver_fit_question"],
                    "lookup_request_params": {
                        "type": "array",
                        "items": (
                            param_variants[0]
                            if len(param_variants) == 1
                            else {"oneOf": param_variants}
                        ),
                        "minItems": 1,
                        "maxItems": len(param_variants),
                    },
                    "returned_identity_verification_fields": {
                        **copy.deepcopy(response_fields),
                        "minItems": 1,
                    },
                    "because": properties["because"],
                    "decision": {
                        "type": "string",
                        "enum": ["CAN_RESOLVE_LOOKUP_TEXT"],
                    },
                },
            }
            positive["required"] = list(positive["properties"])
            option_schema.clear()
            option_schema["oneOf"] = [positive, negative]


def add_lookup_request_reasoned_union(schema: dict[str, Any]) -> None:
    reviews = schema["properties"]["known_input_binding_reviews"]["properties"]
    for review in reviews.values():
        option_reviews = review["properties"]["option_reviews"]["properties"]
        for option_schema in option_reviews.values():
            properties = option_schema["properties"]
            request_variants = properties["request_values"]["anyOf"]
            positive_request = next(
                (
                    variant
                    for variant in request_variants
                    if variant.get("properties")
                ),
                None,
            )
            response_fields = properties["response_match_alternatives"]
            positive_allowed = (
                "CAN_RESOLVE_LOOKUP_TEXT"
                in properties["decision"]["enum"]
                and positive_request is not None
                and bool(response_fields["items"].get("enum"))
            )
            empty_lookup_params = {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
                "maxItems": 0,
            }
            empty_verification_fields = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            }
            negative_resolution = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["CANNOT_RESOLVE_LOOKUP_TEXT"],
                    },
                    "lookup_request_params": empty_lookup_params,
                    "returned_identity_verification_fields": (
                        empty_verification_fields
                    ),
                },
            }
            negative_resolution["required"] = list(
                negative_resolution["properties"]
            )
            resolution_schema: dict[str, Any] = negative_resolution
            if positive_allowed:
                param_variants = [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "param_ref": {
                                "type": "string",
                                "enum": [param_ref],
                            },
                            "value": value_schema,
                        },
                        "required": ["param_ref", "value"],
                    }
                    for param_ref, value_schema in positive_request[
                        "properties"
                    ].items()
                ]
                positive_resolution = {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["CAN_RESOLVE_LOOKUP_TEXT"],
                        },
                        "lookup_request_params": {
                            "type": "array",
                            "items": (
                                param_variants[0]
                                if len(param_variants) == 1
                                else {"oneOf": param_variants}
                            ),
                            "minItems": 1,
                            "maxItems": len(param_variants),
                        },
                        "returned_identity_verification_fields": {
                            **copy.deepcopy(response_fields),
                            "minItems": 1,
                        },
                    },
                }
                positive_resolution["required"] = list(
                    positive_resolution["properties"]
                )
                resolution_schema = {
                    "oneOf": [negative_resolution, positive_resolution]
                }
            option_schema.clear()
            option_schema.update(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "resource_type": properties["resource_type"],
                        "resource_type_match": properties[
                            "resource_type_match"
                        ],
                        "resolver_fit_question": properties[
                            "resolver_fit_question"
                        ],
                        "because": properties["because"],
                        "resolution": resolution_schema,
                    },
                }
            )
            option_schema["required"] = list(option_schema["properties"])


def add_lookup_request_prompt_contract(
    prompt: str,
    *,
    closed_union: bool,
) -> str:
    old_order = (
        "Within each option review, write fields in this order: resource_type, "
        "resource_type_match, resolver_fit_question, because, decision, "
        "request_values, response_match_alternatives."
    )
    new_order = (
        "Within each option review, write fields in this order: resource_type, "
        "resource_type_match, resolver_fit_question, lookup_request_params, "
        "returned_identity_verification_fields, because, decision."
    )
    if old_order not in prompt:
        raise ValueError(f"missing baseline text: {old_order}")
    prompt = prompt.replace(old_order, new_order)
    positive_output = (
        "For CAN_RESOLVE_LOOKUP_TEXT, return request_values keyed by param_ref and "
        "at least one response_match_alternative."
    )
    positive_replacement = (
        "lookup_request_params answers: “Which shown request parameter or "
        "parameters exactly match lookup_text’s identifier meaning for "
        "resource_type_x?” Return those parameter-value pairs, or "
        + ("[]" if closed_union else "{}")
        + " when none "
        "match.\n"
        "For CAN_RESOLVE_LOOKUP_TEXT, return at least one lookup_request_param and "
        "at least one returned_identity_verification_field."
    )
    if positive_output not in prompt:
        raise ValueError(f"missing baseline text: {positive_output}")
    prompt = prompt.replace(positive_output, positive_replacement)
    negative_output = (
        "For CANNOT_RESOLVE_LOOKUP_TEXT, return an empty request_values object and "
        "an empty response_match_alternatives array."
    )
    negative_replacement = (
        "For CANNOT_RESOLVE_LOOKUP_TEXT, return an empty lookup_request_params "
        + ("array" if closed_union else "object")
        + " and an empty returned_identity_verification_fields array."
    )
    if negative_output not in prompt:
        raise ValueError(f"missing baseline text: {negative_output}")
    prompt = prompt.replace(negative_output, negative_replacement)
    prompt = prompt.replace(
        "Match fields never become computation values.",
        "Returned identity verification fields never become computation values.",
    )
    return prompt


def add_lookup_request_reasoned_prompt_contract(prompt: str) -> str:
    old_order = (
        "Within each option review, write fields in this order: resource_type, "
        "resource_type_match, resolver_fit_question, because, decision, "
        "request_values, response_match_alternatives."
    )
    new_order = (
        "Within each option review, write fields in this order: resource_type, "
        "resource_type_match, resolver_fit_question, because, resolution. "
        "Within resolution, write decision, lookup_request_params, then "
        "returned_identity_verification_fields."
    )
    if old_order not in prompt:
        raise ValueError(f"missing baseline text: {old_order}")
    prompt = prompt.replace(old_order, new_order)
    positive_output = (
        "For CAN_RESOLVE_LOOKUP_TEXT, return request_values keyed by param_ref and "
        "at least one response_match_alternative."
    )
    positive_replacement = (
        "lookup_request_params answers: “Which shown request parameter or "
        "parameters exactly match lookup_text’s identifier meaning for "
        "resource_type_x?” Return those parameter-value pairs, or [] when none "
        "match.\n"
        "returned_identity_verification_fields are returned-resource fields that "
        "may exactly equal lookup_text. For PRIMARY_KEY, only fields declared by "
        "canonical_result.components are valid. For DESCRIPTIVE, a field is valid "
        "only when its declared type and choices accept lookup_text and it "
        "describes the returned resource itself.\n"
        "For CAN_RESOLVE_LOOKUP_TEXT, resolution must contain at least one "
        "lookup_request_param and at least one "
        "returned_identity_verification_field."
    )
    if positive_output not in prompt:
        raise ValueError(f"missing baseline text: {positive_output}")
    prompt = prompt.replace(positive_output, positive_replacement)
    negative_output = (
        "For CANNOT_RESOLVE_LOOKUP_TEXT, return an empty request_values object and "
        "an empty response_match_alternatives array."
    )
    negative_replacement = (
        "For CANNOT_RESOLVE_LOOKUP_TEXT, resolution must contain empty "
        "lookup_request_params and returned_identity_verification_fields arrays."
    )
    if negative_output not in prompt:
        raise ValueError(f"missing baseline text: {negative_output}")
    prompt = prompt.replace(negative_output, negative_replacement)
    return prompt.replace(
        "Match fields never become computation values.",
        "Returned identity verification fields never become computation values.",
    )


def rewrite(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {key: rewrite(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [rewrite(child, mapping) for child in value]
    return value


def prepare(
    path: Path,
    clarification: str,
    *,
    derive_resource_type: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    [turn] = [
        turn for run in payload["runs"] for turn in run["model_turns"]
        if turn.get("purpose") == "grounding"
    ]
    turn = copy.deepcopy(turn)
    mapping, option_resource_types = option_contracts(
        turn["prompt"],
        derive_resource_type=derive_resource_type,
    )
    turn = rewrite(turn, mapping)
    add_resource_type_output_contract(
        turn["tool_specs"][0]["input_schema"],
        option_resource_types=option_resource_types,
        derive_resource_type=derive_resource_type,
    )
    if clarification == "lookup_request_params":
        add_lookup_request_output_contract(turn["tool_specs"][0]["input_schema"])
    elif clarification == "lookup_request_closed_union":
        add_lookup_request_closed_union(turn["tool_specs"][0]["input_schema"])
    elif clarification == "lookup_request_reasoned_union":
        add_lookup_request_reasoned_union(
            turn["tool_specs"][0]["input_schema"]
        )
    prompt = turn["prompt"]
    if not derive_resource_type:
        prompt = add_resource_type_evidence(
            prompt,
            option_resource_types=option_resource_types,
        )
    else:
        for option_id, resource_type in option_resource_types.items():
            opening = f'<binding_option id="{option_id}">'
            replacement = (
                f'{opening}\n      <resource_type>{resource_type}</resource_type>'
            )
            if opening not in prompt:
                raise ValueError(f"missing option card: {option_id}")
            prompt = prompt.replace(opening, replacement)
        shown_resource_types = "".join(
            f"\n      <resource_type>{resource_type}</resource_type>"
            for resource_type in sorted(set(option_resource_types.values()))
        )
        prompt = re.sub(
            r'(<known_input\b[^>]*>)',
            (
                r'\1\n    <shown_resource_types>'
                + shown_resource_types
                + "\n    </shown_resource_types>"
            ),
            prompt,
        )
    if OLD_TASK_DESCRIPTION not in prompt:
        raise ValueError(f"missing baseline text: {OLD_TASK_DESCRIPTION}")
    prompt = prompt.replace(OLD_TASK_DESCRIPTION, NEUTRAL_TASK_DESCRIPTION)
    for text in DELETE:
        if text not in prompt:
            raise ValueError(f"missing baseline text: {text}")
        prompt = prompt.replace(text, "")
    if OLD_MATCH_RULE not in prompt:
        raise ValueError(f"missing baseline text: {OLD_MATCH_RULE}")
    prompt = prompt.replace(OLD_MATCH_RULE, "Match fields never become computation values.")
    resource_type_instruction = (
        "Before option_reviews, use question_text, field_label_text, and "
        "value_meaning_hint as catalog-blind semantic evidence. Write "
        "resource_type_basis first, stating which resource the input identifies. "
        "Then set resource_type_x to exactly one shown_resource_type. Use "
        "NO_SHOWN_RESOURCE_TYPE only when none represents that resource. Keep "
        "resource_type_x fixed across every option. Copy each option's shown "
        "resource_type first. SAME_RESOURCE_TYPE means it exactly equals "
        "resource_type_x; otherwise use DIFFERENT_RESOURCE_TYPE."
        if derive_resource_type
        else "resource_type_x is catalog-blind semantic direction. Copy it before "
        "option_reviews. Copy each option's shown resource_type first, then decide "
        "whether it represents resource_type_x by business meaning rather than "
        "spelling."
    )
    prompt = prompt.replace(
        "Use field_label_text and value_meaning_hint together to understand what the supplied text means. Both are catalog-blind approximations, not authoritative catalog names.",
        resource_type_instruction,
    )
    prompt = prompt.replace(
        "For every option, answer the shown resolver_fit_question.",
        "SAME_RESOURCE_TYPE means an instance returned by the resolver is itself an instance of resource_type_x. A resource that references, contains information about, records activity for, or otherwise relates to resource_type_x is DIFFERENT_RESOURCE_TYPE.\nBefore option_reviews, write identifier_kind_basis and then identifier_kind once for the known input. Use PRIMARY_KEY when lookup_text is intended as the complete primary-key value of resource_type_x. Use DESCRIPTIVE when lookup_text is a non-primary-key value used to refer to resource_type_x.\nFor every option, first copy its shown resource_type, then write resource_type_match. DIFFERENT_RESOURCE_TYPE always requires CANNOT_RESOLVE_LOOKUP_TEXT. Only SAME_RESOURCE_TYPE proceeds to resolver_fit_question and route-mechanics assessment.\nFor every option, answer the shown resolver_fit_question.",
    )
    prompt = prompt.replace(
        "Copy all IDs and each resolver_fit_question exactly.",
        "Copy all IDs and each resolver_fit_question exactly.\n"
        "Within each known-input review, write fields in this order: "
        + (
            "resource_type_basis, resource_type_x, identifier_kind_basis, "
            if derive_resource_type
            else "resource_type_x, identifier_kind_basis, "
        )
        + "identifier_kind, option_reviews.\n"
        "Within each option review, write fields in this order: resource_type, resource_type_match, resolver_fit_question, because, decision, request_values, response_match_alternatives.",
    )
    if clarification and clarification not in {
        "lookup_request_params",
        "lookup_request_closed_union",
        "lookup_request_reasoned_union",
    }:
        prompt = prompt.replace(
            "For every option, answer the shown resolver_fit_question.",
            "For every option, answer the shown resolver_fit_question.\n" + clarification,
        )
    turn["prompt"] = prompt.replace(
        "For CAN_RESOLVE_LOOKUP_TEXT, return request_values",
        "For CAN_RESOLVE_LOOKUP_TEXT, write because as: \"{resource_type} represents {resource_type_x}. With identifier_kind={identifier_kind}, this route can resolve {lookup_text} because {route evidence}.\"\n"
        "For CANNOT_RESOLVE_LOOKUP_TEXT, write because as either: \"{resource_type} does not represent {resource_type_x}.\" or \"{resource_type} represents {resource_type_x}, but with identifier_kind={identifier_kind}, this route cannot resolve {lookup_text} because {route evidence}.\"\n"
        "Write decision after because.\n"
        "For CAN_RESOLVE_LOOKUP_TEXT, return request_values",
    )
    if clarification in {"lookup_request_params", "lookup_request_closed_union"}:
        turn["prompt"] = add_lookup_request_prompt_contract(
            turn["prompt"],
            closed_union=clarification == "lookup_request_closed_union",
        )
    elif clarification == "lookup_request_reasoned_union":
        turn["prompt"] = add_lookup_request_reasoned_prompt_contract(
            turn["prompt"]
        )
    return turn, mapping


def model_call(turn: dict[str, Any]) -> dict[str, Any]:
    spec = turn["tool_specs"][0]
    response = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        timeout=provider_timeout_seconds(),
        max_retries=provider_max_retries(),
    ).chat.completions.create(
        model=turn["model_key"].split(":", 1)[-1],
        max_completion_tokens=provider_max_output_tokens(), temperature=0.0,
        messages=[
            {"role": "system", "content": chat_tool_system_prompt(turn["system_prompt"])},
            {"role": "user", "content": turn["prompt"]},
        ],
        tools=[{"type": "function", "function": {
            "name": spec["name"], "description": spec["description"],
            "parameters": _openai_strict_schema(spec["input_schema"]),
            "strict": spec["strict"],
        }}],
        tool_choice={"type": "function", "function": {"name": spec["name"]}},
        parallel_tool_calls=False,
    )
    [tool_call] = list(response.choices[0].message.tool_calls or [])
    return json.loads(tool_call.function.arguments)


def decisions(result: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    reads = {new: OLD_QUESTION.fullmatch(old).group("read") for old, new in mapping.items()}  # type: ignore[union-attr]
    return {
        input_id: {
            "resource_type_x": input_review.get("resource_type_x"),
            "identifier_kind": input_review.get("identifier_kind"),
            "options": {
                reads[review["resolver_fit_question"]]: _decision_payload(review)
                for review in input_review["option_reviews"].values()
            },
        }
        for input_id, input_review in result["known_input_binding_reviews"].items()
    }


def _decision_payload(review: dict[str, Any]) -> dict[str, Any]:
    resolution = review.get("resolution", review)
    return {
        "resource_type": review.get("resource_type"),
        "resource_type_match": review.get("resource_type_match"),
        "lookup_request_params": resolution.get("lookup_request_params"),
        "returned_identity_verification_fields": resolution.get(
            "returned_identity_verification_fields"
        ),
        "because": review.get("because"),
        "decision": resolution["decision"],
    }


def run(label: str) -> None:
    clarification = CLARIFICATIONS[label]
    for capture in CAPTURES:
        turn, mapping = prepare(
            capture,
            (
                label
                if label in {
                    "lookup_request_params",
                    "lookup_request_closed_union",
                    "lookup_request_reasoned_union",
                }
                else clarification
            ),
            derive_resource_type=label in {
                "derived_resource_type",
                "lookup_request_params",
                "lookup_request_closed_union",
                "lookup_request_reasoned_union",
            },
        )
        results: list[dict[str, dict[str, str]]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(model_call, turn): index for index in range(1, 11)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    value = decisions(future.result(), mapping)
                    results.append(value)
                    print(json.dumps({"recommendation": label, "capture": str(capture), "call": index, "decisions": value}, sort_keys=True), flush=True)
                except Exception as exc:
                    print(json.dumps({"recommendation": label, "capture": str(capture), "call": index, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        frequencies: dict[str, int] = {}
        for value in results:
            key = json.dumps(value, sort_keys=True)
            frequencies[key] = frequencies.get(key, 0) + 1
        print(json.dumps({"recommendation": label, "capture": str(capture), "successful_calls": len(results), "frequencies": frequencies}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recommendation", choices=tuple(CLARIFICATIONS))
    run(parser.parse_args().recommendation)
