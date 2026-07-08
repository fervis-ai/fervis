"""Readable API response-shape projections for lookup prompts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable

from fervis.lookup.relation_catalog import (
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    EndpointRead,
    RowCardinality,
    RowPath,
    catalog_input_param_token,
)
from fervis.lookup.fact_plan.row_sources import (
    executable_field_ids_for_row_path,
)
from fervis.lookup.fact_planning.executable_support import ScopedRowPredicate


@dataclass(frozen=True)
class ApiReadResponseShapeProjector:
    """Project an endpoint read into prompt-facing API-read shapes."""

    read: EndpointRead

    def input_params(
        self, *, include_param_tokens: bool = False
    ) -> list[dict[str, Any]]:
        return [
            _input_param_payload(
                read_id=self.read.id,
                param=param,
                include_param_tokens=include_param_tokens,
            )
            for param in self.read.params
        ]

    def response_rows(
        self,
        *,
        row_path_ids: Iterable[str] = (),
        source_candidate_id: str = "",
        include_evidence_tokens: bool = False,
    ) -> list[dict[str, Any]]:
        selected_row_path_ids = tuple(
            dict.fromkeys(str(item) for item in row_path_ids if str(item))
        )
        rows = self._selected_row_paths(selected_row_path_ids)
        return [
            row
            for row_path in rows
            for row in (
                self._response_row(
                    row_path,
                    source_candidate_id=source_candidate_id,
                    include_evidence_tokens=include_evidence_tokens,
                ),
            )
            if row is not None
        ]

    def row_predicates(
        self,
        *,
        row_path_ids: Iterable[str] = (),
        source_candidate_id: str = "",
        field_refs: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_row_path_ids = tuple(
            dict.fromkeys(str(item) for item in row_path_ids if str(item))
        )
        selected_paths = {
            row_path.id for row_path in self._selected_row_paths(selected_row_path_ids)
        }
        field_ids_by_ref = _executable_field_ids_by_ref(
            self.read,
            row_paths=self._selected_row_paths(selected_row_path_ids),
        )
        blocked_field_refs = _blocked_field_refs(self.read)
        output: list[dict[str, Any]] = []
        for field in self.read.fields:
            if field_refs is not None and field.ref not in field_refs:
                continue
            if field.ref in blocked_field_refs:
                continue
            if (field.row_path_id or "root") not in selected_paths:
                continue
            if _field_is_row_container(field_path=field.path, read=self.read):
                continue
            values = _row_predicate_values(field.type, field.choices)
            if not values:
                continue
            field_id = field_ids_by_ref.get(field.ref, "")
            if not field_id or _field_covered_by_query_param(
                field,
                field_id=field_id,
                read=self.read,
            ):
                continue
            output.append(
                ScopedRowPredicate(
                    source_candidate_id=source_candidate_id,
                    row_path_id=field.row_path_id or "root",
                    field_id=field_id,
                    field_path=field.path,
                    type=_row_predicate_type(field.type),
                    allowed_values=values,
                ).to_prompt_payload()
            )
        return output

    def xml(
        self,
        *,
        source_candidate_id: str,
        read_id: str = "",
        row_path_ids: Iterable[str] = (),
        include_evidence_tokens: bool = False,
        extra_attributes: dict[str, object] | None = None,
        indent: str = "  ",
    ) -> str:
        attributes: dict[str, object] = {
            "id": source_candidate_id,
            "read": read_id or self.read.id,
        }
        if extra_attributes:
            attributes.update(extra_attributes)
        lines = [f"<api_read{_xml_attrs(attributes)}>"]
        input_params = self.input_params(include_param_tokens=include_evidence_tokens)
        if input_params:
            lines.append(f"{indent}<input_params>")
            for param in input_params:
                attrs = {
                    key: param[key]
                    for key in (
                        "name",
                        "source",
                        "type",
                        "required",
                        "param_ref",
                        "param_token",
                    )
                    if key in param
                }
                lines.append(f"{indent * 2}<param{_xml_attrs(attrs)}>")
                if param.get("choices"):
                    lines.append(f"{indent * 3}<choices>")
                    for choice in param.get("choices") or ():
                        choice_attrs: dict[str, object] = {"value": choice}
                        labels = param.get("choice_labels")
                        if isinstance(labels, dict) and choice in labels:
                            choice_attrs["label"] = labels[choice]
                        lines.append(
                            f"{indent * 4}<choice{_xml_attrs(choice_attrs)} />"
                        )
                    lines.append(f"{indent * 3}</choices>")
                lines.append(f"{indent * 2}</param>")
            lines.append(f"{indent}</input_params>")
        response_rows = self.response_rows(
            row_path_ids=row_path_ids,
            source_candidate_id=source_candidate_id,
            include_evidence_tokens=include_evidence_tokens,
        )
        if response_rows:
            lines.append(f"{indent}<response>")
            lines.extend(_response_row_xml_lines(response_rows, indent=indent * 2))
            lines.append(f"{indent}</response>")
        row_predicates = self.row_predicates(
            row_path_ids=row_path_ids,
            source_candidate_id=source_candidate_id,
        )
        lines.extend(_row_predicates_xml_lines(row_predicates, indent=indent))
        lines.append("</api_read>")
        return "\n".join(lines)

    def _selected_row_paths(
        self,
        row_path_ids: tuple[str, ...],
    ) -> tuple[RowPath, ...]:
        if not row_path_ids:
            return self.read.row_paths or (
                RowPath(id="root", path="root", cardinality=RowCardinality.ONE),
            )
        selected = set(row_path_ids)
        if not self.read.row_paths and selected == {"root"}:
            return (RowPath(id="root", path="root", cardinality=RowCardinality.ONE),)
        return tuple(
            row_path for row_path in self.read.row_paths if row_path.id in selected
        )

    def _response_row(
        self,
        row_path: RowPath,
        *,
        source_candidate_id: str,
        include_evidence_tokens: bool,
    ) -> dict[str, Any] | None:
        fields = self._response_fields(
            row_path,
            source_candidate_id=source_candidate_id,
            include_evidence_tokens=include_evidence_tokens,
        )
        if not fields:
            return None
        row: dict[str, Any] = {
            "path": row_path.path or row_path.id,
            "cardinality": row_path.cardinality.value,
            "fields": fields,
        }
        if include_evidence_tokens and source_candidate_id:
            row["evidence_token"] = _row_evidence_token(
                source_candidate_id=source_candidate_id,
                row_path_id=row_path.id,
            )
        if row_path.parent_path:
            row["parent_path"] = row_path.parent_path
        return row

    def _response_fields(
        self,
        row_path: RowPath,
        *,
        source_candidate_id: str,
        include_evidence_tokens: bool,
    ) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        seen: set[str] = set()
        blocked_field_refs = _blocked_field_refs(self.read)
        for field in self.read.fields:
            if field.ref in blocked_field_refs:
                continue
            if (field.row_path_id or "root") != row_path.id:
                continue
            if _field_is_row_container(field_path=field.path, read=self.read):
                continue
            field_id = _field_id(field.path)
            if not field_id or field_id in seen:
                continue
            seen.add(field_id)
            payload: dict[str, Any] = {
                "field_id": field_id,
                "path": field.path,
                "type": field.type,
            }
            if include_evidence_tokens and source_candidate_id:
                payload["evidence_token"] = _field_evidence_token(
                    source_candidate_id=source_candidate_id,
                    field_path=field.path,
                )
            fields.append(payload)
        return fields


def _blocked_field_refs(read: EndpointRead) -> set[str]:
    return {
        fact.field_ref
        for fact in read.facts
        if fact.availability != CatalogFactAvailability.AVAILABLE and fact.field_ref
    }


def _field_id(path: str) -> str:
    return str(path or "").split(".")[-1]


def _executable_field_ids_by_ref(
    read: EndpointRead,
    *,
    row_paths: tuple[RowPath, ...],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row_path in row_paths:
        output.update(
            executable_field_ids_for_row_path(
                tuple(
                    field
                    for field in read.fields
                    if (field.row_path_id or "root") == row_path.id
                ),
                row_path=row_path.path,
                row_paths=row_paths,
            )
        )
    return output


def _row_predicate_values(field_type: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    if choices:
        return tuple(str(choice) for choice in choices if str(choice))
    if _row_predicate_type(field_type) == "boolean":
        return ("true", "false")
    return ()


def _row_predicate_type(field_type: str) -> str:
    normalized = str(field_type or "").strip().lower()
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"choice", "enum"}:
        return "choice"
    return normalized


def _field_covered_by_query_param(
    field: CatalogField,
    *,
    field_id: str,
    read: EndpointRead,
) -> bool:
    field_key = _dedupe_name(field_id)
    field_values = _choice_value_set(field.choices)
    for param in read.params:
        if _dedupe_name(param.name) == field_key:
            return True
        if field.ref and param.ref == field.ref:
            return True
        if field_values and field_values == _choice_value_set(param.choices):
            return True
    return False


def _choice_value_set(values: tuple[str, ...]) -> frozenset[str]:
    return frozenset(_dedupe_name(value) for value in values if _dedupe_name(value))


def _dedupe_name(value: object) -> str:
    return str(value or "").strip().lower()


def api_read_cards_xml(payload: dict[str, Any]) -> str:
    lines = ["<candidate_api_reads>"]
    for group in payload.get("requested_fact_read_candidates") or ():
        if not isinstance(group, dict):
            continue
        lines.append(
            f"  <requested_fact id={_xml_quote(group.get('requested_fact_id'))}>"
        )
        for card in group.get("read_candidates") or ():
            if not isinstance(card, dict):
                continue
            lines.extend(_api_read_card_xml_lines(card, indent="    "))
        lines.append("  </requested_fact>")
    lines.append("</candidate_api_reads>")
    return "\n".join(lines)


def source_strategy_candidates_xml(payload: dict[str, Any]) -> str:
    lines = ["<plan_selection_source_strategies>"]
    for group in payload.get("requested_fact_source_strategies") or ():
        if not isinstance(group, dict):
            continue
        lines.append(
            f"  <requested_fact id={_xml_quote(group.get('requested_fact_id'))}>"
        )
        for strategy in group.get("source_strategies") or ():
            if not isinstance(strategy, dict):
                continue
            attrs = {
                "id": strategy.get("source_strategy_id"),
                "plan_shape": strategy.get("plan_shape"),
                "answer_outputs": _space_separated(
                    strategy.get("required_answer_output_ids")
                ),
            }
            lines.append(f"    <source_strategy{_xml_attrs(attrs)}>")
            for member in strategy.get("source_members") or ():
                if isinstance(member, dict):
                    lines.extend(_source_member_xml_lines(member, indent="      "))
            lines.append("    </source_strategy>")
        lines.append("  </requested_fact>")
    lines.append("</plan_selection_source_strategies>")
    return "\n".join(lines)


def source_alignment_reviews_xml(payload: dict[str, Any]) -> str:
    lines = ["<source_alignment_reviews>"]
    for group in payload.get("requested_fact_source_candidates") or ():
        if not isinstance(group, dict):
            continue
        lines.append(
            f"  <requested_fact id={_xml_quote(group.get('requested_fact_id'))}>"
        )
        lines.extend(
            _text_node_xml_lines("fact_text", group.get("fact_text"), indent="    ")
        )
        answer_outputs = tuple(
            item for item in group.get("answer_outputs") or () if isinstance(item, dict)
        )
        if answer_outputs:
            lines.append("    <answer_outputs>")
            for output in answer_outputs:
                lines.append(
                    f"      <answer_output id={_xml_quote(output.get('answer_output_id'))}>"
                )
                lines.extend(
                    _text_node_xml_lines(
                        "description",
                        output.get("description"),
                        indent="        ",
                    )
                )
                lines.append("      </answer_output>")
            lines.append("    </answer_outputs>")
        lines.append("    <source_candidates>")
        for candidate in group.get("source_candidates") or ():
            if not isinstance(candidate, dict):
                continue
            lines.extend(
                _source_alignment_candidate_xml_lines(candidate, indent="      ")
            )
        lines.append("    </source_candidates>")
        lines.append("  </requested_fact>")
    lines.append("</source_alignment_reviews>")
    return "\n".join(lines)


def source_binding_candidates_xml(payload: dict[str, Any]) -> str:
    lines = ["<candidate_evidence_sources>"]
    for group in payload.get("requested_fact_sources") or ():
        if not isinstance(group, dict):
            continue
        lines.append(
            f"  <requested_fact id={_xml_quote(group.get('requested_fact_id'))}>"
        )
        for context in group.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            context_attrs = {
                "id": context.get("context_id"),
                "kind": context.get("kind"),
            }
            lines.append(f"    <source_context{_xml_attrs(context_attrs)}>")
            for candidate in context.get("source_options") or ():
                if isinstance(candidate, dict):
                    lines.extend(_source_member_xml_lines(candidate, indent="      "))
            lines.append("    </source_context>")
        lines.append("  </requested_fact>")
    for payload_key, tag in (
        ("memory_source_candidates", "memory_sources"),
        ("utility_source_candidates", "utility_sources"),
        ("value_source_candidates", "value_sources"),
    ):
        candidates = tuple(
            item for item in payload.get(payload_key) or () if isinstance(item, dict)
        )
        if not candidates:
            continue
        lines.append(f"  <{tag}>")
        for candidate in candidates:
            lines.extend(_source_member_xml_lines(candidate, indent="    "))
        lines.append(f"  </{tag}>")
    lines.append("</candidate_evidence_sources>")
    return "\n".join(lines)


def _source_alignment_candidate_xml_lines(
    candidate: dict[str, Any],
    *,
    indent: str,
) -> list[str]:
    attrs = {
        "id": candidate.get("source_candidate_id"),
        "kind": candidate.get("kind"),
        "read": candidate.get("read_id"),
    }
    lines = [f"{indent}<source_candidate{_xml_attrs(attrs)}>"]
    read_id = candidate.get("read_id")
    if read_id:
        lines.append(f"{indent}  <api_read read={_xml_quote(read_id)}>")
        input_params = candidate.get("input_params")
        if input_params:
            lines.extend(_input_params_xml_lines(input_params, indent=indent + "    "))
        response_rows = candidate.get("response_rows")
        if response_rows:
            lines.append(f"{indent}    <response>")
            lines.extend(
                _response_row_xml_lines(response_rows, indent=indent + "      ")
            )
            lines.append(f"{indent}    </response>")
        lines.append(f"{indent}  </api_read>")
    else:
        lines.extend(_source_member_xml_lines(candidate, indent=indent + "  "))
    lines.append(f"{indent}</source_candidate>")
    return lines


def _api_read_card_xml_lines(card: dict[str, Any], *, indent: str) -> list[str]:
    attrs = {
        "id": card.get("source_candidate_id"),
        "read": card.get("read_id"),
        "endpoint": card.get("endpoint_name"),
        "row_source": card.get("row_source_id"),
        "resources": _space_separated(card.get("resource_names")),
    }
    lines = [f"{indent}<api_read{_xml_attrs(attrs)}>"]
    lines.extend(
        _text_node_xml_lines(
            "description",
            card.get("docstring") or card.get("description"),
            indent=indent + "  ",
        )
    )
    lines.extend(
        _input_params_xml_lines(card.get("input_params"), indent=indent + "  ")
    )
    lines.extend(
        _applicable_known_inputs_xml_lines(
            card.get("applicable_known_inputs"),
            indent=indent + "  ",
        )
    )
    rows = tuple(
        row for row in card.get("response_rows") or () if isinstance(row, dict)
    )
    if rows:
        lines.append(f"{indent}  <response>")
        lines.extend(_response_row_xml_lines(rows, indent=indent + "    "))
        lines.append(f"{indent}  </response>")
    lines.extend(
        _row_predicates_xml_lines(card.get("row_predicates"), indent=indent + "  ")
    )
    lines.append(f"{indent}</api_read>")
    return lines


def _source_member_xml_lines(member: dict[str, Any], *, indent: str) -> list[str]:
    if member.get("kind") not in {"new_api_read", "same_scope_api_read"}:
        attrs = {
            "id": member.get("source_candidate_id"),
            "kind": member.get("kind"),
            "value": member.get("value_id"),
            "relation": member.get("source_relation_id"),
            "memory_relation": member.get("memory_relation_id"),
            "field": member.get("source_field_id"),
            "calendar": member.get("calendar_id"),
            "cardinality": member.get("cardinality"),
        }
        lines = [f"{indent}<source{_xml_attrs(attrs)}>"]
        lines.extend(
            _text_node_xml_lines(
                "description",
                member.get("description") or member.get("meaning"),
                indent=indent + "  ",
            )
        )
        lines.extend(_flat_fields_xml_lines(member.get("fields"), indent=indent + "  "))
        lines.extend(
            _flat_fields_xml_lines(
                member.get("evidence_items"),
                tag="evidence",
                container_tag="evidence_items",
                indent=indent + "  ",
            )
        )
        if member.get("population_bindings"):
            lines.append(f"{indent}  <population_bindings>")
            for binding in member.get("population_bindings") or ():
                if isinstance(binding, dict):
                    lines.append(f"{indent}    <population{_xml_attrs(binding)} />")
            lines.append(f"{indent}  </population_bindings>")
        lines.extend(
            _fulfillment_choices_xml_lines(
                member.get("fulfillment_support_sets"),
                candidate=member,
                indent=indent + "  ",
            )
        )
        lines.append(f"{indent}</source>")
        return lines
    attrs = {
        "id": member.get("source_candidate_id"),
        "kind": member.get("kind"),
        "read": member.get("read_id"),
        "row_source": member.get("row_source_id"),
        "resources": _space_separated(member.get("resource_names")),
    }
    if member.get("memory_relation_id"):
        attrs["memory_relation"] = member.get("memory_relation_id")
    lines = [f"{indent}<api_read{_xml_attrs(attrs)}>"]
    lines.extend(
        _text_node_xml_lines(
            "description",
            member.get("docstring") or member.get("description"),
            indent=indent + "  ",
        )
    )
    lines.extend(
        _text_node_xml_lines(
            "selection_note",
            member.get("selection_note"),
            indent=indent + "  ",
        )
    )
    lines.extend(
        _input_params_xml_lines(member.get("input_params"), indent=indent + "  ")
    )
    if member.get("bound_params"):
        lines.append(
            f"{indent}  <bound_params count={_xml_quote(len(member['bound_params']))} />"
        )
    lines.extend(_binding_params_xml_lines(member.get("params"), indent=indent + "  "))
    rows = tuple(
        row for row in member.get("response_rows") or () if isinstance(row, dict)
    )
    if rows:
        lines.append(f"{indent}  <response>")
        lines.extend(_response_row_xml_lines(rows, indent=indent + "    "))
        lines.append(f"{indent}  </response>")
    lines.extend(
        _row_predicates_xml_lines(member.get("row_predicates"), indent=indent + "  ")
    )
    lines.extend(
        _applied_filters_xml_lines(member.get("applied_filters"), indent=indent + "  ")
    )
    lines.extend(
        _population_roles_xml_lines(
            member.get("population_roles"), indent=indent + "  "
        )
    )
    if member.get("population_bindings"):
        lines.append(f"{indent}  <population_bindings>")
        for binding in member.get("population_bindings") or ():
            if isinstance(binding, dict):
                lines.append(f"{indent}    <population{_xml_attrs(binding)} />")
        lines.append(f"{indent}  </population_bindings>")
    lines.extend(
        _fulfillment_choices_xml_lines(
            member.get("fulfillment_choices"),
            candidate=member,
            indent=indent + "  ",
        )
    )
    lines.append(f"{indent}</api_read>")
    return lines


def _binding_params_xml_lines(params: object, *, indent: str) -> list[str]:
    param_items = tuple(param for param in params or () if isinstance(param, dict))
    if not param_items:
        return []
    lines = [f"{indent}<binding_params>"]
    for param in param_items:
        attrs = {
            key: param[key]
            for key in (
                "param_id",
                "name",
                "type",
                "required",
                "decision_surface",
            )
            if key in param
        }
        lines.append(f"{indent}  <param{_xml_attrs(attrs)}>")
        choices = tuple(str(choice) for choice in param.get("choices") or ())
        if choices:
            lines.append(f"{indent}    <choices>")
            for choice in choices:
                lines.append(f"{indent}      <choice value={_xml_quote(choice)} />")
            lines.append(f"{indent}    </choices>")
        binding_values = tuple(
            item for item in param.get("binding_values") or () if isinstance(item, dict)
        )
        if binding_values:
            lines.append(f"{indent}    <binding_values>")
            for value in binding_values:
                attrs = {
                    "value": value.get("value"),
                    "label": value.get("label"),
                    "source": value.get("source"),
                    "kind": value.get("kind"),
                }
                lines.append(f"{indent}      <value{_xml_attrs(attrs)} />")
            lines.append(f"{indent}    </binding_values>")
        decision_options = tuple(
            option
            for option in param.get("decision_options") or ()
            if isinstance(option, dict)
        )
        if decision_options:
            lines.append(f"{indent}    <decision_options>")
            for option in decision_options:
                lines.append(
                    f"{indent}      <option{_xml_attrs(_decision_option_attrs(option))} />"
                )
            lines.append(f"{indent}    </decision_options>")
        lines.extend(
            _population_contract_xml_lines(
                param.get("population_contract"),
                indent=indent + "    ",
            )
        )
        profiles = tuple(
            item
            for item in param.get("normal_instance_role_profiles") or ()
            if isinstance(item, dict)
        )
        if profiles:
            lines.append(f"{indent}    <normal_instance_role_profiles>")
            for profile in profiles:
                profile_attrs = {
                    "test_id": profile.get("test_id"),
                    "subject": profile.get("subject_text"),
                }
                lines.append(f"{indent}      <profile{_xml_attrs(profile_attrs)}>")
                lines.extend(
                    _text_node_xml_lines(
                        "match_policy",
                        profile.get("match_policy"),
                        indent=indent + "        ",
                    )
                )
                excluded_roles = tuple(
                    item
                    for item in profile.get("excluded_state_roles") or ()
                    if isinstance(item, dict)
                )
                if excluded_roles:
                    lines.append(f"{indent}        <excluded_state_roles>")
                    for role in excluded_roles:
                        attrs = {
                            "role": role.get("role"),
                            "definition": role.get("role_definition"),
                        }
                        lines.append(
                            f"{indent}          <excluded_state{_xml_attrs(attrs)} />"
                        )
                    lines.append(f"{indent}        </excluded_state_roles>")
                lines.append(f"{indent}      </profile>")
            lines.append(f"{indent}    </normal_instance_role_profiles>")
        lines.append(f"{indent}  </param>")
    lines.append(f"{indent}</binding_params>")
    return lines


def _row_predicates_xml_lines(predicates: object, *, indent: str) -> list[str]:
    predicate_items = tuple(item for item in predicates or () if isinstance(item, dict))
    if not predicate_items:
        return []
    lines = [f"{indent}<row_predicates>"]
    for predicate in predicate_items:
        attrs = {
            "id": predicate.get("predicate_id"),
            "field": predicate.get("field_id"),
            "path": predicate.get("field_path"),
            "row": predicate.get("row_path_id"),
            "type": predicate.get("type"),
            "operator": predicate.get("operator"),
            "default": predicate.get("default"),
        }
        lines.append(f"{indent}  <predicate{_xml_attrs(attrs)}>")
        values = tuple(str(value) for value in predicate.get("allowed_values") or ())
        if values:
            lines.append(f"{indent}    <values>")
            for value in values:
                lines.append(f"{indent}      <value>{escape(value)}</value>")
            lines.append(f"{indent}    </values>")
        lines.append(f"{indent}  </predicate>")
    lines.append(f"{indent}</row_predicates>")
    return lines


def _decision_option_attrs(option: dict[str, Any]) -> dict[str, object]:
    return {
        "id": option.get("param_decision_id"),
        "decision": option.get("decision"),
        "value": option.get("value"),
        "value_component": option.get("value_component"),
        "meaning": option.get("meaning"),
    }


def _population_contract_xml_lines(contract: object, *, indent: str) -> list[str]:
    if not isinstance(contract, dict):
        return []
    attrs = {
        "axis_kind": contract.get("axis_kind"),
    }
    lines = [f"{indent}<population_contract{_xml_attrs(attrs)}>"]
    axis_field = contract.get("axis_field")
    if isinstance(axis_field, dict):
        lines.append(f"{indent}  <axis_field{_xml_attrs(axis_field)} />")
    omission = contract.get("omission_behavior")
    if isinstance(omission, dict):
        omission_attrs = {
            "kind": omission.get("kind"),
            "default_value": omission.get("default_value"),
            "default_label": omission.get("default_label"),
        }
        lines.append(f"{indent}  <omission_behavior{_xml_attrs(omission_attrs)}>")
        for consequence in omission.get("omission_consequence_by_requested_fact") or ():
            if isinstance(consequence, dict):
                lines.append(
                    f"{indent}    <requested_fact_effect{_xml_attrs(consequence)} />"
                )
        lines.append(f"{indent}  </omission_behavior>")
    lines.append(f"{indent}</population_contract>")
    return lines


def _applied_filters_xml_lines(filters: object, *, indent: str) -> list[str]:
    filter_items = tuple(item for item in filters or () if isinstance(item, dict))
    if not filter_items:
        return []
    lines = [f"{indent}<applied_filters>"]
    for item in filter_items:
        lines.append(f"{indent}  <filter{_xml_attrs(item)} />")
    lines.append(f"{indent}</applied_filters>")
    return lines


def _applicable_known_inputs_xml_lines(
    inputs: object,
    *,
    indent: str,
) -> list[str]:
    input_items = tuple(item for item in inputs or () if isinstance(item, dict))
    if not input_items:
        return []
    lines = [f"{indent}<applicable_known_inputs>"]
    for item in input_items:
        lines.append(f"{indent}  <known_input{_xml_attrs(item)} />")
    lines.append(f"{indent}</applicable_known_inputs>")
    return lines


def _population_roles_xml_lines(roles: object, *, indent: str) -> list[str]:
    role_items = tuple(item for item in roles or () if isinstance(item, dict))
    if not role_items:
        return []
    lines = [f"{indent}<population_roles>"]
    for role in role_items:
        attrs = {
            "id": role.get("role_id"),
            "row_path": role.get("row_path_id"),
            "kind": role.get("role_kind"),
            "text": role.get("role_text"),
        }
        lines.append(f"{indent}  <role{_xml_attrs(attrs)} />")
    lines.append(f"{indent}</population_roles>")
    return lines


def _fulfillment_choices_xml_lines(
    choices: object,
    *,
    candidate: dict[str, Any],
    indent: str,
) -> list[str]:
    from fervis.lookup.source_binding.candidates.model_visible_evidence import (
        model_visible_fulfillment_evidence,
    )

    choice_items = tuple(choice for choice in choices or () if isinstance(choice, dict))
    if not choice_items:
        return []
    lines = [f"{indent}<fulfillment_choices>"]
    for choice in choice_items:
        attrs = {
            "id": choice.get("fulfillment_choice_id"),
            "answer_output": choice.get("answer_output_id"),
        }
        lines.append(f"{indent}  <choice{_xml_attrs(attrs)}>")
        for evidence in model_visible_fulfillment_evidence(choice, candidate=candidate):
            lines.append(f"{indent}    <evidence{_xml_attrs(evidence)} />")
        lines.append(f"{indent}  </choice>")
    lines.append(f"{indent}</fulfillment_choices>")
    return lines


def _input_params_xml_lines(params: object, *, indent: str) -> list[str]:
    param_items = tuple(param for param in params or () if isinstance(param, dict))
    if not param_items:
        return []
    lines = [f"{indent}<input_params>"]
    for param in param_items:
        attrs = {
            key: param[key]
            for key in (
                "name",
                "source",
                "type",
                "required",
                "param_ref",
                "param_token",
            )
            if key in param
        }
        lines.append(f"{indent}  <param{_xml_attrs(attrs)} />")
    lines.append(f"{indent}</input_params>")
    return lines


def _flat_fields_xml_lines(
    fields: object,
    *,
    indent: str,
    tag: str = "field",
    container_tag: str = "fields",
) -> list[str]:
    field_items = tuple(item for item in fields or () if isinstance(item, dict))
    if not field_items:
        return []
    lines = [f"{indent}<{container_tag}>"]
    for field in field_items:
        attrs = {
            "name": field.get("field_id") or field.get("id"),
            "id": field.get("evidence_id"),
            "path": field.get("field_path") or field.get("path"),
            "type": field.get("type") or field.get("value_type"),
        }
        lines.append(f"{indent}  <{tag}{_xml_attrs(attrs)} />")
    lines.append(f"{indent}</{container_tag}>")
    return lines


def _text_node_xml_lines(tag: str, value: object, *, indent: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [f"{indent}<{tag}>{escape(text)}</{tag}>"]


def _response_row_xml_lines(
    rows: Iterable[dict[str, Any]],
    *,
    indent: str,
) -> list[str]:
    rows_by_parent = _rows_by_parent(tuple(rows))
    return [
        line
        for row in rows_by_parent.get("", ())
        for line in _row_xml_lines(row, rows_by_parent=rows_by_parent, indent=indent)
    ]


def _row_xml_lines(
    row: dict[str, Any],
    *,
    rows_by_parent: dict[str, tuple[dict[str, Any], ...]],
    indent: str,
) -> list[str]:
    attrs = {
        "path": row.get("path"),
        "cardinality": row.get("cardinality"),
        "evidence_token": row.get("evidence_token"),
    }
    lines = [f"{indent}<row{_xml_attrs(attrs)}>"]
    for field in row.get("fields") or ():
        if not isinstance(field, dict):
            continue
        field_attrs = {
            "name": field.get("field_id"),
            "path": field.get("path"),
            "type": field.get("type"),
            "evidence_token": field.get("evidence_token"),
        }
        lines.append(f"{indent}  <field{_xml_attrs(field_attrs)} />")
    for child in rows_by_parent.get(str(row.get("path") or ""), ()):
        lines.extend(
            _row_xml_lines(child, rows_by_parent=rows_by_parent, indent=indent + "  ")
        )
    lines.append(f"{indent}</row>")
    return lines


def _rows_by_parent(
    rows: tuple[dict[str, Any], ...],
) -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, list[dict[str, Any]]] = {}
    paths = {str(row.get("path") or "") for row in rows}
    for row in rows:
        parent = str(row.get("parent_path") or "")
        if parent not in paths:
            parent = ""
        output.setdefault(parent, []).append(row)
    return {key: tuple(value) for key, value in output.items()}


def _xml_attrs(attrs: dict[str, object]) -> str:
    rendered = [
        f"{key}={_xml_quote(value)}"
        for key, value in attrs.items()
        if value not in (None, "", [], ())
    ]
    return (" " + " ".join(rendered)) if rendered else ""


def _xml_quote(value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f'"{escape(text, quote=True)}"'


def _space_separated(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if str(item))
    return str(value or "")


def _input_param_payload(
    *,
    read_id: str,
    param: CatalogParam,
    include_param_tokens: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "param_ref": param.ref,
        "name": param.name,
        "source": param.source.value,
        "type": param.type,
        "required": param.required,
    }
    if include_param_tokens:
        payload["param_token"] = catalog_input_param_token(
            read_id=read_id,
            param=param,
        )
    if param.description:
        payload["description"] = param.description
    if param.choices:
        payload["choices"] = list(param.choices)
    if param.choice_labels:
        payload["choice_labels"] = dict(param.choice_labels)
    if param.default is not None:
        payload["default"] = param.default
    if param.semantics:
        payload["semantics"] = param.semantics
    return payload


def _field_is_row_container(*, field_path: str, read: EndpointRead) -> bool:
    return any(
        row_path.path and field_path == row_path.path for row_path in read.row_paths
    )


def _field_evidence_token(
    *,
    source_candidate_id: str,
    field_path: str,
) -> str:
    return f"{source_candidate_id}.field.{field_path}"


def _row_evidence_token(
    *,
    source_candidate_id: str,
    row_path_id: str,
) -> str:
    return f"{source_candidate_id}.row.{row_path_id}"
