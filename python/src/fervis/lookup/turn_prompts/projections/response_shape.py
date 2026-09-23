"""Readable API response-shape projections for lookup prompts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from collections.abc import Mapping
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
from fervis.lookup.relation_catalog.row_sources import (
    executable_field_ids_for_row_path,
)


@dataclass(frozen=True)
class ScopedRowPredicate:
    source_candidate_id: str
    row_path_id: str
    field_id: str
    field_path: str
    type: str
    allowed_values: tuple[str, ...]
    operator: str = "in"

    def __post_init__(self) -> None:
        if not self.source_candidate_id:
            raise ValueError("row predicate requires source candidate")
        if not self.field_id:
            raise ValueError("row predicate requires field")
        if not self.field_path:
            raise ValueError("row predicate requires field path")
        if not self.allowed_values:
            raise ValueError("row predicate requires allowed values")

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "predicate_id": ".".join(
                (
                    "rp",
                    self.source_candidate_id,
                    "row",
                    self.row_path_id or "root",
                    self.field_id,
                )
            ),
            "field_id": self.field_id,
            "field_path": self.field_path,
            "row_path_id": self.row_path_id or "root",
            "type": self.type,
            "operator": self.operator,
            "allowed_values": list(self.allowed_values),
            "default": "all_values",
        }


@dataclass(frozen=True)
class ApiReadResponseShapeProjector:
    """Project an endpoint read into prompt-facing API-read shapes."""

    read: EndpointRead

    def prompt_payload(
        self,
        *,
        row_path_ids: Iterable[str] = (),
        source_candidate_id: str = "",
        include_evidence_tokens: bool = False,
    ) -> dict[str, object]:
        """Return the shared model-facing shape for one declared API read."""

        return {
            "read_id": self.read.id,
            "endpoint_name": self.read.endpoint_name,
            "resource_names": list(self.read.resource_names),
            "description": self.read.description,
            "input_params": self.input_params(
                include_param_tokens=include_evidence_tokens
            ),
            "response_rows": self.response_rows(
                row_path_ids=row_path_ids,
                source_candidate_id=source_candidate_id,
                include_evidence_tokens=include_evidence_tokens,
            ),
        }

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
        lines.extend(_input_params_xml_lines(input_params, indent=indent))
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
        seen_paths: set[str] = set()
        blocked_field_refs = _blocked_field_refs(self.read)
        for field in self.read.fields:
            if field.ref in blocked_field_refs:
                continue
            if (field.row_path_id or "root") != row_path.id:
                continue
            if _field_is_row_container(field_path=field.path, read=self.read):
                continue
            field_id = _field_id(field.path)
            if not field_id or field.path in seen_paths:
                continue
            seen_paths.add(field.path)
            payload: dict[str, Any] = {
                "field_id": field_id,
                "path": field.path,
                "type": field.type,
            }
            if field.choices:
                payload["choices"] = list(field.choices)
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


def semantic_grounding_tasks_xml(payload: dict[str, Any]) -> str:
    lines = ["<reference_grounding_tasks>"]
    lines.extend(
        _text_node_xml_lines(
            "question",
            payload.get("question"),
            indent="  ",
        )
    )
    for task in _array(payload.get("reference_grounding_tasks")):
        if not isinstance(task, Mapping):
            continue
        lines.append(
            f"  <reference_task{_xml_attrs({'ref': task.get('task_ref'), 'input_ref': task.get('input_ref'), 'use_refs': _space_separated(task.get('use_refs'))})}>"
        )
        lines.extend(
            _text_node_xml_lines(
                "input_text",
                task.get("input_text"),
                indent="    ",
            )
        )
        lines.extend(
            _text_node_xml_lines("operand", task.get("operand"), indent="    ")
        )
        lines.extend(
            _text_node_xml_lines(
                "operand_meaning", task.get("operand_meaning"), indent="    "
            )
        )
        lines.extend(
            _text_node_xml_lines(
                "denoted_instance_kind",
                task.get("denoted_instance_kind"),
                indent="    ",
            )
        )
        if task.get("reference_fact_ref") is not None:
            lines.append(
                f"    <reference_fact{_xml_attrs({'ref': task.get('reference_fact_ref')})} />"
            )
        if task.get("expected_set_ref") is not None:
            lines.append(
                f"    <expected_set{_xml_attrs({'ref': task.get('expected_set_ref')})}>"
            )
            lines.extend(
                _text_node_xml_lines(
                    "meaning",
                    task.get("expected_set_meaning"),
                    indent="      ",
                )
            )
            lines.append("    </expected_set>")
        for option in _array(task.get("resolver_routes")):
            if not isinstance(option, Mapping):
                continue
            lines.append(
                f"    <resolver_route{_xml_attrs({'ref': option.get('route_ref'), 'purpose': option.get('purpose')})}>"
            )
            lines.extend(
                _text_node_xml_lines(
                    "resource_type",
                    option.get("resource_type"),
                    indent="      ",
                )
            )
            api_read = option.get("api_read")
            if isinstance(api_read, Mapping):
                lines.extend(_resolver_api_read_xml_lines(api_read, indent="      "))
            lines.extend(_complete_source_access_xml_lines(option, indent="      "))
            canonical_result = option.get("canonical_result")
            if isinstance(canonical_result, Mapping):
                lines.extend(
                    _canonical_result_xml_lines(canonical_result, indent="      ")
                )
            lines.append("    </resolver_route>")
        lines.append("  </reference_task>")
    lines.append("</reference_grounding_tasks>")
    return "\n".join(lines)


def semantic_read_sources_xml(payload: dict[str, Any]) -> str:
    lines = ["<semantic_read_sources>"]
    for source in _array(payload.get("sources")):
        if not isinstance(source, Mapping):
            continue
        lines.append(f"  <source{_xml_attrs({'ref': source.get('source_ref')})}>")
        api_read = source.get("api_read")
        if isinstance(api_read, Mapping):
            lines.extend(_resolver_api_read_xml_lines(api_read, indent="    "))
        row_source = source.get("row_source")
        if isinstance(row_source, Mapping):
            lines.append(
                f"    <row_source{_xml_attrs({'kind': row_source.get('kind'), 'label': row_source.get('label')})}>"
            )
            lines.extend(
                _text_node_xml_lines(
                    "description", row_source.get("description"), indent="      "
                )
            )
            for field in _array(row_source.get("fields")):
                if not isinstance(field, Mapping):
                    continue
                lines.append(
                    f"      <field{_xml_attrs({'ref': field.get('field_ref'), 'label': field.get('label'), 'type': field.get('type')})} />"
                )
            lines.append("    </row_source>")
        identity_evidence = tuple(
            item
            for item in _array(source.get("identity_evidence"))
            if isinstance(item, Mapping)
        )
        if identity_evidence:
            lines.append("    <identity_evidence>")
            for evidence in identity_evidence:
                lines.append(
                    f"      <identity{_xml_attrs({'ref': evidence.get('identity_ref'), 'entity_kind': evidence.get('entity_kind'), 'key_id': evidence.get('key_id'), 'field_refs': _space_separated(evidence.get('field_refs'))})} />"
                )
            lines.append("    </identity_evidence>")
        set_support_mechanics = tuple(
            item
            for item in _array(source.get("set_support_mechanics"))
            if isinstance(item, Mapping)
        )
        if set_support_mechanics:
            lines.append("    <set_support_mechanics>")
            for mechanic in set_support_mechanics:
                lines.append(
                    f"      <mechanic{_xml_attrs({'ref': mechanic.get('ref'), 'kind': mechanic.get('kind'), 'entity_kind': mechanic.get('entity_kind'), 'param_ref': mechanic.get('param_ref'), 'field_ref': mechanic.get('field_ref'), 'value': mechanic.get('value')})} />"
                )
            lines.append("    </set_support_mechanics>")
        lines.append("  </source>")
    lines.append("</semantic_read_sources>")
    return "\n".join(lines)


def semantic_read_relations_xml(payload: dict[str, Any]) -> str:
    lines = ["<source_relations>"]
    for relation in _array(payload.get("relations")):
        if not isinstance(relation, Mapping):
            continue
        lines.append(
            "  <relation"
            + _xml_attrs(
                {
                    "left_source": relation.get("left_source_ref"),
                    "left_field_refs": _space_separated(
                        relation.get("left_field_refs")
                    ),
                    "right_source": relation.get("right_source_ref"),
                    "right_field_refs": _space_separated(
                        relation.get("right_field_refs")
                    ),
                }
            )
            + " />"
        )
    lines.append("</source_relations>")
    return "\n".join(lines)


def semantic_identity_resolution_tasks_xml(payload: dict[str, Any]) -> str:
    lines = ["<identity_resolution_tasks>"]
    for task in _array(payload.get("identity_resolution_tasks")):
        if not isinstance(task, Mapping):
            continue
        lines.append(
            f"  <identity_task{_xml_attrs({'ref': task.get('task_ref'), 'input_ref': task.get('input_ref'), 'expected_set_ref': task.get('expected_set_ref')})}>"
        )
        for option in _array(task.get("canonical_options")):
            if not isinstance(option, Mapping):
                continue
            lines.append(
                f"    <canonical_option{_xml_attrs({'id': option.get('canonical_option_id'), 'result': option.get('identity_ref'), 'identifier_kind': option.get('identifier_kind')})}>"
            )
            for route in _array(option.get("resolver_routes")):
                if not isinstance(route, Mapping):
                    continue
                lines.extend(_resolver_xml_lines(route, indent="      "))
            lines.append("    </canonical_option>")
        lines.append("  </identity_task>")
    lines.append("</identity_resolution_tasks>")
    return "\n".join(lines)


def semantic_canonical_identity_uses_xml(payload: dict[str, Any]) -> str:
    lines = ["<canonical_identity_uses>"]
    for option in _array(payload.get("canonical_options")):
        if not isinstance(option, Mapping):
            continue
        attrs = {
            "id": option.get("canonical_option_id"),
            "result": option.get("identity_ref"),
        }
        answer_reads = tuple(
            item
            for item in _array(option.get("answer_reads"))
            if isinstance(item, Mapping)
        )
        if not answer_reads:
            lines.append(f"  <canonical_option{_xml_attrs(attrs)} />")
            continue
        lines.append(f"  <canonical_option{_xml_attrs(attrs)}>")
        for answer_read in answer_reads:
            lines.append(
                f"    <answer_read{_xml_attrs({'read': answer_read.get('read_id')})}>"
            )
            for param_ref in _array(answer_read.get("request_param_refs")):
                lines.append(
                    f"      <request_target{_xml_attrs({'param_ref': param_ref})} />"
                )
            for identity in _array(answer_read.get("returned_identities")):
                if not isinstance(identity, Mapping):
                    continue
                lines.append(
                    f"      <returned_identity"
                    f"{_xml_attrs({'identity_ref': identity.get('identity_ref'), 'field_refs': _array(identity.get('field_refs'))})} />"
                )
            lines.append("    </answer_read>")
        lines.append("  </canonical_option>")
    lines.append("</canonical_identity_uses>")
    return "\n".join(lines)


def _complete_source_access_xml_lines(
    resolver: Mapping[str, object], *, indent: str,
) -> list[str]:
    if not resolver.get("complete_source_access"):
        return []
    return [
        f'{indent}<complete_source_access available="true">'
        'The complete source rows can be retrieved without caller-supplied '
        'request inputs. Any required path arguments are supplied by the '
        'established traversal. Match the lookup text against the returned '
        'verification fields.</complete_source_access>'
    ]


def _resolver_xml_lines(resolver: Mapping[str, object], *, indent: str) -> list[str]:
    lines = [
        f"{indent}<resolver{_xml_attrs({'option_id': resolver.get('binding_option_id'), 'purpose': resolver.get('purpose'), 'resolution_method': resolver.get('resolution_method')})}>"
    ]
    api_read = resolver.get("api_read")
    if isinstance(api_read, Mapping):
        lines.extend(_resolver_api_read_xml_lines(api_read, indent=indent + "  "))
    lines.extend(_complete_source_access_xml_lines(resolver, indent=indent + "  "))
    lookup_parameters = tuple(
        item
        for item in _array(resolver.get("lookup_request_parameters"))
        if isinstance(item, Mapping)
    )
    if lookup_parameters:
        lines.append(f"{indent}  <lookup_request_parameters>")
        for item in lookup_parameters:
            lines.append(
                f"{indent}    <parameter"
                f"{_xml_attrs({'param_ref': item.get('param_ref'), 'source': item.get('source'), 'value': item.get('value')})} />"
            )
        lines.append(f"{indent}  </lookup_request_parameters>")
    verification_fields = _array(resolver.get("returned_identity_verification_fields"))
    if verification_fields:
        lines.append(f"{indent}  <returned_identity_verification_fields>")
        for field_path in verification_fields:
            lines.append(f"{indent}    <field{_xml_attrs({'path': field_path})} />")
        lines.append(f"{indent}  </returned_identity_verification_fields>")
    canonical_result = resolver.get("canonical_result")
    if isinstance(canonical_result, Mapping):
        lines.extend(
            _canonical_result_xml_lines(canonical_result, indent=indent + "  ")
        )
    lines.append(f"{indent}</resolver>")
    return lines


def _resolver_api_read_xml_lines(
    api_read: Mapping[str, object],
    *,
    indent: str,
) -> list[str]:
    attrs = {
        "read": api_read.get("read_id"),
        "endpoint": api_read.get("endpoint_name"),
        "resources": _space_separated(api_read.get("resource_names")),
    }
    lines = [f"{indent}<api_read{_xml_attrs(attrs)}>"]
    lines.extend(_text_node_xml_lines("description", api_read.get("description"), indent=indent + "  "))
    lines.extend(
        _input_params_xml_lines(api_read.get("input_params"), indent=indent + "  ")
    )
    response_rows = tuple(
        row for row in _array(api_read.get("response_rows")) if isinstance(row, dict)
    )
    if response_rows:
        lines.append(f"{indent}  <response>")
        lines.extend(_response_row_xml_lines(response_rows, indent=indent + "    "))
        lines.append(f"{indent}  </response>")
    lines.append(f"{indent}</api_read>")
    return lines


def _canonical_result_xml_lines(
    canonical_result: Mapping[str, object],
    *,
    indent: str,
) -> list[str]:
    attrs = {
        "entity_kind": canonical_result.get("entity_kind"),
        "key_id": canonical_result.get("key_id"),
    }
    lines = [f"{indent}<canonical_result{_xml_attrs(attrs)}>"]
    for component in _array(canonical_result.get("components")):
        if isinstance(component, Mapping):
            lines.append(f"{indent}  <component{_xml_attrs(component)} />")
    lines.append(f"{indent}</canonical_result>")
    return lines


def _structured_xml_lines(tag: str, value: object, *, indent: str) -> list[str]:
    if isinstance(value, Mapping):
        lines = [f"{indent}<{tag}>"]
        for key, child in value.items():
            lines.extend(_structured_xml_lines(str(key), child, indent=indent + "  "))
        lines.append(f"{indent}</{tag}>")
        return lines
    if isinstance(value, list | tuple):
        lines = [f"{indent}<{tag}>"]
        item_tag = tag[:-1] if tag.endswith("s") else "item"
        for item in value:
            lines.extend(_structured_xml_lines(item_tag, item, indent=indent + "  "))
        lines.append(f"{indent}</{tag}>")
        return lines
    if value is None:
        return []
    text = str(value).lower() if isinstance(value, bool) else str(value)
    return [f"{indent}<{tag}>{escape(text)}</{tag}>"]


def _population_roles_xml_lines(roles: object, *, indent: str) -> list[str]:
    role_items = tuple(item for item in _array(roles) if isinstance(item, dict))
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


def _input_params_xml_lines(params: object, *, indent: str) -> list[str]:
    param_items = tuple(param for param in _array(params) if isinstance(param, dict))
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
                "description",
                "default",
            )
            if key in param
        }
        choices = tuple(str(choice) for choice in _array(param.get("choices")))
        if not choices:
            lines.append(f"{indent}  <param{_xml_attrs(attrs)} />")
            continue
        lines.append(f"{indent}  <param{_xml_attrs(attrs)}>")
        labels = param.get("choice_labels")
        choice_labels = labels if isinstance(labels, Mapping) else {}
        lines.extend(
            _choices_xml_lines(
                choices,
                labels=choice_labels,
                indent=indent + "    ",
            )
        )
        lines.append(f"{indent}  </param>")
    lines.append(f"{indent}</input_params>")
    return lines


def _row_predicates_xml_lines(predicates: object, *, indent: str) -> list[str]:
    predicate_items = tuple(
        item for item in _array(predicates) if isinstance(item, dict)
    )
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
        values = tuple(str(value) for value in _array(predicate.get("allowed_values")))
        if values:
            lines.append(f"{indent}    <values>")
            for value in values:
                lines.append(f"{indent}      <value>{escape(value)}</value>")
            lines.append(f"{indent}    </values>")
        lines.append(f"{indent}  </predicate>")
    lines.append(f"{indent}</row_predicates>")
    return lines


def _flat_fields_xml_lines(
    fields: object,
    *,
    indent: str,
    tag: str = "field",
    container_tag: str = "fields",
) -> list[str]:
    field_items = tuple(item for item in _array(fields) if isinstance(item, dict))
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
        choices = tuple(str(choice) for choice in _array(field.get("choices")))
        if not choices:
            lines.append(f"{indent}  <field{_xml_attrs(field_attrs)} />")
            continue
        lines.append(f"{indent}  <field{_xml_attrs(field_attrs)}>")
        lines.extend(_choices_xml_lines(choices, indent=indent + "    "))
        lines.append(f"{indent}  </field>")
    for child in rows_by_parent.get(str(row.get("path") or ""), ()):
        lines.extend(
            _row_xml_lines(child, rows_by_parent=rows_by_parent, indent=indent + "  ")
        )
    lines.append(f"{indent}</row>")
    return lines


def _choices_xml_lines(
    choices: tuple[str, ...],
    *,
    indent: str,
    labels: Mapping[object, object] | None = None,
) -> list[str]:
    lines = [f"{indent}<choices>"]
    for choice in choices:
        attrs: dict[str, object] = {"value": choice}
        if labels is not None and choice in labels:
            attrs["label"] = labels[choice]
        lines.append(f"{indent}  <choice{_xml_attrs(attrs)} />")
    lines.append(f"{indent}</choices>")
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


def _array(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _xml_attrs(attrs: Mapping[str, object]) -> str:
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
