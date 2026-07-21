"""Model-facing request for one typed Lookup fact plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog import entity_identity_field_ids
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.fact_planning.available_relations import (
    operation_input_values_payload,
)
from fervis.lookup.fact_planning.fulfillment_evidence import (
    evidence_is_compatible_with_plan_shape,
    field_id_for_fulfillment_evidence,
    required_fulfillment_evidence_ids,
    source_cardinality_by_evidence_id,
    source_field_id_by_evidence_id,
)
from fervis.lookup.fact_planning.fact_requirements import (
    fact_endpoint_requirements,
)
from fervis.lookup.fact_planning.metric_options import (
    scalar_aggregate_choices_for_source,
    scalar_aggregate_choices_prompt,
)
from fervis.lookup.fact_planning.grouped_aggregate_choices import (
    GROUPED_AGGREGATE_PLAN_SHAPES,
    group_key_source_field_candidates,
    grouped_aggregate_choices_by_requested_fact_id,
    grouped_aggregate_choices_prompt,
)
from fervis.lookup.fact_planning.prompt_sections import (
    fact_plan_instruction_sections,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    LiteralValuePayload,
    ValueKind,
)
from fervis.lookup.turn_prompts import (
    HostPromptContext,
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.question_contract import (
    GroupKeySourceKind,
    QuestionContract,
    RequestedFact,
    requested_fact_evidence_ref,
)
from fervis.lookup.plan_selection import BoundPlanSelectionSet
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.source_binding import (
    BoundSource,
    bound_sources_prompt_payload,
)
from fervis.lookup.fact_planning.schema import (
    build_fact_plan_schema,
)
from fervis.lookup.fact_planning.scalar_values import (
    SourceDerivedScalarValue,
    source_derived_scalar_values_by_fact,
)
from fervis.model_io.structured_output.specs import required_tool_spec


PATTERN_FACT_PLAN_TOOL_NAME = "submit_pattern_fact_plan"


@dataclass(frozen=True)
class RuntimeValueContext:
    runtime_date: str
    timezone: str


@dataclass(frozen=True)
class FactPlanRequest:
    question: str
    question_contract: QuestionContract
    relation_catalog: RelationCatalog
    bound_sources: tuple[BoundSource, ...] = ()
    same_scope_relation_catalog: RelationCatalog | None = None
    memory_inputs: dict[str, Any] = field(default_factory=dict)
    memory_relations: tuple[RelationRows, ...] = ()
    catalog_selection: CatalogSelectionResult | None = None
    available_values: tuple[FactValue, ...] = ()
    available_value_uses: tuple[Any, ...] = ()
    conversation_context: dict[str, Any] = field(default_factory=dict)
    host: HostPromptContext = field(default_factory=HostPromptContext)


class PatternFactPlanTurnPrompt(TurnPromptBase):
    turn_name = "pattern fact planning"
    turn_task = "author typed fact plan details for preselected plan shapes"

    def __init__(
        self,
        request: FactPlanRequest,
        *,
        plan_selection: BoundPlanSelectionSet,
    ) -> None:
        self.request = request
        self.plan_selection = plan_selection
        self._grouped_aggregate_choices = (
            _grouped_aggregate_choices_by_requested_fact_id(
                request,
                plan_selection=plan_selection,
            )
        )
        self._scalar_aggregate_choices = _scalar_aggregate_choices_by_requested_fact_id(
            request,
            plan_selection=plan_selection,
        )
        self._source_derived_scalar_values = source_derived_scalar_values_by_fact(
            bound_sources=request.bound_sources,
            plan_selection=plan_selection,
        )

    def prompt_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        sections: list[PromptSection] = [
            builder.json_section(
                "Requested facts:",
                _question_contract_payload(self.request.question_contract),
            ),
            builder.json_section(
                "Selected plan shapes:",
                _plan_selection_payload(self.plan_selection),
            ),
            builder.json_section(
                "Operation input values:",
                _operation_input_values_payload(
                    self.request,
                    source_values_by_fact=self._source_derived_scalar_values,
                ),
            ),
            builder.json_section(
                "Bound sources:",
                _bound_sources_payload(
                    self.request, plan_selection=self.plan_selection
                ),
            ),
        ]
        grouped_aggregate_prompt = grouped_aggregate_choices_prompt(
            self._grouped_aggregate_choices
        )
        if grouped_aggregate_prompt:
            sections.append(
                builder.text_section(
                    "Grouped aggregate operation choices:",
                    grouped_aggregate_prompt,
                )
            )
        scalar_aggregate_prompt = scalar_aggregate_choices_prompt(
            self._scalar_aggregate_choices
        )
        if scalar_aggregate_prompt:
            sections.append(
                builder.text_section(
                    "Scalar aggregate operation choices:",
                    scalar_aggregate_prompt,
                )
            )
        required_evidence_payload = _required_fulfillment_evidence_payload(
            self.request,
            plan_selection=self.plan_selection,
        )
        if required_evidence_payload["required_fulfillment_evidence"]:
            sections.append(
                builder.json_section(
                    "Required fulfillment evidence:",
                    required_evidence_payload,
                )
            )
        sections.extend(
            fact_plan_instruction_sections(
                builder,
                tool_name=PATTERN_FACT_PLAN_TOOL_NAME,
                plan_shapes=frozenset(self.plan_selection.pattern_names()),
            )
        )
        return tuple(sections)

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(provider_schema=self._schema())

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=PATTERN_FACT_PLAN_TOOL_NAME,
                    tool_description="Submit the typed pattern fact plan.",
                    input_schema=self._schema(),
                ),
            )
        )

    def _schema(self) -> dict[str, Any]:
        return build_fact_plan_schema(
            **_schema_clarification_inputs(self.request),
            requested_fact_ids=tuple(
                fact.id for fact in self.request.question_contract.requested_facts
            ),
            pattern_names=self.plan_selection.pattern_names(),
            require_pattern=False,
            field_ids_by_source_binding_id=_field_ids_by_source_binding_id(
                self.request,
                plan_selection=self.plan_selection,
            ),
            identity_field_ids_by_source_binding_id=(
                _identity_field_ids_by_source_binding_id(
                    self.request,
                    plan_selection=self.plan_selection,
                )
            ),
            selected_plan_shapes_by_requested_fact_id=(
                self.plan_selection.plan_shapes_by_requested_fact_id()
            ),
            source_binding_ids_by_requested_fact_id={
                plan.requested_fact_id: plan.source_binding_ids
                for plan in self.plan_selection.plan_selections
            },
            answer_output_ids_by_requested_fact_id=(
                self.plan_selection.answer_output_ids_by_requested_fact_id()
            ),
            answer_output_ids_by_source_binding_id=(
                _answer_output_ids_by_source_binding_id(self.request)
            ),
            source_binding_ids_by_requirement_by_requested_fact_id=(
                self.plan_selection.source_binding_ids_by_requirement_by_requested_fact_id()
            ),
            grouped_aggregate_choices_by_requested_fact_id=(
                self._grouped_aggregate_choices
            ),
            scalar_aggregate_choices_by_requested_fact_id=(
                self._scalar_aggregate_choices
            ),
            ordering_required_by_requested_fact_id={
                fact.id: bool(
                    fact.answer_expression
                    and fact.answer_expression.ordering_direction is not None
                )
                for fact in self.request.question_contract.requested_facts
            },
            value_ids_by_requested_fact_id=_value_ids_by_requested_fact_id(
                self.request,
                source_values_by_fact=self._source_derived_scalar_values,
            ),
        )

    def source_derived_scalar_values(self) -> tuple[SourceDerivedScalarValue, ...]:
        return tuple(
            value
            for values in self._source_derived_scalar_values.values()
            for value in values
        )


def _plan_selection_payload(plan_selection: BoundPlanSelectionSet) -> dict[str, Any]:
    return {
        "plan_selections": [
            _selected_plan_payload(plan) for plan in plan_selection.plan_selections
        ]
    }


def _selected_plan_payload(plan: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested_fact_id": plan.requested_fact_id,
        "plan_selection_id": plan.plan_selection_id,
        "source_strategy_id": plan.source_strategy_id,
        "plan_shape": plan.plan_shape,
        "source_binding_ids": list(plan.source_binding_ids),
    }
    if len(plan.required_answer_output_ids) > 1:
        payload["required_answer_output_ids"] = list(plan.required_answer_output_ids)
    if any(member.requirement_ids for member in plan.source_members):
        payload["source_members"] = [
            {
                "requirement_ids": list(member.requirement_ids),
                "source_candidate_id": member.source_candidate_id,
                "source_binding_ids": list(member.source_binding_ids),
            }
            for member in plan.source_members
        ]
    return payload


def _field_ids_by_source_binding_id(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, tuple[str, ...]]:
    selected_source_ids = {
        source_binding_id
        for plan in plan_selection.plan_selections
        for source_binding_id in plan.source_binding_ids
    }
    output: dict[str, tuple[str, ...]] = {}
    for source in request.bound_sources:
        if source.id not in selected_source_ids:
            continue
        field_ids = tuple(
            dict.fromkeys(
                (
                    *source.available_field_ids,
                    *(
                        field.field_id
                        for field in source.available_fields
                        if field.field_id
                    ),
                )
            )
        )
        if field_ids:
            output[source.id] = field_ids
    return output


def _identity_field_ids_by_source_binding_id(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, tuple[str, ...]]:
    selected_source_ids = {
        source_binding_id
        for plan in plan_selection.plan_selections
        for source_binding_id in plan.source_binding_ids
    }
    return {
        source.id: entity_identity_field_ids(source.available_fields)
        for source in request.bound_sources
        if source.id in selected_source_ids
    }


def _required_fulfillment_evidence_payload(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, Any]:
    requested_fact_ids = {
        plan.requested_fact_id for plan in plan_selection.plan_selections
    }
    allowed_by_fact = {
        fact_id: set(plan_selection.source_binding_ids_for(fact_id))
        for fact_id in requested_fact_ids
    }
    shape_by_fact = {
        plan.requested_fact_id: plan.plan_shape
        for plan in plan_selection.plan_selections
    }
    return {
        "required_fulfillment_evidence": [
            item
            for source in request.bound_sources
            for item in _required_fulfillment_evidence_items(
                source,
                allowed_source_ids_by_fact=allowed_by_fact,
                shape_by_fact=shape_by_fact,
            )
        ]
    }


def _required_fulfillment_evidence_items(
    source: BoundSource,
    *,
    allowed_source_ids_by_fact: dict[str, set[str]],
    shape_by_fact: dict[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    field_id_by_evidence_id = source_field_id_by_evidence_id(source)
    cardinality_by_evidence_id = source_cardinality_by_evidence_id(source)
    output: list[dict[str, Any]] = []
    for fulfillment in source.fulfillments:
        allowed_source_ids = allowed_source_ids_by_fact.get(
            fulfillment.requested_fact_id
        )
        if allowed_source_ids is not None and source.id not in allowed_source_ids:
            continue
        plan_shape = (shape_by_fact or {}).get(fulfillment.requested_fact_id, "")
        if plan_shape in GROUPED_AGGREGATE_PLAN_SHAPES:
            continue
        if _plan_shape_uses_count_metric_for_source(
            source,
            requested_fact_id=fulfillment.requested_fact_id,
            plan_shape=plan_shape,
        ):
            continue
        evidence = []
        fulfillment_evidence_ids = required_fulfillment_evidence_ids(
            fulfillment,
            plan_shape=plan_shape,
        )
        for evidence_id in fulfillment_evidence_ids:
            cardinality = cardinality_by_evidence_id.get(evidence_id, "")
            if not evidence_is_compatible_with_plan_shape(
                cardinality,
                plan_shape=plan_shape,
            ):
                continue
            field_id = field_id_for_fulfillment_evidence(
                evidence_id,
                field_id_by_evidence_id=field_id_by_evidence_id,
                available_field_ids=set(source.available_field_ids),
            )
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "field_id": field_id or evidence_id,
                }
            )
        if not evidence:
            continue
        output.append(
            {
                "requested_fact_id": fulfillment.requested_fact_id,
                "answer_output_id": fulfillment.answer_output_id,
                "source_binding_id": source.id,
                "must_use_evidence": evidence,
            }
        )
    return tuple(output)


def _grouped_aggregate_choices_by_requested_fact_id(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, Any]:
    choices = grouped_aggregate_choices_by_requested_fact_id(
        request.bound_sources,
        selected_plan_shapes_by_requested_fact_id=(
            plan_selection.plan_shapes_by_requested_fact_id()
        ),
        source_binding_ids_by_requested_fact_id={
            plan.requested_fact_id: plan.source_binding_ids
            for plan in plan_selection.plan_selections
        },
    )
    facts_by_id = {fact.id: fact for fact in request.question_contract.requested_facts}
    sources_by_id = {source.id: source for source in request.bound_sources}
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for requested_fact_id, fact_choices in choices.items():
        projected = tuple(
            candidate
            for choice in fact_choices
            if (
                candidate := _with_group_key_source_fields(
                    choice,
                    requested_fact=facts_by_id.get(requested_fact_id),
                    sources_by_id=sources_by_id,
                )
            )
            is not None
        )
        if projected:
            output[requested_fact_id] = projected
    return output


def _with_group_key_source_fields(
    choice: dict[str, Any],
    *,
    requested_fact: RequestedFact | None,
    sources_by_id: dict[str, BoundSource],
) -> dict[str, Any] | None:
    if requested_fact is None or requested_fact.answer_expression is None:
        return choice
    group_key = requested_fact.answer_expression.group_key
    if (
        group_key is None
        or group_key.source_kind is not GroupKeySourceKind.TEMPORAL_BUCKET
    ):
        return choice
    source = sources_by_id.get(str(choice.get("source_binding_id") or ""))
    if source is None:
        return choice
    grain_value_id = group_key.temporal_grain_value_id(
        requested_fact_id=requested_fact.id,
    )
    if source.source is not None and any(
        binding.value_id == grain_value_id
        for binding in source.source.param_bindings
    ):
        return choice
    fields = group_key_source_field_candidates(choice)
    return {**choice, "group_key_source_fields": fields} if fields else None


def _plan_shape_uses_count_metric_for_source(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
) -> bool:
    if plan_shape not in {
        "aggregate_scalar",
        "aggregate_by_group",
    }:
        return False
    choice = scalar_aggregate_choices_for_source(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
    )
    metrics = tuple((choice or {}).get("metric_candidates") or ())
    return bool(metrics) and all(
        isinstance(metric, dict) and str(metric.get("kind") or "") == "count_records"
        for metric in metrics
    )


def _scalar_aggregate_choices_by_requested_fact_id(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, tuple[dict[str, Any], ...]]:
    source_by_id = {source.id: source for source in request.bound_sources}
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for plan in plan_selection.plan_selections:
        if plan.plan_shape != "aggregate_scalar":
            continue
        choices: list[dict[str, Any]] = []
        for source_id in plan.source_binding_ids:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            choice = scalar_aggregate_choices_for_source(
                source,
                requested_fact_id=plan.requested_fact_id,
                plan_shape=plan.plan_shape,
            )
            if choice is not None:
                choices.append(choice)
        if choices:
            output[plan.requested_fact_id] = tuple(choices)
    return output


def _answer_output_ids_by_source_binding_id(
    request: FactPlanRequest,
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for source in request.bound_sources:
        answer_output_ids: list[str] = []
        for fulfillment in source.fulfillments:
            answer_output_id = fulfillment.answer_output_id
            if answer_output_id and answer_output_id not in answer_output_ids:
                answer_output_ids.append(answer_output_id)
        if answer_output_ids:
            output[source.id] = tuple(answer_output_ids)
    return output


def _rank_limit_value_ids(values: tuple[FactValue, ...]) -> tuple[str, ...]:
    return tuple(
        value.id
        for value in values
        if value.kind == ValueKind.LITERAL
        and isinstance(value.payload, LiteralValuePayload)
        and value.payload.literal_type == LiteralType.NUMBER
        and _positive_integer_text(value.payload.value)
    )


def _positive_integer_text(value: object) -> bool:
    text = str(value).strip()
    if not text.isdigit():
        return False
    return int(text) > 0


def _bound_sources_payload(
    request: FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, Any]:
    payload = bound_sources_prompt_payload(bound_sources=request.bound_sources)
    return _shape_compatible_bound_sources_payload(
        payload, plan_selection=plan_selection
    )


_MANY_ROW_PLAN_SHAPES = frozenset(
    {
        "list_rows",
        "grouped_rows",
        "aggregate_scalar",
        "aggregate_by_group",
    }
)


def _shape_compatible_bound_sources_payload(
    payload: dict[str, Any],
    *,
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, Any]:
    output: dict[str, Any] = {"bound_sources": []}
    for source in payload.get("bound_sources") or ():
        if not isinstance(source, dict):
            continue
        requested_fact_id = str(source.get("requested_fact_id") or "")
        source_binding_id = str(source.get("source_binding_id") or "")
        allowed_source_ids = plan_selection.source_binding_ids_for(requested_fact_id)
        if allowed_source_ids and source_binding_id not in allowed_source_ids:
            continue
        plan_shapes = plan_selection.plan_shapes_for(requested_fact_id)
        if len(plan_shapes) == 1:
            source = _shape_compatible_bound_source(
                source,
                plan_shape=plan_shapes[0],
            )
        output["bound_sources"].append(source)
    return output


def _shape_compatible_bound_source(
    source: dict[str, Any],
    *,
    plan_shape: str,
) -> dict[str, Any]:
    if plan_shape not in _MANY_ROW_PLAN_SHAPES:
        return source
    fields = source.get("fields")
    if not isinstance(fields, list):
        return source
    has_many_cardinality = any(
        isinstance(field, dict) and field.get("row_cardinality") == "many"
        for field in fields
    )
    if not has_many_cardinality:
        return source
    fulfilled_evidence_ids = _fulfilled_operation_field_evidence_ids(source)
    many_fields = [
        field
        for field in fields
        if isinstance(field, dict)
        and (
            field.get("row_cardinality") == "many"
            or str(field.get("evidence_id") or "") in fulfilled_evidence_ids
        )
    ]
    if not many_fields:
        return source
    return {**source, "fields": many_fields}


def _fulfilled_operation_field_evidence_ids(source: dict[str, Any]) -> set[str]:
    return {
        evidence_id
        for fulfillment in source.get("fulfills") or ()
        if isinstance(fulfillment, dict)
        for evidence_id in (
            *(fulfillment.get("metric_measure_evidence_ids") or ()),
            *(fulfillment.get("value_evidence_ids") or ()),
            *(fulfillment.get("row_count_basis_evidence_ids") or ()),
            *(
                tuple(
                    str(component.get("field_evidence_id") or "")
                    for component in fulfillment.get("entity_evidence", {}).get(
                        "components",
                        (),
                    )
                    if isinstance(component, dict)
                )
                if isinstance(fulfillment.get("entity_evidence"), dict)
                else ()
            ),
        )
        if isinstance(evidence_id, str) and evidence_id
    }


def _question_contract_payload(contract: QuestionContract) -> dict[str, Any]:
    return {
        "requested_facts": [
            {
                "id": fact.id,
                "evidence_ref": requested_fact_evidence_ref(fact.id),
                "description": fact.description,
                **({"required_for": fact.required_for} if fact.required_for else {}),
                "answer_outputs": [
                    {
                        "id": output.id,
                        **(
                            {"description": output.description}
                            if output.description
                            else {}
                        ),
                    }
                    for output in fact.support_answer_outputs
                ],
            }
            for fact in contract.requested_facts
        ]
    }


def _schema_clarification_inputs(
    request: FactPlanRequest,
) -> dict[str, tuple[str, ...]]:
    required_input_ids: list[str] = []
    choice_input_ids: list[str] = []
    requirements = fact_endpoint_requirements(
        catalog=request.relation_catalog,
        catalog_selection=request.catalog_selection,
        available_values=request.available_values,
        available_value_uses=request.available_value_uses,
    )
    for item in requirements.clarifiable_missing_inputs:
        if item.choices:
            choice_input_ids.append(item.id)
            continue
        required_input_ids.append(item.id)
    return {
        "required_catalog_input_ids": tuple(required_input_ids),
        "required_catalog_choice_input_ids": tuple(choice_input_ids),
    }


def _value_ids_by_requested_fact_id(
    request: FactPlanRequest,
    *,
    source_values_by_fact: dict[str, tuple[SourceDerivedScalarValue, ...]],
) -> dict[str, tuple[str, ...]]:
    return {
        fact.id: (
            *tuple(
                value.id
                for value in request.available_values
                if not value.applies_to_requested_fact_ids
                or fact.id in value.applies_to_requested_fact_ids
            ),
            *(value.value_id for value in source_values_by_fact.get(fact.id, ())),
        )
        for fact in request.question_contract.requested_facts
    }


def _operation_input_values_payload(
    request: FactPlanRequest,
    *,
    source_values_by_fact: dict[str, tuple[SourceDerivedScalarValue, ...]],
) -> dict[str, Any]:
    literal_values = operation_input_values_payload(
        available_values=request.available_values,
        available_value_uses=request.available_value_uses,
    )
    return {
        "values": [
            *(literal_values.get("values") or ()),
            *(
                value.payload
                for values in source_values_by_fact.values()
                for value in values
            ),
        ]
    }
