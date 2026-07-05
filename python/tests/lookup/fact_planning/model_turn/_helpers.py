import json
from dataclasses import replace

import pytest
from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.model_io.turn_artifacts import ModelTurnArtifact
from fervis.lookup.fact_plan.relations import FieldBindingRole
from fervis.lookup.fact_plan.row_sources import api_row_source_id
from fervis.lookup.fact_plan.values import (
    FactValue,
    TimeComponent,
)
from fervis.lookup.fact_planning.request import (
    FactPlanRequest as _FactPlanRequest,
    PatternFactPlanTurnPrompt,
)
from fervis.lookup.fact_planning.turn import (
    FactPlanGenerationError,
    generate_pattern_fact_plan,
)
from fervis.lookup.fact_planning.request import _shape_compatible_bound_source
from fervis.lookup.fact_planning.fulfillment_evidence import (
    field_id_for_fulfillment_evidence,
)
from fervis.lookup.fact_planning.metric_options import (
    scalar_aggregate_choices_for_source,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.fact_plan.relations import (
    EndpointParamBinding,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_planning.pattern_plan import (
    compile_pattern_answer_plan as _compile_pattern_answer_plan,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
)
from fervis.lookup.source_binding import (
    AnswerPopulation,
    BoundSource,
    SourceEvidenceItem,
    SourceField,
    SourceFulfillment,
)
from fervis.lookup.source_binding.candidates import source_candidates
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.plan_selection import (
    SourceStrategyMember,
    BoundRoleTarget,
    BoundSourceStrategyMember,
    BoundSelectedSourceStrategy,
    BoundPlanSelectionSet,
    PlanSelectionSet,
    SelectedSourceStrategy,
)
from fervis.model_io.backbone.dto import ToolSpec
from fervis.model_io.telemetry import MODEL_TURN_PROMPT_BUDGET_CHARS


def FactPlanRequest(*args, **kwargs) -> _FactPlanRequest:
    request = _FactPlanRequest(*args, **kwargs)
    catalog_selection = request.catalog_selection or _all_read_catalog_selection(
        request
    )
    bound_sources = request.bound_sources or _default_bound_sources(
        replace(request, catalog_selection=catalog_selection)
    )
    return replace(
        request,
        catalog_selection=catalog_selection,
        bound_sources=bound_sources,
    )

def _answer_population() -> AnswerPopulation:
    return AnswerPopulation(
        population_binding_id="pop.source_1.candidate_population",
        intent_text="sales",
        match_basis_explanation="sales defines the source population",
    )

def _bound_source_fixture(source: BoundSource) -> BoundSource:
    return source

def _fact_plan_prompt(request: _FactPlanRequest) -> str:
    return _pattern_fact_plan_prompt(request)

def _pattern_fact_plan_prompt(
    request: _FactPlanRequest,
    *,
    plan_selection: BoundPlanSelectionSet | None = None,
) -> str:
    return (
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=plan_selection or _plan_selection_for_request(request),
        )
        .to_model_payload(
            build_turn_prompt_context(
                current_question=request.question,
                conversation_context=request.conversation_context,
                memory_payload=request.memory_inputs,
            )
        )
        .prompt_text
    )

def _plan_selection_for_request(
    request: _FactPlanRequest,
    *,
    plan_shape: str = "list_rows",
) -> BoundPlanSelectionSet:
    requested_fact_ids: list[str] = [
        fact.id for fact in request.question_contract.requested_facts
    ]
    for source in request.bound_sources:
        if (
            source.requested_fact_id
            and source.requested_fact_id not in requested_fact_ids
        ):
            requested_fact_ids.append(source.requested_fact_id)
        for fulfillment in source.fulfillments:
            if fulfillment.requested_fact_id not in requested_fact_ids:
                requested_fact_ids.append(fulfillment.requested_fact_id)
    source_ids_by_fact: dict[str, tuple[str, ...]] = {}
    for requested_fact_id in requested_fact_ids:
        source_ids = tuple(
            source.id
            for source in request.bound_sources
            if source.requested_fact_id == requested_fact_id
            or any(
                fulfillment.requested_fact_id == requested_fact_id
                for fulfillment in source.fulfillments
            )
        )
        source_ids_by_fact[requested_fact_id] = source_ids or ("sb_1",)
    return BoundPlanSelectionSet(
        plan_selections=tuple(
            BoundSelectedSourceStrategy(
                plan_selection_id=f"{requested_fact_id}.{plan_shape}.sb_1",
                requested_fact_id=requested_fact_id,
                source_strategy_id=f"source_strategy.{requested_fact_id}.{plan_shape}.1",
                plan_shape=plan_shape,
                required_answer_output_ids=tuple(
                    output.id
                    for fact in request.question_contract.requested_facts
                    if fact.id == requested_fact_id
                    for output in fact.answer_outputs
                )
                or ("answer_1",),
                source_members=(
                    _bound_plan_member(
                        request,
                        source_binding_ids=source_ids_by_fact[requested_fact_id],
                    ),
                ),
            )
            for requested_fact_id in requested_fact_ids
        )
    )

def _bound_plan_member(
    request: _FactPlanRequest,
    *,
    source_binding_ids: tuple[str, ...],
    source_candidate_id: str = "source_1",
) -> BoundSourceStrategyMember:
    field_ids: list[str] = []
    for source in request.bound_sources:
        if source.id not in source_binding_ids:
            continue
        evidence_field_ids = _fulfillment_field_ids(source)
        if evidence_field_ids:
            for field_id in evidence_field_ids:
                if field_id and field_id not in field_ids:
                    field_ids.append(field_id)
            continue
        for field_id in source.available_field_ids:
            if field_id and field_id not in field_ids:
                field_ids.append(field_id)
    return BoundSourceStrategyMember(
        source_candidate_id=source_candidate_id,
        role_targets=(
            BoundRoleTarget(
                requirement_id="source",
                source_candidate_id=source_candidate_id,
                source_binding_ids=source_binding_ids,
            ),
        ),
        field_ids=tuple(field_ids),
    )

def _fulfillment_field_ids(source: BoundSource) -> tuple[str, ...]:
    field_id_by_evidence_id = {
        item.evidence_id: item.field_id for item in source.evidence_items
    }
    output: list[str] = []
    for fulfillment in source.fulfillments:
        for evidence_id in (
            *fulfillment.metric_measure_evidence_ids,
            *fulfillment.row_count_basis_evidence_ids,
            *fulfillment.group_key_evidence_ids,
        ):
            field_id = field_id_by_evidence_id.get(evidence_id, "")
            if field_id and field_id not in output:
                output.append(field_id)
    return tuple(output)

def _default_bound_sources(request: _FactPlanRequest) -> tuple[BoundSource, ...]:
    candidates = source_candidates(
        SourceBindingRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.question_contract.requested_facts,
            relation_catalog=request.relation_catalog,
            catalog_selection=request.catalog_selection
            or _all_read_catalog_selection(request),
            plan_selection=_selected_plan_for_request(request),
            memory_inputs=request.memory_inputs,
            available_values=request.available_values,
            available_value_uses=request.available_value_uses,
            conversation_context=request.conversation_context,
        )
    )
    return tuple(
        BoundSource(
            id=f"sb_{index}",
            requested_fact_id=request.question_contract.requested_facts[0].id,
            answer_population=_answer_population(),
            source=candidate.source,
            value_id=candidate.value_id,
            available_field_ids=_candidate_field_ids(candidate),
            available_fields=_candidate_source_fields(candidate),
            applied_filters=_candidate_applied_filters(candidate),
            fulfillments=_source_fulfillments(
                request,
                evidence_ids=(_candidate_field_ids(candidate) or ("value",)),
            ),
        )
        for index, candidate in enumerate(candidates.values(), start=1)
        if candidate.source is not None or candidate.value_id
    )

def _selected_plan_for_request(request: _FactPlanRequest) -> PlanSelectionSet:
    fact_id = request.question_contract.requested_facts[0].id
    selection = request.catalog_selection or _all_read_catalog_selection(request)
    source_count = max(1, len(selection.selected_read_ids))
    return PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id=f"plan.{fact_id}",
                requested_fact_id=fact_id,
                source_strategy_id=f"source_strategy.{fact_id}.direct_field_value.1",
                plan_shape="direct_field_value",
                required_answer_output_ids=("answer_1",),
                source_members=tuple(
                    SourceStrategyMember(source_candidate_id=f"source_{index}")
                    for index in range(1, source_count + 1)
                ),
                basis="Selected by test fixture.",
            ),
        )
    )

def compile_pattern_answer_plan(
    payload: dict[str, object],
    *,
    bound_sources: tuple[BoundSource, ...],
    source_binding_ids_by_requested_fact_id: dict[str, tuple[str, ...]] | None = None,
    source_binding_ids_by_requirement_by_requested_fact_id: (
        dict[str, dict[str, tuple[str, ...]]] | None
    ) = None,
):
    if source_binding_ids_by_requested_fact_id is None:
        source_binding_ids_by_requested_fact_id = _selected_source_ids_by_fact(
            bound_sources
        )
    if source_binding_ids_by_requirement_by_requested_fact_id is None:
        source_binding_ids_by_requirement_by_requested_fact_id = {}
    return _compile_pattern_answer_plan(
        payload,
        bound_sources=bound_sources,
        source_binding_ids_by_requested_fact_id=(
            source_binding_ids_by_requested_fact_id
        ),
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
    )

def _selected_source_ids_by_fact(
    bound_sources: tuple[BoundSource, ...],
) -> dict[str, tuple[str, ...]]:
    output: dict[str, list[str]] = {}
    for source in bound_sources:
        output.setdefault(source.requested_fact_id, []).append(source.id)
    return {key: tuple(value) for key, value in output.items()}

def _source_fulfillments(
    request: _FactPlanRequest,
    *,
    evidence_ids: tuple[str, ...],
) -> tuple[SourceFulfillment, ...]:
    return tuple(
        SourceFulfillment(
            requested_fact_id=fact.id,
            answer_output_id=answer.id,
            group_key_evidence_ids=evidence_ids,
            match_basis_explanation=(
                f"{answer.id} is fulfilled by source evidence because "
                "the selected source contains the answer output evidence."
            ),
        )
        for fact in request.question_contract.requested_facts
        for answer in fact.answer_outputs
    )

def _candidate_field_ids(candidate: object) -> tuple[str, ...]:
    fields = getattr(candidate, "fields", ())
    return tuple(
        str(field.get("field_id") or field.get("id") or "")
        for field in fields
        if isinstance(field, dict)
        and str(field.get("field_id") or field.get("id") or "")
    )

def _candidate_source_fields(candidate: object) -> tuple[SourceField, ...]:
    fields = getattr(candidate, "fields", ())
    return tuple(
        SourceField(
            field_id=field_id,
            type=str(field.get("type") or ""),
            roles=tuple(str(role) for role in field.get("roles") or ()),
            label=str(field.get("label") or ""),
            row_cardinality=str(field.get("row_cardinality") or ""),
        )
        for field in fields
        if isinstance(field, dict)
        for field_id in (str(field.get("field_id") or field.get("id") or ""),)
        if field_id
    )

def _candidate_applied_filters(candidate: object) -> tuple[dict[str, object], ...]:
    payload = getattr(candidate, "payload", None)
    filters = payload.get("applied_filters") if isinstance(payload, dict) else ()
    return tuple(dict(item) for item in filters or () if isinstance(item, dict))

def _selected_read_ids(selection: object) -> frozenset[str]:
    if not isinstance(selection, CatalogSelectionResult):
        return frozenset()
    return frozenset(selection.selected_read_ids)

def _api_bound_source_for_memory_boundary_test() -> BoundSource:
    return BoundSource(
        id="sb_1",
        requested_fact_id="rf_answer",
        answer_population=_answer_population(),
        source=RelationSource(kind=SourceKind.API_READ, read_id="sales"),
        cardinality="many",
        available_field_ids=("value",),
        available_fields=(SourceField(field_id="value", type="string"),),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="source_1_evidence_1",
                field_id="value",
                row_cardinality="many",
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="rf_answer",
                answer_output_id="answer",
                match_basis_explanation="value answers answer",
                group_key_evidence_ids=("source_1_evidence_1",),
            ),
        ),
    )

def _two_output_aggregate_bound_source() -> BoundSource:
    return _bound_source_fixture(
        BoundSource(
            id="sb_1",
            requested_fact_id="rf_answer",
            answer_population=_answer_population(),
            source=RelationSource(
                kind=SourceKind.API_READ,
                read_id="list_metric_by_location",
            ),
            cardinality="many",
            available_field_ids=("location_name", "metric_total"),
            available_fields=(
                SourceField(field_id="location_name", type="string"),
                SourceField(field_id="metric_total", type="decimal"),
            ),
            evidence_items=(
                SourceEvidenceItem(
                    evidence_id="source_1.data.location_name",
                    field_id="location_name",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.metric_total",
                    field_id="metric_total",
                    row_cardinality="many",
                ),
            ),
            fulfillments=(
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_1",
                    match_basis_explanation="location_name is the displayed group.",
                    group_key_evidence_ids=("source_1.data.location_name",),
                ),
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_2",
                    match_basis_explanation="metric_total is the measured value.",
                    metric_measure_evidence_ids=("source_1.data.metric_total",),
                ),
            ),
        )
    )

def _two_output_aggregate_bound_source_pair() -> tuple[BoundSource, BoundSource]:
    first = _two_output_aggregate_bound_source()
    second = replace(
        first,
        id="sb_2",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id="list_observed_metric_by_location",
        ),
    )
    return (first, second)

def _ranked_group_key_with_display_bound_source() -> BoundSource:
    return _bound_source_fixture(
        BoundSource(
            id="sb_1",
            requested_fact_id="rf_answer",
            answer_population=_answer_population(),
            source=RelationSource(
                kind=SourceKind.API_READ,
                read_id="list_metric_by_location",
            ),
            cardinality="many",
            available_field_ids=("location_id", "location_name", "metric_total"),
            available_fields=(
                SourceField(field_id="location_id", type="uuid", roles=("identity",)),
                SourceField(field_id="location_name", type="string"),
                SourceField(field_id="metric_total", type="decimal"),
            ),
            evidence_items=(
                SourceEvidenceItem(
                    evidence_id="source_1.data.location_id",
                    field_id="location_id",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.location_name",
                    field_id="location_name",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.metric_total",
                    field_id="metric_total",
                    row_cardinality="many",
                ),
            ),
            fulfillments=(
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_1",
                    match_basis_explanation="location_id identifies the grouped location.",
                    group_key_evidence_ids=("source_1.data.location_id",),
                ),
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_2",
                    match_basis_explanation="metric_total is the measured value.",
                    metric_measure_evidence_ids=("source_1.data.metric_total",),
                ),
            ),
        )
    )

def _ranked_count_by_store_bound_source() -> BoundSource:
    return _bound_source_fixture(
        BoundSource(
            id="sb_1",
            requested_fact_id="rf_answer",
            answer_population=_answer_population(),
            source=RelationSource(
                kind=SourceKind.API_READ,
                read_id="list_order",
            ),
            cardinality="many",
            available_field_ids=("store_id", "store_name", "order_id"),
            available_fields=(
                SourceField(field_id="store_id", type="uuid", roles=("identity",)),
                SourceField(field_id="store_name", type="string"),
                SourceField(
                    field_id="order_id",
                    type="uuid",
                    roles=("identity",),
                    identity=IdentityMetadata(
                        entity_ref="order",
                        identity_field="order_id",
                        primary_key=True,
                        stable=True,
                    ),
                ),
            ),
            evidence_items=(
                SourceEvidenceItem(
                    evidence_id="source_1.data.store_id",
                    field_id="store_id",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.store_name",
                    field_id="store_name",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.order_id",
                    field_id="order_id",
                    row_cardinality="many",
                ),
            ),
            fulfillments=(
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_1",
                    match_basis_explanation="store_id identifies the grouped store.",
                    group_key_evidence_ids=("source_1.data.store_id",),
                ),
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_2",
                    match_basis_explanation="order_id provides the count basis.",
                    row_count_basis_evidence_ids=("source_1.data.order_id",),
                ),
            ),
        )
    )

def _single_output_ranked_aggregate_bound_source() -> BoundSource:
    return _bound_source_fixture(
        BoundSource(
            id="sb_1",
            requested_fact_id="rf_answer",
            answer_population=_answer_population(),
            source=RelationSource(
                kind=SourceKind.API_READ,
                read_id="list_metric_by_location",
            ),
            cardinality="many",
            available_field_ids=("location_id", "metric_total"),
            available_fields=(
                SourceField(field_id="location_id", type="string"),
                SourceField(field_id="metric_total", type="decimal"),
            ),
            evidence_items=(
                SourceEvidenceItem(
                    evidence_id="source_1.data.location_id",
                    field_id="location_id",
                    row_cardinality="many",
                ),
                SourceEvidenceItem(
                    evidence_id="source_1.data.metric_total",
                    field_id="metric_total",
                    row_cardinality="many",
                ),
            ),
            fulfillments=(
                SourceFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer_1",
                    match_basis_explanation=(
                        "The answer output is the winning location; metric_total "
                        "ranks locations but is not the location display value."
                    ),
                    metric_measure_evidence_ids=("source_1.data.metric_total",),
                    group_key_evidence_ids=("source_1.data.location_id",),
                ),
            ),
        )
    )

def _display_only_location_bound_source() -> BoundSource:
    return BoundSource(
        id="sb_1",
        requested_fact_id="rf_answer",
        answer_population=_answer_population(),
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id="list_location",
        ),
        cardinality="many",
        available_field_ids=("location_id", "name"),
        available_fields=(
            SourceField(field_id="location_id", type="uuid", roles=("identity",)),
            SourceField(field_id="name", type="string"),
        ),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="source_1.data.location_id",
                field_id="location_id",
                row_cardinality="many",
            ),
            SourceEvidenceItem(
                evidence_id="source_1.data.name",
                field_id="name",
                row_cardinality="many",
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="rf_answer",
                answer_output_id="answer_1",
                match_basis_explanation="name is the location display value.",
                group_key_evidence_ids=("source_1.data.name",),
                row_count_basis_evidence_ids=("source_1.data.location_id",),
            ),
        ),
    )

def _question_contract() -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="answer",
                answer_outputs=(RequestedFactAnswerOutput(id="answer"),),
            ),
        )
    )

def _all_read_catalog_selection(request: _FactPlanRequest) -> CatalogSelectionResult:
    requested_fact_id = request.question_contract.requested_facts[0].id
    read_ids = tuple(read.id for read in request.relation_catalog.reads)
    return CatalogSelectionResult(
        relation_catalog=request.relation_catalog,
        selected_read_ids=read_ids,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id=requested_fact_id,
                query_terms=(),
                rankings=tuple(
                    CatalogSelectionRanking(read_id=read_id, score=1)
                    for read_id in read_ids
                ),
                selected_read_ids=read_ids,
            ),
        ),
    )

def _request_with_executable_relation_and_required_detail() -> FactPlanRequest:
    relation_catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sales",
                fields=(CatalogField(ref="field.amount", type="decimal"),),
            ),
            EndpointRead(
                id="sale_detail",
                endpoint_name="get_sale",
                params=(
                    CatalogParam(
                        ref="sale.path.sale_id",
                        name="sale_id",
                        source=ParamSource.PATH,
                        type="uuid",
                        required=True,
                    ),
                ),
                fields=(CatalogField(ref="field.amount", type="decimal"),),
            ),
        )
    )
    return FactPlanRequest(
        question="How much were sales?",
        question_contract=_question_contract(),
        relation_catalog=relation_catalog,
        catalog_selection=CatalogSelectionResult(
            relation_catalog=relation_catalog,
            selected_read_ids=("sales", "sale_detail"),
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="rf_answer",
                    query_terms=("sales", "amount"),
                    rankings=(
                        CatalogSelectionRanking(
                            read_id="sales",
                            score=2,
                            matched_terms=("sales", "amount"),
                        ),
                        CatalogSelectionRanking(
                            read_id="sale_detail",
                            score=2,
                            matched_terms=("sales", "amount"),
                        ),
                    ),
                    selected_read_ids=("sales", "sale_detail"),
                ),
            ),
        ),
    )

def _json_prompt_section(prompt: str, *, label: str, next_label: str) -> dict:
    start = prompt.index(f"{label}:\n") + len(f"{label}:\n")
    candidate_labels = (
        next_label,
        "Requested facts",
        "Operation input values",
        "Relation catalog",
        "Required fulfillment evidence",
        "Scalar aggregate operation choices",
        "Grouped/ranked operation choices",
        "Memory inputs",
    )
    ends = [
        _section_index(prompt, candidate, start)
        for candidate in candidate_labels
        if candidate != label and _section_index(prompt, candidate, start) >= 0
    ]
    end = min(ends)
    return json.loads(prompt[start:end])


def _text_prompt_section(prompt: str, *, label: str, next_label: str) -> str:
    start = prompt.index(f"{label}:\n") + len(f"{label}:\n")
    candidate_labels = (
        next_label,
        "Requested facts",
        "Operation input values",
        "Bound sources",
        "Required fulfillment evidence",
        "Scalar aggregate operation choices",
        "Grouped/ranked operation choices",
        "Decision Scope",
    )
    ends = [
        _section_index(prompt, candidate, start)
        for candidate in candidate_labels
        if candidate != label and _section_index(prompt, candidate, start) >= 0
    ]
    end = min(ends)
    return prompt[start:end].strip()

def _section_index(prompt: str, label: str, start: int) -> int:
    colon_index = prompt.find(f"\n\n{label}:", start)
    text_index = prompt.find(f"\n\n{label}\n", start)
    candidates = [index for index in (colon_index, text_index) if index >= 0]
    return min(candidates) if candidates else -1

def _available_relations(payload: dict) -> list[dict]:
    relations: list[dict] = []
    for fact_relations in payload.get("requested_fact_relations") or ():
        relations.extend(fact_relations.get("available_relations") or ())
    relations.extend(payload.get("utility_relations") or ())
    relations.extend(payload.get("memory_relations") or ())
    return relations

def _bound_sources(payload: dict) -> list[dict]:
    return list(payload.get("bound_sources") or ())

__all__ = [name for name in globals() if not name.startswith("__")]
