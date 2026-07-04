from tests.lookup.orchestrator._catalogs import *  # noqa: F403


def _join_key(left: str, right: str):
    from fervis.lookup.fact_plan.operations import JoinKey

    return JoinKey(left=left, right=right)


def _answer_plan(**kwargs) -> AnswerPlan:
    render_spec = kwargs.get("render_spec")
    operations = tuple(kwargs.get("operations", ()))
    kwargs.pop("requested_facts", None)
    return AnswerPlan(
        fulfillment=kwargs.pop(
            "fulfillment",
            _default_fulfillment(render_spec, operations),
        ),
        **kwargs,
    )


def _plan_clarification(
    choice_id: str,
    *,
    requested_fact_id: str = "rf_answer",
) -> PlanClarification:
    input_id = f"{api_row_source_id('clarification_read', 'data')}.selector"
    return PlanClarification(
        missing_catalog_inputs=(
            MissingCatalogRequiredInput(
                id=f"clarify_{choice_id}",
                requested_fact_id=requested_fact_id,
                required_catalog_input_id=input_id,
            ),
        )
    )


def _default_fulfillment(
    render_spec: RenderSpec | None,
    operations: tuple[Operation, ...],
) -> tuple[FactFulfillment, ...]:
    if render_spec is not None:
        outputs = (*render_spec.relation_outputs, *render_spec.scalar_outputs)
        if outputs:
            return tuple(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id=output.id,
                    render_output_id=output.id,
                )
                for output in outputs
            )
    return (
        FactFulfillment(
            requested_fact_id="rf_answer",
            answer_output_id="answer",
            render_output_id="answer",
        ),
    )


def _metric_answer_plan() -> FactPlan:
    return FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="metric_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="location_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(
                            ProjectField(source="location_name", output="location"),
                            ProjectField(source="metric_total"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="location",
                        relation_id="answer_rows",
                        field_id="location",
                    ),
                    RenderRelationOutput(
                        id="metric_total",
                        relation_id="answer_rows",
                        field_id="metric_total",
                    ),
                )
            ),
        )
    )


def _default_question_contract(*, description: str = "answer") -> QuestionContract:
    return _question_contract_for("rf_answer", description=description)


def _known_reference_input(
    input_id: str,
    text: str,
    *,
    value_meaning_hint: str = "",
    resolved_value_text: str | None = None,
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        role=LiteralInputRole.REFERENCE_VALUE,
        resolved_value_text=resolved_value_text or text,
        value_meaning_hint=value_meaning_hint,
    )


def _known_time_input(
    input_id: str,
    text: str,
    *,
    requirement_id: str | None = None,
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        role=LiteralInputRole.TIME_VALUE,
        resolved_value_text=text,
        satisfies_requirement_id=requirement_id or f"{input_id}_requirement",
    )


def _known_result_limit_input(
    input_id: str,
    text: str,
    *,
    value_text: str,
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        role=LiteralInputRole.RESULT_LIMIT,
        resolved_value_text=value_text,
    )


def _question_contract_for(
    requested_fact_id: str,
    *,
    description: str | None = None,
    subject_text: str | None = None,
    binding_target_ids: tuple[str, ...] = ("answer",),
    known_inputs: tuple[RequestedFactKnownInput, ...] = (),
    answer_expression_family: RequestedFactAnswerExpressionFamily = (
        RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE
    ),
) -> QuestionContract:
    text = description or requested_fact_id
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id=requested_fact_id,
                description=text,
                answer_expression=RequestedFactAnswerExpression(
                    family=answer_expression_family
                ),
                answer_subject=RequestedFactAnswerSubject(
                    subject_text=subject_text or text
                ),
                answer_outputs=tuple(
                    RequestedFactAnswerOutput(id=binding_target_id)
                    for binding_target_id in binding_target_ids
                ),
                known_inputs=known_inputs,
            ),
        )
    )


def _pattern_fact_plan_payload(
    *,
    requested_fact_id: str,
    answer_output_ids: tuple[str, ...],
    read_id: str,
    output_fields: tuple[dict[str, Any], ...] = (),
    group_fields: tuple[dict[str, Any], ...] = (),
    metric: dict[str, Any] | None = None,
    pattern: str = "list_rows",
) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "requested_fact_id": requested_fact_id,
        "answer_output_ids": list(answer_output_ids),
        "pattern": pattern,
        "source": {"kind": "read", "read_id": read_id},
    }
    if output_fields:
        answer["output_fields"] = list(output_fields)
    if group_fields:
        answer["group_fields"] = list(group_fields)
    if metric is not None:
        answer["metric"] = metric
    return {"outcome": {"kind": "fact_plan", "answers": [answer]}}


def _question_contract_for_plan(
    plan: FactPlan,
    *,
    description: str | None = None,
) -> QuestionContract:
    binding_target_ids = _render_output_ids(plan)
    return _question_contract_for(
        "rf_answer",
        description=description or _default_description(plan),
        binding_target_ids=binding_target_ids or ("answer",),
        answer_expression_family=_answer_expression_family_for_plan(plan),
    )


def _answer_expression_family_for_plan(
    plan: FactPlan,
) -> RequestedFactAnswerExpressionFamily:
    outcome = plan.outcome
    if not isinstance(outcome, AnswerPlan):
        return RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE
    operation_specs = tuple(operation.spec for operation in outcome.operations)
    if any(isinstance(spec, AntiJoinSpec) for spec in operation_specs):
        return RequestedFactAnswerExpressionFamily.SET_DIFFERENCE
    if any(isinstance(spec, JoinSpec) for spec in operation_specs):
        return RequestedFactAnswerExpressionFamily.LIST_ROWS
    if any(isinstance(spec, ComputeSpec) for spec in operation_specs):
        return RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR
    if any(isinstance(spec, RankSpec) for spec in operation_specs):
        return RequestedFactAnswerExpressionFamily.RANKED_SELECTION
    aggregate_specs = tuple(
        spec for spec in operation_specs if isinstance(spec, AggregateSpec)
    )
    if aggregate_specs:
        if any(spec.group_by for spec in aggregate_specs):
            return RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE
        return RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE
    if outcome.render_spec and outcome.render_spec.relation_outputs:
        return RequestedFactAnswerExpressionFamily.LIST_ROWS
    if outcome.render_spec and outcome.render_spec.scalar_outputs:
        return RequestedFactAnswerExpressionFamily.SCALAR_VALUE
    return RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE


def _render_output_ids(plan: FactPlan) -> tuple[str, ...]:
    outcome = plan.outcome
    if not isinstance(outcome, AnswerPlan) or outcome.render_spec is None:
        return ()
    return tuple(
        slot.id
        for slot in (
            *outcome.render_spec.relation_outputs,
            *outcome.render_spec.scalar_outputs,
        )
    )


def _default_description(plan: FactPlan) -> str:
    outcome = plan.outcome
    if not isinstance(outcome, AnswerPlan) or outcome.render_spec is None:
        return "field.name"
    render_spec = outcome.render_spec
    if render_spec.scalar_outputs:
        return " ".join(slot.scalar_id for slot in render_spec.scalar_outputs)
    if render_spec.relation_outputs:
        descriptions = [
            _source_description(outcome, slot.relation_id, slot.field_id)
            for slot in render_spec.relation_outputs
        ]
        return " ".join(dict.fromkeys(descriptions))
    return "field.name"


def _source_description(answer: AnswerPlan, relation_id: str, field_id: str) -> str:
    return _source_description_inner(answer, relation_id, field_id, seen=set())


def _source_description_inner(
    answer: AnswerPlan,
    relation_id: str,
    field_id: str,
    *,
    seen: set[tuple[str, str]],
) -> str:
    key = (relation_id, field_id)
    if key in seen:
        return field_id
    seen.add(key)
    bindings = {
        relation.id: {field.field_id: field.field_id for field in relation.fields}
        for relation in answer.relations
    }
    operation = next(
        (item for item in answer.operations if item.output_relation == relation_id),
        None,
    )
    if operation is None:
        return bindings.get(relation_id, {}).get(field_id, field_id)
    spec = operation.spec
    if isinstance(spec, ProjectSpec):
        for field in spec.fields:
            if (field.output or field.source) == field_id:
                return _source_description_inner(
                    answer, spec.input_relation, field.source, seen=seen
                )
    if isinstance(spec, AggregateSpec):
        if field_id in spec.group_by:
            return _source_description_inner(
                answer, spec.input_relation, field_id, seen=seen
            )
        for aggregation in spec.aggregations:
            if aggregation.output_field == field_id and aggregation.input_field:
                return _source_description_inner(
                    answer,
                    spec.input_relation,
                    aggregation.input_field,
                    seen=seen,
                )
    if isinstance(spec, AntiJoinSpec):
        for field in spec.output_fields:
            if (field.output or field.source) == field_id:
                return _source_description_inner(
                    answer,
                    spec.candidate.relation_id,
                    field.source,
                    seen=seen,
                )
    return field_id


__all__ = tuple(name for name in globals() if not name.startswith("__"))
