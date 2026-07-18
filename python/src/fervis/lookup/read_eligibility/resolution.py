"""Resolve and apply Read Eligibility's selected named-input bindings."""

from __future__ import annotations

from dataclasses import replace

from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    GroundedValueCertification,
    GroundingIssue,
    InputBindingOption,
    KnownInputBindingTask,
)
from fervis.lookup.grounding.resolution.references import (
    execute_compatible_reference_bindings,
    reference_binding_issue,
)
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lookup.read_eligibility.model import (
    ReadEligibilityRequest,
    ReadEligibilityResult,
    ResolvedRetainedReadSet,
    RetainedReadAssessment,
)
from fervis.lookup.relation_catalog import (
    RelationCatalog,
    RelationDataAccessPort,
)


def resolve_read_eligibility(
    *,
    request: ReadEligibilityRequest,
    result: ReadEligibilityResult,
    full_catalog: RelationCatalog,
    data_access_port: RelationDataAccessPort,
    source_read_key_prefix: str,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> ResolvedRetainedReadSet:
    retained_reads = tuple(
        assessment
        for assessment in result.read_assessments
        if isinstance(assessment, RetainedReadAssessment)
    )
    if not result.canonical_inputs:
        return ResolvedRetainedReadSet(
            retained_reads=retained_reads,
            ledger=CanonicalInputLedger(),
        )
    tasks_by_id = {task.known_input_id: task for task in request.binding_tasks}
    resolver_options_by_id = {
        option.id: option for task in request.binding_tasks for option in task.options
    }
    selected_resolver_bindings = tuple(
        selection.selected_resolver_binding
        for selection in result.canonical_inputs
        if selection.selected_resolver_binding is not None
    )
    resolver_executions = execute_compatible_reference_bindings(
        tasks=request.binding_tasks,
        bindings=selected_resolver_bindings,
        source_read_key_prefix=source_read_key_prefix,
        full_catalog=full_catalog,
        data_access_port=data_access_port,
        source_read_lineage=source_read_lineage,
    )
    values: list[FactValue] = []
    issues: list[GroundingIssue] = []
    certifications: list[GroundedValueCertification] = []
    canonical_values_by_id = {value.id: value for value in request.canonical_values}
    for selection in result.canonical_inputs:
        authority = selection.option
        if authority.canonical_value_id:
            value = canonical_values_by_id[authority.canonical_value_id]
        else:
            resolver_binding = selection.selected_resolver_binding
            if resolver_binding is None:
                raise ValueError("resolver-backed canonical input lost its binding")
            resolver_option = resolver_options_by_id[resolver_binding.option_id]
            outcome = resolver_executions[resolver_option.id]
            if outcome.failure is not None:
                raise outcome.failure
            assert outcome.ledger is not None
            task = tasks_by_id[selection.known_input_id]
            issue = _reference_issue(
                task,
                resolver_option=resolver_option,
                values=outcome.ledger.values,
                requested_fact_id=selection.requested_fact_id,
                truncated=outcome.truncated,
                matched_field_is_stable_unique=(outcome.matched_field_is_stable_unique),
            )
            if issue is not None:
                issues.append(issue)
                continue
            value = outcome.ledger.values[0]
            certifications.extend(outcome.ledger.certifications)
        value = replace(
            value,
            applies_to_requested_fact_ids=(selection.requested_fact_id,),
        )
        values.append(value)
    return ResolvedRetainedReadSet(
        retained_reads=retained_reads,
        ledger=CanonicalInputLedger(
            values=_dedupe_by_id(tuple(values)),
            issues=tuple(issues),
            certifications=tuple(dict.fromkeys(certifications)),
        ),
    )


def _reference_issue(
    task: KnownInputBindingTask,
    *,
    resolver_option: InputBindingOption,
    values: tuple[FactValue, ...],
    requested_fact_id: str,
    truncated: bool,
    matched_field_is_stable_unique: bool,
) -> GroundingIssue | None:
    issue = reference_binding_issue(
        task,
        candidate=resolver_option.candidate,
        values=values,
        truncated=truncated,
        matched_field_is_stable_unique=matched_field_is_stable_unique,
    )
    if issue is None or issue.requested_fact_id == requested_fact_id:
        return issue
    return replace(issue, requested_fact_id=requested_fact_id)


def _dedupe_by_id(values: tuple[FactValue, ...]) -> tuple[FactValue, ...]:
    output: dict[str, FactValue] = {}
    for value in values:
        existing = output.get(value.id)
        if existing is None:
            output[value.id] = value
            continue
        existing_without_scope = replace(
            existing,
            applies_to_requested_fact_ids=(),
        )
        value_without_scope = replace(
            value,
            applies_to_requested_fact_ids=(),
        )
        if existing_without_scope != value_without_scope:
            raise ValueError("grounded value id has conflicting values")
        output[value.id] = replace(
            existing,
            applies_to_requested_fact_ids=tuple(
                dict.fromkeys(
                    (
                        *existing.applies_to_requested_fact_ids,
                        *value.applies_to_requested_fact_ids,
                    )
                )
            ),
        )
    return tuple(output.values())
