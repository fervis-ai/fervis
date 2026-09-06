"""Every actual REST invocation must satisfy its own required parameters."""

from dataclasses import replace

from fervis.lookup.source_binding.verification import (
    SourceStrategyVerificationFailure,
    SourceStrategyVerificationFailureReason,
    VerifiedSourceStrategy,
    verify_source_strategy,
)
from tests.lookup.relational_engine.test_scoped_compilation import employee_query


def test_manager_parameter_does_not_satisfy_employee_invocation():
    verified = employee_query(manager_minimum=True)
    (source,) = verified.request.source_catalog.sources
    (parameter,) = source.params
    request = replace(
        verified.request,
        source_catalog=replace(
            verified.request.source_catalog,
            sources=(replace(source, params=(replace(parameter, required=True),)),),
        ),
    )
    result = verify_source_strategy(verified.binding_plan, request=request)
    assert isinstance(result, SourceStrategyVerificationFailure)
    assert (
        result.reason is SourceStrategyVerificationFailureReason.INCOMPLETE_INVOCATION
    )
    assert parameter.param_ref in result.failed_requirement_refs


def test_optional_manager_parameter_preserves_independent_employee_read():
    verified = employee_query(manager_minimum=True)
    result = verify_source_strategy(verified.binding_plan, request=verified.request)
    assert isinstance(result, VerifiedSourceStrategy)
