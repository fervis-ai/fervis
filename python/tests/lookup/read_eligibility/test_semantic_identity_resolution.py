from dataclasses import replace

import pytest

from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.grounding.identity import IdentifierKind, reference_binding_options
from fervis.lookup.grounding.semantic import (
    CompatibleIdentityRoute,
    GroundingPartition,
    IdentityExecutionClarification,
    IdentityExecutionFailureReason,
    ResolvedIdentity,
    identity_resolution_tasks,
    reference_grounding_tasks,
)
from fervis.lookup.question_contract.semantic_model import (
    FactLocalKind,
    FactLocalRef,
    InputTerm,
)
from fervis.lookup.read_eligibility.semantic import IdentityRouteSelection
from fervis.lookup.read_eligibility.semantic_resolution import (
    execute_identity_selection,
)
from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    RelationCatalog,
)
from fervis.lookup.semantic_types import (
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TextType,
)
from tests.lookup.grounding._fixtures import _staff_read, _uuid_person_read


@pytest.mark.parametrize(
    ("response_body", "truncated", "reason"),
    (
        ({"data": []}, False, IdentityExecutionFailureReason.NOT_FOUND),
        (
            {
                "data": [
                    {"staff_id": "staff_1", "full_name": "Ada"},
                    {"staff_id": "staff_2", "full_name": "Ada"},
                ]
            },
            False,
            IdentityExecutionFailureReason.AMBIGUOUS_RESULT,
        ),
        (
            {"data": [{"staff_id": "staff_1", "full_name": "Ada"}]},
            True,
            IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT,
        ),
    ),
)
def test_identity_resolution_fails_closed_on_missing_ambiguous_or_truncated_rows(
    response_body: dict[str, object],
    truncated: bool,
    reason: IdentityExecutionFailureReason,
) -> None:
    case = _identity_case("Ada")

    result = case.execute(_DataAccess(response_body, truncated=truncated))

    assert isinstance(result, IdentityExecutionClarification)
    assert result.reason is reason


def test_identity_resolution_uses_only_selected_fields_and_exact_case() -> None:
    case = _identity_case("Ada")

    result = case.execute(
        _DataAccess(
            {
                "data": [
                    {
                        "staff_id": "staff_1",
                        "full_name": "ada",
                        "first_name": "Ada",
                    }
                ]
            }
        )
    )

    assert isinstance(result, IdentityExecutionClarification)
    assert result.reason is IdentityExecutionFailureReason.NOT_FOUND


def test_truncated_identity_resolution_accepts_a_declared_stable_unique_match() -> None:
    read = _staff_read()
    read = replace(
        read,
        candidate_keys=(
            *read.candidate_keys,
            CandidateKey(
                id="full_name",
                entity_kind="staff",
                components=(
                    CandidateKeyComponent(
                        id="full_name",
                        field_ref="field.data.full_name",
                    ),
                ),
                stable=True,
            ),
        ),
    )
    case = _identity_case("Ada", catalog=RelationCatalog(reads=(read,)))

    result = case.execute(
        _DataAccess(
            {"data": [{"staff_id": "staff_1", "full_name": "Ada"}]},
            truncated=True,
        )
    )

    assert isinstance(result, ResolvedIdentity)
    assert result.canonical_value.typed_value.payload.key.component_values() == {
        "staff_id": "staff_1"
    }


def test_identity_resolution_compares_declared_uuid_values_canonically() -> None:
    input_value = "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
    canonical_value = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    catalog = RelationCatalog(reads=(_uuid_person_read(),))
    case = _identity_case(
        input_value,
        catalog=catalog,
        request_param_ref="get_person.query.person_id",
        verification_path="data.person_id",
        identifier_kind=IdentifierKind.PRIMARY_KEY,
    )

    result = case.execute(
        _DataAccess(
            {"data": {"person_id": canonical_value, "name": "Ada"}},
            endpoint_name="get_person",
        )
    )

    assert isinstance(result, ResolvedIdentity)
    assert result.canonical_value.typed_value.payload.key.component_values() == {
        "person_id": canonical_value
    }


def test_identity_resolution_scopes_source_read_failure_to_the_selected_task() -> None:
    case = _identity_case("Ada")

    result = case.execute(_FailingDataAccess())

    assert isinstance(result, IdentityExecutionClarification)
    assert result.reason is IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT
    assert result.input_ref == case.input_term.id


class _IdentityCase:
    def __init__(self, *, catalog, task, selection, input_term) -> None:
        self.catalog = catalog
        self.task = task
        self.selection = selection
        self.input_term = input_term

    def execute(self, data_access):
        return execute_identity_selection(
            task=self.task,
            selection=self.selection,
            input_term=self.input_term,
            full_catalog=self.catalog,
            data_access_port=data_access,
            source_read_key_prefix="test",
        )


def _identity_case(
    operand: str,
    *,
    catalog: RelationCatalog | None = None,
    request_param_ref: str = "list_staff_list.query.name",
    verification_path: str = "data.full_name",
    identifier_kind: IdentifierKind = IdentifierKind.DESCRIPTIVE,
) -> _IdentityCase:
    catalog = catalog or RelationCatalog(reads=(_staff_read(),))
    input_term = InputTerm(
        id="input_1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, operand),
        operand=operand,
        value_type=TextType(),
    )
    set_ref = FactLocalRef("fact_1", FactLocalKind.SET, "s1")
    [option] = reference_binding_options(
        input_id=input_term.id,
        resolver_catalog=catalog,
        resolver_row_sources=build_row_source_catalog(catalog),
        expected_identity=None,
    )
    [grounding_task] = reference_grounding_tasks(
        (
            GroundingPartition(
                input_ref=input_term.id,
                use_refs=("fact_1:input_use:e1:1",),
                expected_value_type=IdentifierType("s1"),
                expected_set_ref=set_ref,
                operand_meaning="the identified instance",
            ),
        ),
        resolver_options_by_use_ref={"fact_1:input_use:e1:1": (option,)},
        denoted_instance_kinds_by_input_ref={input_term.id: "staff member"},
    )
    route = CompatibleIdentityRoute(
        option_id=option.id,
        identifier_kind=identifier_kind,
        lookup_request_param_refs=(request_param_ref,),
        returned_identity_verification_field_paths=(verification_path,),
    )
    [task] = identity_resolution_tasks(
        (grounding_task,),
        compatible_bindings_by_task_ref={grounding_task.task_ref: (route,)},
    )
    [canonical_option] = task.canonical_options
    selection = IdentityRouteSelection(
        task_ref=task.task_ref,
        canonical_option_assessments=(),
        canonical_option_basis="The operand denotes the returned identity.",
        canonical_option_id=canonical_option.canonical_option_id,
        resolver_route_assessments=(),
        resolver_route_basis="The selected route resolves that identity.",
        resolver_route_id=option.id,
    )
    return _IdentityCase(
        catalog=catalog,
        task=task,
        selection=selection,
        input_term=input_term,
    )


class _DataAccess:
    def __init__(
        self,
        response_body: dict[str, object],
        *,
        truncated: bool = False,
        endpoint_name: str = "list_staff_list",
    ) -> None:
        self.response_body = response_body
        self.truncated = truncated
        self.endpoint_name = endpoint_name

    def read(self, *, endpoint_name, args):
        assert endpoint_name == self.endpoint_name
        assert args
        return {
            "responseStatus": 200,
            "responseBody": self.response_body,
            "truncated": self.truncated,
        }


class _FailingDataAccess:
    def read(self, *, endpoint_name, args):
        del endpoint_name, args
        raise RuntimeError("resolver unavailable")
