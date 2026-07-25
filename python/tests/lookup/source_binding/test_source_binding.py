from dataclasses import replace
from copy import deepcopy

from jsonschema import ValidationError, validate
import pytest

from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.answer_program.relations import FieldBindingRole
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCandidateKey,
    RowSourceEntityReference,
    RowSourceEntityReferenceComponent,
    RowSourceField,
    RowSourceCatalog,
    RowSourceKind,
    RowSourceKeyComponent,
    RowSourceParam,
    RowSourceValueType,
    build_row_source_catalog,
)
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.plan_selection.semantic import (
    CandidateSourceStrategy,
    SourceAlignment,
    SourceAlignmentAssessment,
    SourceStrategyBranch,
)
from fervis.lookup.read_eligibility.semantic import (
    ReadRequirementAssessment,
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.lookup.relation_catalog import EntityKeyComponentTarget, RelationCatalog
from fervis.lookup.source_binding.model import (
    AssociationRealization,
    AssociationRealizationKind,
    CatalogProvidedValue,
    FactRealization,
    FactRealizationKind,
    SemanticSourceBindingRequest,
    SetRealization,
    SourceBindingClarification,
    SourceBindingPlan,
    SourceMechanicKind,
    SubjectObligationBinding,
    SubjectObligationRealization,
    source_binding_clarification,
)
from fervis.lookup.source_binding.parser import compile_source_binding_plan
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
)
from fervis.lookup.source_binding.verification import (
    SourceStrategyVerificationFailure,
    SourceStrategyVerificationFailureReason,
    VerifiedSourceStrategy,
    verify_source_strategy,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    AllResults,
    AssociationTerm,
    FactTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TextType,
)
from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
)
from tests.lookup.grounding._fixtures import _staff_read
from tests.lookup.read_eligibility.test_semantic_read_eligibility import (
    _semantic_contract,
)


def test_boolean_requirement_fact_refs_accept_a_direct_boolean_fact() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "canceled")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", BooleanType(), origin),),
        expressions=(),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="f1",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "f1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = RowSource(
        id="source_sales",
        kind=RowSourceKind.API_READ,
        label="sales",
        params=(
            RowSourceParam(
                id="is_canceled",
                param_ref="source_sales.is_canceled",
                name="is_canceled",
                type=RowSourceValueType.BOOLEAN,
                choices=("true", "false"),
            ),
        ),
    )
    branch = SourceStrategyBranch(
        branch_id="fact_1:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=tuple(
            clause.clause_ref for clause in index.qualification.clauses
        ),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=fact.id,
            source_assessments=(
                SourceAlignmentAssessment(
                    source_ref=source.id,
                    basis="The source returns sales.",
                    alignment=SourceAlignment.DIRECT,
                ),
            ),
            strategy_basis="One direct source.",
            branches=(branch,),
        ),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(),
    )
    [requirement] = index.boolean_requirements

    assert request.requirement_fact_refs(requirement.requirement_ref) == (
        "fact_1:fact:f1",
    )
    assert request.requirement_value_refs(requirement.requirement_ref) == ()
    [surface] = request.source_catalog.choice_surfaces
    assert request.choice_requirement_refs(
        surface,
        branch_id=branch.branch_id,
    ) == (requirement.requirement_ref,)


def test_association_realization_schema_encodes_realization_kind_coherence() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "related records")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_left", origin), SetTerm("s_right", origin)),
        associations=(AssociationTerm("a_related", "s_left", "s_right", origin),),
        facts=(FactTerm("f_right_value", "s_right", TextType(), origin),),
        expressions=(),
        subject=Subject("s_left", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "f_right_value", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = RowSource(
        id="source_related",
        kind=RowSourceKind.API_READ,
        label="related records",
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    branch = SourceStrategyBranch(
        branch_id="fact_1:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=(),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=fact.id,
            source_assessments=(
                SourceAlignmentAssessment(
                    source_ref=source.id,
                    basis="The source contains the complete raw ingredients.",
                    alignment=SourceAlignment.DIRECT,
                ),
            ),
            strategy_basis="One co-resident source.",
            branches=(branch,),
        ),
        source_catalog=catalog,
        canonical_values=(),
    )
    association_ref = next(ref.token for ref in index.association_requirement_refs)
    schema = build_semantic_source_binding_schema(request)
    realization_schema = schema["properties"]["association_bindings"]["properties"][
        association_ref
    ]["items"]
    payload = {
        "branch_id": branch.branch_id,
        "mapping_basis": "Both set instances occur in one returned row.",
        "realization_ref": source.id,
    }

    validate(payload, realization_schema)
    with pytest.raises(ValidationError):
        validate({**payload, "contract_evidence_refs": [source.id]}, realization_schema)
    payload["realization_ref"] = "unknown"
    with pytest.raises(ValidationError):
        validate(payload, realization_schema)


def test_fact_field_schema_offers_only_type_compatible_fields() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "record label")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", TextType(), origin),),
        expressions=(),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "f1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    source = RowSource(
        id="source_records",
        kind=RowSourceKind.API_READ,
        label="records",
        fields=(
            RowSourceField(
                id="record_id",
                field_ref="source_records.record_id",
                label="record ID",
                type=RowSourceValueType.UUID,
                allowed_roles=(FieldBindingRole.IDENTITY,),
            ),
            RowSourceField(
                id="label",
                field_ref="source_records.label",
                label="label",
                type=RowSourceValueType.STRING,
                allowed_roles=(FieldBindingRole.OUTPUT,),
            ),
        ),
    )
    branch = SourceStrategyBranch(
        branch_id="fact_1:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=(),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=fact.id,
            source_assessments=(
                SourceAlignmentAssessment(
                    source_ref=source.id,
                    basis="The source returns record labels.",
                    alignment=SourceAlignment.DIRECT,
                ),
            ),
            strategy_basis="One direct source.",
            branches=(branch,),
        ),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(),
    )
    schema = build_semantic_source_binding_schema(request)
    fact_ref = index.fact_local_ref_by_local_id["f1"].token

    assert schema["properties"]["fact_bindings"]["properties"][fact_ref]["items"][
        "properties"
    ]["field_ref"]["enum"] == ["source_records.label"]


def test_identifier_fact_must_use_the_identified_set_identity_contract() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "paid staff")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_payment", origin), SetTerm("s_staff", origin)),
        associations=(AssociationTerm("a_paid_to", "s_payment", "s_staff", origin),),
        facts=(
            FactTerm(
                "f_staff_id",
                "a_paid_to",
                IdentifierType("s_staff"),
                origin,
            ),
        ),
        expressions=(),
        subject=Subject("s_payment", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "f_staff_id", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    payment_id = RowSourceField(
        id="payment_id",
        field_ref="source_payment.payment_id",
        label="payment ID",
        type=RowSourceValueType.UUID,
        allowed_roles=(FieldBindingRole.IDENTITY,),
    )
    staff_id = RowSourceField(
        id="staff_id",
        field_ref="source_payment.staff_id",
        label="staff ID",
        type=RowSourceValueType.UUID,
        allowed_roles=(FieldBindingRole.IDENTITY,),
    )
    source = RowSource(
        id="source_payment",
        kind=RowSourceKind.API_READ,
        label="payments",
        fields=(payment_id, staff_id),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="payment",
                components=(RowSourceKeyComponent("payment_id", payment_id.id),),
                primary=True,
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                id="staff_reference",
                target_entity_kind="staff",
                target_key_id="primary_key",
                components=(
                    RowSourceEntityReferenceComponent("staff_id", staff_id.id),
                ),
            ),
        ),
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    branch_id = "fact_1:source_branch:1"
    strategy = CandidateSourceStrategy(
        requested_fact_id=fact.id,
        source_assessments=(
            SourceAlignmentAssessment(
                source_ref=source.id,
                basis="The source returns payments and their staff reference.",
                alignment=SourceAlignment.DIRECT,
            ),
        ),
        strategy_basis="One source contains the requested relation.",
        branches=(
            SourceStrategyBranch(
                branch_id=branch_id,
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=(),
            ),
        ),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=catalog,
        canonical_values=(),
    )
    payment_identity, staff_identity = source.identity_evidence
    payment_set_ref = index.fact_local_ref_by_local_id["s_payment"].token
    staff_set_ref = index.fact_local_ref_by_local_id["s_staff"].token
    association_ref = index.fact_local_ref_by_local_id["a_paid_to"].token
    identifier_ref = index.fact_local_ref_by_local_id["f_staff_id"].token
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            payment_set_ref: (
                SetRealization(
                    branch_id,
                    "Payment rows represent payments.",
                    source.id,
                    payment_identity.identity_ref,
                    payment_identity.field_refs,
                    (payment_identity.identity_ref,),
                ),
            ),
            staff_set_ref: (
                SetRealization(
                    branch_id,
                    "Staff is mentioned by each payment.",
                    source.id,
                    None,
                    (),
                    (source.id,),
                ),
            ),
        },
        fact_bindings={
            identifier_ref: (
                FactRealization(
                    branch_id,
                    "Use the payment identity for the staff output.",
                    source.id,
                    FactRealizationKind.ENTITY_KEY,
                    payment_identity.identity_ref,
                    payment_identity.field_refs,
                    (payment_identity.identity_ref,),
                ),
            ),
        },
        association_bindings={
            association_ref: (
                AssociationRealization(
                    branch_id,
                    "Payment and staff occur in one row.",
                    AssociationRealizationKind.CO_RESIDENT,
                    (source.id,),
                    None,
                    (source.id,),
                ),
            ),
        },
        invocation_applications=(),
        boolean_bindings={},
        subject_binding=SubjectObligationBinding(
            payment_set_ref,
            (SubjectObligationRealization(branch_id, ()),),
        ),
    )

    failure = verify_source_strategy(plan, request=request)

    assert isinstance(failure, SourceStrategyVerificationFailure)
    assert (
        failure.reason
        is SourceStrategyVerificationFailureReason.INCONSISTENT_IDENTIFIER
    )
    corrected = replace(
        plan,
        set_bindings={
            **plan.set_bindings,
            staff_set_ref: (
                replace(
                    plan.set_bindings[staff_set_ref][0],
                    identity_ref=staff_identity.identity_ref,
                    identity_field_refs=staff_identity.field_refs,
                ),
            ),
        },
        fact_bindings={
            identifier_ref: (
                replace(
                    plan.fact_bindings[identifier_ref][0],
                    identity_ref=staff_identity.identity_ref,
                    field_refs=staff_identity.field_refs,
                ),
            ),
        },
    )
    assert isinstance(
        verify_source_strategy(corrected, request=request), VerifiedSourceStrategy
    )


def test_semantic_binding_maps_requirements_once_per_strategy_branch() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    all_sources = build_row_source_catalog(RelationCatalog(reads=(_staff_read(),)))
    row_sources = RowSourceCatalog(
        sources=tuple(source for source in all_sources.sources if source.read_id)
    )
    [source] = row_sources.sources
    read_result = SemanticReadEligibilityResult(
        read_assessments=(
            ReadRequirementAssessment(
                requested_fact_id=index.requested_fact_id,
                candidate_ref=source.read_id,
                source_refs=(source.id,),
                read_id=source.read_id,
                relevant_field_refs=tuple(field.field_ref for field in source.fields),
                assessment_basis="The source supplies the staff rows and identity.",
                decision=SemanticReadDecision.RETAIN,
            ),
        ),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(
        row_sources, read_eligibility=read_result
    )
    [clause] = index.qualification.clauses
    strategy = CandidateSourceStrategy(
        requested_fact_id=index.requested_fact_id,
        source_assessments=(
            SourceAlignmentAssessment(
                source_ref=source.id,
                basis="The source contains the complete raw ingredients.",
                alignment=SourceAlignment.DIRECT,
            ),
        ),
        strategy_basis="The staff source is direct.",
        branches=(
            SourceStrategyBranch(
                branch_id=f"{index.requested_fact_id}:source_branch:1",
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=(clause.clause_ref,),
            ),
        ),
    )
    [use] = index.input_use_sites
    fact_value = FactValue.identity(
        id="canonical_staff_1",
        known_input_id=input_term.id,
        key=EntityKeyValue(
            entity_kind="staff",
            key_id="primary_key",
            components=(EntityKeyComponentValue("staff_id", "staff_1"),),
        ),
        display_value="Ada",
        proof_refs=("resolver:list_staff_list",),
    )
    canonical_value = CanonicalInputValue(
        canonical_value_id=fact_value.id,
        input_ref=input_term.id,
        use_refs=(use.use_ref,),
        typed_value=fact_value,
        certification_refs=("resolver:list_staff_list",),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=available,
        canonical_values=(canonical_value,),
    )
    invocation_schema = build_semantic_source_binding_schema(request)["properties"][
        "resolved_input_applications"
    ]
    assert tuple(invocation_schema["properties"]) == (strategy.branches[0].branch_id,)
    set_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
    )
    fact_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "fact"
    )
    [boolean] = index.boolean_requirements
    branch_id = strategy.branches[0].branch_id
    [identity_evidence] = available.identity_evidence
    payload = {
        "set_bindings": {
            set_ref: [
                {
                    "branch_id": branch_id,
                    "mapping_basis": "Staff rows represent the requested set.",
                    "source_ref": source.id,
                    "identity_ref": identity_evidence.identity_ref,
                }
            ]
        },
        "fact_bindings": {
            fact_ref: [
                {
                    "branch_id": branch_id,
                    "mapping_basis": "The declared staff key carries staff identity.",
                    "source_ref": source.id,
                }
            ]
        },
        "association_bindings": {},
        "resolved_input_applications": {branch_id: []},
        "finite_choice_applications": {branch_id: {}},
        "subject_binding": {
            "subject_ref": index.subject_obligation.subject_set_ref.token,
            "branch_realizations": [
                {"branch_id": branch_id, "finite_choice_reviews": {}}
            ],
        },
    }

    validate(payload, build_semantic_source_binding_schema(request))
    plan = compile_source_binding_plan(payload, request=request)

    fact_binding = plan.fact_bindings[fact_ref][0]
    assert fact_binding.kind is FactRealizationKind.ENTITY_KEY
    assert fact_binding.contract_evidence_refs == (
        available.contract_snapshot.ref,
        source.id,
        identity_evidence.identity_ref,
        *identity_evidence.field_refs,
    )
    assert (
        plan.boolean_bindings[boolean.requirement_ref][0].mechanics[0].kind
        is SourceMechanicKind.RETURNED_ROW_PREDICATE
    )
    assert isinstance(
        verify_source_strategy(plan, request=request), VerifiedSourceStrategy
    )

    identity_param = replace(
        source.params[0],
        entity_target=EntityKeyComponentTarget(
            entity_kind="staff",
            key_id="primary_key",
            component_id="staff_id",
        ),
    )
    verification_param = RowSourceParam(
        id="is_verified",
        param_ref=f"{source.id}.is_verified",
        name="is_verified",
        type=RowSourceValueType.CHOICE,
        choices=("true", "false"),
    )
    returned_identity_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(replace(source, params=(verification_param,)),),
        ),
    )
    [returned_verification_surface] = (
        returned_identity_request.source_catalog.choice_surfaces
    )
    assert boolean.requirement_ref not in (
        returned_identity_request.choice_requirement_refs(
            returned_verification_surface,
            branch_id=branch_id
        )
    )
    invocation_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(
                replace(
                    source,
                    params=(identity_param, verification_param),
                ),
            ),
        ),
    )
    [invocation_verification_surface] = tuple(
        surface
        for surface in invocation_request.source_catalog.choice_surfaces
        if surface.surface_ref == verification_param.param_ref
    )
    assert boolean.requirement_ref not in (
        invocation_request.choice_requirement_refs(
            invocation_verification_surface,
            branch_id=branch_id,
        )
    )
    [identity_option] = invocation_request.invocation_options_for_owner(
        boolean.requirement_ref,
        branch_id=branch_id,
    )
    invocation_payload = deepcopy(payload)
    invocation_payload["fact_bindings"] = {fact_ref: []}
    invocation_payload["resolved_input_applications"] = {
        branch_id: [
            {
                "mapping_basis": "The request parameter enforces staff identity.",
                "owner_ref": boolean.requirement_ref,
                "value_ref": identity_option.value_ref,
                "value_component": (
                    identity_option.component_ref or identity_option.projection.value
                ),
                "target_ref": identity_option.target_ref,
            }
        ]
    }
    invocation_payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ] = {
        verification_param.param_ref: {
            "surface_mapping_basis": (
                "Verification does not fulfill the staff identity requirement."
            ),
            "choice_reviews": {
                value: {
                    "choice_domain_meaning": (
                        f"Rows whose verification flag is {value}."
                    ),
                    "role_match_basis": (
                        "Verification does not match an excluded staff state."
                    ),
                    "matched_excluded_role": "NONE",
                    "choice_inclusion_basis": (
                        "Verification does not restrict the requested staff rows."
                    ),
                    "choice_inclusion": "INCLUDE",
                }
                for value in verification_param.choices
            },
        }
    }
    validate(
        invocation_payload,
        build_semantic_source_binding_schema(invocation_request),
    )
    invocation_plan = compile_source_binding_plan(
        invocation_payload,
        request=invocation_request,
    )

    assert isinstance(
        verify_source_strategy(invocation_plan, request=invocation_request),
        VerifiedSourceStrategy,
    )

    missing_param = RowSourceParam(
        id="required_scope",
        param_ref=f"{source.id}.required_scope",
        name="required_scope",
        type=RowSourceValueType.STRING,
        required=True,
    )
    incomplete_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(replace(source, params=(*source.params, missing_param)),),
        ),
    )
    failure = verify_source_strategy(plan, request=incomplete_request)
    assert isinstance(failure, SourceStrategyVerificationFailure)
    assert (
        failure.reason is SourceStrategyVerificationFailureReason.INCOMPLETE_INVOCATION
    )
    clarification = source_binding_clarification(incomplete_request)
    assert isinstance(clarification, SourceBindingClarification)
    [missing] = clarification.missing_catalog_values
    assert missing.target_ref == missing_param.param_ref

    supplied = CatalogProvidedValue(
        catalog_input_ref=missing.catalog_input_ref,
        target_ref=missing.target_ref,
        value_id="catalog_required_scope",
        typed_value=FactValue.literal(
            id="catalog_required_scope",
            literal_type=LiteralType.STRING,
            value="current",
            proof_refs=(missing.catalog_input_ref,),
        ),
        certification_refs=(missing.catalog_input_ref,),
    )
    successor = replace(incomplete_request, catalog_values=(supplied,))
    assert source_binding_clarification(successor) is None
    assert any(
        option.target_ref == missing_param.param_ref
        and option.value_ref == supplied.value_id
        for option in successor.invocation_projection_options
    )
    required_owner = f"source_required:{missing_param.param_ref}"
    required_option = next(
        option
        for option in successor.invocation_projection_options
        if option.target_ref == missing_param.param_ref
        and option.value_ref == supplied.value_id
    )
    supplied_payload = deepcopy(payload)
    supplied_payload["resolved_input_applications"] = {
        branch_id: [
            {
                "mapping_basis": "Supply the required scope.",
                "owner_ref": required_owner,
                "value_ref": required_option.value_ref,
                "value_component": (
                    required_option.component_ref or required_option.projection.value
                ),
                "target_ref": required_option.target_ref,
            }
        ]
    }
    validate(supplied_payload, build_semantic_source_binding_schema(successor))
    supplied_plan = compile_source_binding_plan(
        supplied_payload,
        request=successor,
    )
    assert isinstance(
        verify_source_strategy(supplied_plan, request=successor),
        VerifiedSourceStrategy,
    )

    missing_required_owner = deepcopy(supplied_payload)
    missing_required_owner["resolved_input_applications"][branch_id] = []
    validate(
        missing_required_owner,
        build_semantic_source_binding_schema(successor),
    )
    with pytest.raises(ValueError, match="required invocation target"):
        compile_source_binding_plan(
            missing_required_owner,
            request=successor,
        )

    lifecycle_param = RowSourceParam(
        id="state",
        param_ref=f"{source.id}.state",
        name="state",
        type=RowSourceValueType.CHOICE,
        choices=("active", "provisional", "deleted"),
    )
    unreviewed_subject_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(replace(source, params=(*source.params, lifecycle_param)),),
        ),
    )
    subject_failure = verify_source_strategy(plan, request=unreviewed_subject_request)
    assert isinstance(subject_failure, SourceStrategyVerificationFailure)
    assert (
        subject_failure.reason
        is SourceStrategyVerificationFailureReason.INSUFFICIENT_COMPLETENESS
    )
    assert {
        value.value_ref
        for value in unreviewed_subject_request.source_catalog.choice_values
    } <= {
        option.value_ref
        for option in unreviewed_subject_request.invocation_projection_options
    }
    assert not {
        value.value_ref
        for value in unreviewed_subject_request.source_catalog.choice_values
    } & {
        option.value_ref
        for option in unreviewed_subject_request.authored_invocation_projection_options
    }

    [active_choice, provisional_choice, deleted_choice] = (
        unreviewed_subject_request.source_catalog.choice_values
    )
    lifecycle_review = deepcopy(payload)
    lifecycle_review["resolved_input_applications"] = {branch_id: []}
    lifecycle_review["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ] = {
        lifecycle_param.param_ref: {
            "surface_mapping_basis": (
                "The lifecycle surface defines the requested event population."
            ),
            "choice_reviews": {
                active_choice.value: {
                    "choice_domain_meaning": "Active rows are effective events.",
                    "role_match_basis": "Active is not an excluded state.",
                    "matched_excluded_role": "NONE",
                    "choice_inclusion_basis": (
                        "Active events belong in the normal subject set."
                    ),
                    "choice_inclusion": "INCLUDE",
                },
                provisional_choice.value: {
                    "choice_domain_meaning": (
                        "Provisional rows are ordinary effective events in this domain."
                    ),
                    "role_match_basis": (
                        "Provisional is not an excluded state."
                    ),
                    "matched_excluded_role": "NONE",
                    "choice_inclusion_basis": (
                        "Provisional events belong in the normal subject set."
                    ),
                    "choice_inclusion": "INCLUDE",
                },
                deleted_choice.value: {
                    "choice_domain_meaning": "Deleted rows are non-current events.",
                    "role_match_basis": "Deleted is a non-current artifact.",
                    "matched_excluded_role": "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT",
                    "choice_inclusion_basis": (
                        "Deleted events do not belong in the normal subject set."
                    ),
                    "choice_inclusion": "EXCLUDE",
                },
            },
        }
    }
    validate(
        lifecycle_review,
        build_semantic_source_binding_schema(unreviewed_subject_request),
    )
    lifecycle_plan = compile_source_binding_plan(
        lifecycle_review,
        request=unreviewed_subject_request,
    )
    [lifecycle_surface] = lifecycle_plan.subject_binding.branch_realizations[
        0
    ].surface_reviews
    assert lifecycle_surface.included_choice_refs == (
        active_choice.value_ref,
        provisional_choice.value_ref,
    )
    assert isinstance(
        verify_source_strategy(
            lifecycle_plan,
            request=unreviewed_subject_request,
        ),
        VerifiedSourceStrategy,
    )

    subtype_param = RowSourceParam(
        id="kind",
        param_ref=f"{source.id}.kind",
        name="kind",
        type=RowSourceValueType.CHOICE,
        choices=("STAFF", "CONTRACTOR"),
    )
    subtype_request = replace(
        request,
        source_catalog=replace(
            request.source_catalog,
            sources=(replace(source, params=(*source.params, subtype_param)),),
        ),
    )
    [staff_choice, contractor_choice] = subtype_request.source_catalog.choice_values
    subtype_payload = deepcopy(payload)
    subtype_payload["resolved_input_applications"] = {branch_id: []}
    subtype_payload["subject_binding"]["branch_realizations"][0][
        "finite_choice_reviews"
    ] = {
        subtype_param.param_ref: {
            "surface_mapping_basis": (
                "The kind surface defines the requested staff population."
            ),
            "choice_reviews": {
                staff_choice.value: {
                    "choice_domain_meaning": "Staff rows represent staff members.",
                    "role_match_basis": "Staff belongs to the requested subject set.",
                    "matched_excluded_role": "NONE",
                    "choice_inclusion_basis": (
                        "Staff members belong in the normal subject set."
                    ),
                    "choice_inclusion": "INCLUDE",
                },
                contractor_choice.value: {
                    "choice_domain_meaning": (
                        "Contractor rows represent contractor members."
                    ),
                    "role_match_basis": (
                        "Contractors match none of the excluded subject-state roles."
                    ),
                    "matched_excluded_role": "NONE",
                    "choice_inclusion_basis": (
                        "Contractors do not belong in the normal staff subject set."
                    ),
                    "choice_inclusion": "EXCLUDE",
                },
            },
        }
    }

    validate(subtype_payload, build_semantic_source_binding_schema(subtype_request))
    subtype_plan = compile_source_binding_plan(
        subtype_payload,
        request=subtype_request,
    )

    assert {
        subtype_request.source_catalog.choice_value(application.value_ref).value
        for application in subtype_plan.invocation_applications
        if application.value_ref.startswith("source_choice:")
    } == {"STAFF"}
    assert isinstance(
        verify_source_strategy(subtype_plan, request=subtype_request),
        VerifiedSourceStrategy,
    )


def test_input_owned_identifier_fact_only_offers_the_certified_identity_contract() -> (
    None
):
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    [use] = index.input_use_sites
    all_sources = build_row_source_catalog(RelationCatalog(reads=(_staff_read(),)))
    [base_source] = tuple(source for source in all_sources.sources if source.read_id)
    descriptive_field = next(
        field for field in base_source.fields if field.label == "full_name"
    )
    source = replace(
        base_source,
        candidate_keys=(
            *base_source.candidate_keys,
            RowSourceCandidateKey(
                id="unique_name",
                entity_kind="staff",
                components=(
                    RowSourceKeyComponent(
                        id="full_name",
                        field_id=descriptive_field.id,
                    ),
                ),
            ),
        ),
    )
    available = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    canonical_value = CanonicalInputValue(
        canonical_value_id="canonical_staff_1",
        input_ref=input_term.id,
        use_refs=(use.use_ref,),
        typed_value=FactValue.identity(
            id="canonical_staff_1",
            known_input_id=input_term.id,
            key=EntityKeyValue(
                entity_kind="staff",
                key_id="primary_key",
                components=(EntityKeyComponentValue("staff_id", "staff_1"),),
            ),
            display_value="Ada",
            proof_refs=("resolver:list_staff_list",),
        ),
        certification_refs=("resolver:list_staff_list",),
    )
    strategy = CandidateSourceStrategy(
        requested_fact_id=index.requested_fact_id,
        source_assessments=(
            SourceAlignmentAssessment(
                source_ref=source.id,
                basis="The source contains the requested staff rows.",
                alignment=SourceAlignment.DIRECT,
            ),
        ),
        strategy_basis="The source is direct.",
        branches=(
            SourceStrategyBranch(
                branch_id=f"{index.requested_fact_id}:source_branch:1",
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=tuple(
                    clause.clause_ref for clause in index.qualification.clauses
                ),
            ),
        ),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=available,
        canonical_values=(canonical_value,),
    )
    assert use.reference_fact_ref is not None

    schema = build_semantic_source_binding_schema(request)
    identifier_schema = schema["properties"]["fact_bindings"]["properties"][
        use.reference_fact_ref.token
    ]["items"]

    assert set(identifier_schema["properties"]) == {
        "branch_id",
        "mapping_basis",
        "source_ref",
    }
    (SetRealization,)
    (SourceBindingPlan,)
    (SubjectObligationBinding,)
    (SubjectObligationRealization,)
