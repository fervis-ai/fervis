"""Coverage is relational division over complete candidate and required sets."""

from decimal import Decimal

import pytest

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
)
from fervis.lookup.question_contract.model import (
    RequestedFact,
    SetTerm,
    AssociationTerm,
    FactTerm,
    NullCheck,
    Coverage,
    RequestedOutput,
    Subject,
    InstanceInterpretation,
    AllResults,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.semantic_types import (
    SourceOrigin,
    SourceOriginKind,
    IdentifierType,
    DecimalType,
    UnitlessMeasure,
)
from fervis.lookup.expression_operators import ExpressionUnaryOperator
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSource,
    RowSourceKind,
    RowSourceField,
    RowSourceValueType,
    RowSourceCandidateKey,
    RowSourceKeyComponent,
    RowSourceEntityReference,
    RowSourceEntityReferenceComponent,
    row_source_relation_evidence,
)
from fervis.lookup.answer_program.relations import FieldBindingRole
from fervis.lookup.source_binding.model import (
    CandidateSourceStrategy,
    SourceStrategyBranch,
    SemanticSourceBindingRequest,
    SetRealization,
    FactRealization,
    FactRealizationKind,
    AssociationRealization,
    AssociationRealizationKind,
    SourceBindingPlan,
    SubjectObligationBinding,
    SubjectObligationRealization,
    BooleanRequirementRealization,
    SourceMechanic,
    SourceMechanicKind,
)
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)


def coverage_query(*, return_truth=False):
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "Candidates with an observation for every required member",
    )
    fact = RequestedFact(
        "coverage_fact",
        origin,
        tuple(
            SetTerm(name, origin) for name in ("candidate", "required", "observation")
        ),
        (
            AssociationTerm(
                "candidate_observations", "candidate", "observation", origin
            ),
            AssociationTerm("required_observations", "required", "observation", origin),
        ),
        (
            FactTerm("candidate_id", "candidate", IdentifierType("candidate"), origin),
            FactTerm("amount", "observation", DecimalType(UnitlessMeasure()), origin),
        ),
        (
            NullCheck("qualifies", ExpressionUnaryOperator.NOT_NULL, "amount", origin),
            Coverage(
                "covered",
                "candidate",
                "required",
                "observation",
                ("candidate_observations",),
                ("required_observations",),
                None,
                "qualifies",
                origin,
            ),
        ),
        Subject("candidate", InstanceInterpretation.RESOURCE_POPULATION),
        None if return_truth else "covered",
        (),
        (RequestedOutput("candidate", "candidate_id", origin),)
        + ((RequestedOutput("truth", "covered", origin),) if return_truth else ()),
        (),
        AllResults(),
        (),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    roles = (
        FieldBindingRole.IDENTITY,
        FieldBindingRole.OUTPUT,
        FieldBindingRole.PREDICATE,
    )
    sources = []
    for name in ("candidate", "required", "observation"):
        fields = (
            RowSourceField(
                "id", f"{name}.id", "Identifier", RowSourceValueType.STRING, roles
            ),
        )
        references = ()
        if name == "observation":
            fields += tuple(
                RowSourceField(
                    f"{owner}_id",
                    f"{name}.{owner}_id",
                    "Reference",
                    RowSourceValueType.STRING,
                    roles,
                )
                for owner in ("candidate", "required")
            )
            fields += (
                RowSourceField(
                    "amount",
                    "observation.amount",
                    "Amount",
                    RowSourceValueType.DECIMAL,
                    roles,
                ),
            )
            references = tuple(
                RowSourceEntityReference(
                    owner,
                    owner,
                    "pk",
                    (RowSourceEntityReferenceComponent("id", f"{owner}_id"),),
                )
                for owner in ("candidate", "required")
            )
        sources.append(
            RowSource(
                id=name,
                kind=RowSourceKind.API_READ,
                read_id=f"list_{name}",
                label=name,
                fields=fields,
                candidate_keys=(
                    RowSourceCandidateKey(
                        "pk", name, (RowSourceKeyComponent("id", "id"),), primary=True
                    ),
                ),
                entity_references=references,
            )
        )
    links = row_source_relation_evidence(tuple(sources))
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"), tuple(sources), links
    )
    branch = SourceStrategyBranch(
        "branch",
        ("candidate", "observation", "required"),
        tuple(link.evidence_ref for link in links),
        tuple(c.clause_ref for c in index.qualification.clauses),
    )
    strategy = CandidateSourceStrategy(fact.id, (branch,))
    request = SemanticSourceBindingRequest(index, strategy, catalog, ())
    refs = index.fact_local_ref_by_local_id
    sets = {}
    for source in sources:
        identity = source.identity_evidence[0]
        sets[refs[source.id].token] = (
            SetRealization(
                "branch",
                "Complete entity rows.",
                source.id,
                identity.identity_ref,
                identity.field_refs,
                (source.id, identity.identity_ref),
            ),
        )
    identity = sources[0].identity_evidence[0]
    facts = {
        refs["candidate_id"].token: (
            FactRealization(
                "branch",
                "Candidate identity.",
                "candidate",
                FactRealizationKind.ENTITY_KEY,
                identity.identity_ref,
                identity.field_refs,
                ("candidate", identity.identity_ref),
            ),
        ),
        refs["amount"].token: (
            FactRealization(
                "branch",
                "Observed amount.",
                "observation",
                FactRealizationKind.RETURNED_FIELD,
                None,
                ("observation.amount",),
                ("observation", "observation.amount"),
            ),
        ),
    }
    associations = {}
    for owner in ("candidate", "required"):
        link = next(
            link
            for link in links
            if owner in (link.left_source_ref, link.right_source_ref)
        )
        associations[refs[f"{owner}_observations"].token] = (
            AssociationRealization(
                "branch",
                "Declared foreign key.",
                AssociationRealizationKind.DECLARED_RELATION,
                (owner, "observation"),
                link.evidence_ref,
                (owner, "observation", link.evidence_ref),
            ),
        )
    booleans = {
        r.requirement_ref: (
            BooleanRequirementRealization(
                "branch",
                (
                    SourceMechanic(
                        "Evaluate the condition.",
                        "observation",
                        (),
                        ("observation", r.requirement_ref),
                        SourceMechanicKind.RETURNED_ROW_PREDICATE,
                    ),
                ),
            ),
        )
        for r in index.boolean_requirements
    }
    plan = SourceBindingPlan(
        strategy,
        sets,
        facts,
        associations,
        (),
        booleans,
        SubjectObligationBinding(
            refs["candidate"].token, (SubjectObligationRealization("branch", ()),)
        ),
    )
    request.for_bindings(
        plan.set_bindings, plan.fact_bindings, plan.association_bindings
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    return verified


@pytest.mark.parametrize(
    ("required", "observations", "expected"),
    [
        (("x", "y"), (("a", "x", 1), ("a", "y", 1), ("b", "x", 1)), {"a"}),
        ((), (), {"a", "b", "c"}),
        (("x",), (), set()),
        (("x",), (("a", "x", None), ("a", "x", 1), ("b", "x", None)), {"a"}),
    ],
)
@pytest.mark.parametrize("return_truth", [False, True])
def test_coverage_preserves_missing_members_vacuity_and_multiple_observations(
    required, observations, expected, return_truth
):
    program = compile_verified_source_strategy(
        coverage_query(return_truth=return_truth)
    ).answer_program
    data = {
        "candidate": tuple({"id": item} for item in ("a", "b", "c")),
        "required": tuple({"id": item} for item in required),
        "observation": tuple(
            {
                "id": str(i),
                "candidate_id": candidate,
                "required_id": member,
                "amount": None if amount is None else Decimal(amount),
            }
            for i, (candidate, member, amount) in enumerate(observations)
        ),
    }
    types = {
        "candidate": {"id": "string"},
        "required": {"id": "string"},
        "observation": {
            "id": "string",
            "candidate_id": "string",
            "required_id": "string",
            "amount": "decimal",
        },
    }
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    r.id,
                    data[r.source.row_source_id],
                    ("id",),
                    types[r.source.row_source_id],
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for r in program.relations
            ),
            operations=tuple(
                ExecutableOperation(o.id, o.spec, o.output_relation)
                for o in program.operations
            ),
        )
    )
    assert result.issue is None
    output = program.result_projection.relation_outputs[0]
    rows = result.relation(output.relation_id).rows
    if return_truth:
        truth = program.result_projection.relation_outputs[1]
        assert {
            row[output.entity_key.components[0].field_id]: row[truth.field_id]
            for row in rows
        } == {candidate: candidate in expected for candidate in ("a", "b", "c")}
    else:
        assert {
            row[output.entity_key.components[0].field_id] for row in rows
        } == expected


def test_coverage_program_has_verifiable_relation_contracts():
    from fervis.lookup.plan_execution.verification.contracts import _relation_contracts
    from fervis.lookup.plan_execution.verification.execution_proof import (
        ExecutionProofContext,
    )
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceCatalog

    verified = coverage_query()
    program = compile_verified_source_strategy(verified).answer_program
    contracts = _relation_contracts(
        program,
        catalog=None,
        row_sources=RowSourceCatalog(verified.request.source_catalog.sources),
        proof_context=ExecutionProofContext.empty(),
    )
    output = program.result_projection.relation_outputs[0]
    assert contracts[output.relation_id].grain_keys
    assert (
        output.entity_key.components[0].field_id in contracts[output.relation_id].fields
    )
