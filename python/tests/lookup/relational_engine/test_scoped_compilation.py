from decimal import Decimal
from dataclasses import replace

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
    Comparison,
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
from fervis.lookup.expression_operators import ExpressionBinaryOperator as B
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
from fervis.lookup.source_binding.occurrences import occurrence_scope
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


def employee_query(
    quantifier=None, *, manager_minimum=False, return_truth=False, branches=1
):
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT, "employees compared with their managers"
    )
    comparison = Comparison(
        "comparison", B.GT, "manager_salary", "employee_salary", origin
    )
    expressions = (comparison,)
    qualification = "comparison"
    if quantifier is not None:
        from fervis.lookup.question_contract.model import Quantify

        expressions += (
            Quantify(
                "quantified",
                quantifier,
                "manager",
                ("management",),
                "comparison",
                origin,
            ),
        )
        qualification = "quantified"
    inputs, denotations = {}, {}
    if manager_minimum:
        from fervis.lookup.question_contract.model import (
            InputTerm,
            InputDenotation,
            InputDenotationKind,
            BooleanComposition,
            BooleanCompositionOperator,
        )

        term = InputTerm("minimum", origin, "150", DecimalType(UnitlessMeasure()))
        inputs = {term.id: term}
        denotations = {
            term.id: InputDenotation(
                "minimum_denotation",
                term.id,
                "Salary threshold",
                "Literal number",
                None,
                InputDenotationKind.NON_IDENTITY_SCALAR,
            )
        }
        expressions += (
            Comparison(
                "minimum_manager_salary", B.GT, "manager_salary", term.id, origin
            ),
            BooleanComposition(
                "all_conditions",
                BooleanCompositionOperator.AND,
                (qualification, "minimum_manager_salary"),
                origin,
            ),
        )
        qualification = "all_conditions"
    fact = RequestedFact(
        "fact_1",
        origin,
        (SetTerm("employee", origin), SetTerm("manager", origin)),
        (AssociationTerm("management", "employee", "manager", origin),),
        (
            FactTerm("employee_id", "employee", IdentifierType("employee"), origin),
            FactTerm(
                "employee_salary", "employee", DecimalType(UnitlessMeasure()), origin
            ),
            FactTerm(
                "manager_salary", "manager", DecimalType(UnitlessMeasure()), origin
            ),
        ),
        expressions,
        Subject("employee", InstanceInterpretation.RESOURCE_POPULATION),
        None if return_truth else qualification,
        (),
        (RequestedOutput("employee", "employee_id", origin),)
        + ((RequestedOutput("truth", "quantified", origin),) if return_truth else ()),
        (),
        AllResults(),
        (),
    )
    index = analyze_requested_fact(fact, inputs=inputs, input_denotations=denotations)
    roles = (
        FieldBindingRole.IDENTITY,
        FieldBindingRole.OUTPUT,
        FieldBindingRole.PREDICATE,
    )
    source = RowSource(
        id="employees",
        kind=RowSourceKind.API_READ,
        read_id="list_employees",
        label="Employee",
        fields=(
            RowSourceField(
                "id",
                "field.id",
                "Employee identifier",
                RowSourceValueType.STRING,
                roles,
            ),
            RowSourceField(
                "manager_id",
                "field.manager_id",
                "Manager identifier",
                RowSourceValueType.STRING,
                roles,
            ),
            RowSourceField(
                "salary", "field.salary", "Salary", RowSourceValueType.DECIMAL, roles
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                "pk", "employee", (RowSourceKeyComponent("id", "id"),), primary=True
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                "manager",
                "employee",
                "pk",
                (RowSourceEntityReferenceComponent("id", "manager_id"),),
            ),
        ),
    )
    if manager_minimum:
        from fervis.lookup.relation_catalog.row_sources.model import RowSourceParam
        from fervis.host_api.contracts.population import ParameterPopulation

        source = replace(
            source,
            params=(
                RowSourceParam(
                    "min_salary",
                    "employees.min_salary",
                    "Minimum salary",
                    RowSourceValueType.DECIMAL,
                    population=ParameterPopulation(field_path="field.salary", comparison_operator="gt"),
                ),
            ),
        )
    links = row_source_relation_evidence((source,))
    assert len(links) == 1
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"), (source,), links
    )
    branch = SourceStrategyBranch(
        "branch",
        (source.id,),
        (links[0].evidence_ref,),
        tuple(c.clause_ref for c in index.qualification.clauses),
    )
    strategy = CandidateSourceStrategy(fact.id, (branch,))
    canonical = ()
    if manager_minimum:
        from fervis.lookup.grounding.semantic import CanonicalInputValue
        from fervis.lookup.answer_program.values import FactValue, LiteralType

        canonical = (
            CanonicalInputValue(
                "minimum_value",
                "minimum",
                tuple(use.use_ref for use in index.input_use_sites),
                FactValue.literal(
                    id="minimum_value",
                    literal_type=LiteralType.NUMBER,
                    value="150",
                    known_input_id="minimum",
                    proof_refs=("question_input:minimum",),
                ),
                ("question_input:minimum",),
            ),
        )
    request = SemanticSourceBindingRequest(index, strategy, catalog, canonical)
    refs = index.fact_local_ref_by_local_id
    identity = source.identity_evidence[0]
    sets = {
        refs[name].token: (
            SetRealization(
                "branch",
                "This role uses an employee row.",
                source.id,
                identity.identity_ref,
                identity.field_refs,
                (source.id, identity.identity_ref),
            ),
        )
        for name in ("employee", "manager")
    }
    facts = {
        refs[name].token: (
            FactRealization(
                "branch",
                "Salary on this role.",
                source.id,
                FactRealizationKind.RETURNED_FIELD,
                None,
                ("field.salary",),
                (source.id, "field.salary"),
            ),
        )
        for name in ("employee_salary", "manager_salary")
    }
    facts[refs["employee_id"].token] = (
        FactRealization(
            "branch",
            "Employee identity.",
            source.id,
            FactRealizationKind.ENTITY_KEY,
            identity.identity_ref,
            identity.field_refs,
            (source.id, identity.identity_ref),
        ),
    )
    associations = {
        refs["management"].token: (
            AssociationRealization(
                "branch",
                "Employee manager foreign key.",
                AssociationRealizationKind.DECLARED_RELATION,
                (source.id, source.id),
                links[0].evidence_ref,
                (source.id, links[0].evidence_ref),
                reference_from_set_ref=refs["employee"].token,
            ),
        )
    }
    booleans = {
        r.requirement_ref: (
            BooleanRequirementRealization(
                "branch",
                (
                    SourceMechanic(
                        "Evaluate the declared predicate.",
                        source.id,
                        (),
                        (source.id, r.requirement_ref),
                        SourceMechanicKind.RETURNED_ROW_PREDICATE,
                    ),
                ),
            ),
        )
        for r in index.boolean_requirements
    }
    applications = ()
    if manager_minimum:
        from fervis.lookup.source_binding.model import (
            InvocationValueApplication,
            InvocationTargetApplication,
        )
        from fervis.lookup.answer_program.values import ValueProjectionKind

        requirement = next(
            item
            for item in index.boolean_requirements
            if item.atom_ref.value_ref == refs["minimum_manager_salary"].token
        )
        applications = (
            InvocationValueApplication(
                "manager_minimum",
                "branch",
                source.id,
                "minimum_value",
                requirement.requirement_ref,
                (
                    InvocationTargetApplication(
                        "Minimum manager salary.",
                        "employees.min_salary",
                        "minimum_value",
                        ValueProjectionKind.WHOLE_VALUE,
                        None,
                    ),
                ),
            ),
        )
        booleans[requirement.requirement_ref] = (
            BooleanRequirementRealization(
                "branch",
                (
                    SourceMechanic(
                        "Filter manager rows.",
                        source.id,
                        ("manager_minimum",),
                        (source.id, requirement.requirement_ref),
                        SourceMechanicKind.INVOCATION_PREDICATE,
                    ),
                ),
            ),
        )
    plan = SourceBindingPlan(
        strategy,
        sets,
        facts,
        associations,
        applications,
        booleans,
        SubjectObligationBinding(
            refs["employee"].token, (SubjectObligationRealization("branch", ()),)
        ),
    )
    if branches > 1:
        branch_ids = tuple(f"branch_{i}" for i in range(branches))
        strategy = replace(
            strategy,
            branches=tuple(replace(branch, branch_id=bid) for bid in branch_ids),
        )

        def duplicated(bindings):
            return {
                ref: tuple(
                    replace(value, branch_id=bid)
                    for bid in branch_ids
                    for value in values
                )
                for ref, values in bindings.items()
            }

        plan = replace(
            plan,
            strategy=strategy,
            set_bindings=duplicated(plan.set_bindings),
            fact_bindings=duplicated(plan.fact_bindings),
            association_bindings=duplicated(plan.association_bindings),
            boolean_bindings=duplicated(plan.boolean_bindings),
            subject_binding=replace(
                plan.subject_binding,
                branch_realizations=tuple(
                    replace(value, branch_id=bid)
                    for bid in branch_ids
                    for value in plan.subject_binding.branch_realizations
                ),
            ),
        )
        request = replace(request, strategy=strategy)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    return verified


def execute_employee_query(
    quantifier=None, *, manager_minimum=False, rows=None, return_truth=False, branches=1
):
    verified = employee_query(
        quantifier,
        manager_minimum=manager_minimum,
        return_truth=return_truth,
        branches=branches,
    )
    program = compile_verified_source_strategy(verified).answer_program
    rows = (
        rows
        if rows is not None
        else (
            {"id": "e", "manager_id": "m", "salary": Decimal(100)},
            {"id": "m", "manager_id": "c", "salary": Decimal(50)},
            {"id": "c", "manager_id": None, "salary": Decimal(200)},
        )
    )
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    r.id,
                    tuple(
                        row
                        for row in rows
                        if not r.source.param_bindings
                        or row["salary"] is not None
                        and row["salary"] > 150
                    ),
                    ("id",),
                    {"id": "string", "manager_id": "string", "salary": "decimal"},
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
    if return_truth:
        truth = program.result_projection.relation_outputs[1]
        return {
            row[output.entity_key.components[0].field_id]: row[truth.field_id]
            for row in result.relation(output.relation_id).rows
        }
    return {
        row[output.entity_key.components[0].field_id]
        for row in result.relation(output.relation_id).rows
    }


def test_one_producer_has_two_independent_logical_occurrences():
    verified = employee_query()
    scope = occurrence_scope(verified.request, verified.binding_plan, "branch")
    assert len(scope.occurrences) == 2
    assert len({o.source_ref for o in scope.occurrences}) == 1
    assert scope.links[0].left_occurrence != scope.links[0].right_occurrence


def test_self_join_compares_the_employee_with_a_different_manager_row():
    assert execute_employee_query() == {"m"}


@pytest.mark.parametrize(
    ("quantifier", "expected"),
    [("exists", {"m"}), ("not_exists", {"e", "c"}), ("forall", {"m", "c"})],
)
def test_correlated_quantifiers_preserve_scope_and_vacuous_truth(quantifier, expected):
    from fervis.lookup.question_contract.model import Quantifier

    assert execute_employee_query(Quantifier(quantifier)) == expected


def test_parameter_predicate_applies_only_to_its_logical_read_occurrence():
    program = compile_verified_source_strategy(
        employee_query(manager_minimum=True)
    ).answer_program
    employee, manager = program.relations
    assert employee.source.param_bindings == ()
    assert [binding.param_id for binding in manager.source.param_bindings] == [
        "min_salary"
    ]
    assert execute_employee_query(manager_minimum=True) == {"m"}


@pytest.mark.parametrize(
    ("quantifier", "expected"),
    [("exists", set()), ("not_exists", {"c"}), ("forall", {"c"})],
)
def test_unknown_correlated_comparisons_remain_unknown_under_quantification(
    quantifier, expected
):
    from fervis.lookup.question_contract.model import Quantifier

    rows = (
        {"id": "e", "manager_id": "m", "salary": Decimal(100)},
        {"id": "m", "manager_id": "c", "salary": None},
        {"id": "c", "manager_id": None, "salary": Decimal(200)},
    )
    assert execute_employee_query(Quantifier(quantifier), rows=rows) == expected


def test_quantified_boolean_can_be_returned_without_filtering_candidates():
    from fervis.lookup.question_contract.model import Quantifier

    assert execute_employee_query(Quantifier.EXISTS, return_truth=True) == {
        "e": False,
        "m": True,
        "c": False,
    }


@pytest.mark.parametrize(
    ("quantifier", "expected"),
    [("exists", {"m"}), ("not_exists", {"e", "c"}), ("forall", {"m", "c"})],
)
def test_union_branches_preserve_correlated_scope(quantifier, expected):
    from fervis.lookup.question_contract.model import Quantifier

    assert execute_employee_query(Quantifier(quantifier), branches=2) == expected


def test_union_branches_preserve_returned_relational_values():
    from fervis.lookup.question_contract.model import Quantifier

    assert execute_employee_query(Quantifier.EXISTS, return_truth=True, branches=2) == {
        "e": False,
        "m": True,
        "c": False,
    }
