"""Observed joins need typed field mappings, not inferred foreign-key metadata."""

from dataclasses import replace
import pytest
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    RequestedOutput,
    FactLocalRef,
    Quantifier,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
from fervis.lookup.relation_catalog import RelationCatalog, CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.available_sources import snapshot_source_catalog, SourceFieldBinding
from fervis.lookup.orchestration.logical_planning import (
    realize_and_compile_logical_plan,
)
from fervis.lookup.fact_compilation.model import FactCompilationResult
from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize("quantifier", [None, Quantifier.EXISTS, Quantifier.NOT_EXISTS, Quantifier.FORALL])
@pytest.mark.parametrize("split_sources", [False, True])
@pytest.mark.parametrize("duplicate_managers", [False, True])
@pytest.mark.parametrize("duplicate_subjects", [False, True])
def test_observed_self_join_preserves_the_counted_unit(
    duplicate_managers, duplicate_subjects, split_sources, quantifier
):
    _run_observed_join(
        duplicate_managers, duplicate_subjects, split_sources=split_sources, quantifier=quantifier
    )


def _run_observed_join(
    duplicate_managers=False, duplicate_subjects=False, mutate=None, split_sources=False, quantifier=None
):
    original = employee_query(quantifier).request.index.requested_fact
    fact = replace(
        original,
        expressions=(
            *original.expressions,
            Aggregate(
                "n", AggregateFunction.COUNT, "employee", None, False, original.origin
            ),
        ),
        outputs=(RequestedOutput("count", "n", original.origin),),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    logical = ParsedSemanticQuestionContract(
        "Count qualifying staff records.", QuestionContract((), (fact,)), (index,)
    )
    read = replace(
        _read("staff", value_type="string"),
        candidate_keys=(),
        fields=(
            *_read("staff", value_type="string").fields,
            CatalogField(
                "staff.manager_id", "string", path="manager_id", row_path_id="root"
            ),
            CatalogField("staff.salary", "number", path="salary", row_path_id="root"),
        ),
    )
    manager_read = replace(
        _read("managers", value_type="string"),
        candidate_keys=(),
        fields=(
            *_read("managers", value_type="string").fields,
            CatalogField(
                "managers.salary", "number", path="salary", row_path_id="root"
            ),
        ),
    )
    catalog = RelationCatalog(reads=(read, manager_read) if split_sources else (read,))
    sources = build_api_row_source_catalog(catalog).sources
    source = next(item for item in sources if item.read_id == "staff")
    manager_source = (
        next(item for item in sources if item.read_id == "managers")
        if split_sources
        else source
    )
    by_role = {"employee": source, "manager": manager_source}
    available = snapshot_source_catalog(sources)
    assert available.relation_evidence == ()
    fields = {
        field.path: SourceFieldBinding(source.id, field).ref for field in source.fields
    }
    manager_fields = {
        field.path: SourceFieldBinding(manager_source.id, field).ref
        for field in manager_source.fields
    }
    turns = []

    def turn(purpose, prompt, parse):
        turns.append(type(prompt).__name__)
        branch = prompt.request.strategy.branches[0].branch_id
        if len(turns) == 1:
            original_parse = parse

            def parse(payload):
                if mutate is not None:
                    mutate(payload, fields)
                from jsonschema import validate
                from fervis.lookup.source_binding.schema import (
                    build_semantic_source_realization_schema,
                )

                if mutate is None:
                    validate(
                        payload,
                        build_semantic_source_realization_schema(prompt.request),
                    )
                return original_parse(payload)

            return parse(
                {
                    "set_bindings": {
                        f"fact_1:set:{name}": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "One independent staff-row role.",
                                "rows_ref": by_role[name].id,
                                "record_fields": [],
                            }
                        ]
                        for name in ("employee", "manager")
                    },
                    "fact_bindings": {
                        f"fact_1:fact:{name}": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Salary on this logical row.",
                                "field_ref": (
                                    manager_fields
                                    if name == "manager_salary"
                                    else fields
                                )["salary"],
                            }
                        ]
                        for name in ("employee_salary", "manager_salary")
                    },
                    "association_bindings": {
                        "fact_1:association:management": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "Match the observed manager identifier to the other staff row identifier.",
                                "realization_ref": None,
                                "reference_from_set_ref": None,
                                "field_pairs": [
                                    {
                                        "from_field_ref": fields["manager_id"],
                                        "to_field_ref": manager_fields["id"],
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        if len(turns) == 2:
            return parse(
                {
                    "populations": {
                        f"fact_1:set:{name}": [
                            {
                                "branch_id": branch,
                                "logical_set_meaning": index.term_by_ref[
                                    FactLocalRef.from_token(f"fact_1:set:{name}")
                                ].origin.meaning,
                                "mapping_basis": "Both roles range over the staff rows; the association determines matches.",
                                "population": {"kind": "exact_population"},
                            }
                        ]
                        for name in ("employee", "manager")
                    }
                }
            )
        return parse(
            {
                "resolved_input_applications": {branch: []},
                "finite_choice_applications": {branch: {}},
                "choice_requirement_applications": {branch: {}},
            }
        )

    result = realize_and_compile_logical_plan(
        logical, sources_by_fact={"fact_1": available}, canonical_values=(), turn=turn
    )
    assert isinstance(result, FactCompilationResult)
    from fervis.lookup.contract_codec import (
        canonical_answer_program_json,
        decode_answer_program,
    )
    from fervis.lookup.answer_program.invocation import (
        invoke_answer_program,
        RuntimePorts,
    )
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory

    program = decode_answer_program(
        canonical_answer_program_json(result.answer_program)
    )
    rows = [
        {"id": "e", "manager_id": "m", "salary": 100},
        {"id": "m", "manager_id": "c", "salary": 50},
        {"id": "c", "manager_id": None, "salary": 200},
    ]
    if duplicate_managers:
        rows.append(dict(rows[2]))
    if duplicate_subjects:
        rows.append(dict(rows[1]))
    reads = []

    class Port:
        def read(self, *, endpoint_name, args):
            reads.append((endpoint_name, args))
            return {"responseStatus": 200, "responseBody": rows}

    executed = invoke_answer_program(
        program=program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(Port(), LookupMemory()),
    )
    assert executed.issue is None
    expected = (
        2 + int(duplicate_managers)
        if quantifier is Quantifier.NOT_EXISTS
        else 2 + int(duplicate_managers) + int(duplicate_subjects)
        if quantifier is Quantifier.FORALL
        else 1 + int(duplicate_subjects)
    )
    assert next(
        iter(executed.fact_result.outcome.projected_rows[0].values.values())
    ) == expected
    assert reads == (
        [("staff", {}), ("managers", {})] if split_sources else [("staff", {})]
    )


@pytest.mark.parametrize(
    "mutation", ["empty", "duplicate", "unknown", "incompatible", "orientation"]
)
def test_observed_association_rejects_invalid_bindings(mutation):
    def mutate(payload, fields):
        association = payload["association_bindings"]["fact_1:association:management"][
            0
        ]
        pairs = association["field_pairs"]
        if mutation == "empty":
            pairs.clear()
        elif mutation == "duplicate":
            pairs.append(dict(pairs[0]))
        elif mutation == "unknown":
            pairs[0]["from_field_ref"] = "invented_field"
        elif mutation == "incompatible":
            pairs[0]["to_field_ref"] = fields["salary"]
        else:
            association["reference_from_set_ref"] = "fact_1:set:employee"

    with pytest.raises(ValueError):
        _run_observed_join(mutate=mutate)


@pytest.mark.parametrize(
    "mutation",
    ["empty", "unknown", "incompatible", "orientation", "evidence", "source"],
)
def test_verified_observed_association_rechecks_tampered_plan(monkeypatch, mutation):
    import fervis.lookup.source_binding as bindings
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        VerifiedSourceStrategy,
        SourceStrategyVerificationFailure,
    )

    seen = []

    def capture(plan, *, request):
        result = verify_source_strategy(plan, request=request)
        assert isinstance(result, VerifiedSourceStrategy)
        seen.append(result)
        return result

    monkeypatch.setattr(bindings, "verify_source_strategy", capture)
    _run_observed_join()
    verified = seen[0]
    ref = "fact_1:association:management"
    (association,) = verified.binding_plan.association_bindings[ref]
    (pair,) = association.field_pairs
    if mutation == "empty":
        changed = replace(association, field_pairs=())
    elif mutation == "unknown":
        changed = replace(association, field_pairs=(("unknown", pair[1]),))
    elif mutation == "incompatible":
        salary = next(
            binding.ref
            for binding in verified.request.source_catalog.field_bindings
            if binding.field.path == "salary"
        )
        changed = replace(association, field_pairs=((pair[0], salary),))
    elif mutation == "orientation":
        changed = replace(association, reference_from_set_ref="fact_1:set:manager")
    elif mutation == "evidence":
        changed = replace(association, relation_evidence_ref="invented_authority")
    else:
        changed = replace(association, source_refs=("foreign", "foreign"))
    plan = replace(verified.binding_plan, association_bindings={ref: (changed,)})
    result = verify_source_strategy(plan, request=verified.request)
    assert isinstance(result, SourceStrategyVerificationFailure)


def test_observed_association_rejects_a_known_field_from_the_wrong_endpoint():
    def mutate(payload, fields):
        association = payload["association_bindings"]["fact_1:association:management"][
            0
        ]
        association["field_pairs"][0]["to_field_ref"] = fields["id"]

    with pytest.raises(ValueError, match="assigned endpoints"):
        _run_observed_join(mutate=mutate, split_sources=True)


def test_missing_predicate_field_is_rejected_at_source_realization():
    def omit_property(payload, fields):
        payload['fact_bindings']['fact_1:fact:manager_salary'] = []

    with pytest.raises(ValueError, match='requires returned fields.*manager_salary'):
        _run_observed_join(mutate=omit_property, quantifier=Quantifier.EXISTS)
