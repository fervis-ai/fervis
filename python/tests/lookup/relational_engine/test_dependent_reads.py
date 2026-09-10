"""Dependent API calls consume correlated rows, with complete traversal evidence."""

from dataclasses import replace
import pytest
from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
from fervis.lookup.relation_catalog.model import (
    EndpointRead,
    RelationCatalog,
    CatalogField,
    CatalogParam,
    RowPath,
    RowCardinality,
    CandidateKey,
    CandidateKeyComponent,
)
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.answer_program.relations import (
    Relation,
    RelationSource,
    RelationField,
    SourceKind,
    FieldBindingRole,
    EndpointParamBinding,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.compilation import compile_answer_program
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory


def _read(read_id, *, params=(), value_type="integer", paired=False):
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        params=params,
        row_paths=(RowPath("root", "", RowCardinality.MANY),),
        fields=(
            CatalogField(f"{read_id}.id", value_type, path="id", row_path_id="root"),
            *(
                (
                    CatalogField(
                        f"{read_id}.zone", "string", path="zone", row_path_id="root"
                    ),
                )
                if paired
                else ()
            ),
        ),
        candidate_keys=(
            CandidateKey(
                "primary",
                read_id,
                (CandidateKeyComponent("id", f"{read_id}.id"),),
                primary=True,
            ),
        ),
    )


def _program(
    *,
    parent_type="integer",
    bad_field=False,
    paired=False,
    via_projection=False,
    child_incomplete=False,
):
    contract, _, _, compiled, _, _ = _compile_memory_count(
        (), meaning="instrument count"
    )
    catalog = RelationCatalog(
        reads=(
            _read("facilities", value_type=parent_type, paired=paired),
            _read(
                "instruments",
                params=(
                    CatalogParam(
                        "facility_id", "facility_id", "path", "integer", required=True
                    ),
                    *(
                        (CatalogParam("zone", "zone", "path", "string", required=True),)
                        if paired
                        else ()
                    ),
                ),
            ),
        )
    )
    if child_incomplete:
        from fervis.lookup.relation_catalog.model import (
            PaginationMetadata,
            CompletenessPolicy,
        )

        catalog = replace(
            catalog,
            reads=(
                catalog.reads[0],
                replace(
                    catalog.reads[1],
                    pagination=PaginationMetadata(
                        completeness_policy=CompletenessPolicy.INCOMPLETE
                    ),
                ),
            ),
        )
    sources = {
        source.read_id: source
        for source in build_api_row_source_catalog(catalog).sources
    }
    parent = sources["facilities"]
    child = sources["instruments"]
    parent_relation = Relation(
        "facilities",
        RelationSource(
            SourceKind.API_READ, read_id="facilities", row_source_id=parent.id
        ),
        (RelationField(parent.fields[0].id, (FieldBindingRole.IDENTITY,)),),
    )
    if paired:
        zone = next(
            field for field in parent.fields if field.field_ref == "facilities.zone"
        )
        parent_relation = replace(
            parent_relation,
            fields=(
                *parent_relation.fields,
                RelationField(zone.id, (FieldBindingRole.OUTPUT,)),
            ),
        )
    original = compiled.answer_program.relations[0]
    child_relation = replace(
        original,
        source=RelationSource(
            SourceKind.API_READ,
            read_id="instruments",
            row_source_id=child.id,
            argument_relation_id=parent_relation.id,
            proof_refs=("read:instruments",),
            param_bindings=(
                EndpointParamBinding(
                    child.params[0].id,
                    FieldRef("missing" if bad_field else parent.fields[0].id),
                ),
            ),
        ),
    )
    if paired:
        param = next(param for param in child.params if param.name == "zone")
        child_relation = replace(
            child_relation,
            source=replace(
                child_relation.source,
                param_bindings=(
                    *child_relation.source.param_bindings,
                    EndpointParamBinding(param.id, FieldRef(zone.id)),
                ),
            ),
        )
    operations = compiled.answer_program.operations
    if via_projection:
        from fervis.lookup.answer_program.operations import (
            Operation,
            ProjectSpec,
            NamedExpression,
        )

        projections = tuple(
            NamedExpression("argument_" + str(index), binding.value_expr)
            for index, binding in enumerate(child_relation.source.param_bindings)
        )
        operation = Operation(
            "argument_projection",
            ProjectSpec(parent_relation.id, projections),
            output_relation="argument_rows",
        )
        child_relation = replace(
            child_relation,
            source=replace(
                child_relation.source,
                argument_relation_id="argument_rows",
                param_bindings=tuple(
                    replace(
                        binding, value_expr=FieldRef(projections[index].output_field)
                    )
                    for index, binding in enumerate(
                        child_relation.source.param_bindings
                    )
                ),
            ),
        )
        operations = (operation, *operations)
    guarantees = tuple(
        replace(item, subject=replace(item.subject, proof_refs=("read:instruments",)))
        for item in compiled.answer_program.relation_guarantees
    )
    program = replace(
        compiled.answer_program,
        relations=(parent_relation, child_relation),
        relation_guarantees=guarantees,
        operations=operations,
    )
    program, bindings = compile_answer_program(
        program,
        question_contract=contract,
        catalog=catalog,
        bindings=compiled.initial_bindings,
    )
    return program, bindings, catalog


@pytest.mark.parametrize(
    "parent_ids,expected", [((1, 2, 3), 3), ((), 0), ((1, 1, 2), 3)]
)
def test_dependent_read_counts_all_children_and_calls_each_argument_once(
    parent_ids, expected
):
    program, bindings, catalog = _program()
    calls = []

    class Port:
        def read(self, *, endpoint_name, args, **kwargs):
            calls.append((endpoint_name, dict(args)))
            rows = (
                [{"id": value} for value in parent_ids]
                if endpoint_name == "facilities"
                else {1: [{"id": 11}, {"id": 12}], 2: [{"id": 21}], 3: []}[
                    args["facility_id"]
                ]
            )
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {
        "fact_1.output_1": expected
    }
    assert calls == [
        ("facilities", {}),
        *[
            ("instruments", {"facility_id": value})
            for value in dict.fromkeys(parent_ids)
        ],
    ]
    assert "read:facilities" in result.proof_refs
    if not parent_ids:
        assert "read:instruments" not in result.proof_refs


@pytest.mark.parametrize("kwargs", [{"bad_field": True}, {"parent_type": "string"}])
def test_dependent_argument_contract_is_rejected_before_any_transport(kwargs):
    from fervis.lookup.plan_execution.errors import VerificationError

    # No transport exists at compilation: bad fields and incompatible types
    # are now rejected from the declared metadata alone.
    with pytest.raises(VerificationError,match='dependent argument'):
        _program(**kwargs)


def test_source_dependency_cycle_is_rejected_before_any_transport():
    from fervis.lookup.plan_execution.errors import VerificationError

    program, bindings, catalog = _program()
    child = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    cyclic = replace(child, source=replace(child.source, argument_relation_id=child.id))
    program = replace(
        program,
        relations=tuple(
            cyclic if relation.id == child.id else relation
            for relation in program.relations
        ),
    )
    with pytest.raises(VerificationError, match="cyclic"):
        invoke_answer_program(
            program=program,
            bindings=bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(data_access_port=None, memory=LookupMemory()),
        )


@pytest.mark.parametrize("via_projection", [False, True])
def test_dependent_request_preserves_correlated_argument_tuples(via_projection):
    program, bindings, catalog = _program(paired=True, via_projection=via_projection)
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, dict(args)))
            rows = (
                [{"id": 1, "zone": "north"}, {"id": 2, "zone": "south"}]
                if endpoint_name == "facilities"
                else {(1, "north"): [{"id": 11}], (2, "south"): [{"id": 22}]}[
                    (args["facility_id"], args["zone"])
                ]
            )
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 2}
    assert calls == [
        ("facilities", {}),
        ("instruments", {"facility_id": 1, "zone": "north"}),
        ("instruments", {"facility_id": 2, "zone": "south"}),
    ]
    driver = "argument_rows" if via_projection else "facilities"
    child = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    assert any(
        edge.source == "relation:" + driver and edge.target == "relation:" + child.id
        for edge in result.proof_graph.edges
    )


def test_incomplete_parent_cannot_prove_a_complete_child_traversal():
    from fervis.lookup.outcomes.errors import ExecutionIssueKind

    program, bindings, catalog = _program()
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": [{"id": 1}],
                "truncated": True,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue.kind is ExecutionIssueKind.INCOMPLETE_EVIDENCE
    assert result.fact_result is None
    assert calls == ["facilities"]


@pytest.mark.parametrize("static_second", [False, True])
def test_parameter_assignment_is_unique_before_any_read(static_second):
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType
    from fervis.lookup.plan_execution.errors import VerificationError

    program, bindings, catalog = _program()
    child = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    original = child.source.param_bindings[0]
    expression = (
        ConstantRef(
            "other",
            "v1",
            FactValue.literal(id="other", literal_type=LiteralType.NUMBER, value="2"),
        )
        if static_second
        else original.value_expr
    )
    child = replace(
        child,
        source=replace(
            child.source,
            param_bindings=(original, replace(original, value_expr=expression)),
        ),
    )
    program = replace(
        program,
        relations=tuple(
            child if relation.id == child.id else relation
            for relation in program.relations
        ),
    )
    with pytest.raises(VerificationError, match="repeats a request parameter"):
        invoke_answer_program(
            program=program,
            bindings=bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(data_access_port=None, memory=LookupMemory()),
        )


def test_direct_source_loading_classifies_incomplete_evidence():
    from fervis.lookup.plan_execution.operation_engine import execute_operations
    from fervis.lookup.plan_execution.operation_runtime import RelationEngineInput
    from fervis.lookup.outcomes.errors import (
        IncompleteEvidenceError,
        ExecutionIssueKind,
    )

    def loader(relation_id, require_relation):
        raise IncompleteEvidenceError(
            relation_id=relation_id, proof_refs=("read:facilities",)
        )

    result = execute_operations(
        RelationEngineInput(source_relation_ids=("facilities",), relation_loader=loader)
    )
    assert result.issue.kind is ExecutionIssueKind.INCOMPLETE_EVIDENCE
    assert result.issue.proof_refs == ("read:facilities",)


@pytest.mark.parametrize("compile_first", [False, True])
def test_computed_request_values_use_the_existing_projection_engine(compile_first):
    from fervis.lookup.answer_program.expressions import (
        BinaryExpression,
        ExpressionBinaryOperator,
    )
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.plan_execution.errors import VerificationError

    program, bindings, catalog = _program()
    child = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    original = child.source.param_bindings[0]
    expression = BinaryExpression(
        ExpressionBinaryOperator.ADD,
        original.value_expr,
        ConstantRef(
            "offset",
            "v1",
            FactValue.literal(id="offset", literal_type=LiteralType.NUMBER, value="1"),
        ),
    )
    child = replace(
        child,
        source=replace(
            child.source, param_bindings=(replace(original, value_expr=expression),)
        ),
    )
    program = replace(
        program,
        relations=tuple(
            child if relation.id == child.id else relation
            for relation in program.relations
        ),
    )
    if not compile_first:
        with pytest.raises(VerificationError, match="preceding projection"):
            invoke_answer_program(
                program=program,
                bindings=bindings,
                environment=ExecutionEnvironment(catalog=catalog),
                ports=RuntimePorts(data_access_port=None, memory=LookupMemory()),
            )
        return
    program, bindings = compile_answer_program(
        program,
        question_contract=QuestionContract(
            program.inputs, program.fact_template, program.input_denotations
        ),
        catalog=catalog,
        bindings=bindings,
    )
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, dict(args)))
            rows = (
                [{"id": 1}, {"id": 2}]
                if endpoint_name == "facilities"
                else {2: [{"id": 22}], 3: [{"id": 33}]}[args["facility_id"]]
            )
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 2}
    assert calls == [
        ("facilities", {}),
        ("instruments", {"facility_id": 2}),
        ("instruments", {"facility_id": 3}),
    ]


def test_empty_parent_proves_empty_traversal_without_requiring_child_read_coverage():
    program, bindings, catalog = _program(child_incomplete=True)
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            assert endpoint_name == "facilities"
            return {"responseStatus": 200, "responseFormat": "json", "responseBody": []}

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 0}
    assert calls == ["facilities"]
    assert "read:instruments" not in result.proof_refs


def test_nested_reads_retain_parent_path_values_that_are_absent_from_json():
    from fervis.lookup.question_contract import QuestionContract

    program, bindings, catalog = _program()
    instrument_read = replace(catalog.read("instruments"), candidate_keys=())
    sample_read = _read(
        "samples",
        params=(
            CatalogParam(
                "facility_id", "facility_id", "path", "integer", required=True
            ),
            CatalogParam(
                "instrument_id", "instrument_id", "path", "integer", required=True
            ),
        ),
    )
    catalog = replace(
        catalog, reads=(catalog.read("facilities"), instrument_read, sample_read)
    )
    sources = {
        source.read_id: source
        for source in build_api_row_source_catalog(catalog).sources
    }
    instrument = sources["instruments"]
    sample = sources["samples"]
    context_field = instrument.request_argument_fields[0]
    assert context_field not in instrument.fields
    original = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    driver = replace(
        original,
        id="instrument_rows",
        fields=(
            RelationField(instrument.fields[0].id, (FieldBindingRole.OUTPUT,)),
            RelationField(context_field.id, (FieldBindingRole.REQUEST_ARGUMENT,)),
        ),
    )
    mappings = {
        "facility_id": context_field.id,
        "instrument_id": instrument.fields[0].id,
    }
    root = replace(
        original,
        source=RelationSource(
            SourceKind.API_READ,
            read_id="samples",
            row_source_id=sample.id,
            argument_relation_id=driver.id,
            proof_refs=("read:samples",),
            param_bindings=tuple(
                EndpointParamBinding(param.id, FieldRef(mappings[param.name]))
                for param in sample.params
            ),
        ),
    )
    guarantees = tuple(
        replace(item, subject=replace(item.subject, proof_refs=("read:samples",)))
        for item in program.relation_guarantees
    )
    program = replace(
        program,
        relations=(
            *[relation for relation in program.relations if relation.id != original.id],
            driver,
            root,
        ),
        relation_guarantees=guarantees,
    )
    program, bindings = compile_answer_program(
        program,
        question_contract=QuestionContract(
            program.inputs, program.fact_template, program.input_denotations
        ),
        catalog=catalog,
        bindings=bindings,
    )
    from fervis.lookup.plan_execution.errors import VerificationError

    invalid_driver = replace(
        driver,
        fields=(
            driver.fields[0],
            replace(driver.fields[1], roles=(FieldBindingRole.OUTPUT,)),
        ),
    )
    invalid = replace(
        program,
        relations=tuple(
            invalid_driver if relation.id == driver.id else relation
            for relation in program.relations
        ),
    )
    with pytest.raises(VerificationError, match="field role"):
        invoke_answer_program(
            program=invalid,
            bindings=bindings,
            environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(data_access_port=None, memory=LookupMemory()),
        )
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, dict(args)))
            if endpoint_name == "facilities":
                rows = [{"id": 1}, {"id": 2}]
            elif endpoint_name == "instruments":
                rows = [{"id": 7}]
                args[
                    "facility_id"
                ] = -999  # Connector-local mutation must not rewrite retained context.
            else:
                rows = {(1, 7): [{"id": 101}], (2, 7): [{"id": 102}]}[
                    (args["facility_id"], args["instrument_id"])
                ]
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {"fact_1.output_1": 2}
    assert calls == [
        ("facilities", {}),
        ("instruments", {"facility_id": 1}),
        ("instruments", {"facility_id": 2}),
        ("samples", {"facility_id": 1, "instrument_id": 7}),
        ("samples", {"facility_id": 2, "instrument_id": 7}),
    ]


def test_identity_access_uses_the_same_physical_graph_without_an_answer_template():
    from fervis.lookup.answer_program.model import RelationProgram
    from fervis.lookup.answer_program.api_reads import ApiReadSession
    from fervis.lookup.source_reads.access_execution import execute_access_program

    program, bindings, catalog = _program()
    graph = RelationProgram(
        parameters=program.parameters,
        relations=program.relations,
        operations=program.operations,
    )
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            rows = [{"id": 1}] if endpoint_name == "facilities" else [{"id": 11}]
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = execute_access_program(
        graph, catalog=catalog, read_session=ApiReadSession(Port()), bindings=bindings
    )
    assert result.engine_output.issue is None
    assert result.engine_output.relation(program.operations[-1].output_relation).rows
    assert calls == ["facilities", "instruments"]


@pytest.mark.parametrize("fixed_parent", [False, True])
def test_access_mapping_expands_requests_and_preserves_supplied_scope(fixed_parent):
    from fervis.lookup.source_reads.access_model import (
        ReadAccessCatalog,
        ReadDependency,
        AccessArgument,
    )
    from fervis.lookup.source_reads.access_compilation import expand_read_access
    from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType
    from fervis.lookup.question_contract import QuestionContract

    program, bindings, catalog = _program(paired=True)
    sources = {
        source.read_id: source
        for source in build_api_row_source_catalog(catalog).sources
    }
    parent = sources["facilities"]
    child = sources["instruments"]
    original = next(
        relation
        for relation in program.relations
        if relation.source.argument_relation_id
    )
    fixed = ()
    if fixed_parent:
        param = next(param for param in child.params if param.name == "facility_id")
        fixed = (
            EndpointParamBinding(
                param.id,
                ConstantRef(
                    "selected_facility",
                    "v1",
                    FactValue.literal(
                        id="selected_facility",
                        literal_type=LiteralType.NUMBER,
                        value="1",
                    ),
                ),
            ),
        )
    root = replace(
        original,
        source=replace(original.source, argument_relation_id="", param_bindings=fixed),
    )
    program = replace(program, relations=(root,))
    fields = {field.field_ref: field for field in parent.fields}
    access = ReadAccessCatalog(
        tuple(sources.values()),
        (
            ReadDependency(
                child.id,
                parent.id,
                (
                    AccessArgument("facility_id", fields["facilities.id"].field_ref),
                    AccessArgument("zone", fields["facilities.zone"].field_ref),
                ),
                "The complete parent rows supply each child request tuple.",
            ),
        ),
    )
    program = expand_read_access(program, access)
    program, bindings = compile_answer_program(
        program,
        question_contract=QuestionContract(
            program.inputs, program.fact_template, program.input_denotations
        ),
        catalog=catalog,
        bindings=bindings,
    )
    calls = []

    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, dict(args)))
            rows = (
                [{"id": 1, "zone": "north"}, {"id": 2, "zone": "south"}]
                if endpoint_name == "facilities"
                else {(1, "north"): [{"id": 11}], (2, "south"): [{"id": 22}]}[
                    (args["facility_id"], args["zone"])
                ]
            )
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": rows,
            }

    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(data_access_port=Port(), memory=LookupMemory()),
    )
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {
        "fact_1.output_1": 1 if fixed_parent else 2
    }
    assert calls == [
        ("facilities", {}),
        ("instruments", {"facility_id": 1, "zone": "north"}),
        *(
            []
            if fixed_parent
            else [("instruments", {"facility_id": 2, "zone": "south"})]
        ),
    ]


def test_recursive_access_emits_intermediate_argument_filters():
    from fervis.lookup.answer_program.model import RelationProgram
    from fervis.lookup.answer_program.api_reads import ApiReadSession
    from fervis.lookup.source_reads.access_model import ReadAccessCatalog, ReadDependency, AccessArgument
    from fervis.lookup.source_reads.access_compilation import expand_read_access
    from fervis.lookup.source_reads.access_execution import execute_access_program
    catalog = RelationCatalog(reads=(
        _read('facilities', paired=True),
        _read('instruments', params=(
            CatalogParam('facility_id', 'facility_id', 'path', 'integer', required=True),
            CatalogParam('zone', 'zone', 'query', 'string', default='north', semantics='response_shape'),
        )),
        _read('samples', params=(CatalogParam('instrument_id', 'instrument_id', 'path', 'integer', required=True),)),
    ))
    sources = build_api_row_source_catalog(catalog).sources
    facility, instrument, sample = sources
    access = ReadAccessCatalog(sources, (
        ReadDependency(instrument.id, facility.id, (
            AccessArgument('facility_id', 'facilities.id'), AccessArgument('zone', 'facilities.zone'),
        ), 'The same parent row supplies the argument tuple.'),
        ReadDependency(sample.id, instrument.id, (AccessArgument('instrument_id', 'instruments.id'),),
                       'Every sample belongs to a listed instrument.'),
    ))
    program = expand_read_access(RelationProgram(relations=(Relation(
        'samples', RelationSource(SourceKind.API_READ, read_id='samples', row_source_id=sample.id),
        (RelationField(sample.fields[0].id, (FieldBindingRole.OUTPUT,)),),
    ),)), access)
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, args))
            if endpoint_name == 'facilities':
                rows = [{'id': 1, 'zone': 'north'}, {'id': 2, 'zone': 'south'}]
            elif endpoint_name == 'instruments':
                assert args == {'zone': 'north', 'facility_id': 1}
                rows = [{'id': 10}]
            else:
                assert args == {'instrument_id': 10}
                rows = [{'id': 100}]
            return {'responseStatus': 200, 'responseBody': rows}
    result = execute_access_program(program, catalog=catalog, read_session=ApiReadSession(Port()))
    assert result.row_context.rows_for_relation('samples') == ({sample.fields[0].id: 100},)
    assert [name for name, _ in calls] == ['facilities', 'instruments', 'samples']


@pytest.mark.parametrize('inspected_child', [False, True])
def test_pipeline_retains_operational_parent_outside_semantic_selection(monkeypatch, inspected_child):
    from types import SimpleNamespace
    from fervis.lookup.orchestration import pipeline
    from fervis.lookup.relation_catalog.selection.model import CatalogSelectionResult
    program, bindings, catalog = _program()
    child = catalog.read('instruments')
    full = replace(catalog, reads=(catalog.read('facilities'),
        replace(child, fields=(), row_paths=(), candidate_keys=()) if inspected_child else child))
    selection = CatalogSelectionResult(replace(catalog, reads=(child,)), (), ('instruments',))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name, dict(args)))
            rows = [{'id': 1}, {'id': 2}] if endpoint_name == 'facilities' else [{'id': args['facility_id'] * 10}]
            return {'responseStatus': 200, 'responseBody': rows}
    port = Port()
    def execute(**kwargs):
        return invoke_answer_program(program=kwargs['program'], bindings=kwargs['bindings'],
            environment=kwargs['environment'], ports=RuntimePorts(data_access_port=port, memory=LookupMemory()))
    monkeypatch.setattr(pipeline, 'run_answer_program_execution', execute)
    state = SimpleNamespace(
        semantic_compilation=SimpleNamespace(catalog_selection=selection, canonical_values=(),
            compilation=SimpleNamespace(answer_program=program, initial_bindings=bindings)),
        full_catalog=full, semantic_turn_numbers={}, semantic_usage={}, memory=LookupMemory(),
        ports=SimpleNamespace(data_access_port=port, lineage_step_sink=None, lineage_required=False,
                              program_invocation_binding=None),
        request=SimpleNamespace(authority_ref='', runtime_values=None),
        conversation_resolution=None, compiled_conversation_resolution=None,
    )
    result = pipeline._run_semantic_execution_phase(state)
    assert result.issue is None
    assert result.fact_result.outcome.projected_rows[0].values == {'fact_1.output_1': 2}
    assert calls == [('facilities', {}), ('instruments', {'facility_id': 1}), ('instruments', {'facility_id': 2})]
