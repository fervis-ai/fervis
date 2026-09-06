"""Observed JSON structure is useful evidence without becoming key authority."""

import pytest

from fervis.host_api.contracts import EndpointContract
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.source_reads.representation import observe_read_representation


def test_route_without_response_schema_remains_in_catalog():
    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items/",
        docstring="List items",
        view_class="items",
        resource_names=("items",),
    )
    catalog = relation_catalog_from_endpoint_contracts((contract,))
    assert [read.id for read in catalog.reads] == ["items"]
    assert catalog.reads[0].row_paths == ()
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog

    assert build_api_row_source_catalog(catalog).sources == ()


def test_observation_records_open_structural_fields_without_identity_claims():
    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items/",
        docstring="List items",
        view_class="items",
        resource_names=("items",),
    )
    (read,) = relation_catalog_from_endpoint_contracts((contract,)).reads
    observed = observe_read_representation(
        read,
        {
            "responseStatus": 200,
            "responseFormat": "json",
            "responseBody": [
                {"id": "a", "active": True, "amount": 2},
                {"id": "b", "amount": 2.5},
            ],
        },
    )
    assert observed.read.row_paths[0].cardinality.value == "many"
    fields = {field.path: field for field in observed.read.fields}
    assert fields["active"].nullable is True
    assert fields["amount"].type == "number"
    assert all(field.choices == () for field in fields.values())
    assert observed.read.candidate_keys == ()
    assert observed.read.entity_references == ()
    assert observed.response_hash.startswith("sha256:")
    assert read.fields == ()


def test_inspection_only_reads_selected_directly_invokable_routes():
    from dataclasses import replace
    from fervis.lookup.relation_catalog import CatalogParam, ParamSource
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )

    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items/",
        docstring="List items",
        view_class="items",
        resource_names=("items",),
    )
    (read,) = relation_catalog_from_endpoint_contracts((contract,)).reads
    detail = replace(
        read,
        id="item",
        endpoint_name="item",
        params=(
            CatalogParam(
                ref="item.path.id",
                name="id",
                source=ParamSource.PATH,
                type="string",
                required=True,
            ),
        ),
    )
    unrelated = replace(read, id="unrelated", endpoint_name="unrelated")

    class Port:
        calls = []

        def read(self, *, endpoint_name, args):
            self.calls.append((endpoint_name, args))
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": [{"name": "A"}],
            }

    port = Port()
    seen = []
    result = inspect_selected_representations(
        RelationCatalog(reads=(read, detail, unrelated)),
        read_ids=("items", "item"),
        data_access_port=port,
        on_response=lambda read, response: seen.append(read.id),
    )
    assert port.calls == [("items", {})]
    assert seen == ["items"]
    assert result.read("items").row_paths[0].cardinality.value == "many"
    assert result.read("item").row_paths == ()
    assert result.read("unrelated").row_paths == ()


def test_json_root_property_cannot_collide_with_root_row_identifier():
    from fervis.lookup.relation_catalog import EndpointRead, parse_relation_catalog
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.source_reads.response import extract_row_source_rows

    body = [{"root": [{"x": 1}]}]
    observed = observe_read_representation(
        EndpointRead("items", "items", resource_names=("items",)),
        {"responseStatus": 200, "responseFormat": "json", "responseBody": body},
    )
    catalog = parse_relation_catalog(RelationCatalog(reads=(observed.read,)))
    sources = build_api_row_source_catalog(catalog).sources
    assert len({source.id for source in sources}) == len(sources) == 2
    child = next(source for source in sources if source.row_path == "root")
    assert extract_row_source_rows(body, row_source=child) == ({"x": 1},)
    assert (
        next(field for field in child.fields if field.label == "x").response_path == "x"
    )


def test_inspection_failure_is_not_an_unavailable_schema_claim():
    import pytest
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )
    from fervis.lookup.source_reads.response import EndpointResponseError

    read = EndpointRead("items", "items", resource_names=("items",))

    class Port:
        def read(self, **kwargs):
            return {
                "responseStatus": 503,
                "responseFormat": "json",
                "responseBody": {"error": "unavailable"},
            }

    seen = []
    with pytest.raises(EndpointResponseError, match="HTTP 503"):
        inspect_selected_representations(
            RelationCatalog(reads=(read,)),
            read_ids=("items",),
            data_access_port=Port(),
            on_response=lambda read, result: seen.append(result["responseStatus"]),
        )
    assert seen == [503]


def test_current_inspection_reestablishes_compatible_structure_without_pinning_rows():
    from dataclasses import replace
    import pytest
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.answer_program.model import AnswerProgram
    from fervis.lookup.answer_program.relations import (
        Relation,
        RelationSource,
        SourceKind,
        RelationField,
        FieldBindingRole,
    )
    from fervis.lookup.answer_program.compatibility import (
        build_program_compatibility,
        verify_program_compatibility,
    )
    from fervis.lookup.plan_execution.errors import VerificationError
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )

    raw = RelationCatalog(
        reads=(EndpointRead("items", "items", resource_names=("items",)),)
    )

    class Port:
        rows = [{"name": "A"}]

        def read(self, **kwargs):
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": self.rows,
            }

    port = Port()
    first = inspect_selected_representations(
        raw, read_ids=("items",), data_access_port=port
    )
    sources = build_api_row_source_catalog(first)
    (source,) = sources.sources
    program = AnswerProgram(
        relations=(
            Relation(
                "items",
                RelationSource(
                    SourceKind.API_READ, read_id="items", row_source_id=source.id
                ),
                fields=(RelationField("name", (FieldBindingRole.OUTPUT,)),),
            ),
        )
    )
    program = replace(
        program, compatibility=build_program_compatibility(program, row_sources=sources)
    )
    port.rows = [{"name": "B"}, {"name": "C"}]
    current = inspect_selected_representations(
        raw, read_ids=("items",), data_access_port=port
    )
    verify_program_compatibility(program, catalog=current, memory_relations=())
    port.rows = [{"renamed": "B"}]
    changed = inspect_selected_representations(
        raw, read_ids=("items",), data_access_port=port
    )
    with pytest.raises(VerificationError, match="incompatible_source_contract"):
        verify_program_compatibility(program, catalog=changed, memory_relations=())


def test_changed_observed_source_recompiles_the_current_question(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from fervis.lookup.orchestration import pipeline
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.answer_program.model import AnswerProgram
    from fervis.lookup.answer_program.relations import (
        Relation,
        RelationSource,
        SourceKind,
        RelationField,
        FieldBindingRole,
    )
    from fervis.lookup.answer_program.compatibility import build_program_compatibility
    from fervis.lookup.answer_program.values import BindingSet
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )

    raw = RelationCatalog(
        reads=(EndpointRead("items", "items", resource_names=("items",)),)
    )

    class Port:
        rows = [{"name": "A"}]

        def read(self, **kwargs):
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": self.rows,
            }

    port = Port()
    old = inspect_selected_representations(
        raw, read_ids=("items",), data_access_port=port
    )
    sources = build_api_row_source_catalog(old)
    program = AnswerProgram(
        relations=(
            Relation(
                "items",
                RelationSource(
                    SourceKind.API_READ,
                    read_id="items",
                    row_source_id=sources.sources[0].id,
                ),
                fields=(RelationField("name", (FieldBindingRole.OUTPUT,)),),
            ),
        )
    )
    program = replace(
        program, compatibility=build_program_compatibility(program, row_sources=sources)
    )
    state = SimpleNamespace(
        full_catalog=raw,
        memory=LookupMemory(),
        request=SimpleNamespace(run_id="current_run", question="What about now?"),
        ports=SimpleNamespace(data_access_port=port, lineage_step_sink=None),
    )
    monkeypatch.setattr(
        pipeline, "callable_frame_bindings", lambda *args, **kwargs: BindingSet()
    )
    seen = []

    def compile_current(current, *, catalog):
        seen.append(
            (
                current.request.question,
                tuple(field.path for field in catalog.read("items").fields),
            )
        )
        return "fresh compilation"

    monkeypatch.setattr(pipeline, "_run_semantic_compile_question", compile_current)
    port.rows = [{"renamed": "B"}]
    result = pipeline._run_continue_prior_request_program(
        state,
        SimpleNamespace(frame=SimpleNamespace(program=program)),
        grounded_values=(),
    )
    assert result == "fresh compilation"
    assert seen == [("What about now?", ("renamed",))]


def test_empty_json_array_retains_an_empty_many_row_source():
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.source_reads.response import extract_row_source_rows

    observed = observe_read_representation(
        EndpointRead("items", "items", resource_names=("items",)),
        {"responseStatus": 200, "responseFormat": "json", "responseBody": []},
    )
    (source,) = build_api_row_source_catalog(
        RelationCatalog(reads=(observed.read,))
    ).sources
    assert source.row_cardinality.value == "many"
    assert source.fields == ()
    assert extract_row_source_rows([], row_source=source) == ()


def test_inspection_respects_execution_read_scope_before_any_request():
    import pytest
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.orchestration.execution_sources import prepare_execution_catalog
    from fervis.lookup.answer_program.model import AnswerProgram
    from fervis.lookup.answer_program.relations import (
        Relation,
        RelationSource,
        SourceKind,
    )
    from fervis.lookup.plan_execution.errors import VerificationError

    class Port:
        def read(self, **kwargs):
            raise AssertionError("Out-of-scope route was requested")

    program = AnswerProgram(
        relations=(
            Relation("r", RelationSource(SourceKind.API_READ, read_id="private")),
        )
    )
    with pytest.raises(VerificationError, match="outside selected catalog"):
        prepare_execution_catalog(
            run_id="r",
            catalog=RelationCatalog(
                reads=(EndpointRead("private", "private", resource_names=("private",)),)
            ),
            program=program,
            data_access_port=Port(),
            lineage_step_sink=None,
            allowed_read_ids=frozenset({"public"}),
        )


def test_uninspected_positive_recall_is_retained_for_the_next_batch():
    from dataclasses import replace
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.orchestration.semantic_compilation import _bound_recall_selection
    from fervis.lookup.relation_catalog.selection.model import (
        CatalogSelectionResult,
        RequestedFactCatalogSelection,
    )
    from fervis.lookup.relation_catalog.selection.batches import (
        next_catalog_selection_batch,
    )
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )

    raw = RelationCatalog(
        reads=tuple(
            EndpointRead(name, name, resource_names=("items",)) for name in ("a", "b")
        )
    )

    class Port:
        calls = []

        def read(self, *, endpoint_name, args):
            self.calls.append(endpoint_name)
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": [{"name": endpoint_name}],
            }

    port = Port()
    current = inspect_selected_representations(
        raw, read_ids=("a",), data_access_port=port
    )
    selection = CatalogSelectionResult(
        RelationCatalog(reads=(current.read("a"),)),
        (RequestedFactCatalogSelection("fact_1", ("items",), (), ("a",), ("b",)),),
        ("a",),
    )
    bounded = _bound_recall_selection(selection, full_catalog=current, values=())
    next_batch = next_catalog_selection_batch(
        catalog_selection=bounded, full_catalog=current, max_reads_per_fact=1
    )
    assert next_batch is not None
    assert next_batch.selected_read_ids == ("b",)
    current = inspect_selected_representations(
        current, read_ids=next_batch.selected_read_ids, data_access_port=port
    )
    assert port.calls == ["a", "b"]
    assert [field.path for field in current.read("b").fields] == ["name"]


@pytest.mark.parametrize("first_retained", [False, True])
def test_source_preparation_inspects_a_later_useful_candidate(
    monkeypatch, first_retained
):
    from fervis.lookup.orchestration import semantic_compilation as compilation
    from fervis.lookup.relation_catalog import EndpointRead
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.relation_catalog.selection.model import (
        CatalogSelectionResult,
        RequestedFactCatalogSelection,
    )
    from fervis.lookup.read_eligibility.semantic import (
        ReadRequirementAssessment,
        SemanticReadDecision,
        SemanticReadEligibilityResult,
    )
    from fervis.lookup.question_contract import QuestionContractRequest
    from fervis.lookup.turn_prompts import HostPromptContext, build_turn_prompt_context
    from fervis.lookup.source_reads.representation import (
        inspect_selected_representations,
    )
    from tests.lookup.fact_compilation.test_compiler import _compile_memory_count

    _, _, _, _, _, count = _compile_memory_count(())

    class Port:
        calls = []

        def read(self, *, endpoint_name, args):
            self.calls.append(endpoint_name)
            return {
                "responseStatus": 200,
                "responseFormat": "json",
                "responseBody": [{"name": endpoint_name}],
            }

    port = Port()
    raw = RelationCatalog(
        reads=tuple(
            EndpointRead(name, name, resource_names=("items",)) for name in ("a", "b")
        )
    )
    current = inspect_selected_representations(
        raw, read_ids=("a",), data_access_port=port
    )
    selection = CatalogSelectionResult(
        RelationCatalog(reads=(current.read("a"),)),
        (RequestedFactCatalogSelection("fact_1", ("items",), (), ("a",), ("b",)),),
        ("a",),
    )
    initial_sources = build_api_row_source_catalog(selection.relation_catalog)

    def assessed(source, decision):
        return SemanticReadEligibilityResult(
            (
                ReadRequirementAssessment(
                    requested_fact_id="fact_1",
                    candidate_ref=source.read_id,
                    source_refs=(source.id,),
                    read_id=source.read_id,
                    relevant_field_refs=tuple(f.field_ref for f in source.fields),
                    assessment_basis="This fixture selects the later matching resource.",
                    decision=decision,
                ),
            ),
            (),
        )

    def later(eligibility_request, **kwargs):
        (source,) = eligibility_request.source_catalog.sources
        assert source.read_id == "b"
        assert any(field.label == "name" for field in source.fields)
        return assessed(source, SemanticReadDecision.RETAIN)

    monkeypatch.setattr(compilation, "_read_eligibility_turn", later)
    question = "How many items?"
    request = compilation.SemanticCompilationRequest(
        run_id="test",
        question=question,
        question_contract_request=QuestionContractRequest(
            current_question=question, conversation_context={}
        ),
        full_catalog=current,
        memory_relations=(),
        data_access_port=port,
        model_port=None,
        provider="openai",
        max_thinking_tokens=1,
        max_catalog_reads_per_fact=1,
        runtime_values=None,
        conversation_context={},
        host=HostPromptContext(),
    )
    result = compilation._prepare_source_candidates(
        initial_catalog_selection=selection,
        initial_answer_sources=initial_sources,
        initial_eligibility=assessed(
            initial_sources.sources[0],
            SemanticReadDecision.RETAIN
            if first_retained
            else SemanticReadDecision.DROP,
        ),
        canonical_values=(),
        indexes=(count.request.index,),
        context=build_turn_prompt_context(
            current_question=question, conversation_context={}
        ),
        request=request,
        on_turn=None,
    )
    assert port.calls == ["a", "b"]
    assert {source.read_id for source in result[3].sources} == (
        {"a", "b"} if first_retained else {"b"}
    )


def test_empty_paginated_observation_preserves_declared_results_structure():
    from fervis.host_api.contracts import PaginationContract, PaginationKind
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog

    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items/",
        docstring="List items",
        view_class="items",
        resource_names=("items",),
        pagination=PaginationContract(
            kind=PaginationKind.OFFSET,
            position_query_param="offset",
            page_size_query_param="limit",
            results_path="items",
            total_path="total",
            page_size=10,
            max_page_size=100,
        ),
    )
    (read,) = relation_catalog_from_endpoint_contracts((contract,)).reads
    assert (
        next(path for path in read.row_paths if path.path == "data").cardinality.value
        == "many"
    )
    observed = observe_read_representation(
        read,
        {
            "responseStatus": 200,
            "responseFormat": "json",
            "responseBody": {"data": [], "pagination": {"count": 0, "has_more": False}},
        },
    )
    sources = build_api_row_source_catalog(
        RelationCatalog(reads=(observed.read,))
    ).sources
    assert (
        next(
            source for source in sources if source.row_path == "data"
        ).row_cardinality.value
        == "many"
    )
    assert all(
        not field.path.startswith("pagination") for field in observed.read.fields
    )


def test_saved_count_executes_when_paginated_data_becomes_empty():
    from fervis.host_api.contracts import PaginationContract, PaginationKind
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.available_sources import (
        AvailableSourceCatalog,
        SourceContractSnapshot,
    )
    from fervis.lookup.source_binding.model import (
        SemanticSourceBindingRequest,
        CandidateSourceStrategy,
        SourceStrategyBranch,
    )
    from fervis.lookup.source_binding.parser import (
        compile_source_realization,
        compile_source_binding_plan,
    )
    from fervis.lookup.source_binding.membership import parse_source_membership
    from fervis.lookup.source_binding.verification import (
        verify_source_strategy,
        VerifiedSourceStrategy,
    )
    from fervis.lookup.fact_compilation import compile_verified_source_strategy
    from fervis.lookup.answer_program.invocation import (
        invoke_answer_program,
        RuntimePorts,
    )
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from tests.lookup.fact_compilation.test_compiler import _compile_memory_count

    _, _, _, _, _, base = _compile_memory_count((), meaning="item count")
    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items/",
        docstring="List items",
        view_class="items",
        resource_names=("items",),
        pagination=PaginationContract(
            kind=PaginationKind.OFFSET,
            position_query_param="offset",
            page_size_query_param="limit",
            results_path="items",
            total_path="total",
            page_size=10,
            max_page_size=100,
        ),
    )
    (read,) = relation_catalog_from_endpoint_contracts((contract,)).reads

    def response(rows):
        return {
            "responseStatus": 200,
            "responseFormat": "json",
            "responseBody": {
                "data": rows,
                "pagination": {"count": len(rows), "has_more": False},
            },
        }

    initial = observe_read_representation(read, response([{"name": "A"}])).read
    source = next(
        source
        for source in build_api_row_source_catalog(
            RelationCatalog(reads=(initial,))
        ).sources
        if source.row_path == "data"
    )
    index = base.request.index
    branch = SourceStrategyBranch(
        "branch", (source.id,), (), (index.qualification.clauses[0].clause_ref,)
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(index.requested_fact_id, (branch,)),
        source_catalog=AvailableSourceCatalog(
            SourceContractSnapshot.from_content("{}"), (source,), ()
        ),
        canonical_values=(),
    )
    realization = compile_source_realization(
        {
            "set_bindings": {
                index.subject_obligation.subject_set_ref.token: [
                    {
                        "branch_id": "branch",
                        "mapping_basis": "The API returns item rows.",
                        "rows_ref": source.id,
                    }
                ]
            },
            "fact_bindings": {},
            "association_bindings": {},
        },
        request=request,
    )
    membership = parse_source_membership({"branch": {}}, realization=realization)
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {"branch": []},
            "finite_choice_applications": {"branch": {}},
            "choice_requirement_applications": {"branch": {}},
        },
        membership=membership,
    )
    verified = verify_source_strategy(plan, request=membership.realization.request)
    assert isinstance(verified, VerifiedSourceStrategy)
    compiled = compile_verified_source_strategy(verified)

    class Port:
        rows = []

        def read(self, **kwargs):
            return response(self.rows)

    from fervis.lookup.memory.projection import LookupMemory

    port = Port()
    for rows in ([], [{"name": "B"}, {"name": "C"}]):
        port.rows = rows
        current = observe_read_representation(read, response(rows)).read
        executed = invoke_answer_program(
            program=compiled.answer_program,
            bindings=compiled.initial_bindings,
            environment=ExecutionEnvironment(catalog=RelationCatalog(reads=(current,))),
            ports=RuntimePorts(data_access_port=port, memory=LookupMemory()),
        )
        assert executed.issue is None
        assert executed.fact_result.outcome.projected_rows[0].values == {
            "fact_1.output_1": len(rows)
        }


@pytest.mark.parametrize(
    "cardinality,body", [("many", [{"name": "A"}]), ("one", {"name": "A"})]
)
def test_declared_cardinality_without_fields_remains_authoritative(cardinality, body):
    contract = EndpointContract(
        endpoint_name="items",
        url_name="items",
        method="GET",
        path_template="/items",
        docstring="",
        view_class="items",
        response_cardinality=cardinality,
        resource_names=("items",),
    )
    (read,) = relation_catalog_from_endpoint_contracts((contract,)).reads
    assert read.row_paths[0].cardinality.value == cardinality
    result = {"responseStatus": 200, "responseFormat": "json", "responseBody": body}
    assert (
        observe_read_representation(read, result).read.row_paths[0].cardinality.value
        == cardinality
    )
    result["responseBody"] = {} if cardinality == "many" else []
    with pytest.raises(ValueError, match="contradict declared response cardinality"):
        observe_read_representation(read, result)


@pytest.mark.parametrize("transport_exception", [False, True])
def test_continuation_inspection_failure_retains_terminal_usage_and_lineage(
    monkeypatch, transport_exception
):
    from types import SimpleNamespace
    from fervis.lookup.answer_program import AnswerProgram
    from fervis.lookup.answer_program.relations import (
        Relation,
        RelationSource,
        SourceKind,
    )
    from fervis.lookup.answer_program.values import BindingSet
    from fervis.lookup.lineage.steps import LineageRuntimeStepSink
    from fervis.lookup.orchestration import pipeline
    from fervis.lookup.relation_catalog import EndpointRead, CatalogEndpointMetadata

    class Recorder:
        reads = ()
        errors = []

        def record_step_with_source_context(
            self, step, catalog_endpoints, source_reads, artifacts
        ):
            self.reads = source_reads
            return step

        def record_runtime_error_result(self, error):
            self.errors.append(error)
            return error

    class Port:
        def read(self, **kwargs):
            if transport_exception:
                raise ConnectionError("connection lost")
            return {"responseStatus": 503, "responseFormat": "json", "responseBody": {}}

    recorder = Recorder()
    state = SimpleNamespace(
        request=SimpleNamespace(run_id="continuation_failure"),
        conversation_turn=SimpleNamespace(usage={"costUsd": 0.01}),
        semantic_usage={"costUsd": 0.02},
        full_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    "items",
                    "items",
                    resource_names=("items",),
                    catalog_endpoint=CatalogEndpointMetadata(
                        "items",
                        "items",
                        "fastapi",
                        "fastapi_app",
                        ("sample",),
                        "GET",
                        "/items",
                        "sample.items",
                    ),
                ),
            )
        ),
        ports=SimpleNamespace(
            data_access_port=Port(),
            lineage_required=True,
            lineage_step_sink=LineageRuntimeStepSink(
                run_id="continuation_failure", recorder=recorder
            ),
        ),
    )
    program = AnswerProgram(
        relations=(
            Relation(
                "items",
                RelationSource(
                    SourceKind.API_READ, read_id="items", row_source_id="items"
                ),
            ),
        )
    )
    monkeypatch.setattr(
        pipeline, "callable_frame_bindings", lambda *args, **kwargs: BindingSet()
    )
    result = pipeline._run_continue_prior_request_program(
        state,
        SimpleNamespace(frame=SimpleNamespace(program=program)),
        grounded_values=(),
    )
    assert result.status == "FAILED"
    assert result.error == "program_execution_failed"
    assert result.usage["costUsd"] == pytest.approx(0.03)
    assert len(recorder.reads) == 1
    assert len(recorder.errors) == 1
    assert recorder.errors[0].result.run_id == "continuation_failure"
