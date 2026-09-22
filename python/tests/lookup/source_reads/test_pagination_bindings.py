"""Discovered traversal survives typed compilation and canonical replay."""

from dataclasses import replace

import pytest

from fervis.host_api.adapters.get_execution import (
    prepare_get_endpoint,
    execute_prepared_get,
)
from fervis.host_api.context import HostApiContext
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.response_page import ResponsePage
from fervis.lookup.orchestration.host_runtime import HostRelationDataAccess
from fervis.lookup.orchestration.logical_planning import (
    realize_and_compile_logical_plan,
)
from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.available_sources import snapshot_source_catalog
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from tests.host_api.test_bound_pagination import contract, policy
from fervis.host_api.contracts import PaginationContract
from tests.lookup.fact_compilation.test_compiler import _compile_memory_count


def candidate():
    host = replace(contract(), resource_names=("records",))
    catalog = relation_catalog_from_endpoint_contracts((host,))
    pagination = PaginationContract.from_public_dict(policy()["pagination_contract"])
    discovered = replace(
        catalog,
        reads=tuple(
            replace(read, pagination_binding=pagination) for read in catalog.reads
        ),
    )
    sources = build_api_row_source_catalog(discovered).sources
    source = next(source for source in sources if source.row_path == "records")
    logical_contract, _, _, _, _, verified = _compile_memory_count(())
    logical = ParsedSemanticQuestionContract(
        "Count all records.", logical_contract, (verified.request.index,)
    )
    calls = []

    def turn(purpose, prompt, parse):
        calls.append(prompt)
        branch = prompt.request.strategy.branches[0].branch_id
        if len(calls) == 1:
            return parse(
                {
                    "set_bindings": {
                        "fact_1:set:s1": [
                            {
                                "branch_id": branch,
                                "mapping_basis": "All returned record occurrences.",
                                "rows_ref": source.id,
                                "record_fields": [],
                            }
                        ]
                    },
                    "fact_bindings": {},
                    "association_bindings": {},
                }
            )
        return parse(
            {
                "populations": {
                    "fact_1:set:s1": [
                        {
                            "branch_id": branch,
                            "logical_set_meaning": "event count",
                            "mapping_basis": "Every record occurrence is in this set.",
                            "population": {"kind": "exact_population"},
                        }
                    ]
                }
            }
        )

    result = realize_and_compile_logical_plan(
        logical,
        sources_by_fact={"fact_1": snapshot_source_catalog((source,))},
        canonical_values=(),
        turn=turn,
    )
    program = decode_answer_program(
        canonical_answer_program_json(result.answer_program)
    )
    return host, catalog, program, result.initial_bindings


@pytest.mark.parametrize(
    "failure", [None, "foreign_binding", "changed_type", "later_page"]
)
def test_persisted_plan_traverses_all_pages_and_rechecks_authority(failure):
    host, catalog, program, bindings = candidate()
    if failure == "foreign_binding":
        (relation,) = program.relations
        program = replace(
            program,
            relations=(
                replace(
                    relation,
                    source=replace(
                        relation.source,
                        pagination_binding=replace(
                            relation.source.pagination_binding,
                            position_query_param="foreign",
                        ),
                    ),
                ),
            ),
        )
    elif failure == "changed_type":
        catalog = replace(
            catalog,
            reads=tuple(
                replace(
                    read,
                    params=tuple(
                        replace(param, type="string")
                        if param.name == "segment"
                        else param
                        for param in read.params
                    ),
                )
                for read in catalog.reads
            ),
        )
    reads = []

    class Adapter:
        def describe_sources(self):
            return (host,)

        def execute_read(self, *, authority, invocation):
            prepared = prepare_get_endpoint(
                host,
                path_params=dict(invocation.path_params),
                query_params=dict(invocation.query_params),
                page_policy=invocation.page_policy,
            )

            def page(url, args):
                reads.append(dict(args))
                if failure == "later_page" and args["segment"] == 2:
                    return ResponsePage(500, {"error": "page unavailable"})
                count = 1 if args["segment"] == 3 else 2
                return ResponsePage(
                    200, {"records": [{"id": 1} for _ in range(count)], "matched": 5}
                )

            return execute_prepared_get(
                contract=host,
                prepared=prepared,
                page_policy=invocation.page_policy,
                get_page=page,
            )

    port = HostRelationDataAccess(
        HostApiContext(Adapter()),
        ReadAuthority("test", ReadContextRef("anonymous", "")),
    )
    if failure in {"foreign_binding", "changed_type"}:
        with pytest.raises(ValueError):
            invoke_answer_program(
                program=program,
                bindings=bindings,
                environment=ExecutionEnvironment(catalog=catalog),
                ports=RuntimePorts(port, LookupMemory()),
            )
        assert reads == []
        return
    if failure == "later_page":
        from fervis.lookup.plan_execution.errors import RelationEngineError

        with pytest.raises(RelationEngineError, match="HTTP 500"):
            invoke_answer_program(
                program=program,
                bindings=bindings,
                environment=ExecutionEnvironment(catalog=catalog),
                ports=RuntimePorts(port, LookupMemory()),
            )
        assert len(reads) == 2
        return
    result = invoke_answer_program(
        program=program,
        bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),
        ports=RuntimePorts(port, LookupMemory()),
    )
    assert result.issue is None
    assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == 5
    assert reads == [{"segment": i, "width": 2} for i in (1, 2, 3)]
