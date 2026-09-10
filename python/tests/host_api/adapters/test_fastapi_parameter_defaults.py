from enum import Enum, IntEnum
from datetime import date
from fastapi.testclient import TestClient

from fastapi import FastAPI, Query
from fastapi.routing import APIRoute

from fervis.host_api.adapters.fastapi.schema_introspection import (
    fastapi_route_parameters,
)
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.host_api.contracts import EndpointContract


class Scope(str, Enum):
    ALL = "all"
    SOME = "some"


class Status(IntEnum):
    ACTIVE = 1
    HIDDEN = 2


def test_literal_defaults_reach_logical_read_contracts_without_running_factories():
    app = FastAPI()

    def forbidden_factory():
        raise AssertionError("Catalog discovery must not run a default factory")

    @app.get("/entries")
    def entries(
        required: str,
        active: bool = Query(True),
        hidden: bool = Query(False),
        offset: int = Query(0),
        term: str = Query(""),
        scope: Scope = Query(Scope.ALL),
        status: Status = Query(Status.ACTIVE),
        generated: str = Query(default_factory=forbidden_factory),
    ):
        return []

    route = next(route for route in app.routes if isinstance(route, APIRoute))
    params = fastapi_route_parameters(route, source="query")
    assert {param.name: param.default for param in params} == {
        "required": None,
        "active": True,
        "hidden": False,
        "offset": 0,
        "term": "",
        "scope": "all",
        "status": "1",
        "generated": None,
    }
    assert not next(
        param for param in params if param.name == "generated"
    ).default_is_known
    assert next(param for param in params if param.name == "required").required
    endpoint = EndpointContract(
        endpoint_name="entries",
        url_name="entries",
        method="GET",
        path_template="/entries",
        docstring="",
        view_class="",
        query_params=params,
        resource_names=("entries",),
    )
    [read] = relation_catalog_from_endpoint_contracts((endpoint,)).reads
    assert {param.name: param.default for param in read.params} == {
        param.name: param.default for param in params
    }


def test_collection_and_temporal_defaults_match_native_request_behavior():
    app = FastAPI()

    @app.get("/entries")
    def entries(
        states: list[str] = Query(["active"]), since: date = Query(date(2026, 1, 1))
    ):
        return {"states": states, "since": since}

    route = next(route for route in app.routes if isinstance(route, APIRoute))
    params = fastapi_route_parameters(route, source="query")
    assert {param.name: param.default for param in params} == TestClient(app).get(
        "/entries"
    ).json()


def test_unknown_dynamic_default_is_not_proven_to_preserve_population():
    from dataclasses import replace
    from fervis.lookup.source_binding.population_interpretation import (
        population_interpretation_targets,
    )
    from fervis.lookup.source_binding.parameter_coverage import (
        uncovered_parameter_scopes,
    )
    from fervis.lookup.source_binding.parser import (
        compile_source_realization,
        compile_source_binding_plan,
    )
    from tests.lookup.source_binding.test_population_interpretation import _request

    request = _request()
    source = request.source_catalog.sources[0]
    source = replace(
        source,
        params=(
            replace(
                source.params[0], required=False, default=None, default_is_known=False
            ),
        ),
    )
    request = replace(
        request, source_catalog=replace(request.source_catalog, sources=(source,))
    )
    assert population_interpretation_targets(request)
    branch = request.strategy.branches[0].branch_id
    set_ref = request.index.subject_obligation.subject_set_ref.token
    realized = compile_source_realization(
        {
            "set_bindings": {
                str(set_ref): [
                    {
                        "branch_id": branch,
                        "mapping_basis": "Rows",
                        "rows_ref": source.id,
                    }
                ]
            },
            "fact_bindings": {},
            "association_bindings": {},
        },
        request=request,
    )
    plan = compile_source_binding_plan(
        {
            "resolved_input_applications": {branch: []},
            "choice_requirement_applications": {branch: {}},
            "finite_choice_applications": {branch: {}},
        },
        realization=realized,
    )
    assert uncovered_parameter_scopes(plan, request=realized.request)
