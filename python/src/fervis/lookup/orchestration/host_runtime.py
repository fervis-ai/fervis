"""Current host catalog and authorized data access for answer execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.host_api.context import HostApiContext
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.host_api.contracts.read import ReadInvocation
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)


def host_relation_catalog(host_api_context: HostApiContext) -> RelationCatalog:
    return relation_catalog_from_endpoint_contracts(
        host_api_context.describe_sources()
    )


@dataclass(frozen=True)
class HostRelationDataAccess:
    host_api_context: HostApiContext
    authority: ReadAuthority

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        contract = self.host_api_context.endpoint_contract(endpoint_name)
        if contract is None:
            raise ValueError(f"Unknown endpoint contract: {endpoint_name}")
        path_params: dict[str, Any] = {}
        query_params: dict[str, Any] = {}
        params = {
            f"{endpoint_name}.{param.source}.{param.name}": param
            for param in (*contract.path_params, *contract.query_params)
        }
        for param_ref, value in args.items():
            param = params.get(str(param_ref))
            if param is None:
                raise ValueError(f"Unknown endpoint parameter: {param_ref}")
            if param.source == "path":
                path_params[param.name] = value
            elif param.source == "query":
                query_params[param.name] = value
            else:
                raise ValueError(
                    f"Unsupported endpoint parameter source: {param_ref}"
                )
        return self.host_api_context.execute_read(
            authority=self.authority,
            invocation=ReadInvocation(
                endpoint_name=endpoint_name,
                path_params=path_params,
                query_params=query_params,
                page_policy={
                    "mode": "all_pages" if contract.paginated else "single_page"
                },
            ),
        ).to_public_dict()
