"""Same-scope API-read source candidates derived from memory proof."""

from ._shared import (
    Any,
    DraftEndpointParamBinding,
    RelationCatalog,
    RowSource,
    build_row_source_catalog,
    canonical_param_value,
    dataclass,
    json,
    row_sources_for_read_id,
)
from .evidence import _read_field_payloads
from fervis.lookup.relation_catalog.parameter_values import (
    parse_catalog_parameter_value,
)


@dataclass(frozen=True)
class _SameScopeReadScope:
    memory_relation_id: str
    read_id: str
    param_bindings: tuple[DraftEndpointParamBinding, ...]


def _same_scope_api_candidate_payloads(
    memory_inputs: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
) -> list[dict[str, Any]]:
    row_sources = build_row_source_catalog(relation_catalog)
    items: list[tuple[_SameScopeReadScope, RowSource]] = []
    for scope in _same_scope_read_scopes(
        memory_inputs,
        relation_catalog=relation_catalog,
    ):
        for source in row_sources_for_read_id(scope.read_id, row_sources=row_sources):
            if not source.fields:
                continue
            if not _row_source_scope_compatible(source, scope=scope):
                continue
            items.append((scope, source))

    candidate_counts: dict[tuple[str, str], int] = {}
    for scope, _source in items:
        count_key = (scope.memory_relation_id, scope.read_id)
        candidate_counts[count_key] = candidate_counts.get(count_key, 0) + 1

    output: list[dict[str, Any]] = []
    grouped: dict[
        tuple[str, str, str], tuple[list[_SameScopeReadScope], RowSource]
    ] = {}
    for scope, source in items:
        group_key = (scope.memory_relation_id, scope.read_id, source.id)
        scopes, _source = grouped.setdefault(group_key, ([], source))
        scopes.append(scope)
    seen: set[str] = set()
    for (memory_relation_id, read_id, _source_id), (scopes, source) in grouped.items():
        scope = scopes[0]
        candidate_id = _same_scope_candidate_id(
            scope,
            source=source,
            ambiguous=candidate_counts[(memory_relation_id, read_id)] > len(scopes),
        )
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        invocation_payloads = [
            {
                "bound_params": [
                    _prior_scope_bound_param_payload(binding)
                    for binding in item.param_bindings
                ]
            }
            for item in scopes
        ]
        invocation_payload = (
            {"bound_params": invocation_payloads[0]["bound_params"]}
            if len(invocation_payloads) == 1
            else {"source_invocations": invocation_payloads}
        )
        output.append(
            {
                "source_candidate_id": candidate_id,
                "kind": "same_scope_api_read",
                "memory_relation_id": scope.memory_relation_id,
                "read_id": scope.read_id,
                "row_source_id": source.id,
                "row_path_id": source.row_path_id,
                "cardinality": source.row_cardinality.value,
                "result_grains": [
                    {
                        "grain_id": source.row_path_id or "root",
                        "row_path_id": source.row_path_id or "root",
                        "row_source_id": source.id,
                        "cardinality": source.row_cardinality.value,
                        "evidence_items": _read_field_payloads((source,)),
                    }
                ],
                **invocation_payload,
                "fields": _read_field_payloads((source,)),
            }
        )
    return output


def _prior_scope_bound_param_payload(
    binding: DraftEndpointParamBinding,
) -> dict[str, Any]:
    output = {
        "param_id": binding.param_id,
        "value": binding.compiler_value,
        "source": "prior_scope",
    }
    if binding.proof_refs:
        output["proof_refs"] = list(binding.proof_refs)
    return output


def _row_source_scope_compatible(
    source: RowSource,
    *,
    scope: _SameScopeReadScope,
) -> bool:
    bindings_by_param_id = {
        binding.param_id: binding.compiler_value for binding in scope.param_bindings
    }
    for param in source.params:
        if param.default is None:
            continue
        value = bindings_by_param_id.get(param.id)
        if value is None:
            return False
        if canonical_param_value(value) != canonical_param_value(param.default):
            return False
    return True


def _same_scope_candidate_id(
    scope: _SameScopeReadScope,
    *,
    source: RowSource,
    ambiguous: bool,
) -> str:
    if not ambiguous:
        return f"{scope.memory_relation_id}.{scope.read_id}.same_scope"
    row_path_id = source.row_path_id or "root"
    return f"{scope.memory_relation_id}.{scope.read_id}.{row_path_id}.same_scope"


def _same_scope_read_scopes(
    memory_inputs: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
) -> tuple[_SameScopeReadScope, ...]:
    output: list[_SameScopeReadScope] = []
    row_sources = build_row_source_catalog(relation_catalog)
    for relation in memory_inputs.get("memoryRelations", ()) or ():
        if not isinstance(relation, dict):
            continue
        relation_id = str(relation.get("id") or "")
        if not relation_id:
            continue
        completeness = relation.get("completeness")
        if not isinstance(completeness, dict):
            continue
        proof_read_names = _proof_read_names(completeness.get("proofRefs"))
        for scope_payload in _scope_fingerprint_payloads(
            completeness.get("scopeFingerprint")
        ):
            endpoint_args = scope_payload.get("endpointArgs")
            endpoint_arg_proof_refs = scope_payload.get("endpointArgProofRefs")
            row_filters = scope_payload.get("rowFilters")
            if not isinstance(endpoint_args, dict):
                continue
            if not isinstance(endpoint_arg_proof_refs, dict):
                continue
            if row_filters:
                continue
            for read in relation_catalog.reads:
                if not _scope_mentions_read(
                    read_id=read.id,
                    endpoint_name=read.endpoint_name,
                    endpoint_args=endpoint_args,
                    proof_read_names=proof_read_names,
                ):
                    continue
                bindings = _scope_param_bindings(
                    read.id,
                    endpoint_args=endpoint_args,
                    endpoint_arg_proof_refs=endpoint_arg_proof_refs,
                    row_sources=row_sources,
                )
                output.append(
                    _SameScopeReadScope(
                        memory_relation_id=relation_id,
                        read_id=read.id,
                        param_bindings=bindings,
                    )
                )
    return tuple(output)


def _scope_mentions_read(
    *,
    read_id: str,
    endpoint_name: str,
    endpoint_args: dict[str, Any],
    proof_read_names: frozenset[str],
) -> bool:
    if read_id in proof_read_names or endpoint_name in proof_read_names:
        return True
    prefix = f"{read_id}."
    endpoint_prefix = f"{endpoint_name}."
    return any(
        str(param_ref).startswith(prefix) or str(param_ref).startswith(endpoint_prefix)
        for param_ref in endpoint_args
    )


def _scope_param_bindings(
    read_id: str,
    *,
    endpoint_args: dict[str, Any],
    endpoint_arg_proof_refs: dict[str, Any],
    row_sources: Any,
) -> tuple[DraftEndpointParamBinding, ...]:
    params_by_ref: dict[str, Any] = {}
    for source in row_sources_for_read_id(read_id, row_sources=row_sources):
        for param in source.params:
            params_by_ref.setdefault(param.param_ref, param)
    output: list[DraftEndpointParamBinding] = []
    seen: set[str] = set()
    for param_ref, value in endpoint_args.items():
        param = params_by_ref.get(str(param_ref))
        if param is None or param.id in seen:
            continue
        seen.add(param.id)
        output.append(
            DraftEndpointParamBinding(
                param_id=param.id,
                value=parse_catalog_parameter_value(
                    value,
                    type_name=param.type.value,
                    choices=param.choices,
                ),
                proof_refs=_endpoint_arg_proof_refs(
                    endpoint_arg_proof_refs.get(str(param_ref))
                ),
            )
        )
    return tuple(output)


def _endpoint_arg_proof_refs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(ref) for ref in value if str(ref))


def _scope_fingerprint_payloads(value: Any) -> tuple[dict[str, Any], ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    parsed = _json_object(text)
    if parsed is not None:
        return (parsed,)
    return tuple(
        payload
        for part in text.split("|")
        for payload in (_json_object(part),)
        if payload is not None
    )


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _proof_read_names(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    prefix = "read:"
    return frozenset(
        text[len(prefix) :]
        for item in value
        for text in (str(item),)
        if text.startswith(prefix) and text[len(prefix) :]
    )
