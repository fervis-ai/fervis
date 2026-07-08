"""Row-predicate source-binding candidate projection."""

from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector

from ._shared import Any, RelationCatalog
from .candidate_tree import map_source_candidate_tree
from .field_scope import SourceBindingFieldScope


def with_row_predicates(
    payload: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
    field_scope: SourceBindingFieldScope | None = None,
) -> dict[str, Any]:
    resolved_field_scope = field_scope or SourceBindingFieldScope.unscoped()
    return map_source_candidate_tree(
        payload,
        lambda candidate, _context: candidate_with_row_predicates(
            candidate,
            relation_catalog=relation_catalog,
            field_scope=resolved_field_scope,
        ),
        top_level_keys=("memory_source_candidates", "utility_source_candidates"),
    )


def candidate_with_row_predicates(
    candidate: dict[str, Any],
    *,
    relation_catalog: RelationCatalog,
    field_scope: SourceBindingFieldScope | None = None,
) -> dict[str, Any]:
    if candidate.get("kind") not in {"new_api_read", "same_scope_api_read"}:
        return candidate
    output = dict(candidate)
    read_id = str(candidate.get("read_id") or "")
    if not read_id:
        return output
    source_candidate_id = str(candidate.get("source_candidate_id") or "")
    resolved_field_scope = field_scope or SourceBindingFieldScope.unscoped()
    predicates = ApiReadResponseShapeProjector(
        relation_catalog.read(read_id)
    ).row_predicates(
        row_path_ids=candidate_row_predicate_path_ids(candidate),
        source_candidate_id=source_candidate_id,
        field_refs=resolved_field_scope.field_refs_for_candidate(source_candidate_id),
    )
    if predicates:
        output["row_predicates"] = predicates
    else:
        output.pop("row_predicates", None)
    return output


def candidate_row_path_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            row_path_id
            for grain in candidate.get("result_grains") or ()
            if isinstance(grain, dict)
            for row_path_id in (str(grain.get("row_path_id") or ""),)
            if row_path_id
        )
    )


def candidate_row_predicate_path_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            row_path_id
            for grain in candidate.get("result_grains") or ()
            if isinstance(grain, dict)
            and str(grain.get("cardinality") or "").lower() == "many"
            for row_path_id in (str(grain.get("row_path_id") or ""),)
            if row_path_id
        )
    )
