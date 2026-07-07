"""Terminal outcome checks for fact-plan verification."""

from ._shared import (
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    CatalogSelectionResult,
    FactValue,
    MissingCatalogChoiceInput,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
    QuestionContract,
    RelationCatalog,
    RelationRows,
    RequestedFact,
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    VerificationError,
    build_row_source_catalog,
    catalog_selection_evidence_ref,
    clarifiable_required_inputs,
    grounded_required_input_ids,
    read_evidence_ref,
    read_field_evidence_ref,
)
from .blocked import (
    _policy_blocked_evidence_refs,
    _required_reviewed_read_ids,
)
from .question_contract import _verify_question_contract


def _verify_plan_clarification(
    outcome: PlanClarification,
    *,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None,
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[object, ...],
    memory_relations: tuple[RelationRows, ...],
) -> None:
    _verify_question_contract(question_contract)
    requested = {fact.id: fact for fact in question_contract.requested_facts}
    row_sources = (
        build_row_source_catalog(catalog, memory_relations=memory_relations)
        if catalog is not None
        else RowSourceCatalog()
    )
    available_required_inputs = {
        item.id: item for item in clarifiable_required_inputs(row_sources)
    }
    satisfied_required_inputs = grounded_required_input_ids(
        row_sources,
        values=available_values,
        grounded_input_uses=available_value_uses,
    )
    seen: set[str] = set()
    for item in outcome.missing_catalog_inputs:
        if item.id in seen:
            raise VerificationError("duplicate clarification")
        seen.add(item.id)
        fact = requested.get(item.requested_fact_id)
        if fact is None:
            raise VerificationError("clarification references unknown requested fact")
        if item.id in {output.id for output in fact.support_answer_outputs}:
            raise VerificationError("clarification id must not be an answer output")
        _verify_missing_catalog_input(
            item,
            fact=fact,
            required_inputs_by_id=available_required_inputs,
            satisfied_required_input_ids=satisfied_required_inputs,
        )


def _verify_missing_catalog_input(
    item,
    *,
    fact: RequestedFact,
    required_inputs_by_id: dict[str, object],
    satisfied_required_input_ids: frozenset[str],
) -> None:
    protected_ids = {
        *(known.id for known in fact.known_inputs),
        *(output.id for output in fact.support_answer_outputs),
    }
    if isinstance(item, MissingCatalogRequiredInput):
        if item.required_catalog_input_id in satisfied_required_input_ids:
            raise VerificationError("missing catalog input is already satisfied")
        if item.required_catalog_input_id in protected_ids:
            raise VerificationError(
                "missing catalog input cannot target known input or answer output"
            )
        required_input = required_inputs_by_id.get(item.required_catalog_input_id)
        if required_input is None:
            raise VerificationError(
                "missing catalog input references unknown required input"
            )
        if tuple(getattr(required_input, "choices", ()) or ()):
            raise VerificationError(
                "missing catalog input must use choice input for choices"
            )
        return
    if isinstance(item, MissingCatalogChoiceInput):
        if item.required_catalog_choice_input_id in satisfied_required_input_ids:
            raise VerificationError("missing catalog choice input is already satisfied")
        if item.required_catalog_choice_input_id in protected_ids:
            raise VerificationError(
                "missing catalog choice input cannot target known input or answer output"
            )
        required_input = required_inputs_by_id.get(
            item.required_catalog_choice_input_id
        )
        if required_input is None:
            raise VerificationError(
                "missing catalog choice input references unknown required input"
            )
        if not tuple(getattr(required_input, "choices", ()) or ()):
            raise VerificationError(
                "missing catalog choice input requires choice-bearing required input"
            )
        return
    raise VerificationError("unsupported missing catalog input")


def _verify_plan_impossible(
    outcome: PlanImpossible,
    *,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    _verify_question_contract(question_contract)
    requested = {fact.id for fact in question_contract.requested_facts}
    blocked: set[str] = set()
    for item in outcome.blocked_facts:
        _verify_blocked_fact(
            item,
            requested=requested,
            catalog=catalog,
            catalog_selection=catalog_selection,
        )
        if item.requested_fact_id in blocked:
            raise VerificationError("duplicate blocked fact")
        blocked.add(item.requested_fact_id)
    missing = requested - blocked
    if missing:
        raise VerificationError("requested fact is neither fulfilled nor blocked")


def _verify_blocked_fact(
    item: BlockedFact,
    *,
    requested: set[str],
    catalog: RelationCatalog | None,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    if item.requested_fact_id not in requested:
        raise VerificationError("blocked fact references unknown requested fact")
    if item.basis not in {
        BlockedFactBasis.CATALOG_ACCESS,
        BlockedFactBasis.POLICY_ACCESS,
    }:
        raise VerificationError("blocked fact basis is not supported")
    if not item.evidence_refs:
        raise VerificationError("blocked fact requires catalog evidence")
    if _blocked_fact_has_zero_catalog_selection(
        item,
        catalog_selection=catalog_selection,
    ):
        _verify_zero_selection_blocked_fact(item)
        return
    if not item.reviewed_read_ids:
        raise VerificationError("blocked fact requires reviewed reads")
    if catalog is None:
        raise VerificationError("catalog is required to verify blocked facts")
    row_sources = build_row_source_catalog(catalog)
    _verify_blocked_fact_reviews_catalog(
        item,
        row_sources=row_sources,
        catalog_selection=catalog_selection,
    )
    reviewed = _reviewed_row_sources(item, row_sources=row_sources)
    reviewed_catalog = RowSourceCatalog(sources=tuple(reviewed.values()))
    _verify_blocked_fact_evidence_refs(item, row_sources=reviewed_catalog)
    _verify_blocked_fact_basis(item, row_sources=reviewed_catalog)
    for field in item.nearest_fields:
        if not _blocked_field_exists(field, reviewed=reviewed):
            raise VerificationError("blocked fact references unknown nearest field")


def _blocked_fact_has_zero_catalog_selection(
    item: BlockedFact,
    *,
    catalog_selection: CatalogSelectionResult | None,
) -> bool:
    if catalog_selection is None:
        return False
    expected_ref = catalog_selection_evidence_ref(
        requested_fact_id=item.requested_fact_id
    )
    if expected_ref not in item.evidence_refs:
        return False
    for selection in catalog_selection.requested_fact_selections:
        if selection.requested_fact_id != item.requested_fact_id:
            continue
        return not selection.selected_read_ids
    return False


def _verify_zero_selection_blocked_fact(item: BlockedFact) -> None:
    if item.basis != BlockedFactBasis.CATALOG_ACCESS:
        raise VerificationError("zero catalog selection requires catalog basis")
    expected_ref = catalog_selection_evidence_ref(
        requested_fact_id=item.requested_fact_id
    )
    if set(item.evidence_refs) != {expected_ref}:
        raise VerificationError("zero catalog selection evidence must be exact")
    if item.reviewed_read_ids or item.nearest_fields:
        raise VerificationError("zero catalog selection cannot review reads")


def _verify_blocked_fact_reviews_catalog(
    item: BlockedFact,
    *,
    row_sources: RowSourceCatalog,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    required = set(
        _required_reviewed_read_ids(
            item.requested_fact_id,
            row_sources=row_sources,
            catalog_selection=catalog_selection,
        )
        or ()
    )
    reviewed = set(item.reviewed_read_ids)
    if reviewed != required:
        raise VerificationError("blocked fact must review every available API read")


def _reviewed_row_sources(
    item: BlockedFact,
    *,
    row_sources: RowSourceCatalog,
) -> dict[str, RowSource]:
    reviewed: dict[str, RowSource] = {}
    for read_id in item.reviewed_read_ids:
        sources = tuple(
            source
            for source in row_sources.sources
            if source.kind == RowSourceKind.API_READ and source.read_id == read_id
        )
        if not sources:
            raise VerificationError("blocked fact references unknown reviewed read")
        for source in sources:
            reviewed[source.id] = source
    return reviewed


def _blocked_field_exists(
    field: BlockedFactField,
    *,
    reviewed: dict[str, RowSource],
) -> bool:
    return any(
        source.read_id == field.read_id
        and any(item.id == field.field_id for item in source.fields)
        for source in reviewed.values()
    )


def _verify_blocked_fact_evidence_refs(
    item: BlockedFact,
    *,
    row_sources: RowSourceCatalog,
) -> None:
    available = _blocked_fact_available_evidence_refs(row_sources)
    if not set(item.evidence_refs) <= available:
        raise VerificationError("blocked fact references unknown catalog evidence")


def _blocked_fact_available_evidence_refs(
    row_sources: RowSourceCatalog,
) -> frozenset[str]:
    refs = {
        read_evidence_ref(source.read_id)
        for source in row_sources.sources
        if source.kind == RowSourceKind.API_READ and source.read_id
    }
    for source in row_sources.sources:
        refs.update(
            read_field_evidence_ref(read_id=source.read_id, field_id=field.id)
            for field in source.fields
            if source.read_id
        )
        for fact in source.blocked_facts:
            refs.update(fact.proof_refs)
    return frozenset(refs)


def _verify_blocked_fact_basis(
    item: BlockedFact,
    *,
    row_sources: RowSourceCatalog,
) -> None:
    if item.basis != BlockedFactBasis.POLICY_ACCESS:
        return
    policy_refs = _policy_blocked_evidence_refs(row_sources)
    if not policy_refs.intersection(item.evidence_refs):
        raise VerificationError("policy blocked fact requires policy evidence")
