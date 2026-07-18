"""Parse read-eligibility retention output."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.read_eligibility.candidate_scope import (
    ReadEligibilityCandidateScope,
)
from fervis.lookup.read_eligibility.model import (
    RETENTION_DECISION_VALUES,
    CanonicalInputOption,
    CanonicalInputSelection,
    DroppedReadAssessment,
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
    RetainedReadAssessment,
)
from fervis.lookup.provider_contract import ProviderObject
from fervis.lookup.read_eligibility import provider_contract as provider_output
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)
from fervis.lookup.read_eligibility.input_bindings import interpretation_question


@dataclass(frozen=True)
class _ExpectedCandidate:
    requested_fact_id: str
    source_candidate_id: str
    source_candidate_signature: str
    read_id: str


@dataclass(frozen=True)
class _ReadEligibilityParseContext:
    requested_fact_ids: set[str]
    requested_fact_order: tuple[str, ...]
    expected_candidates: dict[tuple[str, str], _ExpectedCandidate]
    expected_candidate_order: tuple[tuple[str, str], ...]
    expected_candidate_ids_by_fact: dict[str, set[str]]
    field_refs_by_candidate_signature_and_token: dict[str, dict[str, str]]
    row_path_ids_by_candidate_signature_and_token: dict[str, dict[str, str]]
    canonical_options_by_fact_and_input: dict[
        tuple[str, str],
        tuple[CanonicalInputOption, ...],
    ]
    known_input_order_by_fact: dict[str, tuple[str, ...]]
    interpretation_questions_by_fact_and_input: dict[tuple[str, str], str]

    @classmethod
    def for_request(
        cls,
        request: ReadEligibilityRequest,
    ) -> "_ReadEligibilityParseContext":
        surface = read_eligibility_candidate_surface(request)
        scopes = surface.candidate_scopes
        expected_candidates = _expected_read_candidates(scopes)
        return cls(
            requested_fact_ids={fact.id for fact in request.requested_facts},
            requested_fact_order=tuple(fact.id for fact in request.requested_facts),
            expected_candidates=expected_candidates,
            expected_candidate_order=tuple(expected_candidates),
            expected_candidate_ids_by_fact=_expected_candidate_ids_by_fact(
                request,
                expected_candidates,
            ),
            field_refs_by_candidate_signature_and_token=(
                _field_refs_by_candidate_signature_and_token(scopes)
            ),
            row_path_ids_by_candidate_signature_and_token=(
                _row_path_ids_by_candidate_signature_and_token(scopes)
            ),
            canonical_options_by_fact_and_input=(
                _canonical_options_by_fact_and_input(surface.canonical_options)
            ),
            known_input_order_by_fact=_known_input_order_by_fact(
                surface.canonical_options
            ),
            interpretation_questions_by_fact_and_input=(
                _interpretation_questions_by_fact_and_input(
                    request,
                    surface.canonical_options,
                )
            ),
        )

    def expected_candidate(
        self,
        requested_fact_id: str,
        source_candidate_id: str,
    ) -> _ExpectedCandidate | None:
        return self.expected_candidates.get((requested_fact_id, source_candidate_id))

    def expected_candidate_ids_for_fact(self, requested_fact_id: str) -> set[str]:
        return self.expected_candidate_ids_by_fact.get(requested_fact_id, set())

    def expected_candidate_order_for_fact(
        self,
        requested_fact_id: str,
    ) -> tuple[str, ...]:
        return tuple(
            source_candidate_id
            for fact_id, source_candidate_id in self.expected_candidate_order
            if fact_id == requested_fact_id
        )

    def field_ref_for_token(
        self,
        *,
        source_candidate_signature: str,
        evidence_token: str,
    ) -> str:
        return self.field_refs_by_candidate_signature_and_token.get(
            source_candidate_signature,
            {},
        ).get(evidence_token, "")

    def row_path_id_for_token(
        self,
        *,
        source_candidate_signature: str,
        evidence_token: str,
    ) -> str:
        return self.row_path_ids_by_candidate_signature_and_token.get(
            source_candidate_signature,
            {},
        ).get(evidence_token, "")

    def canonical_options(
        self,
        *,
        requested_fact_id: str,
        known_input_token: str,
    ) -> tuple[CanonicalInputOption, ...]:
        return self.canonical_options_by_fact_and_input.get(
            (requested_fact_id, known_input_token),
            (),
        )


def parse_read_eligibility(
    payload: dict[str, object],
    *,
    request: ReadEligibilityRequest,
) -> ReadEligibilityResult:
    output = provider_output.ReadEligibilityOutput.parse(payload)
    context = _ReadEligibilityParseContext.for_request(request)
    assessments_by_key, canonical_inputs = _parsed_assessments(
        output.requested_fact_assessments,
        context=context,
    )
    return ReadEligibilityResult(
        read_assessments=tuple(
            assessments_by_key[key]
            for key in context.expected_candidate_order
            if key in assessments_by_key
        ),
        canonical_inputs=canonical_inputs,
    )


def _parsed_assessments(
    items: dict[str, provider_output.RequestedFactAssessmentOutput],
    *,
    context: _ReadEligibilityParseContext,
) -> tuple[
    dict[tuple[str, str], ReadAssessment],
    tuple[CanonicalInputSelection, ...],
]:
    if set(items) != context.requested_fact_ids:
        raise ValueError("read eligibility must assess every requested fact")
    output: dict[tuple[str, str], ReadAssessment] = {}
    canonical_inputs: list[CanonicalInputSelection] = []
    for requested_fact_id in context.requested_fact_order:
        item = items[requested_fact_id]
        fact_assessments = _read_candidate_reviews(
            item.read_candidate_reviews,
            context=context,
            requested_fact_id=requested_fact_id,
        )
        canonical_selections = _canonical_selections(
            item.canonical_inputs,
            requested_fact_id=requested_fact_id,
            context=context,
        )
        for source_candidate_id, assessment in fact_assessments.items():
            output[(requested_fact_id, source_candidate_id)] = assessment
        canonical_inputs.extend(canonical_selections.values())
    return output, tuple(canonical_inputs)


def _canonical_selections(
    items: dict[str, provider_output.CanonicalInputSelectionOutput],
    *,
    requested_fact_id: str,
    context: _ReadEligibilityParseContext,
) -> dict[str, CanonicalInputSelection]:
    expected_order = context.known_input_order_by_fact.get(requested_fact_id, ())
    if set(items) != set(expected_order):
        raise ValueError("known input bindings must cover every shown named input")
    output: dict[str, CanonicalInputSelection] = {}
    for known_input_token in expected_order:
        item = items[known_input_token]
        options = context.canonical_options(
            requested_fact_id=requested_fact_id,
            known_input_token=known_input_token,
        )
        canonical_option_id = _required_text(item.canonical_option_id)
        matching = tuple(
            option for option in options if option.id == canonical_option_id
        )
        if not matching:
            raise ValueError("known input binding references unknown canonical option")
        expected_option_ids = tuple(option.id for option in options)
        option_assessments = item.canonical_option_assessments
        if set(option_assessments) != set(expected_option_ids):
            raise ValueError(
                "canonical option assessments must cover every shown option"
            )
        expected_question = context.interpretation_questions_by_fact_and_input[
            (requested_fact_id, known_input_token)
        ]
        if _required_text(item.interpretation_question) != expected_question:
            raise ValueError("known input interpretation question mismatch")
        output[known_input_token] = CanonicalInputSelection(
            option=matching[0],
            interpretation_question=expected_question,
            canonical_option_assessments=tuple(
                (option_id, _required_text(option_assessments[option_id]))
                for option_id in expected_option_ids
            ),
            because=_required_text(item.because),
        )
    return output


def _read_candidate_reviews(
    items: dict[str, ProviderObject],
    *,
    context: _ReadEligibilityParseContext,
    requested_fact_id: str,
) -> dict[str, ReadAssessment]:
    expected_ids = context.expected_candidate_ids_for_fact(requested_fact_id)
    if set(items) != expected_ids:
        raise ValueError("requested fact assessment must assess every shown read")
    output: dict[str, ReadAssessment] = {}
    for source_candidate_id in context.expected_candidate_order_for_fact(
        requested_fact_id
    ):
        raw_item = items[source_candidate_id]
        retention_decision = raw_item.discriminator("retention_decision")
        if retention_decision not in RETENTION_DECISION_VALUES:
            raise ValueError("read candidate review has unsupported retention decision")
        if retention_decision == "RETAIN":
            retained_item = raw_item.parse_as(provider_output.RetainedReadReviewOutput)
        else:
            dropped_item = raw_item.parse_as(provider_output.DroppedReadReviewOutput)
        expected = context.expected_candidate(requested_fact_id, source_candidate_id)
        if expected is None:
            raise ValueError("read candidate review references unknown candidate")
        if retention_decision == "DROP":
            output[source_candidate_id] = DroppedReadAssessment(
                source_candidate_id=expected.source_candidate_id,
                source_candidate_signature=expected.source_candidate_signature,
                requested_fact_id=expected.requested_fact_id,
                read_id=expected.read_id,
                retention_basis=_required_text(dropped_item.retention_basis),
            )
            continue
        output[source_candidate_id] = _retained_read_assessment(
            retained_item,
            expected=expected,
            context=context,
        )
    return output


def _retained_read_assessment(
    item: provider_output.RetainedReadReviewOutput,
    *,
    expected: _ExpectedCandidate,
    context: _ReadEligibilityParseContext,
) -> RetainedReadAssessment:
    return RetainedReadAssessment(
        source_candidate_id=expected.source_candidate_id,
        source_candidate_signature=expected.source_candidate_signature,
        requested_fact_id=expected.requested_fact_id,
        read_id=expected.read_id,
        retention_basis=_required_text(item.retention_basis),
        relevant_row_path_ids=_row_path_ids(
            _required_texts(item.relevant_row_path_tokens),
            context=context,
            expected=expected,
        ),
        relevant_field_refs=_field_refs(
            _required_texts(item.relevant_field_tokens),
            context=context,
            expected=expected,
        ),
    )


def _row_path_ids(
    tokens: tuple[str, ...],
    *,
    context: _ReadEligibilityParseContext,
    expected: _ExpectedCandidate,
) -> tuple[str, ...]:
    output: list[str] = []
    for token in tokens:
        row_path_id = context.row_path_id_for_token(
            source_candidate_signature=expected.source_candidate_signature,
            evidence_token=token,
        )
        if not row_path_id:
            raise ValueError("read candidate review references unknown row path token")
        output.append(row_path_id)
    return tuple(dict.fromkeys(output))


def _field_refs(
    tokens: tuple[str, ...],
    *,
    context: _ReadEligibilityParseContext,
    expected: _ExpectedCandidate,
) -> tuple[str, ...]:
    output: list[str] = []
    for token in tokens:
        field_ref = context.field_ref_for_token(
            source_candidate_signature=expected.source_candidate_signature,
            evidence_token=token,
        )
        if not field_ref:
            raise ValueError("read candidate review references unknown field token")
        output.append(field_ref)
    return tuple(dict.fromkeys(output))


def _expected_read_candidates(
    scopes: tuple[ReadEligibilityCandidateScope, ...],
) -> dict[tuple[str, str], _ExpectedCandidate]:
    return {
        (scope.requested_fact_id, scope.source_candidate_id): _ExpectedCandidate(
            requested_fact_id=scope.requested_fact_id,
            source_candidate_id=scope.source_candidate_id,
            source_candidate_signature=scope.source_candidate_signature,
            read_id=scope.read_id,
        )
        for scope in scopes
    }


def _expected_candidate_ids_by_fact(
    request: ReadEligibilityRequest,
    expected_candidates: dict[tuple[str, str], _ExpectedCandidate],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {fact.id: set() for fact in request.requested_facts}
    for expected in expected_candidates.values():
        output.setdefault(expected.requested_fact_id, set()).add(
            expected.source_candidate_id
        )
    return output


def _field_refs_by_candidate_signature_and_token(
    scopes: tuple[ReadEligibilityCandidateScope, ...],
) -> dict[str, dict[str, str]]:
    return {
        scope.source_candidate_signature: dict(scope.field_refs_by_evidence_token)
        for scope in scopes
    }


def _row_path_ids_by_candidate_signature_and_token(
    scopes: tuple[ReadEligibilityCandidateScope, ...],
) -> dict[str, dict[str, str]]:
    return {
        scope.source_candidate_signature: dict(scope.row_path_ids_by_evidence_token)
        for scope in scopes
    }


def _known_input_order_by_fact(
    options: tuple[CanonicalInputOption, ...],
) -> dict[str, tuple[str, ...]]:
    output: dict[str, list[str]] = {}
    for option in options:
        order = output.setdefault(option.requested_fact_id, [])
        if option.known_input_token not in order:
            order.append(option.known_input_token)
    return {key: tuple(value) for key, value in output.items()}


def _canonical_options_by_fact_and_input(
    options: tuple[CanonicalInputOption, ...],
) -> dict[tuple[str, str], tuple[CanonicalInputOption, ...]]:
    grouped: dict[tuple[str, str], list[CanonicalInputOption]] = {}
    for option in options:
        grouped.setdefault(
            (option.requested_fact_id, option.known_input_token),
            [],
        ).append(option)
    return {key: tuple(value) for key, value in grouped.items()}


def _interpretation_questions_by_fact_and_input(
    request: ReadEligibilityRequest,
    options: tuple[CanonicalInputOption, ...],
) -> dict[tuple[str, str], str]:
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    known_inputs_by_id = {
        known_input.id: known_input
        for fact in request.requested_facts
        for known_input in fact.known_inputs
    }
    return {
        (option.requested_fact_id, option.known_input_token): (
            interpretation_question(
                known_input_text=known_inputs_by_id[option.known_input_id].text,
                answer_fact=facts_by_id[option.requested_fact_id].description,
            )
        )
        for option in options
    }


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("expected non-empty text")
    return text


def _required_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_required_text(value) for value in values)
