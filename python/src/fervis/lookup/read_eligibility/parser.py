"""Parse read-eligibility retention output."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.read_eligibility.candidate_scope import (
    ReadEligibilityCandidateScope,
)
from fervis.lookup.read_eligibility.model import (
    RETENTION_DECISION_VALUES,
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility import provider_contract as provider_output
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)


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

    @classmethod
    def for_request(
        cls,
        request: ReadEligibilityRequest,
    ) -> "_ReadEligibilityParseContext":
        scopes = read_eligibility_candidate_surface(request).candidate_scopes
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


def parse_read_eligibility(
    payload: dict[str, object],
    *,
    request: ReadEligibilityRequest,
) -> ReadEligibilityResult:
    output = provider_output.ReadEligibilityOutput.parse(payload)
    context = _ReadEligibilityParseContext.for_request(request)
    assessments_by_key = _read_assessments_by_key(
        output.requested_fact_assessments,
        context=context,
    )
    return ReadEligibilityResult(
        read_assessments=tuple(
            assessments_by_key[key]
            for key in context.expected_candidate_order
            if key in assessments_by_key
        )
    )


def _read_assessments_by_key(
    items: tuple[provider_output.RequestedFactAssessmentOutput, ...],
    *,
    context: _ReadEligibilityParseContext,
) -> dict[tuple[str, str], ReadAssessment]:
    seen_requested_facts: set[str] = set()
    seen_requested_fact_order: list[str] = []
    output: dict[tuple[str, str], ReadAssessment] = {}
    for item in items:
        requested_fact_id = _required_text(item.requested_fact_id)
        if requested_fact_id not in context.requested_fact_ids:
            raise ValueError("requested fact assessment references unknown fact")
        if requested_fact_id in seen_requested_facts:
            raise ValueError("duplicate requested fact assessment")
        seen_requested_facts.add(requested_fact_id)
        seen_requested_fact_order.append(requested_fact_id)
        fact_assessments = _read_candidate_reviews(
            item.read_candidate_reviews,
            context=context,
            requested_fact_id=requested_fact_id,
        )
        expected_candidate_ids = context.expected_candidate_ids_for_fact(
            requested_fact_id
        )
        if set(fact_assessments) != expected_candidate_ids:
            raise ValueError("requested fact assessment must assess every shown read")
        if tuple(fact_assessments) != context.expected_candidate_order_for_fact(
            requested_fact_id
        ):
            raise ValueError(
                "requested fact assessment must assess reads in the same order shown"
            )
        for source_candidate_id, assessment in fact_assessments.items():
            output[(requested_fact_id, source_candidate_id)] = assessment
    if seen_requested_facts != context.requested_fact_ids:
        raise ValueError("read eligibility must assess every requested fact")
    if tuple(seen_requested_fact_order) != context.requested_fact_order:
        raise ValueError(
            "read eligibility must assess requested facts in the same order shown"
        )
    return output


def _read_candidate_reviews(
    items: tuple[provider_output.ReadCandidateReviewOutput, ...],
    *,
    context: _ReadEligibilityParseContext,
    requested_fact_id: str,
) -> dict[str, ReadAssessment]:
    output: dict[str, ReadAssessment] = {}
    for item in items:
        source_candidate_id = _required_text(item.source_candidate_id)
        if source_candidate_id in output:
            raise ValueError("read candidate assessed more than once")
        read_id = _required_text(item.read_id)
        expected = context.expected_candidate(requested_fact_id, source_candidate_id)
        if expected is None:
            raise ValueError("read candidate review references unknown candidate")
        if expected.read_id != read_id:
            raise ValueError("read candidate review read_id does not match candidate")
        row_path_ids = _row_path_ids(
            _required_texts(item.relevant_row_path_tokens),
            context=context,
            expected=expected,
        )
        field_refs = _field_refs(
            _required_texts(item.relevant_field_tokens),
            context=context,
            expected=expected,
        )
        retention_decision = _required_text(item.retention_decision)
        if retention_decision not in RETENTION_DECISION_VALUES:
            raise ValueError("read candidate review has unsupported retention decision")
        output[source_candidate_id] = ReadAssessment(
            source_candidate_id=expected.source_candidate_id,
            source_candidate_signature=expected.source_candidate_signature,
            requested_fact_id=expected.requested_fact_id,
            read_id=expected.read_id,
            relevant_row_path_ids=row_path_ids,
            relevant_field_refs=field_refs,
            retention_basis=_required_text(item.retention_basis),
            retention_decision=retention_decision,
        )
    return output


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


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("expected non-empty text")
    return text


def _required_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_required_text(value) for value in values)
