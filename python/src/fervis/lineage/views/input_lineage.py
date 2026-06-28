"""End-user input lineage projection."""

from __future__ import annotations

from fervis.lineage.enums import ContributionOrigin
from fervis.lineage.views.detail import LineageRenderDetail
from fervis.lineage.views.model import (
    InputLineageResultView,
    InputLineageView,
    LineageView,
)


def input_lineage_view(
    view: LineageView, *, answer_output: str | None = None
) -> InputLineageView:
    return InputLineageView(
        root_kind=view.root_kind,
        root_id=view.root_id,
        results=tuple(_input_results(view, answer_output=answer_output)),
    )


def render_input_lineage(
    view: InputLineageView,
    *,
    detail: LineageRenderDetail = LineageRenderDetail.COMPACT,
) -> str:
    lines: list[str] = []
    for result in view.results:
        if lines:
            lines.append("")
        lines.append(f"Inputs used for {result.requested_fact_id}")
        lines.append(f"Fact: {result.fact_description}")
        _append_group(lines, "Explicit", result.explicit)
        _append_group(lines, "Derived", result.derived)
        _append_group(lines, "Contextual", result.contextual)
        _append_group(lines, "Applied in execution", result.applied)
        if detail.includes_verbose():
            _append_group(lines, "Evidence", result.evidence_refs)
        if detail.includes_debug():
            _append_group(lines, "Proof handles", result.proof_handles)
    if not lines:
        lines.append("No input lineage is available for this view.")
    return "\n".join(lines)


def _input_results(view: LineageView, *, answer_output: str | None):
    for question in view.questions:
        for run in question.runs:
            for fact in run.requested_facts:
                for result in fact.fact_results:
                    if result.proof is None:
                        continue
                    reachable = _reachable_proof_nodes(
                        result.proof.computation_links,
                        _selected_answer_output_refs(
                            fact.answer_outputs,
                            answer_output=answer_output,
                        ),
                    )
                    if not reachable:
                        continue
                    yield InputLineageResultView(
                        fact_result_id=result.fact_result_id,
                        requested_fact_id=fact.requested_fact_id,
                        fact_description=fact.description,
                        explicit=_contribution_labels(
                            result.proof.contributions,
                            origin=ContributionOrigin.EXPLICIT,
                            reachable=reachable,
                        ),
                        derived=_contribution_labels(
                            result.proof.contributions,
                            origin=ContributionOrigin.DERIVED,
                            reachable=reachable,
                        ),
                        contextual=_contribution_labels(
                            result.proof.contributions,
                            origin=ContributionOrigin.CONTEXTUAL,
                            reachable=reachable,
                        ),
                        applied=_applied_execution_lines(
                            result.proof.applied_inputs,
                            reachable=reachable,
                        ),
                        evidence_refs=_evidence_refs(result.proof, reachable=reachable),
                        proof_handles=tuple(sorted(reachable)),
                    )


def _selected_answer_output_refs(
    outputs, *, answer_output: str | None
) -> tuple[str, ...]:
    return tuple(
        proof_ref
        for output in outputs
        if answer_output is None or output.output_key == answer_output
        for proof_ref in output.proof_node_refs
    )


def _reachable_proof_nodes(links, start_refs: tuple[str, ...]) -> frozenset[str]:
    if not start_refs:
        return frozenset()
    sources_by_target: dict[str, list[str]] = {}
    for link in links:
        sources_by_target.setdefault(link.target, []).append(link.source)
    reachable: set[str] = set()
    pending = list(start_refs)
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(sources_by_target.get(node_id, ()))
    return frozenset(reachable)


def _contribution_labels(
    contributions, *, origin: ContributionOrigin, reachable: frozenset[str]
) -> tuple[str, ...]:
    return _dedupe_keep_order(
        item.label
        for item in contributions
        if item.origin is origin and _has_reachable_ref(item.node_refs, reachable)
    )


def _applied_execution_lines(
    applied_inputs, *, reachable: frozenset[str]
) -> tuple[str, ...]:
    return tuple(
        f"{item.label} {item.action}"
        for item in applied_inputs
        if item.handle in reachable
    )


def _evidence_refs(proof, *, reachable: frozenset[str]) -> tuple[str, ...]:
    return _dedupe_keep_order(
        ref
        for item in (*proof.contributions, *proof.applied_inputs)
        if _has_reachable_ref(_item_node_refs(item), reachable)
        for ref in item.proof_refs
    )


def _item_node_refs(item) -> tuple[str, ...]:
    node_refs = getattr(item, "node_refs", ())
    if node_refs:
        return node_refs
    handle = str(getattr(item, "handle", "") or "")
    return (handle,) if handle else ()


def _has_reachable_ref(proof_refs: tuple[str, ...], reachable: frozenset[str]) -> bool:
    if not proof_refs:
        return False
    return any(ref in reachable for ref in proof_refs)


def _append_group(lines: list[str], title: str, values: tuple[str, ...]) -> None:
    if not values:
        return
    lines.append(title)
    for value in values:
        lines.append(f"- {value}")


def _dedupe_keep_order(values) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
