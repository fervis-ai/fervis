from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator

from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    AllResults,
    FactLocalRef,
    FactTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.query_enrichment.semantic import (
    ReferenceInputRecallTask,
    RecallBucketMatch,
    SemanticQueryEnrichmentRequest,
    SemanticRecallBucket,
    SemanticRecallBucketKind,
)
from fervis.lookup.query_enrichment.semantic_parser import (
    parse_semantic_query_enrichment,
)
from fervis.lookup.query_enrichment.semantic_schema import (
    build_semantic_query_enrichment_schema,
)
from fervis.lookup.relation_catalog.selection.selector.semantic_selection import (
    SemanticCatalogSelectionRequest,
    select_semantic_relation_catalog,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.catalog import catalog_from_payload


def run_semantic_query_enrichment_case(payload: dict[str, Any]) -> list[str]:
    value = payload["input"]
    if value.get("mode") == "select_catalog":
        return _run_semantic_catalog_selection(payload)
    request = SemanticQueryEnrichmentRequest(
        recall_buckets=tuple(_bucket(item) for item in value["recall_buckets"]),
        reference_tasks=tuple(
            _reference_task(item) for item in value["reference_tasks"]
        ),
        resource_names=tuple(value["resource_names"]),
    )
    errors = tuple(
        Draft7Validator(build_semantic_query_enrichment_schema(request)).iter_errors(
            value["payload"]
        )
    )
    if errors:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        return [f"schema rejected payload: {errors[0].message}"]
    try:
        result = parse_semantic_query_enrichment(value["payload"], request=request)
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        return [f"parser rejected payload: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual = {
        "recall_bucket_matches": [
            {
                "bucket_ref": item.bucket_ref,
                "exhaustive_resource_names": list(item.exhaustive_resource_names),
                "matching_resource_names": list(item.matching_resource_names),
            }
            for item in result.recall_bucket_matches
        ],
        "input_resource_search_terms": [
            {
                "input_use_ref": item.input_use_ref,
                "catalog_search_terms": list(item.catalog_search_terms),
            }
            for item in result.input_resource_search_terms
        ],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )


def _run_semantic_catalog_selection(payload: dict[str, Any]) -> list[str]:
    value = payload["input"]
    origin = _origin("alpha")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(
            FactTerm("f1", "s1", TextType(), _origin("beta")),
            FactTerm("f2", "s1", TextType(), _origin("gamma")),
        ),
        expressions=(),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(
            RequestedOutput("output_1", "f1", _origin("beta")),
            RequestedOutput("output_2", "f2", _origin("gamma")),
        ),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    result = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            relation_catalog=catalog_from_payload(value["catalog"]),
            indexes=(index,),
            resource_matches=tuple(
                RecallBucketMatch(
                    bucket_ref=item["bucket_ref"],
                    exhaustive_resource_names=tuple(item["exhaustive_resource_names"]),
                    matching_resource_names=tuple(item["matching_resource_names"]),
                )
                for item in value["resource_matches"]
            ),
            max_reads_per_fact=int(value["max_reads_per_fact"]),
        )
    )
    return subset_mismatches(
        actual={
            "selected_read_ids": list(result.selected_read_ids),
            "selected_read_membership": {
                read_id: True for read_id in result.selected_read_ids
            },
        },
        expected_subset=payload["expect"].get("result_contains") or {},
    )


def _origin(meaning: str) -> SourceOrigin:
    return SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, meaning)


def _bucket(item: dict[str, Any]) -> SemanticRecallBucket:
    return SemanticRecallBucket(
        bucket_ref=item["bucket_ref"],
        kind=SemanticRecallBucketKind(item["kind"]),
        origin=_origin(item["meaning"]),
        requirement_refs=tuple(
            FactLocalRef.from_token(ref) for ref in item["requirement_refs"]
        ),
    )


def _reference_task(item: dict[str, Any]) -> ReferenceInputRecallTask:
    return ReferenceInputRecallTask(
        input_use_ref=item["input_use_ref"],
        input_ref=item["input_ref"],
        input_origin=_origin(item["input_text"]),
        operand_meaning=item["operand_meaning"],
        reference_fact_ref=FactLocalRef.from_token(item["reference_fact_ref"]),
        expected_set_ref=(
            FactLocalRef.from_token(item["expected_set_ref"])
            if item.get("expected_set_ref")
            else None
        ),
    )


__all__ = ["run_semantic_query_enrichment_case"]
