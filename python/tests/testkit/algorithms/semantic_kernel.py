from __future__ import annotations

from typing import Any

from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceValueType,
)
from fervis.lookup.read_eligibility.semantic import (
    ReadRequirementAssessment,
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.lookup.plan_execution.verification.contract_types import (
    RelationSemanticGuarantee,
    _combine_semantic_guarantee_maps,
)
from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
    infer_operator_result,
)
from fervis.lookup.qualification import (
    BooleanAtomRef,
    BooleanPolarity,
    QualificationAtomProof,
    QualificationGuarantee,
    QualificationDNF,
    SubjectGuarantee,
    and_dnf,
    atom_dnf,
    not_dnf,
    or_dnf,
    qualification_entails,
    relative_dnf,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    CollectionType,
    DateTimeType,
    DateType,
    DecimalType,
    IntegerType,
    RatioMeasure,
    TemporalScopeType,
    TextType,
    UnitlessMeasure,
    ScalarType,
    ValueType,
)

from tests.testkit.assertions import subset_mismatches


def run_semantic_kernel_case(payload: dict[str, Any]) -> list[str]:
    request = payload["input"]
    mode = str(request["mode"])
    actual: dict[str, Any]
    if mode == "infer_operator":
        operator_text = str(request["operator"])
        operator = (
            ExpressionUnaryOperator(operator_text)
            if len(request["operands"]) == 1
            else ExpressionBinaryOperator(operator_text)
        )
        result = infer_operator_result(
            operator,
            tuple(_value_type(item) for item in request["operands"]),
        )
        actual = {"result_type": _type_payload(result)}
    elif mode == "qualification":
        fact_id = str(request["requested_fact_id"])
        actual_formula = _formula(fact_id, request["actual"])
        requested_formula = _formula(fact_id, request["requested"])
        actual = {
            "actual_clauses": _clauses(actual_formula),
            "entails": qualification_entails(actual_formula, requested_formula),
        }
    elif mode == "relative_qualification":
        fact_id = str(request["requested_fact_id"])
        actual = {
            "residual_clauses": _clauses(
                relative_dnf(
                    fact_id,
                    given=_formula(fact_id, request["given"]),
                    condition=_formula(fact_id, request["condition"]),
                )
            )
        }
    elif mode == "retained_read_row_sources":
        sources = tuple(
            RowSource(
                id=str(item["id"]),
                kind=RowSourceKind.API_READ,
                label=str(item["id"]),
                read_id=str(item["read_id"]),
                row_path_id=str(item["row_path_id"]),
                row_path=str(item["row_path"]),
                fields=tuple(
                    RowSourceField(
                        id=str(field),
                        field_ref=str(field),
                        label=str(field),
                        type=RowSourceValueType.STRING,
                        allowed_roles=(),
                    )
                    for field in item["field_refs"]
                ),
            )
            for item in request["sources"]
        )
        assessment = ReadRequirementAssessment(
            requested_fact_id="fact_1",
            candidate_ref="read_candidate",
            source_refs=tuple(source.id for source in sources),
            read_id=sources[0].read_id,
            relevant_field_refs=tuple(request["retained_field_tokens"]),
            assessment_basis="The retained fields supply the required value.",
            decision=SemanticReadDecision.RETAIN,
        )
        catalog = build_available_source_catalog(
            RowSourceCatalog(sources=sources),
            read_eligibility=SemanticReadEligibilityResult(
                read_assessments=(assessment,),
                identity_outcomes=(),
            ),
        )
        actual = {
            "retained_source_refs": [source.id for source in catalog.sources],
            "retained_source_count": len(catalog.sources),
        }
    elif mode == "union_semantic_guarantees":
        fact_id = str(request["requested_fact_id"])
        atom = BooleanAtomRef(str(request["atom_ref"]), BooleanPolarity.POSITIVE)
        guarantee = RelationSemanticGuarantee(
            qualification=QualificationGuarantee(
                requested_fact_id=fact_id,
                formula=atom_dnf(fact_id, atom),
                atom_proofs=(QualificationAtomProof(atom, ("proof",)),),
            ),
            subject=SubjectGuarantee(
                requested_fact_id=fact_id,
                subject_set_ref=str(request["subject_set_ref"]),
                interpretation="normal_business_instance",
                proof_refs=("proof",),
            ),
        )
        combined = _combine_semantic_guarantee_maps(
            tuple(
                {fact_id: guarantee} if present else {}
                for present in request["input_guarantees"]
            ),
            disjoin=True,
        )
        actual = {"guaranteed_fact_refs": sorted(combined)}
    else:
        return [f"unsupported semantic-kernel mode: {mode}"]
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _formula(requested_fact_id: str, payload: dict[str, Any]) -> QualificationDNF:
    kind = str(payload["kind"])
    if kind == "atom":
        return atom_dnf(
            requested_fact_id,
            BooleanAtomRef(
                value_ref=str(payload["ref"]),
                polarity=BooleanPolarity(str(payload.get("polarity") or "positive")),
            ),
        )
    arguments = tuple(
        _formula(requested_fact_id, item) for item in payload.get("arguments") or ()
    )
    if kind == "not" and len(arguments) == 1:
        return not_dnf(requested_fact_id, arguments[0])
    if kind in {"and", "or"} and len(arguments) >= 2:
        result = arguments[0]
        operation = and_dnf if kind == "and" else or_dnf
        for argument in arguments[1:]:
            result = operation(requested_fact_id, result, argument)
        return result
    raise ValueError("invalid Boolean test formula")


def _clauses(value: QualificationDNF) -> list[list[str]]:
    return [
        [
            f"{'!' if atom.polarity is BooleanPolarity.NEGATIVE else ''}{atom.value_ref}"
            for atom in sorted(clause.atom_refs)
        ]
        for clause in value.clauses
    ]


def _value_type(payload: dict[str, Any]) -> ValueType:
    kind = str(payload["kind"])
    if kind == "boolean":
        return BooleanType()
    if kind == "integer":
        return IntegerType()
    if kind == "text":
        return TextType()
    if kind == "date":
        return DateType()
    if kind == "datetime":
        return DateTimeType()
    if kind == "temporal_scope":
        return TemporalScopeType()
    if kind == "decimal":
        measure = str(payload.get("measure") or "unitless")
        return DecimalType(RatioMeasure() if measure == "ratio" else UnitlessMeasure())
    if kind == "collection":
        element_type = _value_type(payload["element_type"])
        if not isinstance(element_type, ScalarType):
            raise ValueError("collection element type must be scalar")
        return CollectionType(element_type)
    raise ValueError(f"unsupported test value type {kind}")


def _type_payload(value_type: ValueType) -> dict[str, Any]:
    if isinstance(value_type, BooleanType):
        return {"kind": "boolean"}
    if isinstance(value_type, IntegerType):
        return {"kind": "integer"}
    if isinstance(value_type, DecimalType):
        return {
            "kind": "decimal",
            "measure": (
                "ratio" if isinstance(value_type.measure, RatioMeasure) else "unitless"
            ),
        }
    raise ValueError("test serializer does not support inferred type")
