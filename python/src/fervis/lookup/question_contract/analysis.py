"""Deterministic analysis of one semantic relational requested fact."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, TypeAlias

from fervis.lookup.expression_operators import (
    infer_aggregate_result,
    infer_operator_result,
)
from fervis.lookup.qualification import (
    BooleanRequirement,
    BooleanRequirementUseSite,
    BooleanFormula,
    QualificationDNF,
    boolean_requirements,
    normalize_qualification,
)
from fervis.lookup.question_contract.model import (
    Aggregate,
    Arithmetic,
    AssociationTerm,
    BooleanComposition,
    BooleanCompositionOperator,
    Comparison,
    Coverage,
    ExpressionNode,
    FactLocalKind,
    FactLocalRef,
    FactTerm,
    FirstRankWithTies,
    InputDenotation,
    InputDenotationKind,
    InputTerm,
    NullCheck,
    Ordering,
    Quantify,
    RelatedRow,
    RequestedFact,
    ResultSelection,
    SetTerm,
    TakeWithBoundaryTies,
    PositionWithTies,
    TemporalBucket,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    DateTimeType,
    DateType,
    IdentifierType,
    IntegerType,
    TemporalPointType,
    ValueType,
)


SemanticValueRef: TypeAlias = FactLocalRef | str


@dataclass(frozen=True)
class ConstantDomain:
    pass


@dataclass(frozen=True)
class RowDomain:
    owner_ref: FactLocalRef


@dataclass(frozen=True)
class AssociationExpandedDomain:
    path_refs: tuple[FactLocalRef, ...]


@dataclass(frozen=True)
class AggregateDomain:
    grouping_refs: tuple[FactLocalRef, ...]


@dataclass(frozen=True)
class ResultDomain:
    pass


EvaluationDomain: TypeAlias = (
    ConstantDomain
    | RowDomain
    | AssociationExpandedDomain
    | AggregateDomain
    | ResultDomain
)


@dataclass(frozen=True)
class SubjectRows:
    subject_set_ref: FactLocalRef


@dataclass(frozen=True)
class Groups:
    grouping_refs: tuple[FactLocalRef, ...]


@dataclass(frozen=True)
class Singleton:
    pass


ResultGrain: TypeAlias = SubjectRows | Groups | Singleton


@dataclass(frozen=True)
class InputUseSite:
    use_ref: str
    input_ref: str
    operand_meaning: str
    requested_fact_id: str
    structural_location: str
    expression_ref: FactLocalRef | None
    expected_value_type: ValueType
    identity_set_ref: FactLocalRef | None
    reference_fact_ref: FactLocalRef | None


@dataclass(frozen=True)
class OutputRequirement:
    output_ref: FactLocalRef
    value_ref: SemanticValueRef
    dependencies: frozenset[SemanticValueRef]


@dataclass(frozen=True)
class ResourcePopulation:
    subject_set_ref: FactLocalRef


@dataclass(frozen=True)
class RawDataRecord:
    subject_set_ref: FactLocalRef


SubjectObligation: TypeAlias = ResourcePopulation | RawDataRecord


@dataclass(frozen=True)
class RequestedFactSemanticIndex:
    requested_fact: RequestedFact
    input_by_ref: Mapping[str, InputTerm]
    input_denotation_by_ref: Mapping[str, InputDenotation]
    requested_fact_id: str
    fact_local_ref_by_local_id: Mapping[str, FactLocalRef]
    term_by_ref: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm]
    expression_by_ref: Mapping[FactLocalRef, ExpressionNode]
    inferred_type_by_ref: Mapping[SemanticValueRef, ValueType]
    evaluation_domain_by_ref: Mapping[SemanticValueRef, EvaluationDomain]
    direct_dependencies_by_ref: Mapping[FactLocalRef, frozenset[SemanticValueRef]]
    transitive_dependencies_by_ref: Mapping[FactLocalRef, frozenset[SemanticValueRef]]
    input_use_sites: tuple[InputUseSite, ...]
    output_requirements: tuple[OutputRequirement, ...]
    source_requirement_refs: frozenset[FactLocalRef]
    association_requirement_refs: frozenset[FactLocalRef]
    qualification: QualificationDNF
    boolean_requirements: tuple[BooleanRequirement, ...]
    subject_obligation: SubjectObligation
    grouping_refs: tuple[FactLocalRef, ...]
    ordering_refs: tuple[FactLocalRef, ...]
    selection: ResultSelection
    result_grain: ResultGrain

    def output_qualification(self, output_id: str) -> QualificationDNF:
        """Require subject filtering only for outputs evaluated over that subject."""
        from fervis.lookup.question_contract.domains import (
            value_population_dependencies,
        )

        output = next(
            item for item in self.requested_fact.outputs if item.id == output_id
        )
        ref = self.fact_local_ref_by_local_id.get(
            output.expression_ref, output.expression_ref
        )
        if (
            isinstance(self.result_grain, Singleton)
            and self.subject_obligation.subject_set_ref.token
            not in value_population_dependencies(self, ref)
        ):
            return QualificationDNF.true(self.requested_fact_id)
        return self.qualification

    def condition_qualification(self, expression_ref: str) -> QualificationDNF:
        """Normalize a scoped condition with the same algebra as qualification."""
        return normalize_qualification(
            requested_fact_id=self.requested_fact_id,
            root_ref=self.fact_local_ref_by_local_id[expression_ref].token,
            formulas={
                ref.token: _qualification_node(
                    node,
                    local_refs=self.fact_local_ref_by_local_id,
                    inputs=self.input_by_ref,
                )
                for ref, node in self.expression_by_ref.items()
            },
        )

    @property
    def observed_fact_refs(self) -> frozenset[FactLocalRef]:
        """Facts whose values must be read to produce, group, or order results.

        Qualification-only facts may instead be satisfied by an authorized
        parameter application. Observed values cannot be replaced by a filter.
        """

        roots = (
            *tuple(item.value_ref for item in self.output_requirements),
            *self.grouping_refs,
            *self.ordering_refs,
        )
        return frozenset().union(*(self.value_fact_refs(ref) for ref in roots))

    def value_fact_refs(self, ref: SemanticValueRef) -> frozenset[FactLocalRef]:
        """Raw value dependencies, excluding separately owned scoped predicates."""
        if isinstance(ref, str):
            return frozenset()
        if ref.kind is FactLocalKind.FACT:
            return frozenset((ref,))
        node = self.expression_by_ref.get(ref)
        if isinstance(node, (Quantify, RelatedRow, Coverage)):
            return frozenset()
        dependencies = (
            (self.fact_local_ref_by_local_id[node.argument_ref],)
            if isinstance(node, Aggregate)
            else self.direct_dependencies_by_ref.get(ref, ())
        )
        return frozenset().union(
            *(self.value_fact_refs(child) for child in dependencies)
        )

    def value_type(self, ref: SemanticValueRef) -> ValueType:
        if isinstance(ref, str):
            return self.input_by_ref[ref].value_type
        value = self.term_by_ref.get(ref) or self.expression_by_ref.get(ref)
        if value is None:
            raise KeyError(ref)
        return semantic_value_type(
            value,
            inferred_type=self.inferred_type_by_ref.get(ref),
        )

    @property
    def term_requirement_refs(self) -> frozenset[FactLocalRef]:
        """Source-backed set and fact terms, excluding associations."""

        return self.source_requirement_refs - self.association_requirement_refs

    def source_support_closure(
        self, requirement_refs: frozenset[FactLocalRef]
    ) -> frozenset[FactLocalRef]:
        """Expand identity facts through their declared semantic relationship."""

        expanded = set(requirement_refs)
        for ref in requirement_refs:
            term = self.term_by_ref.get(ref)
            if not isinstance(term, FactTerm):
                continue
            expanded.add(self.fact_local_ref_by_local_id[term.owner_ref])
            if not isinstance(term.value_type, IdentifierType):
                continue
            identity_set_ref = self.fact_local_ref_by_local_id[term.value_type.set_ref]
            expanded.add(identity_set_ref)
            association_ref = _unique_identifier_association(
                self.fact_local_ref_by_local_id[term.owner_ref],
                identity_set_ref=identity_set_ref,
                terms=self.term_by_ref,
                local_refs=self.fact_local_ref_by_local_id,
            )
            if association_ref is not None:
                expanded.add(association_ref)
        subject_ref = self.subject_obligation.subject_set_ref
        for ref in tuple(expanded):
            if ref.kind is FactLocalKind.SET and ref != subject_ref:
                related_row_refs = tuple(
                    self.fact_local_ref_by_local_id[item]
                    for expression in self.expression_by_ref.values()
                    if isinstance(expression, RelatedRow)
                    and expression.set_ref == ref.local_id
                    for item in expression.association_refs
                )
                if related_row_refs:
                    expanded.update(related_row_refs)
                    continue
                expanded.update(
                    _unique_association_path_refs(
                        subject_ref,
                        destination=ref,
                        terms=self.term_by_ref,
                        local_refs=self.fact_local_ref_by_local_id,
                    )
                )
        return frozenset(expanded) & self.source_requirement_refs


def analyze_requested_fact(
    requested_fact: RequestedFact,
    *,
    inputs: Mapping[str, InputTerm],
    input_denotations: Mapping[str, InputDenotation],
) -> RequestedFactSemanticIndex:
    local_refs = _local_refs(requested_fact)
    terms = _terms(requested_fact, local_refs=local_refs)
    expressions = _expressions(requested_fact, local_refs=local_refs)
    subject_ref = _required_local_ref(
        requested_fact.subject.set_ref,
        local_refs=local_refs,
        kind=FactLocalKind.SET,
    )
    grouping_refs = tuple(
        _required_local_scalar_ref(item, local_refs=local_refs)
        for item in requested_fact.grouping_refs
    )
    inferred_types: dict[SemanticValueRef, ValueType] = {
        input_id: item.value_type for input_id, item in inputs.items()
    }
    domains: dict[SemanticValueRef, EvaluationDomain] = {
        input_id: ConstantDomain() for input_id in inputs
    }
    for fact in requested_fact.facts:
        fact_ref = local_refs[fact.id]
        inferred_types[fact_ref] = fact.value_type
        owner_ref = local_refs[fact.owner_ref]
        domains[fact_ref] = _fact_domain(owner_ref, terms=terms)

    direct: dict[FactLocalRef, frozenset[SemanticValueRef]] = {}
    transitive: dict[FactLocalRef, frozenset[SemanticValueRef]] = {}
    for node in requested_fact.expressions:
        node_ref = local_refs[node.id]
        dependencies = tuple(
            _value_ref(value_ref, local_refs=local_refs, inputs=inputs)
            for value_ref in _node_dependency_ids(node)
        )
        for dependency in dependencies:
            if (
                isinstance(dependency, FactLocalRef)
                and dependency.kind is FactLocalKind.EXPRESSION
                and dependency not in inferred_types
            ):
                raise ValueError(f"expression {node.id} contains a forward reference")
        direct[node_ref] = frozenset(dependencies)
        transitive[node_ref] = frozenset(
            dependency
            for item in dependencies
            for dependency in (
                {item}
                if isinstance(item, str)
                else {item, *transitive.get(item, frozenset())}
            )
        )
        inferred_types[node_ref] = _infer_node_type(
            node,
            dependencies=dependencies,
            inferred_types=inferred_types,
            terms=terms,
        )
        domains[node_ref] = _infer_node_domain(
            node,
            dependencies=dependencies,
            domains=domains,
            local_refs=local_refs,
            terms=terms,
            subject_ref=subject_ref,
            grouping_refs=grouping_refs,
        )

    qualification_ref = (
        None
        if requested_fact.qualification_ref is None
        else _required_boolean_ref(
            requested_fact.qualification_ref,
            local_refs=local_refs,
            inferred_types=inferred_types,
        )
    )
    if qualification_ref is not None and not _domain_belongs_to_subject(
        domains[qualification_ref], subject_ref=subject_ref, terms=terms
    ):
        raise ValueError("qualification is not evaluated for the subject")
    if grouping_refs:
        _validate_grouping_domains(
            tuple(domains[item] for item in grouping_refs),
            subject_ref=subject_ref,
            terms=terms,
            local_refs=local_refs,
        )
    result_grain = _result_grain(
        requested_fact,
        subject_ref=subject_ref,
        grouping_refs=grouping_refs,
        local_refs=local_refs,
        inputs=inputs,
        domains=domains,
    )
    _validate_result_surfaces(
        requested_fact,
        result_grain=result_grain,
        grouping_refs=grouping_refs,
        local_refs=local_refs,
        inputs=inputs,
        inferred_types=inferred_types,
        domains=domains,
    )
    formulas = {
        ref.token: _qualification_node(node, local_refs=local_refs, inputs=inputs)
        for ref, node in expressions.items()
        if isinstance(inferred_types.get(ref), BooleanType)
    }
    qualification = normalize_qualification(
        requested_fact_id=requested_fact.id,
        root_ref=None if qualification_ref is None else qualification_ref.token,
        formulas=formulas,
    )
    requirements = _boolean_requirements(
        requested_fact,
        qualification=qualification,
        formulas=formulas,
        local_refs=local_refs,
    )
    input_use_sites = _input_use_sites(
        requested_fact,
        inputs=inputs,
        input_denotations=input_denotations,
        local_refs=local_refs,
        inferred_types=inferred_types,
        direct=direct,
    )
    output_requirements = tuple(
        _output_requirement(
            requested_fact,
            output_index=index,
            local_refs=local_refs,
            inputs=inputs,
            transitive=transitive,
        )
        for index, _ in enumerate(requested_fact.outputs, start=1)
    )
    source_refs = _source_requirements(
        output_requirements=output_requirements,
        subject_ref=subject_ref,
        qualification_ref=qualification_ref,
        grouping_refs=grouping_refs,
        ordering=requested_fact.ordering,
        local_refs=local_refs,
        transitive=transitive,
        terms=terms,
        expressions=expressions,
        domains=domains,
    )
    association_refs = frozenset(
        ref for ref in source_refs if ref.kind is FactLocalKind.ASSOCIATION
    )
    obligation: SubjectObligation
    if requested_fact.subject.instance_interpretation.value == "raw_data_record":
        obligation = RawDataRecord(subject_ref)
    else:
        obligation = ResourcePopulation(subject_ref)
    return RequestedFactSemanticIndex(
        requested_fact=requested_fact,
        input_by_ref=MappingProxyType(dict(inputs)),
        input_denotation_by_ref=MappingProxyType(dict(input_denotations)),
        requested_fact_id=requested_fact.id,
        fact_local_ref_by_local_id=MappingProxyType(local_refs),
        term_by_ref=MappingProxyType(terms),
        expression_by_ref=MappingProxyType(expressions),
        inferred_type_by_ref=MappingProxyType(inferred_types),
        evaluation_domain_by_ref=MappingProxyType(domains),
        direct_dependencies_by_ref=MappingProxyType(direct),
        transitive_dependencies_by_ref=MappingProxyType(transitive),
        input_use_sites=input_use_sites,
        output_requirements=output_requirements,
        source_requirement_refs=source_refs,
        association_requirement_refs=association_refs,
        qualification=qualification,
        boolean_requirements=requirements,
        subject_obligation=obligation,
        grouping_refs=grouping_refs,
        ordering_refs=tuple(
            _required_local_scalar_ref(item.expression_ref, local_refs=local_refs)
            for item in requested_fact.ordering
        ),
        selection=requested_fact.selection,
        result_grain=result_grain,
    )


def infer_expression_types(
    index: RequestedFactSemanticIndex,
    *,
    fact_type_by_ref: Mapping[FactLocalRef, ValueType],
) -> Mapping[SemanticValueRef, ValueType]:
    """Revalidate one expression graph with concrete bound fact types."""

    inferred_types: dict[SemanticValueRef, ValueType] = {
        input_ref: item.value_type for input_ref, item in index.input_by_ref.items()
    }
    for ref, term in index.term_by_ref.items():
        if isinstance(term, FactTerm):
            inferred_types[ref] = fact_type_by_ref.get(ref, term.value_type)
    for node in index.requested_fact.expressions:
        node_ref = index.fact_local_ref_by_local_id[node.id]
        dependencies = tuple(
            _value_ref(
                value_ref,
                local_refs=index.fact_local_ref_by_local_id,
                inputs=index.input_by_ref,
            )
            for value_ref in _node_dependency_ids(node)
        )
        inferred_types[node_ref] = _infer_node_type(
            node,
            dependencies=dependencies,
            inferred_types=inferred_types,
            terms=index.term_by_ref,
        )
    return MappingProxyType(inferred_types)


def semantic_value_type(
    value: SetTerm | AssociationTerm | FactTerm | ExpressionNode,
    *,
    inferred_type: ValueType | None,
) -> ValueType:
    """Return the canonical semantic type for a declared value."""

    if isinstance(value, SetTerm):
        return IdentifierType(value.id)
    if inferred_type is None:
        raise ValueError("semantic value lacks an inferred type")
    return inferred_type


def _local_refs(requested_fact: RequestedFact) -> dict[str, FactLocalRef]:
    refs: dict[str, FactLocalRef] = {}
    for kind, items in (
        (FactLocalKind.SET, requested_fact.sets),
        (FactLocalKind.ASSOCIATION, requested_fact.associations),
        (FactLocalKind.FACT, requested_fact.facts),
        (FactLocalKind.EXPRESSION, requested_fact.expressions),
    ):
        for item in items:
            if item.id in refs:
                raise ValueError(f"duplicate fact-local id: {item.id}")
            refs[item.id] = FactLocalRef(requested_fact.id, kind, item.id)
    return refs


def _terms(
    requested_fact: RequestedFact, *, local_refs: Mapping[str, FactLocalRef]
) -> dict[FactLocalRef, SetTerm | AssociationTerm | FactTerm]:
    terms: dict[FactLocalRef, SetTerm | AssociationTerm | FactTerm] = {}
    for set_term in requested_fact.sets:
        terms[local_refs[set_term.id]] = set_term
    for association_term in requested_fact.associations:
        terms[local_refs[association_term.id]] = association_term
    for fact_term in requested_fact.facts:
        terms[local_refs[fact_term.id]] = fact_term
    for association in requested_fact.associations:
        _required_local_ref(
            association.from_set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
        _required_local_ref(
            association.to_set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
    for fact in requested_fact.facts:
        owner = local_refs.get(fact.owner_ref)
        if owner is None:
            raise ValueError(f"unknown fact owner: {fact.owner_ref}")
        if owner.kind not in {FactLocalKind.SET, FactLocalKind.ASSOCIATION}:
            raise ValueError("fact owner must be a set or association")
        if isinstance(fact.value_type, IdentifierType):
            _required_local_ref(
                fact.value_type.set_ref, local_refs=local_refs, kind=FactLocalKind.SET
            )
    return terms


def _expressions(
    requested_fact: RequestedFact, *, local_refs: Mapping[str, FactLocalRef]
) -> dict[FactLocalRef, ExpressionNode]:
    return {local_refs[item.id]: item for item in requested_fact.expressions}


def _fact_domain(
    owner_ref: FactLocalRef,
    *,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> EvaluationDomain:
    if owner_ref.kind is FactLocalKind.SET:
        return RowDomain(owner_ref)
    if owner_ref.kind is FactLocalKind.ASSOCIATION:
        return AssociationExpandedDomain((owner_ref,))
    raise ValueError("invalid fact owner")


def _node_dependency_ids(node: ExpressionNode) -> tuple[str, ...]:
    if isinstance(node, BooleanComposition | Arithmetic):
        return node.argument_refs
    if isinstance(node, Comparison):
        return (node.left_ref, node.right_ref)
    if isinstance(node, NullCheck):
        return (node.argument_ref,)
    if isinstance(node, TemporalBucket):
        return (node.value_ref,)
    if isinstance(node, Aggregate):
        return (node.argument_ref,) + (
            () if node.filter_ref is None else (node.filter_ref,)
        )
    if isinstance(node, Quantify):
        return (node.condition_ref,)
    if isinstance(node, RelatedRow):
        return () if node.condition_ref is None else (node.condition_ref,)
    if isinstance(node, Coverage):
        return (
            ()
            if node.required_member_condition_ref is None
            else (node.required_member_condition_ref,)
        ) + (node.condition_ref,)
    raise TypeError(f"unsupported semantic expression {type(node).__name__}")


def _infer_node_type(
    node: ExpressionNode,
    *,
    dependencies: tuple[SemanticValueRef, ...],
    inferred_types: Mapping[SemanticValueRef, ValueType],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> ValueType:
    if not isinstance(node, Aggregate) and any(
        isinstance(item, FactLocalRef)
        and item.kind in {FactLocalKind.SET, FactLocalKind.ASSOCIATION}
        for item in dependencies
    ):
        raise ValueError("expression requires scalar operands")
    if isinstance(node, BooleanComposition):
        operator = {
            BooleanCompositionOperator.AND: "and",
            BooleanCompositionOperator.OR: "or",
            BooleanCompositionOperator.NOT: "not",
        }[node.operator]
        from fervis.lookup.expression_operators import (  # local: closes import cycle
            ExpressionBinaryOperator,
            ExpressionUnaryOperator,
        )

        typed_operator = (
            ExpressionUnaryOperator.NOT
            if operator == "not"
            else ExpressionBinaryOperator(operator)
        )
        if node.operator in {
            BooleanCompositionOperator.AND,
            BooleanCompositionOperator.OR,
        }:
            if len(dependencies) < 2:
                raise ValueError(f"{operator} requires at least two arguments")
            if any(
                not isinstance(inferred_types[item], BooleanType)
                for item in dependencies
            ):
                raise ValueError(f"{operator} requires Boolean arguments")
            return BooleanType()
        return infer_operator_result(
            typed_operator, tuple(inferred_types[item] for item in dependencies)
        )
    if isinstance(node, Comparison | Arithmetic | NullCheck):
        return infer_operator_result(
            node.operator,
            tuple(inferred_types[item] for item in dependencies),
        )
    if isinstance(node, TemporalBucket):
        argument_type = inferred_types[dependencies[0]]
        if not isinstance(argument_type, (DateType, DateTimeType, TemporalPointType)):
            raise ValueError("temporal bucket requires Date or DateTime")
        return argument_type
    if isinstance(node, Aggregate):
        argument = dependencies[0]
        if node.filter_ref is not None and not isinstance(
            inferred_types[dependencies[1]], BooleanType
        ):
            raise ValueError("aggregate filter must be Boolean")
        if isinstance(argument, FactLocalRef) and argument.kind is FactLocalKind.SET:
            if node.function.value != "count":
                raise ValueError("only count accepts a set argument")
            if node.distinct_argument:
                raise ValueError("count(SetTerm) cannot use distinct_argument")
            return IntegerType()
        return infer_aggregate_result(node.function.value, inferred_types[argument])
    if isinstance(node, Quantify | Coverage | RelatedRow):
        if dependencies and not isinstance(
            inferred_types[dependencies[0]], BooleanType
        ):
            raise ValueError("relational condition must be Boolean")
        return BooleanType()
    raise TypeError(f"unsupported semantic expression {type(node).__name__}")


def _infer_node_domain(
    node: ExpressionNode,
    *,
    dependencies: tuple[SemanticValueRef, ...],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
    local_refs: Mapping[str, FactLocalRef],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    subject_ref: FactLocalRef,
    grouping_refs: tuple[FactLocalRef, ...],
) -> EvaluationDomain:
    if isinstance(node, BooleanComposition) and node.operator in {
        BooleanCompositionOperator.AND,
        BooleanCompositionOperator.OR,
    }:
        shared_source = _shared_source_boolean_domain(
            tuple(domains[item] for item in dependencies),
            terms=terms,
        )
        if shared_source is not None:
            return shared_source
    if isinstance(node, Aggregate):
        if node.filter_ref is not None:
            filter_domain = domains[dependencies[1]]
            argument_ref = dependencies[0]
            if (
                isinstance(argument_ref, FactLocalRef)
                and argument_ref.kind is FactLocalKind.SET
            ):
                expected = RowDomain(argument_ref)
                if (
                    not isinstance(filter_domain, ConstantDomain)
                    and filter_domain != expected
                ):
                    raise ValueError("aggregate filter has an incompatible row domain")
            else:
                _combine_domains((domains[argument_ref], filter_domain), terms=terms)
        if not grouping_refs and _aggregate_is_correlated_to_subject(
            dependencies[0],
            subject_ref=subject_ref,
            local_refs=local_refs,
            domains=domains,
            terms=terms,
        ):
            return RowDomain(subject_ref)
        return AggregateDomain(grouping_refs)
    if isinstance(node, Quantify):
        over_ref = _required_local_ref(
            node.over_set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
        _validate_association_path(
            node.association_refs,
            local_refs=local_refs,
            terms=terms,
            destination=over_ref,
        )
        condition_domain = domains[dependencies[0]]
        correlated = isinstance(condition_domain, AssociationExpandedDomain) and (
            {ref.local_id for ref in condition_domain.path_refs}
            <= set(node.association_refs)
        )
        if not correlated and not _domain_belongs_to_subject(
            condition_domain, subject_ref=over_ref, terms=terms
        ):
            raise ValueError("quantifier condition has the wrong row domain")
        return (
            AggregateDomain(())
            if not node.association_refs
            else RowDomain(
                _path_source(
                    node.association_refs,
                    local_refs=local_refs,
                    terms=terms,
                    destination=over_ref,
                )
            )
        )
    if isinstance(node, RelatedRow):
        related = _required_local_ref(
            node.set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
        if len(node.association_refs) < 2:
            raise ValueError("related row requires at least two associations")
        association_terms = tuple(
            terms[
                _required_local_ref(
                    ref,
                    local_refs=local_refs,
                    kind=FactLocalKind.ASSOCIATION,
                )
            ]
            for ref in node.association_refs
        )
        if any(not isinstance(item, AssociationTerm) for item in association_terms):
            raise ValueError("related row references a non-association")
        sources = {
            item.from_set_ref
            for item in association_terms
            if isinstance(item, AssociationTerm) and item.to_set_ref == related.local_id
        }
        if len(sources) != 1 or len(association_terms) != len(node.association_refs):
            raise ValueError(
                "related-row associations must connect one source to one related set"
            )
        if dependencies and not _domain_belongs_to_subject(
            domains[dependencies[0]], subject_ref=related, terms=terms
        ):
            raise ValueError("related-row condition has the wrong row domain")
        return RowDomain(local_refs[next(iter(sources))])
    if isinstance(node, Coverage):
        candidate = _required_local_ref(
            node.candidate_set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
        required = _required_local_ref(
            node.required_dimension_set_ref,
            local_refs=local_refs,
            kind=FactLocalKind.SET,
        )
        observation = _required_local_ref(
            node.observation_set_ref, local_refs=local_refs, kind=FactLocalKind.SET
        )
        if len({candidate, required, observation}) != 3:
            raise ValueError("coverage set refs must be distinct")
        _validate_association_path(
            node.candidate_observation_association_refs,
            local_refs=local_refs,
            terms=terms,
            source=candidate,
            destination=observation,
        )
        _validate_association_path(
            node.dimension_observation_association_refs,
            local_refs=local_refs,
            terms=terms,
            source=required,
            destination=observation,
        )
        observation_condition = dependencies[-1]
        if (
            node.required_member_condition_ref is not None
            and not _domain_belongs_to_subject(
                domains[dependencies[0]], subject_ref=required, terms=terms
            )
        ):
            raise ValueError(
                "coverage required-member condition has the wrong row domain"
            )
        if not _domain_belongs_to_subject(
            domains[observation_condition], subject_ref=observation, terms=terms
        ):
            raise ValueError("coverage condition has the wrong row domain")
        return RowDomain(candidate)
    return _combine_domains(tuple(domains[item] for item in dependencies), terms=terms)


def _shared_source_boolean_domain(
    domains: tuple[EvaluationDomain, ...],
    *,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> RowDomain | None:
    expanded = tuple(
        item for item in domains if isinstance(item, AssociationExpandedDomain)
    )
    if len(set(expanded)) < 2:
        return None
    first_associations = tuple(terms[item.path_refs[0]] for item in expanded)
    if any(not isinstance(item, AssociationTerm) for item in first_associations):
        return None
    source_refs = {
        item.from_set_ref
        for item in first_associations
        if isinstance(item, AssociationTerm)
    }
    if len(source_refs) != 1:
        return None
    source_ref = next(iter(source_refs))
    source_local_ref = next(
        (
            ref
            for ref, term in terms.items()
            if isinstance(term, SetTerm) and term.id == source_ref
        ),
        None,
    )
    if source_local_ref is None:
        raise ValueError("association source set is undeclared")
    rows = tuple(item for item in domains if isinstance(item, RowDomain))
    if any(item.owner_ref != source_local_ref for item in rows):
        return None
    return RowDomain(source_local_ref)


def _validate_grouping_domains(
    domains: tuple[EvaluationDomain, ...],
    *,
    subject_ref: FactLocalRef,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    local_refs: Mapping[str, FactLocalRef],
) -> None:
    """A grouping tuple combines connected row keys, not scalar operands."""
    varying = tuple(item for item in domains if not isinstance(item, ConstantDomain))
    if not varying or any(
        not isinstance(item, (RowDomain, AssociationExpandedDomain)) for item in varying
    ):
        raise ValueError("grouping requires row-level values")
    for domain in varying:
        if isinstance(domain, RowDomain):
            owner = domain.owner_ref
        else:
            assert isinstance(domain, AssociationExpandedDomain)
            association = terms[domain.path_refs[0]]
            assert isinstance(association, AssociationTerm)
            owner = local_refs[association.from_set_ref]
        if owner != subject_ref and not _unique_association_path_refs(
            subject_ref,
            destination=owner,
            terms=terms,
            local_refs=local_refs,
        ):
            raise ValueError("grouping has an unrelated row domain")


def _combine_domains(
    domains: tuple[EvaluationDomain, ...],
    *,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> EvaluationDomain:
    nonconstant = tuple(
        item for item in domains if not isinstance(item, ConstantDomain)
    )
    if not nonconstant:
        return ConstantDomain()
    first = nonconstant[0]
    if all(item == first for item in nonconstant[1:]):
        return first
    expanded = tuple(
        item for item in nonconstant if isinstance(item, AssociationExpandedDomain)
    )
    rows = tuple(item for item in nonconstant if isinstance(item, RowDomain))
    if len(set(expanded)) == 1 and rows:
        expansion = expanded[0]
        members = {
            member
            for ref in expansion.path_refs
            for term in (terms[ref],)
            if isinstance(term, AssociationTerm)
            for member in (term.from_set_ref, term.to_set_ref)
        }
        if all(row.owner_ref.local_id in members for row in rows):
            return expansion
    if not expanded and len({row.owner_ref for row in rows}) == 2:
        owners = {row.owner_ref.local_id for row in rows}
        connections = tuple(
            ref
            for ref, term in terms.items()
            if isinstance(term, AssociationTerm)
            and {term.from_set_ref, term.to_set_ref} == owners
        )
        if len(connections) == 1:
            return AssociationExpandedDomain(connections)
    raise ValueError("expression mixes unrelated evaluation domains")


def _domain_belongs_to_subject(
    domain: EvaluationDomain,
    *,
    subject_ref: FactLocalRef,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> bool:
    if isinstance(domain, ConstantDomain):
        return True
    if isinstance(domain, RowDomain):
        return domain.owner_ref == subject_ref
    if isinstance(domain, AssociationExpandedDomain):
        first = terms[domain.path_refs[0]]
        return (
            isinstance(first, AssociationTerm)
            and first.from_set_ref == subject_ref.local_id
        )
    return False


def _result_grain(
    requested_fact: RequestedFact,
    *,
    subject_ref: FactLocalRef,
    grouping_refs: tuple[FactLocalRef, ...],
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
) -> ResultGrain:
    if grouping_refs:
        return Groups(grouping_refs)
    output_refs = tuple(
        _value_ref(item.expression_ref, local_refs=local_refs, inputs=inputs)
        for item in requested_fact.outputs
    )
    if any(
        isinstance(item, FactLocalRef) and item.kind is FactLocalKind.ASSOCIATION
        for item in output_refs
    ):
        raise ValueError("output requires a scalar value")
    set_outputs = tuple(
        item
        for item in output_refs
        if isinstance(item, FactLocalRef) and item.kind is FactLocalKind.SET
    )
    if any(item != subject_ref for item in set_outputs):
        raise ValueError("row output can project only the requested subject set")
    output_domains = tuple(
        RowDomain(subject_ref) if item in set_outputs else domains[item]
        for item in output_refs
    )
    if any(
        isinstance(item, (RowDomain, AssociationExpandedDomain))
        for item in output_domains
    ):
        return SubjectRows(subject_ref)
    return Singleton()


def _validate_result_surfaces(
    requested_fact: RequestedFact,
    *,
    result_grain: ResultGrain,
    grouping_refs: tuple[FactLocalRef, ...],
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
    inferred_types: Mapping[SemanticValueRef, ValueType],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
) -> None:
    output_refs = tuple(
        _value_ref(item.expression_ref, local_refs=local_refs, inputs=inputs)
        for item in requested_fact.outputs
    )
    if isinstance(result_grain, Singleton) and len(output_refs) != 1:
        raise ValueError("scalar requested fact requires exactly one output")
    for ref in output_refs:
        if isinstance(ref, FactLocalRef) and ref.kind is FactLocalKind.SET:
            if not isinstance(result_grain, SubjectRows):
                raise ValueError("set projection requires subject rows")
            continue
        _validate_result_ref(
            ref, result_grain=result_grain, grouping_refs=grouping_refs, domains=domains
        )
    for ordering in requested_fact.ordering:
        ref = _value_ref(ordering.expression_ref, local_refs=local_refs, inputs=inputs)
        _validate_result_ref(
            ref, result_grain=result_grain, grouping_refs=grouping_refs, domains=domains
        )
    if (
        isinstance(
            requested_fact.selection,
            (FirstRankWithTies, TakeWithBoundaryTies, PositionWithTies),
        )
        and not requested_fact.ordering
    ):
        raise ValueError("bounded selection requires ordering")
    if isinstance(result_grain, Singleton) and requested_fact.ordering:
        raise ValueError("scalar result cannot be ordered")
    if isinstance(requested_fact.selection, (TakeWithBoundaryTies, PositionWithTies)):
        input_term = inputs.get(requested_fact.selection.limit_input_ref)
        if input_term is None or not isinstance(input_term.value_type, IntegerType):
            raise ValueError("take limit must reference an integer input")
    if requested_fact.distinct_by:
        if isinstance(result_grain, Singleton | Groups):
            raise ValueError("distinct_by is invalid for scalar or grouped results")
        if tuple(requested_fact.distinct_by) != tuple(
            item.expression_ref for item in requested_fact.outputs
        ):
            raise ValueError("distinct_by must equal the complete output tuple")
    del inferred_types


def _validate_result_ref(
    ref: SemanticValueRef,
    *,
    result_grain: ResultGrain,
    grouping_refs: tuple[FactLocalRef, ...],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
) -> None:
    domain = domains[ref]
    if isinstance(result_grain, Groups) and not (
        ref in grouping_refs
        or isinstance(domain, (AggregateDomain, ResultDomain, ConstantDomain))
    ):
        raise ValueError("group result contains an ungrouped row value")
    if isinstance(result_grain, SubjectRows) and isinstance(
        domain, (AggregateDomain, ResultDomain)
    ):
        raise ValueError("row result contains a whole-population aggregate")


def _aggregate_is_correlated_to_subject(
    argument_ref: SemanticValueRef,
    *,
    subject_ref: FactLocalRef,
    local_refs: Mapping[str, FactLocalRef],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
) -> bool:
    if isinstance(argument_ref, str):
        return False
    if argument_ref.kind is FactLocalKind.SET:
        argument_set_ref = argument_ref
    else:
        domain = domains[argument_ref]
        if isinstance(domain, AssociationExpandedDomain):
            return _domain_belongs_to_subject(
                domain,
                subject_ref=subject_ref,
                terms=terms,
            )
        if not isinstance(domain, RowDomain):
            return False
        argument_set_ref = domain.owner_ref
    if argument_set_ref == subject_ref:
        return False
    return bool(
        _unique_association_path_refs(
            subject_ref,
            destination=argument_set_ref,
            terms=terms,
            local_refs=local_refs,
        )
    )


def _qualification_node(
    node: ExpressionNode,
    *,
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
) -> BooleanFormula | None:
    if isinstance(node, BooleanComposition):
        return BooleanFormula(
            operator=node.operator.value,
            argument_refs=tuple(
                _boolean_ref_token(item, local_refs=local_refs, inputs=inputs)
                for item in node.argument_refs
            ),
        )
    return None


def _boolean_requirements(
    requested_fact: RequestedFact,
    *,
    qualification: QualificationDNF,
    formulas: Mapping[str, BooleanFormula | None],
    local_refs: Mapping[str, FactLocalRef],
) -> tuple[BooleanRequirement, ...]:
    requirements = list(
        boolean_requirements(
            requested_fact_id=requested_fact.id,
            formula=qualification,
            use_site=BooleanRequirementUseSite.POPULATION,
            owner_expression_ref=None,
        )
    )
    for node in requested_fact.expressions:
        owner_ref = local_refs[node.id].token
        scoped: tuple[str, BooleanRequirementUseSite] | None = None
        if isinstance(node, Aggregate) and node.filter_ref is not None:
            scoped = (node.filter_ref, BooleanRequirementUseSite.AGGREGATE_FILTER)
        elif isinstance(node, Quantify):
            scoped = (
                node.condition_ref,
                BooleanRequirementUseSite.QUANTIFIER_CONDITION,
            )
        elif isinstance(node, RelatedRow) and node.condition_ref is not None:
            scoped = (
                node.condition_ref,
                BooleanRequirementUseSite.QUANTIFIER_CONDITION,
            )
        elif isinstance(node, Coverage):
            if node.required_member_condition_ref is not None:
                required_condition = normalize_qualification(
                    requested_fact_id=requested_fact.id,
                    root_ref=local_refs[node.required_member_condition_ref].token,
                    formulas=formulas,
                )
                requirements.extend(
                    boolean_requirements(
                        requested_fact_id=requested_fact.id,
                        formula=required_condition,
                        use_site=BooleanRequirementUseSite.COVERAGE_REQUIRED_MEMBER,
                        owner_expression_ref=owner_ref,
                    )
                )
            scoped = (node.condition_ref, BooleanRequirementUseSite.COVERAGE_CONDITION)
        if scoped is None:
            continue
        condition_ref, use_site = scoped
        condition = normalize_qualification(
            requested_fact_id=requested_fact.id,
            root_ref=local_refs[condition_ref].token,
            formulas=formulas,
        )
        requirements.extend(
            boolean_requirements(
                requested_fact_id=requested_fact.id,
                formula=condition,
                use_site=use_site,
                owner_expression_ref=owner_ref,
            )
        )
    return tuple(requirements)


def _input_use_sites(
    requested_fact: RequestedFact,
    *,
    inputs: Mapping[str, InputTerm],
    input_denotations: Mapping[str, InputDenotation],
    local_refs: Mapping[str, FactLocalRef],
    inferred_types: Mapping[SemanticValueRef, ValueType],
    direct: Mapping[FactLocalRef, frozenset[SemanticValueRef]],
) -> tuple[InputUseSite, ...]:
    uses: list[InputUseSite] = []
    for expression_ref, dependencies in direct.items():
        node = next(
            item
            for item in requested_fact.expressions
            if item.id == expression_ref.local_id
        )
        ordered = tuple(
            _value_ref(item, local_refs=local_refs, inputs=inputs)
            for item in _node_dependency_ids(node)
        )
        for index, dependency in enumerate(ordered):
            if not isinstance(dependency, str):
                continue
            denotation = _required_input_denotation(
                dependency, input_denotations=input_denotations
            )
            identity_peer = _identity_peer(
                ordered,
                index=index,
                inferred_types=inferred_types,
                local_refs=local_refs,
            )
            if denotation.kind is InputDenotationKind.IDENTITY_REFERENCE:
                if identity_peer is None:
                    raise ValueError(
                        "identity input must compare with an identifier fact"
                    )
                identity_set_ref = identity_peer
                reference_fact_ref = _identifier_fact_peer(
                    ordered,
                    index=index,
                    inferred_types=inferred_types,
                )
            else:
                if identity_peer is not None:
                    raise ValueError(
                        "non-identity input cannot supply identifier authority"
                    )
                identity_set_ref = None
                reference_fact_ref = None
            uses.append(
                InputUseSite(
                    use_ref=f"{requested_fact.id}:input_use:{expression_ref.local_id}:{index}",
                    input_ref=dependency,
                    operand_meaning=denotation.operand_meaning,
                    requested_fact_id=requested_fact.id,
                    structural_location=f"expression:{expression_ref.local_id}:argument:{index}",
                    expression_ref=expression_ref,
                    expected_value_type=inputs[dependency].value_type,
                    identity_set_ref=identity_set_ref,
                    reference_fact_ref=reference_fact_ref,
                )
            )
    if isinstance(requested_fact.selection, (TakeWithBoundaryTies, PositionWithTies)):
        input_ref = requested_fact.selection.limit_input_ref
        denotation = _required_input_denotation(
            input_ref, input_denotations=input_denotations
        )
        if denotation.kind is not InputDenotationKind.NON_IDENTITY_SCALAR:
            raise ValueError("result limit must be a non-identity scalar")
        uses.append(
            InputUseSite(
                use_ref=f"{requested_fact.id}:input_use:selection_limit",
                input_ref=input_ref,
                operand_meaning=denotation.operand_meaning,
                requested_fact_id=requested_fact.id,
                structural_location="selection:limit",
                expression_ref=None,
                expected_value_type=inputs[input_ref].value_type,
                identity_set_ref=None,
                reference_fact_ref=None,
            )
        )
    return tuple(uses)


def _required_input_denotation(
    input_ref: str,
    *,
    input_denotations: Mapping[str, InputDenotation],
) -> InputDenotation:
    denotation = input_denotations.get(input_ref)
    if denotation is None:
        raise ValueError("input use lacks denotation")
    return denotation


def _identity_peer(
    dependencies: tuple[SemanticValueRef, ...],
    *,
    index: int,
    inferred_types: Mapping[SemanticValueRef, ValueType],
    local_refs: Mapping[str, FactLocalRef],
) -> FactLocalRef | None:
    for peer_index, peer in enumerate(dependencies):
        if peer_index == index:
            continue
        peer_type = inferred_types[peer]
        if isinstance(peer_type, IdentifierType):
            return _required_local_ref(
                peer_type.set_ref, local_refs=local_refs, kind=FactLocalKind.SET
            )
    return None


def _identifier_fact_peer(
    dependencies: tuple[SemanticValueRef, ...],
    *,
    index: int,
    inferred_types: Mapping[SemanticValueRef, ValueType],
) -> FactLocalRef | None:
    for peer_index, peer in enumerate(dependencies):
        if peer_index == index or not isinstance(peer, FactLocalRef):
            continue
        peer_type = inferred_types[peer]
        if peer.kind is FactLocalKind.FACT and isinstance(peer_type, IdentifierType):
            return peer
    return None


def _output_requirement(
    requested_fact: RequestedFact,
    *,
    output_index: int,
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
    transitive: Mapping[FactLocalRef, frozenset[SemanticValueRef]],
) -> OutputRequirement:
    output = requested_fact.outputs[output_index - 1]
    output_ref = FactLocalRef(requested_fact.id, FactLocalKind.OUTPUT, output.id)
    value_ref = _value_ref(output.expression_ref, local_refs=local_refs, inputs=inputs)
    dependencies = (
        frozenset({value_ref})
        if isinstance(value_ref, str)
        else frozenset({value_ref, *transitive.get(value_ref, frozenset())})
    )
    return OutputRequirement(output_ref, value_ref, dependencies)


def _source_requirements(
    *,
    output_requirements: tuple[OutputRequirement, ...],
    subject_ref: FactLocalRef,
    qualification_ref: FactLocalRef | None,
    grouping_refs: tuple[FactLocalRef, ...],
    ordering: tuple[Ordering, ...],
    local_refs: Mapping[str, FactLocalRef],
    transitive: Mapping[FactLocalRef, frozenset[SemanticValueRef]],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    expressions: Mapping[FactLocalRef, ExpressionNode],
    domains: Mapping[SemanticValueRef, EvaluationDomain],
) -> frozenset[FactLocalRef]:
    # Scalar domains can be independent. Row-valued surfaces still need a
    # declared path connecting their owner to the requested subject.
    row_surfaces = (
        *tuple(item.value_ref for item in output_requirements),
        *grouping_refs,
        *tuple(local_refs[item.expression_ref] for item in ordering),
    )
    for surface in row_surfaces:
        domain = domains.get(surface)
        if isinstance(domain, RowDomain) and domain.owner_ref != subject_ref:
            if not _unique_association_path_refs(
                subject_ref,
                destination=domain.owner_ref,
                terms=terms,
                local_refs=local_refs,
            ):
                raise ValueError("result surface has an unrelated row domain")
    refs: set[FactLocalRef] = {subject_ref, *grouping_refs}
    explicitly_connected_sets: set[FactLocalRef] = set()
    for requirement in output_requirements:
        refs.update(
            item for item in requirement.dependencies if isinstance(item, FactLocalRef)
        )
    if qualification_ref is not None:
        refs.add(qualification_ref)
        refs.update(
            item
            for item in transitive.get(qualification_ref, frozenset())
            if isinstance(item, FactLocalRef)
        )
    for item in ordering:
        ref = local_refs[item.expression_ref]
        refs.add(ref)
        refs.update(
            dep
            for dep in transitive.get(ref, frozenset())
            if isinstance(dep, FactLocalRef)
        )
    for ref, fact_term in (
        (local_refs[item.id], item)
        for item in terms.values()
        if isinstance(item, FactTerm)
    ):
        if ref not in refs:
            continue
        refs.add(local_refs[fact_term.owner_ref])
    for ref in tuple(refs):
        expression_node = expressions.get(ref)
        if isinstance(expression_node, Quantify):
            refs.add(local_refs[expression_node.over_set_ref])
            refs.update(local_refs[item] for item in expression_node.association_refs)
        elif isinstance(expression_node, RelatedRow):
            related_set_ref = local_refs[expression_node.set_ref]
            refs.add(related_set_ref)
            explicitly_connected_sets.add(related_set_ref)
            refs.update(local_refs[item] for item in expression_node.association_refs)
        elif isinstance(expression_node, Coverage):
            refs.update(
                {
                    local_refs[expression_node.candidate_set_ref],
                    local_refs[expression_node.required_dimension_set_ref],
                    local_refs[expression_node.observation_set_ref],
                    *(
                        local_refs[item]
                        for item in expression_node.candidate_observation_association_refs
                    ),
                    *(
                        local_refs[item]
                        for item in expression_node.dimension_observation_association_refs
                    ),
                }
            )
    expanded = set(refs)
    for ref in tuple(refs):
        term = terms.get(ref)
        if isinstance(term, FactTerm):
            owner_ref = local_refs[term.owner_ref]
            expanded.add(owner_ref)
            if isinstance(term.value_type, IdentifierType):
                identity_set_ref = local_refs[term.value_type.set_ref]
                expanded.add(identity_set_ref)
                association_ref = _unique_identifier_association(
                    owner_ref,
                    identity_set_ref=identity_set_ref,
                    terms=terms,
                    local_refs=local_refs,
                )
                if association_ref is not None:
                    expanded.add(association_ref)
    for ref in tuple(expanded):
        if (
            ref.kind is FactLocalKind.SET
            and ref != subject_ref
            and ref not in explicitly_connected_sets
        ):
            expanded.update(
                _unique_association_path_refs(
                    subject_ref,
                    destination=ref,
                    terms=terms,
                    local_refs=local_refs,
                )
            )
    return frozenset(
        ref
        for ref in expanded
        if ref.kind
        in {FactLocalKind.SET, FactLocalKind.ASSOCIATION, FactLocalKind.FACT}
    )


def _unique_identifier_association(
    owner_ref: FactLocalRef,
    *,
    identity_set_ref: FactLocalRef,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    local_refs: Mapping[str, FactLocalRef],
) -> FactLocalRef | None:
    if owner_ref.kind is not FactLocalKind.SET or owner_ref == identity_set_ref:
        return None
    endpoint_refs = frozenset((owner_ref, identity_set_ref))
    matches = tuple(
        ref
        for ref, term in terms.items()
        if isinstance(term, AssociationTerm)
        and frozenset(
            (
                local_refs[term.from_set_ref],
                local_refs[term.to_set_ref],
            )
        )
        == endpoint_refs
    )
    if len(matches) != 1:
        raise ValueError(
            "set-owned identifier requires one declared association to its identity set"
        )
    return matches[0]


def _unique_association_path_refs(
    source: FactLocalRef,
    *,
    destination: FactLocalRef,
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    local_refs: Mapping[str, FactLocalRef],
) -> tuple[FactLocalRef, ...]:
    adjacency: dict[FactLocalRef, list[tuple[FactLocalRef, FactLocalRef]]] = {}
    for ref, term in terms.items():
        if not isinstance(term, AssociationTerm):
            continue
        left = local_refs[term.from_set_ref]
        right = local_refs[term.to_set_ref]
        adjacency.setdefault(left, []).append((ref, right))
        adjacency.setdefault(right, []).append((ref, left))

    paths: list[tuple[FactLocalRef, ...]] = []

    def visit(
        current: FactLocalRef,
        *,
        visited: frozenset[FactLocalRef],
        path: tuple[FactLocalRef, ...],
    ) -> None:
        if current == destination:
            paths.append(path)
            return
        for association_ref, next_set_ref in adjacency.get(current, ()):
            if next_set_ref in visited:
                continue
            visit(
                next_set_ref,
                visited=visited | {next_set_ref},
                path=(*path, association_ref),
            )

    visit(source, visited=frozenset((source,)), path=())
    if not paths:
        return ()
    if len(paths) != 1:
        raise ValueError(
            "source-backed set has ambiguous association paths from subject"
        )
    return paths[0]


def _boolean_ref_token(
    value_ref: str,
    *,
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
) -> str:
    if value_ref in local_refs:
        return local_refs[value_ref].token
    if value_ref in inputs:
        return value_ref
    raise ValueError(f"unknown Boolean reference: {value_ref}")


def _value_ref(
    value_ref: str,
    *,
    local_refs: Mapping[str, FactLocalRef],
    inputs: Mapping[str, InputTerm],
) -> SemanticValueRef:
    if value_ref in local_refs:
        ref = local_refs[value_ref]
        if ref.kind is FactLocalKind.ASSOCIATION:
            raise ValueError("association is not a value reference")
        return ref
    if value_ref in inputs:
        return value_ref
    raise ValueError(f"unknown value reference: {value_ref}")


def _required_local_scalar_ref(
    value_ref: str, *, local_refs: Mapping[str, FactLocalRef]
) -> FactLocalRef:
    ref = local_refs.get(value_ref)
    if ref is None:
        raise ValueError(f"unknown local scalar reference: {value_ref}")
    if ref.kind in {FactLocalKind.SET, FactLocalKind.ASSOCIATION}:
        raise ValueError(f"{value_ref} does not reference a scalar value")
    return ref


def _required_boolean_ref(
    value_ref: str,
    *,
    local_refs: Mapping[str, FactLocalRef],
    inferred_types: Mapping[SemanticValueRef, ValueType],
) -> FactLocalRef:
    ref = _required_local_scalar_ref(value_ref, local_refs=local_refs)
    if not isinstance(inferred_types.get(ref), BooleanType):
        raise ValueError("qualification must reference a Boolean value")
    return ref


def _required_local_ref(
    local_id: str,
    *,
    local_refs: Mapping[str, FactLocalRef],
    kind: FactLocalKind,
) -> FactLocalRef:
    ref = local_refs.get(local_id)
    if ref is None or ref.kind is not kind:
        raise ValueError(f"{local_id} must reference a {kind.value}")
    return ref


def _validate_association_path(
    association_ids: tuple[str, ...],
    *,
    local_refs: Mapping[str, FactLocalRef],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    source: FactLocalRef | None = None,
    destination: FactLocalRef | None = None,
) -> None:
    if not association_ids:
        return
    associations: list[AssociationTerm] = []
    for item in association_ids:
        term = terms[
            _required_local_ref(
                item, local_refs=local_refs, kind=FactLocalKind.ASSOCIATION
            )
        ]
        if not isinstance(term, AssociationTerm):
            raise ValueError("association path contains a non-association")
        associations.append(term)
    candidates = _association_path_endpoints(associations)
    if not any(
        (source is None or start == source.local_id)
        and (destination is None or end == destination.local_id)
        for start, end in candidates
    ):
        raise ValueError("association path does not connect the required sets")


def _association_path_endpoints(
    associations: list[AssociationTerm],
) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[str, str]] = []
    first = associations[0]
    for start, current in (
        (first.from_set_ref, first.to_set_ref),
        (first.to_set_ref, first.from_set_ref),
    ):
        for association in associations[1:]:
            if association.from_set_ref == current:
                current = association.to_set_ref
            elif association.to_set_ref == current:
                current = association.from_set_ref
            else:
                break
        else:
            candidates.append((start, current))
    return tuple(dict.fromkeys(candidates))


def _path_source(
    association_ids: tuple[str, ...],
    *,
    local_refs: Mapping[str, FactLocalRef],
    terms: Mapping[FactLocalRef, SetTerm | AssociationTerm | FactTerm],
    destination: FactLocalRef | None = None,
) -> FactLocalRef:
    associations = [terms[local_refs[item]] for item in association_ids]
    if any(not isinstance(item, AssociationTerm) for item in associations):
        raise ValueError("association path contains a non-association")
    endpoints = _association_path_endpoints(
        [item for item in associations if isinstance(item, AssociationTerm)]
    )
    matching = [
        start
        for start, end in endpoints
        if destination is None or end == destination.local_id
    ]
    if len(set(matching)) != 1:
        raise ValueError("association path source is ambiguous")
    return local_refs[matching[0]]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
