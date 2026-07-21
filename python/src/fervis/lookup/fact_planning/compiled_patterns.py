"""Typed intermediate results shared by pattern selection and compilation."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.operations import (
    AggregationFunction,
    KeepAll,
    Operation,
    OrderSelection,
    SortDirection,
    Take,
)
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    LiteralType,
)
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ScalarResultOutput,
)
from fervis.lookup.fact_planning.executable_support import RowPopulationBasis
from fervis.lookup.question_contract import (
    RequestedFact,
    RequestedFactOrderingDirection,
    ResultSelectionKind,
)


@dataclass(frozen=True)
class CompiledPattern:
    fulfillment: tuple[FactFulfillment, ...]
    relations: tuple[Relation, ...]
    operations: tuple[Operation, ...]
    relation_outputs: tuple[RelationResultOutput, ...]
    scalar_outputs: tuple[ScalarResultOutput, ...]


@dataclass(frozen=True)
class PatternAddress:
    requested_fact_id: str
    answer_output_ids: tuple[str, ...]
    plan_shape: str
    source_binding_id: str


@dataclass(frozen=True)
class CompiledMetric:
    field_id: str
    row_population_basis: RowPopulationBasis | None
    label: str
    output_field_id: str
    function: AggregationFunction
    answer_output_id: str


@dataclass(frozen=True)
class CompiledOrdering:
    direction: SortDirection
    selection: OrderSelection

    @classmethod
    def from_requested_fact(
        cls,
        fact: RequestedFact,
        *,
        input_context: CompilerInputContext,
    ) -> CompiledOrdering | None:
        expression = fact.answer_expression
        if expression is None or expression.ordering_direction is None:
            return None
        direction = (
            SortDirection.ASC
            if expression.ordering_direction
            is RequestedFactOrderingDirection.ASCENDING
            else SortDirection.DESC
        )
        if expression.selection_kind is ResultSelectionKind.ALL_RESULTS:
            selection: OrderSelection = KeepAll()
        elif expression.selection_kind is ResultSelectionKind.TAKE_ONE:
            selection = Take(
                ConstantRef(
                    constant_id="selection.take-one",
                    version_ref="selection@1",
                    value=FactValue.literal(
                        id="selection.take-one",
                        literal_type=LiteralType.NUMBER,
                        value="1",
                    ),
                )
            )
        elif expression.selection_kind is ResultSelectionKind.TAKE:
            selection = Take(
                input_context.expression_for_question_input(
                    expression.limit_input_ref
                )
            )
        else:
            raise ValueError("ordered fact requires a result selection")
        return cls(direction=direction, selection=selection)
