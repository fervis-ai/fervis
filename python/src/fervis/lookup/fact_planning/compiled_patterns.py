"""Typed intermediate results shared by pattern selection and compilation."""

from dataclasses import dataclass

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.operations import (
    AggregationFunction,
    Operation,
    SortDirection,
)
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    LiteralType,
)
from fervis.lookup.answer_program.expressions import Expression
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ScalarResultOutput,
)
from fervis.lookup.fact_planning.executable_support import RowPopulationBasis


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
class CompiledRank:
    direction: SortDirection
    limit: int
    limit_value_id: str

    def limit_expression(
        self,
        input_context: CompilerInputContext,
    ) -> Expression:
        if self.limit_value_id:
            return input_context.expression_for_value(self.limit_value_id)
        return ConstantRef(
            constant_id=f"rank-limit.{self.limit}",
            version_ref="rank@1",
            value=FactValue.literal(
                id=f"rank-limit.{self.limit}",
                literal_type=LiteralType.NUMBER,
                value=str(self.limit),
            ),
        )
