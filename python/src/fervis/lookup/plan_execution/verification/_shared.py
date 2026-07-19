"""Shared imports for fact-plan verification internals."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, replace

from fervis.lookup.relation_catalog.model import (
    CatalogFactAvailability,
    CatalogField,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    catalog_selection_evidence_ref,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.authorized_sources import (
    AuthorizedExecutionSources,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.answer_program.expression_instantiation import (
    instantiate_program_expressions,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    FactPlan,
    MissingCatalogChoiceInput,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
)
from .operation_invariants import verify_operation
from fervis.lookup.fact_planning.grounded_params import (
    unique_grounded_param_ids_by_row_source,
)
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AntiJoinSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    Operation,
    OrderSpec,
    NamedExpression,
    ProjectSpec,
    ProjectToKeySpec,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_planning.required_inputs import (
    clarifiable_required_inputs,
    grounded_required_input_ids,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    build_row_source_catalog,
    read_evidence_ref,
    read_field_evidence_ref,
    row_source_evidence_ref,
    row_source_field_evidence_ref,
    row_source_for_relation,
)
from fervis.lookup.fact_planning.value_validation import verify_value_contract
from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    LiteralValuePayload,
)
from fervis.lookup.question_contract import QuestionContract, RequestedFact
