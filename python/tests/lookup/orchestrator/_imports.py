from __future__ import annotations

# ruff: noqa: F401

from collections import defaultdict
from dataclasses import dataclass, field
import json
import re
from typing import Any

from fervis import errors as api_errors
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    ComputeSpec,
    JoinSpec,
    Operation,
    NamedExpression,
    ProjectSpec,
    OrderSpec,
    RelationRole,
    RelationRoleRef,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
    RelationResultOutput,
    ResultProjection,
    ScalarResultOutput,
)
from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType
from fervis.lookup.answer_rendering import rendered_fact_payload
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    CONVERSATION_RESOLUTION_TOOL_NAMES,
)
from fervis.lookup.errors import ErrorCode
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    FactPlan,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
)
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_ROW_SOURCE_ID,
    api_row_source_id,
    read_field_evidence_ref,
    required_input_evidence_ref,
)
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.orchestration.pipeline import run_lookup_question
from fervis.lookup.orchestration.request import LookupRequest, LookupRuntimePorts
from fervis.lookup.outcomes.model import BlockedRequirementKind, OutcomeKind
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    KnownInputKind,
    KnownInputSource,
    LiteralInputRole,
    NormalInstanceExcludedStateRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactGroupKey,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
    RequestedFactOrderingDirection,
    ResultSelectionKind,
    default_answer_population,
    requested_fact_evidence_ref,
)
from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogFact,
    CatalogFactAvailability,
    CatalogParam,
    CompletenessPolicy,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    FieldRequirement,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.memory.addresses import (
    EvidenceRef,
    FactAddress,
    FactAddressValue,
    RelationSourceKind,
)
from fervis.memory.artifacts import FactOutcome, build_fact_artifact
from tests.lookup.source_binding_helpers import (
    bound_fact_plan_payload_from_fact_plan,
    plan_selection_payload_from_fact_plan,
    source_binding_payload_for_one_call,
    source_binding_payload_from_fact_plan,
    source_binding_target_id_for_candidate,
    source_candidate_answer_population,
    source_candidate_with_fields,
    source_fulfills_by_row_population_for_candidate,
    source_fulfills_for_candidate,
)


__all__ = tuple(name for name in globals() if not name.startswith("__"))
