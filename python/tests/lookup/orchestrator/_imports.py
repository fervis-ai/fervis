from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import re
from typing import Any

from fervis import errors as api_errors

from fervis.lookup.errors import ErrorCode
from fervis.lookup.orchestration.pipeline import run_lookup_question
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogFact,
    CatalogFactAvailability,
    CatalogParam,
    CompletenessPolicy,
    EndpointRead,
    FieldRequirement,
    IdentityMetadata,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.fact_plan.fact_plan import (
    AnswerPlan,
    BlockedFact,
    BlockedFactBasis,
    FactFulfillment,
    FactPlan,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
)
from fervis.lookup.fact_plan.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    ComputeSpec,
    JoinSpec,
    Operation,
    ProjectField,
    ProjectSpec,
    RankSpec,
    RelationRole,
    RelationRoleRef,
)
from fervis.lookup.fact_plan.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
    RenderSpec,
)
from fervis.lookup.fact_plan.row_sources import (
    api_row_source_id,
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_ROW_SOURCE_ID,
    read_field_evidence_ref,
    required_input_evidence_ref,
)
from fervis.lookup.fact_plan.values import (
    RowFilterUse,
    ScalarInputUse,
    ValueFilterOperator,
    ValueUse,
)
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.question_contract import (
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
    RequestedFactKnownInput,
    default_answer_population,
    requested_fact_evidence_ref,
)
from fervis.lookup.outcomes.model import (
    BlockedRequirementKind,
    OutcomeKind,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    CONVERSATION_RESOLUTION_TOOL_NAMES,
)
from fervis.lookup.answer_rendering import (
    rendered_fact_payload,
)
from fervis.memory.addresses import (
    EvidenceRef,
    FactAddress,
    RelationSourceKind,
)
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from tests.lookup.source_binding_helpers import (
    bound_fact_plan_payload_from_fact_plan,
    plan_selection_payload_from_fact_plan,
    source_candidate_answer_population,
    source_candidate_with_fields,
    source_fulfills_by_row_population_for_candidate,
    source_fulfills_for_candidate,
    source_binding_payload_for_one_call,
    source_binding_payload_from_fact_plan,
)

__all__ = tuple(name for name in globals() if not name.startswith("__"))
