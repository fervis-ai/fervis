"""Lookup runtime request and port models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from fervis.lookup.relation_catalog import (
    RelationCatalogProvider,
    RelationDataAccessPort,
)
from fervis.lookup.relation_catalog.selection import (
    DEFAULT_MAX_CATALOG_READS_PER_FACT,
)
from fervis.lookup.turn_prompts.context import HostPromptContext

if TYPE_CHECKING:
    from fervis.lookup.answer_program.persistence import (
        PriorProgramInvocationReader,
        ProgramInvocationBinding,
    )
    from fervis.lookup.fact_planning.request import RuntimeValueContext


class LookupProgressSink(Protocol):
    def emit(self, event: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class LookupRequest:
    question: str
    conversation_context: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    tenant_id: str = ""
    authority_ref: str = ""
    user_context: dict[str, Any] = field(default_factory=dict)
    provider_preferences: dict[str, Any] = field(default_factory=dict)
    max_thinking_tokens: int = 64
    max_catalog_reads_per_fact: int = DEFAULT_MAX_CATALOG_READS_PER_FACT
    runtime_values: RuntimeValueContext | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)
    active_attempt: int | None = None


@dataclass(frozen=True)
class LookupRuntimePorts:
    relation_catalog_port: RelationCatalogProvider
    data_access_port: RelationDataAccessPort
    planner_model_port: Any
    program_invocation_binding: ProgramInvocationBinding | None = None
    prior_program_invocations: PriorProgramInvocationReader | None = None
    lineage_step_sink: Any = None
    progress_sink: LookupProgressSink | None = None
    lineage_required: bool = False
    clock: Any = None
    policy_port: Any = None
