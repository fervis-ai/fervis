"""Shared imports for source-binding candidate internals."""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog, RowCardinality
from fervis.lookup.fact_planning.available_relations import (
    available_relation_catalog_payload,
    operation_input_values_payload,
)
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourceAppliedFilter,
)
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceField,
    build_row_source_catalog,
    row_sources_for_read_id,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralType,
    LiteralValuePayload,
    ValueKind,
    TimeValuePayload,
    known_input_id_for_value,
)
from fervis.lookup.source_binding.model import (
    BoundSource,
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
)
from fervis.lookup.source_binding.param_values import canonical_param_value

SourceCandidateInputRequest = SourceCandidateDiscoveryRequest | SourceBindingRequest
