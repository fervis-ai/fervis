# Fervis Test Migration Ledger

This ledger tracks the move from implementation-shaped Python tests to the
conformance structure described in
`docs/memo/2026-06-05-lookup-runtime-conformance-testing.md`.

Statuses:

- `migrated`: behavior is covered by a portable conformance case.
- `native_keep`: behavior is intentionally Python/framework infrastructure.
- `pending`: behavior still needs classification or migration.
- `delete_redundant`: behavior is covered elsewhere and the old test can go.

## Migrated

### Business Time Resolver

Portable algorithm cases live in:

```text
api/tests/fervis/conformance/cases/algorithms/business_time/
```

They replace portable behavior from:

```text
api/tests/fervis/business_time/test_resolver.py
```

The old file now keeps only native clock/timezone hook behavior:

```text
api/tests/fervis/business_time/test_resolver.py::test_default_anchor_uses_requested_timezone
```

### Relation Catalog Row-Source Projection

Portable algorithm cases live in:

```text
api/tests/fervis/conformance/cases/algorithms/relation_catalog/
```

They replace portable behavior from:

```text
api/tests/fervis/lookup/catalog/test_validation.py
api/tests/fervis/lookup/catalog/test_ports.py
```

The old Python files were deleted after the conformance cases passed.

### Fact Endpoint Requirements

Portable algorithm cases live in:

```text
api/tests/fervis/conformance/cases/algorithms/fact_requirements/
```

They replace portable behavior from:

```text
api/tests/fervis/lookup/fact_planning/test_fact_requirements.py
```

The old Python file was deleted after the conformance cases passed.

### Fact Plan Schema

Portable algorithm cases live in:

```text
api/tests/fervis/conformance/cases/algorithms/fact_plan_schema/
```

They replace portable behavior from:

```text
api/tests/fervis/lookup/fact_planning/test_grounded_schema.py
```

### Core Capabilities

Portable algorithm cases live in:

```text
api/tests/fervis/conformance/cases/algorithms/capabilities/
```

They replace portable behavior from:

```text
api/tests/fervis/core/test_capabilities.py
```

### Host API Endpoint Projection

Adapter conformance cases live in:

```text
api/tests/fervis/conformance/cases/adapters/host_api/
```

They are replacing public-interface behavior from:

```text
api/tests/fervis/host_api/test_relation_catalog_projection.py
```

This file is being migrated incrementally because several old tests combine
multiple public projection behaviors.

## Remaining Large Native Lookup Suites

The manifest currently classifies every live fervis test node. The files
below are still large native suites and should keep shrinking when a behavior
can be expressed through a stable conformance boundary:

```text
api/tests/fervis/lookup/test_source_binding_one_call.py
api/tests/fervis/lookup/test_pattern_fact_plan_contract.py
api/tests/fervis/lookup/test_conversation_resolution_pipeline.py
api/tests/fervis/lookup/orchestrator/test_pipeline.py
api/tests/fervis/lookup/orchestrator/test_memory_and_values.py
```

Keeping a test native is valid only when it tests Python/runtime
infrastructure, provider routing, framework-specific orchestration, or a
prompt-surface smoke boundary. Portable runtime, turn-contract, adapter, and
algorithm behaviors should live in conformance cases.

## Native Infrastructure Buckets

These areas should mostly stay native, with small focused tests:

```text
api/tests/fervis/interfaces/
api/tests/fervis/model_io/
api/tests/fervis/runtime/
```

## First-Class Adapter Buckets

These should migrate toward adapter conformance and framework-specific
introspection tests:

```text
api/tests/fervis/host_api/
```
