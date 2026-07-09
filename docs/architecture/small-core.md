# Small Core Architecture

For Fervis to achieve its long-term goal, which is to become the open source
verifiable and auditable layer between natural-language factual questions and
relational evidence, the core engine must remain very small. Practically, that
means Fervis should become a **small proof/planning kernel**, not a growing pile
of connectors, source-specific logic, prompt variants, and business semantics.

The core should own only the things that must be universally true:

```text
question contract
source catalog contract
bounded model choices
relational plan IR
safe source-read request
RelationRows
deterministic operations
proof / lineage / audit
policy enforcement hooks
```

Everything else should live outside the core:

```text
Django / FastAPI / Flask adapters
REST endpoint discovery
DB table discovery
ERP connectors
semantic catalogs
auth provider integrations
LLM provider details
UI / CLI rendering
goldset loading
host-specific business meaning
```

The rule should be:

> If a feature is specific to a framework, source system, host app, business domain, model provider, or UI, it is not core.

For the long-term goal, “small core” matters because Fervis needs to support many evidence sources without becoming many systems glued together. REST APIs, DB tables, ERP objects, metrics, and governed views should all compile down into the same few primitives:

```text
source -> fields -> bindings -> filters -> rows -> proof
```

The core should not know what “employee salary,” “order revenue,” or “customer churn” means. It should know:

```text
This source is allowed.
These fields are selectable.
These filters are required.
This requester has this authority.
This read produced these rows.
This answer is backed by this proof.
```

So the practical design pressure is:

1. **Keep the relational IR tiny.**
   Projection, filter, join, group, aggregate, order, limit, maybe time/as-of later.

2. **Keep source access behind one contract.**
   API reads and DB reads should both become `RelationReadRequest -> RelationRows`.

3. **Move all source-specific complexity to adapters.**
   REST params, SQL compilation, ERP object APIs, pagination, auth quirks.

4. **Make policy mandatory at the read boundary.**
   Tenant, principal, source allowlist, required filters, limits, audit.

5. **Make proof a core invariant.**
   If the engine cannot prove where an answer came from, it should not answer.

6. **Avoid semantic ambition in core.**
   Fervis can use semantic catalogs later, but the kernel should not become one.

In short:

> Fervis core should be the place where factual questions become constrained relational evidence with proof. Everything else should be an adapter, catalog provider, executor, or evaluation harness.
