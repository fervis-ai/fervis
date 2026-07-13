# Small Trusted Kernel Architecture

Fervis aims to become the open source verifiable and auditable layer between
natural-language factual questions and relational evidence. Reaching that goal
requires a small trusted kernel, not a core that accumulates every connector,
framework integration, prompt variant, provider protocol, and host business
rule.

“Small” means minimizing the trusted computing base, not minimizing line count
or moving essential correctness elsewhere. External components may supply
mechanisms and domain assertions, but the kernel must own every invariant
required for an `answered` outcome to be trustworthy.

The governing principle is:

> The kernel should be ontology-neutral, not semantics-blind.

It must not define what “employee salary,” “order revenue,” or a host lifecycle
value means. It must require explicit, attributable evidence for those meanings
whenever they affect an answer.

## Trust Model

The kernel accepts explicit factual, semantic, source, and authority contracts;
compiles only authorized evidence reads; deterministically derives canonical
answer outputs; and proves that those outputs fulfill the requested facts.

```text
QuestionContract
+ SemanticCatalogSnapshot
+ SourceCatalogSnapshot
+ AuthorityEnvelope
-> VerifiedPlan
-> RelationReadRequests
-> RelationEvidence
-> deterministic relation operations
-> AnswerOutputs + ExecutionProofGraph
```

This diagram is target-state architecture. v0.1.4 preserves explicit typed
interpretation controls but does not yet implement `SemanticCatalogSnapshot` or
independently governed host-semantic version checking. Until that boundary exists, the
kernel must not present hashes derived from a saved program as current semantic
authority. The future saved-pin/current-registry rule is documented in
`biggest-strength-and-weakness.md`.

If an essential link is missing, ambiguous, unauthorized, incomplete, or cannot
be represented in proof, Fervis must clarify, return a typed non-answer outcome,
or fail visibly. It must not manufacture certainty from names, sample rows,
model rationale, or adapter success.

The kernel cannot prove a person’s private intention. It can preserve one
explicit commitment to what the question means and prove that downstream work
does not silently drift from that commitment.

## Architectural Layers

The kernel is not the whole Fervis product:

| Layer | Responsibilities |
| --- | --- |
| Semantic frontend and runtime shell | Conversation resolution, model turns, retrieval, bounded selection, clarification, run lifecycle, retries, usage, and orchestration. |
| Trusted kernel | Canonical contracts, parsing, semantic sufficiency, authority obligations, plan verification and compilation, deterministic relation semantics, canonical outcomes and outputs, and proof. |
| Providers and adapters | Host semantic definitions, source discovery and execution, frameworks, credentials and host policy, model protocols, persistence, interfaces, and evaluation tooling. |

Dependencies point toward the kernel:

```text
semantic frontend / runtime shell -> kernel contracts
providers / adapters             -> kernel contracts
kernel                            -X-> provider, framework, host, or UI code
```

“Outside the kernel” does not mean outside the product or outside the trust
model. Model and adapter outputs are untrusted inputs to the boundary. They must
be parsed into canonical representations and cannot mint factual success merely
by being schema-valid.

## Factual And Semantic Contracts

The kernel owns the canonical question and requested-fact representation,
including:

- requested outputs and answer shapes;
- answer population and counted unit;
- metric, grouping, comparison, ranking, and limit obligations;
- literal, grounded, temporal, and prior-row inputs;
- fact-local ownership of inputs and outputs;
- immutable fact identity within a run.

The semantic frontend may propose this contract from natural language. The
kernel owns its parser and invariants. Downstream phases may satisfy, clarify,
or block the requested fact; they may not rewrite it.

Host-specific semantic definitions live outside the kernel. The kernel owns the
versioned semantic-evidence contract through which those definitions become
admissible, including assertions about:

- stable identities and display fields;
- source and relation grain;
- population and lifecycle-state meaning;
- metric definitions, units, and valid aggregations;
- parameter choice, default, and omission meaning;
- temporal fields, validity windows, and as-of behavior;
- governed relationships and safe joins;
- completeness requirements and known limitations.

Semantic providers own the definitions. The kernel owns assertion identity,
versioning, provenance, applicability, reference integrity, and sufficiency.
Models may map user wording to shown assertions and compose legal choices from
them. Model rationale is useful lineage, but it is not independent certification
of the model’s choice.

Missing or conflicting required semantic evidence must produce clarification, a
typed non-answer outcome, or refusal. Backend code must not replace it with
endpoint-name, field-name, fixture, or host-product heuristics.

## Relation Evidence And Authority

The kernel owns a source-neutral logical protocol for cataloged relation
producers. It does not require APIs, database tables, ERP objects, and governed
views to have identical physical invocation semantics.

A verified read request carries, as applicable:

- source identity, kind, contract version, and capabilities;
- selected fields and expected grain;
- typed bindings, filters, ordering, limits, and page policy;
- required completeness and consistency;
- authority, policy, and resource-budget obligations.

An accepted result is a `RelationEvidence` envelope, not bare rows. It carries:

- normalized `RelationRows` and typed schema;
- actual grain and cardinality;
- completeness, truncation, and pagination state;
- snapshot, freshness, or consistency information;
- an executed-request receipt and catalog/source-contract references;
- authority, policy-decision, and proof references;
- partial or terminal read state.

REST adapters compile endpoint paths and parameters; SQL adapters compile safe
projections and predicates; ERP adapters use object-specific APIs. Those
mechanics remain outside the kernel. Whether their evidence is complete enough
to prove a total, maximum, absence, coverage, or ranking claim is a kernel
decision.

Authentication and host policy implementations also live outside the kernel,
but authority propagation is mandatory. The kernel binds the host decision to:

- catalog visibility and source/field admissibility;
- mandatory predicates, limits, and compilation;
- every source-read request;
- derivation or output checks required by host policy;
- proof.

The host decides what a principal may read and reauthorizes host-data access.
Fervis enforces its allowlists and plan limits and refuses to proceed without the
required host decision. Models may neither author nor broaden authority.
Individually permitted reads do not automatically make every join, count,
aggregate, comparison, or released output permissible.

## Plans, Outcomes, And Proof

The relational IR should be small and orthogonal, not vaguely defined. Every
admitted operation must have exact, versioned semantics for the cases it
supports, including nulls, duplicates, decimals, comparison and tie behavior,
timezones and intervals, join cardinality, grain, and completeness propagation.

Core operations may include projection, filtering, joins, anti-joins, set and
coverage operations, grouping, aggregation, ranking, ordering, limits, and
generated or prior-evidence relations.

An operation-family extension must provide a closed IR, parser, verifier,
compiler or executor semantics, proof projection, and portable conformance
cases. Models never emit SQL or Python for factual derivation; they select and
compose bounded typed choices that the kernel verifies and compiles.

The kernel owns typed factual outcomes and canonical answer-value
materialization. Outcomes include answered, needs clarification, proven empty
data, insufficient or incomplete evidence, policy blocked, unsupported plan,
and visible failure.

Every `AnswerOutput` is bound to the requested output it fulfills and to its
upstream proof. Canonical rendering is deterministic. Interfaces may choose
layout, labels, tables, and accessibility presentation, but may not add factual
claims or reconstruct canonical values outside the proof.

For each successful answer output, proof retains the path through:

- the immutable requested fact and contextual inputs;
- semantic assertions and catalog/source-contract snapshots;
- authority and policy decisions;
- source-read requests and relation evidence;
- population, metric, grain, and completeness choices;
- deterministic operations and the canonical value.

The obligation is stronger than “where did this value come from?” It is:

> Why does this value, derived from these authorized and sufficient relations,
> fulfill this exact requested output?

A successful answer cannot survive missing or invalid required proof.

Proof, lineage, and audit remain distinct:

- the kernel constructs and validates the execution proof graph;
- the runtime records chronological lineage and refuses terminal success if
  required audit truth cannot be persisted;
- persistence adapters store canonical records;
- explanation and audit views project them without creating another truth store.

## Runtime And Adapter Responsibilities

The runtime shell coordinates the kernel without defining factual truth. It owns
conversation and continuation handling, model orchestration, retrieval and
bounded selection, clarification, lifecycle and queues, retries, usage,
operational observability, lineage recording, and port composition.

Memory is not a shortcut around the kernel. A prior contribution enters a new
run as a typed input or relation-evidence contribution with proof references.

Implementations outside the kernel include:

- framework adapters and application interfaces;
- REST, SQL, ERP, memory, and generated-source discovery and execution;
- host semantic catalogs and governed-definition stores;
- credential acquisition and host policy engines;
- LLM provider protocols and structured-output transports;
- persistence implementations and presentation renderers;
- goldset loaders, runners, and reports.

These implementations use versioned kernel contracts, declare capabilities,
fail visibly on unsupported obligations, and pass portable conformance tests.
The kernel proves derivation from admissible provider evidence; it cannot prove
that an external system or host declaration corresponds to external reality.
Provider trust and conformance must therefore be explicit.

## Boundary Test

A responsibility belongs in the kernel when one or more of these are true:

1. It must be universal for an `answered` outcome to be trustworthy.
2. It defines a canonical contract that must survive persistence, replay, or
   provider replacement.
3. It parses or verifies untrusted model, adapter, catalog, policy, or source
   output.
4. It changes deterministic relation, answer, or proof semantics.
5. Multiple implementations would create competing truths about what happened
   or why an answer is valid.

A responsibility normally belongs outside when it is framework-, source-,
host-, domain-, provider-, storage-, or interface-specific and can be replaced
without changing canonical factual or proof semantics. Specificity never permits
bypassing kernel obligations.

Small-core discipline means:

- one canonical owner and representation per factual concept;
- closed, versioned contracts at every trust boundary;
- deterministic derivation wherever semantic judgment is unnecessary;
- no host-product assumptions, field-name heuristics, or generated computation;
- no duplicated proof, lineage, outcome, or persistence truth;
- no hidden compatibility paths that weaken current invariants;
- no cartesian-product model contracts when a parser can compose bounded IDs.

It does not mean a semantics-blind kernel, a lowest-common-denominator row DTO,
optional policy, adapter-owned validation, model self-certification, a diminished
product ecosystem, or avoidance of irreducible correctness complexity.

## General-Source Gate

This document defines a target dependency boundary, not a claim that the current
package layout already satisfies it. The existing REST path should be its first
proof: the frontend produces canonical contracts; the kernel verifies a
source-neutral plan and emits authorized read requests; a REST adapter returns
relation evidence; and the kernel produces canonical outputs plus proof without
endpoint concepts leaking back into it.

Only then should additional first-class source kinds broaden the runtime. Every
database, ERP, view, metric, memory, or generated source must preserve the same:

- factual and semantic obligations;
- authority boundary;
- source capability and completeness evidence;
- deterministic relation and answer semantics;
- proof and terminal-outcome rules.

Adding a source kind should add a catalog provider and executor, not a second
planning, verification, execution, or proof architecture.

## Final Rule

> The kernel does not define host business meaning or host policy. It requires
> versioned evidence for both, verifies their applicability, compiles only
> authorized plans, deterministically derives canonical outputs, and preserves
> proof of every obligation required for those outputs to answer the requested
> facts.
