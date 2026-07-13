# Fervis Architecture Summary

Fervis is a compiler and runtime for verifiable factual answers over operational
APIs. It turns a natural-language question into bounded model decisions, compiles
a successful interpretation into an immutable `AnswerProgram`, and invokes that
program through authorized source reads, deterministic computation, and auditable
proof.

Fervis currently supports REST APIs as its first source of relational evidence.
Its long-term goal is to become the open source verifiable and auditable layer
between natural-language factual questions and relational evidence: API responses
today, and later database tables, views, and other governed sources.

Fervis is a constrained factual-answer runtime, not a general chatbot or
open-ended agent loop. Models interpret bounded prompt surfaces; backend code
owns catalog projection, validation, compilation, execution, persistence, proof,
usage accounting, and interfaces.

## Product Promise

Every factual answer should be explainable from:

1. the question and conversation context;
2. the model's parsed resolution and planning decisions;
3. the configured source catalog;
4. the invoked `AnswerProgram` and typed bindings;
5. the current authorized source reads;
6. the proof graph and final answer outputs.

If Fervis cannot prove the answer, it must ask a clarification, return a factual
terminal outcome, or fail visibly.

## Hard Rules

1. Fail closed. Do not answer from unverified facts.
2. Keep model authority bounded. Models may author bounded text inside strict,
   parsed contracts. When a model refers to backend-owned entities, evidence,
   or legal alternatives, it may use only IDs and choices shown in the current
   prompt surface.
3. Compile once and invoke deterministically. A successful model-assisted
   question produces an immutable `AnswerProgram` and initial typed bindings.
   Initial answers, compatible prior-question continuations, and deterministic
   reruns use the same verification and invocation path. Backend code performs
   source reads, relation operations, time resolution, proof projection, and
   canonical rendering.
4. Keep one audit truth. Lineage is canonical for facts, source reads, proof,
   answers, clarification, and history. Operational lifecycle responses and
   events may project question/run-work state; audit, explanation, and historical
   views project lineage and do not rebuild it.
5. Do not become a semantic oracle. Runtime code must not infer domain meaning
   from endpoint names, field names, fixture names, or host-product assumptions.
6. Use domain-neutral language for internal mechanics. Runtime contracts and
   package names should say source, endpoint, fact, answer, value, context,
   relation, and proof. Model-facing language may remain concretely business-
   oriented where experiments show that abstraction changes meaning; it must
   still avoid host-product assumptions and backend-invented semantics.
7. Prefer small closed contracts. When a prompt grows, remove duplicated context
   before adding new instructions.
8. Parse, do not merely validate. At each boundary, convert less-structured
   input into a more precise Fervis-owned representation that carries the proof
   of its invariants forward. Do not scatter guards that check a value and then
   leave downstream code with the same ambiguous shape. Invalid states should be
   unrepresentable, or at least unreachable except through a parser. This follows
   Alexis King's "Parse, don't validate":
   https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/
9. Do not encode cartesian products in model-facing contracts. If a model can
   compose one valid choice from bounded IDs, expose the IDs and parse the
   composed result into a typed backend representation. Enumerating every
   combination in prompts or JSON schemas is brute force, grows explosively,
   and moves deterministic contract ownership out of the parser.

## Semantic Authority

The kernel should be ontology-neutral, not semantics-blind. It must not define
what revenue, an active employee, a completed sale, or a host field means. Those
meanings ultimately require explicit, attributable host-owned assertions; model
rationale and schema-valid self-attestation are not independent semantic
authority.

Fervis does not yet implement a first-class governed semantic registry.
`AnswerProgram` compatibility pins currently protect executable schema,
function, compiler, and source contracts; they must not be presented as proof
that a host's business definition is still current. Future governed-semantic
compatibility must compare a version pinned by the saved program with a current
version obtained independently from the host-owned registry. Deriving both
sides from the saved program would be meaningless self-comparison.

## Project Configuration

`config/fervis.json` is the canonical project config. It declares:

- schema version;
- framework;
- routes prefix;
- host metadata, including the timezone used for relative-date grounding;
- source allowlist;
- model providers and allowed model keys;
- environment-scoped persistence and model defaults.

`config/fervis_auth.json` is the canonical auth/read-context config. It records
how Fervis captures a host-owned read context and how reads execute later as
that context. It is generated through `fervis auth configure`, not hand-edited
as Python code.

`FERVIS_ENV` selects an environment. If it is not set, Fervis uses
`default_environment`.

## Onboarding Gate

`fervis init` creates the JSON config, records configured sources, and safely
patches framework mounting when the host code shape is provable. If a patch is
not safe, it blocks with a specific next action.

`fervis migrate` prepares Fervis persistence.

`fervis doctor` is the onboarding gate. It validates config, framework mounting,
catalog discovery, auth configuration, persistence, and configured source
readiness. For Flask, doctor also certifies response contract cardinality against
runtime responses so list endpoints are not misread as single-object endpoints.

Only run `fervis runtime ask` after doctor passes.

## Host API Boundary

`host_api/` exposes framework-neutral contracts:

- `EndpointContract` describes method, path template, params, capabilities,
  response shape, and relation metadata, including declared candidate keys and
  entity references.
- `CatalogEndpointContract` records framework endpoint identity observed during
  introspection.
- `ReadContextRef` is the persisted non-secret host-owned handle used to
  reauthorize reads during queued work.
- `ReadAuthority` is the durable Fervis authority for state reads and host reads:
  tenant id plus the submitted read context and, when configured, an encrypted
  delegated read credential.
- `ReadInvocation` is the selected safe read: endpoint name plus path, query,
  and page policy.
- `HostApiAdapter` describes configured sources, captures read context, and
  executes selected reads through that context.

The visible catalog is the configured source allowlist. Fervis does not derive
endpoint visibility from roles or scopes. The host API enforces row and endpoint
visibility when `execute_read(authority, invocation)` runs.

Some hosts require the original request credential to pass through their normal
authorization middleware. Fervis may capture only explicitly configured headers,
encrypt the short-lived delegated credential, and replay it through the same
compiled read-request path. The host still reauthorizes the read; Fervis does not
turn delegated credentials into its own role or permission system.

`in_process` and `http` transports return the same read result contract. Their
authentication and transport details are implementation details behind the host
API adapter.

## Framework Adapters

Fervis currently has first-class Python adapters for:

- Django + DRF;
- FastAPI;
- Flask.

Django/DRF and FastAPI primarily use runtime framework artifacts to describe
routes and execute reads. Static source inspection is only for safe init/mount
edits, not catalog truth.

Framework introspection may expose identity only when framework or model
metadata establishes the complete candidate key or foreign-key relationship.
A path parameter's name, type, or position is not identity authority, and a
response model must not inherit identity from an unrelated model merely because
they share a base class.

Flask has less native introspection. Fervis supports Flask runtime route
discovery plus contract surfaces such as OpenAPI/Swagger, Marshmallow-backed
schemas, JSON:API metadata, and Flask-AppBuilder metadata. A configured Flask
GET route is either read-eligible with a deterministic contract or doctor blocks.

## Question Runtime

External interfaces call one framework-neutral lifecycle:

```text
interface request
-> question admission, authority, and idempotency
-> Question + immutable QuestionRun
-> queued RunWork
-> LookupService
-> answer, factual terminal, visible failure, or waiting for clarification
-> interface projection
```

`questions/` owns question and run lifecycle. `run_work/` owns queued execution,
worker attempts, and terminalization. `lookup/` owns factual interpretation,
planning, execution, and outcomes. Interfaces translate transport; they do not
create a second lifecycle or lookup path.

The runtime has two closed execution specifications:

- `ResolveQuestionRunSpec` starts a model-assisted run from natural language;
- `RerunProgramSpec` starts a deterministic run from a persisted program
  invocation prepared with complete typed bindings.

`QuestionRun.kind` records that execution-level distinction as
`model_assisted` or `deterministic`. A model-assisted run may either compile a
new question or invoke a compatible prior question after conversation
resolution. `ProgramInvocation.kind` records the more precise program origin:
`compiled_question`, `continue_prior_request`, or `rerun_program`.

A model-assisted run follows this flow:

```text
question + visible conversation context
-> optional conversation resolution
-> either:
     raw question + any typed resolution
       -> question contract
       -> query enrichment and catalog selection
       -> grounding
       -> read eligibility
       -> plan selection and source binding
       -> fact planning
       -> AnswerProgram compilation + initial BindingSet
   or:
     canonical callable prior frame
       -> ground changed arguments
       -> persisted AnswerProgram + complete BindingSet
-> program verification and instantiation
-> fresh authorized source reads and deterministic relation operations
-> factual outcome classification
-> canonical answer outputs and proof
```

A deterministic rerun starts after natural-language interpretation:

```text
answered ProgramInvocation
-> typed binding patch or declared capability application
-> new complete BindingSet and, when needed, ProgramRevision
-> new immutable QuestionRun + ProgramInvocation
-> program verification and instantiation
-> fresh authorized reads and deterministic relation operations
-> canonical answer outputs and proof
```

Model turns stop at semantic interpretation and bounded selection. Backend code
parses their closed contracts, resolves backend-owned IDs, compiles the selected
meaning, and owns verification, execution, classification, proof construction,
and canonical rendering. Conversation resolution's contextualized prose is a
reasoning and audit artifact; downstream turns receive the raw question plus
typed resolved meaning, never a freehand integrated rewrite.

Lineage recording is interleaved throughout the run, not appended after the
answer. Model turns, endpoint snapshots, source reads, compile steps, execution,
facts, clarification requests and responses, and terminal results record their
canonical artifacts as they occur. Required lineage is fail-closed: a successful
factual answer cannot outlive failure to persist its audit truth.

## Model Turn Boundaries

| Turn | Model chooses | Backend owns |
| --- | --- | --- |
| Conversation resolution | Complete resolved clauses, attributed values, retained fixed frame parts, an optional complete call to a shown prior frame, or a structured unresolved outcome. | Canonical context sources and frames, source-reference parsing, frame-call signature checking, persisted-program loading, and the execution-path split. |
| Question contract | Requested facts, answer shapes, outputs, populations, and literal or row-set question inputs from the raw question plus typed prior meaning. | Stable IDs, catalog blindness, contract parsing, omission of duplicate current-only values, and preservation of uncompiled time text. |
| Query enrichment | Resource and entity search terms grounded in the current question contract. | Catalog and resolver recall. |
| Grounding | A shown resolver route or candidate when semantic selection is required. | Exact-literal verification, deterministic time resolution, resolver execution, canonical values, and ambiguity detection. |
| Read eligibility | Candidate-local retain/drop decisions plus relevant row paths and field evidence hints. | Read-card construction, candidate identity, structural filtering, and retained-read caps. |
| Plan selection | Which source strategy to evaluate. | Strategy construction from retained reads. |
| Source binding | Source invocations, answer population, params and omission, finite-choice effects, fulfillment evidence, metric fit, and role-qualified targets. | Review-scope construction, ID and param validation, target compatibility, deterministic bindings, and executable consistency. |
| Fact planning | A typed operation plan. | Parsing the planning IR and compiling one closed `AnswerProgram` plus initial bindings. |

`requested_facts` are immutable inside a run. Downstream turns satisfy,
clarify, or block them; they do not rewrite what the user asked.

## Answer Programs And Operations

Fact planning emits typed planning IR. It does not emit Python and is not the
persisted executable contract. Deterministic compilation produces an immutable,
canonically serialized, content-addressed `AnswerProgram` containing:

- requested-fact templates and output fulfillment;
- typed parameter declarations and declared revision capabilities;
- relation and read templates;
- a typed deterministic operation graph;
- canonical render instructions;
- schema, compiler, function, and source-contract compatibility pins.

A `BindingSet` closes every required program parameter. A `ProgramInvocation`
records one program, one complete binding set, its run, its invocation kind, and
its base invocation when derived. Binding patches and program revisions are
immutable derivation records; they never mutate a prior invocation.

Compilation and invocation are deliberately separate. Instantiation verifies a
program and bindings against the current execution environment before any source
read. Invocation then performs fresh authorized reads, deterministic relation
operations, fact materialization, rendering, and proof construction. The first
answer, a callable-frame continuation, and a deterministic rerun all use this
same path.

An entity-valued result carries its declared entity kind, candidate-key ID, and
complete key components. Verification preserves that exact authority through
relation operations and rejects result projections that rename the entity kind,
key, or components. Human-facing display fields are not substitutes for this
computational identity.

A deterministic rerun uses no model calls. It may reuse the same bindings, apply
a typed binding patch, or apply a capability declared by the program. It still
reads current evidence, so deterministic means stable execution meaning—not a
promise that mutable source data yields the same numeric answer. Programs or
values that depend on conversation memory are not admitted to this rerun path.

Operation-family code under `lookup/operation_families/`,
`lookup/fact_planning/`, `lookup/fact_plan/`, `lookup/answer_program/`, and
`lookup/plan_execution/` owns operation support construction, schemas, parsing,
compilation, verification, and execution.

Adding a new operation should add one operation-family implementation and
portable outcome tests. It should not require changing unrelated model turns or
introducing another executor beside `AnswerProgram` invocation.

## Lineage And Proof

Lineage is the canonical audit spine:

```text
Conversation
  -> Question
    -> QuestionRun
      -> RunStep
        -> ClarificationRequest
          -> ClarificationResponse
      -> ModelCall / ModelCallUsage / RunArtifact
      -> CatalogEndpoint / SourceRead
      -> ProgramInvocation (BindingSet / BindingPatch)
        -> AnswerProgram
        -> ProgramRevision
      -> RequestedFact
        -> FactResult
          -> ExecutionProofGraph
          -> AnswerOutput
      -> RunResult
        -> Answer
          -> AnswerOutput
          -> AnswerPresentation
      -> MemoryArtifact
      -> RuntimeErrorDetail
```

`QuestionRun` identity, execution kind, trigger, and ancestry are immutable once
written. A clarification makes the current run wait; its typed response resumes
another execution attempt inside that run. A deterministic rerun creates a new
run on the same `Question`; an ordinary conversational follow-up creates a new
`Question` in the same conversation. Retriable worker or provider execution
attempts remain within one run and are distinguished by attempt metadata; they
do not rewrite question, run, or invocation identity.

`QuestionRun.kind` answers whether the run began from natural language or a
closed program invocation. It intentionally groups full compilation and
callable prior-question continuation as model-assisted, because both use
conversation resolution. `ProgramInvocation.kind` preserves their conceptual
difference. When one exists, frontends use the latest successful model-assisted
run as the question's primary run; deterministic reruns remain selectable
variants and do not silently replace conversation memory.

`ExecutionProofGraph` is the proof truth. Views may project it by answer output,
step, source read, or input contribution, but they do not store another proof
truth.

## Clarification And Memory

Clarification is a waiting state inside the same question run, not a terminal fact
result and not a child run. The producing step records a structured
`ClarificationRequest`; run work enters `WAITING_FOR_CLARIFICATION` without writing a
`RunResult`, `FactResult`, or answer. A typed `ClarificationResponse` belongs to
that request and run, requeues the same run, and begins its next attempt. The run
gets exactly one terminal result after it eventually answers, reaches another
factual terminal, or fails visibly.

Clarification question text, options, and client-facing display are projections
of the structured clarification payload. Lineage preserves the payload rather
than reducing it to prose. A selected option carries its canonical value and
grounding authority into the resumed attempt; the response fragment is not
treated as a fresh natural-language question or reconstructed through
conversation memory.

Memory is not a shortcut around proof. Successful prior requests project typed
context sources and backend-owned canonical frames. A frame separates fixed
question-shape parts—subject, outputs, population, and grouping—from bindable
values such as identities, time scopes, limits, and row sets. When its program
permits later binding, it also exposes a callable signature whose persisted
program fixes the computation.

Conversation resolution uses model reasoning to produce complete resolved
clauses with attributed values, retained fixed frame parts, and optionally one
complete call to a shown prior frame. The backend checks only structural claims:
copied spans, visible source IDs, canonical frame-part IDs, complete arguments,
and persisted-program identity. It does not compare domain nouns, infer edits,
or decide whether the follow-up means the same fact.

A valid frame call bypasses question contract and fact planning, grounds only
changed arguments, and invokes the prior program. A shape-changing follow-up
continues through the ordinary model-assisted compiler using the raw question
plus typed resolved meaning. Any prior-context contribution to the new answer
remains explicit in lineage and proof.

## Interfaces

Interfaces are adapters, not audit stores.

- `interfaces/cli/` owns the `fervis` command surface.
- `interfaces/django/` owns Django + DRF HTTP views, serializers, throttles,
  security, and composition.
- `interfaces/fastapi/` owns FastAPI route adapters.
- `interfaces/flask/` owns Flask route adapters.
- `interfaces/common/` owns shared projections.
- `run_work/` owns package-level queued worker behavior.

Operational interface payloads such as accepted, queued, running, and active-run
conflict project question lifecycle and run-work state. Historical and factual
read views return answers, compact explanation, verbose explanation, usage when
available, clarification requests, and runtime failure details from typed lineage
views rather than reconstructing history from events. When a run needs
clarification, interface views return the structured clarification details
recorded in lineage so clients can render questions, choices, and evidence
consistently.

## Model IO And Usage

`model_io/` owns provider routing, structured output, pricing lookup, usage
recording, and provider-specific adapters. Project config declares providers and
allowed model keys. API keys stay in environment variables.

Usage includes provider, model key, input tokens, output tokens,
reasoning tokens when available, duration, and cost when pricing data is known.

## Goldsets

Goldsets are host-owned evaluation suites. They run questions through the same
question lifecycle as `fervis runtime ask`, against the host API's configured
sources and data. A goldset may define setup questions to create conversation
context before an evaluated question. Its oracle evaluates the runtime result;
it must not call lookup internals or host APIs directly.

## Command Surface

The CLI is an adapter over typed services. Important command groups include:

```text
fervis init
fervis auth configure
fervis migrate
fervis doctor
fervis catalog
fervis runtime ask
fervis worker
fervis explain
fervis inspect
fervis usage
fervis model
fervis goldset run
```

Commands return agent-readable structured results and precise next actions.
Next actions should be minimal, usually one action and never noisy.

## Package Map

```text
django/                     Django public integration imports
fastapi/                    FastAPI public integration imports
flask/                      Flask public integration imports
project/                    Config, init, doctor, mounting, persistence
host_api/                   Framework-neutral API contracts and adapters
questions/                  Question lifecycle ports and orchestration boundary
run_work/                   Queued worker execution
lookup/                     Factual question compiler/runtime
lookup/answer_program/      Canonical program, bindings, revisions, invocation
lineage/                    Canonical audit models, payloads, and views
memory/                     Conversation memory projection
model_io/                   Provider routing, structured output, pricing
observability/              Lineage-backed inspect, usage, prompt views
interfaces/                 CLI and framework HTTP adapters
evaluation/goldsets/        Path-loaded runtime goldset runner
storage/                    SQL-backed storage helpers
```

## Testing Direction

Most behavior tests should be outcome-level and portable. YAML conformance cases
define stable inputs and expected outcomes. Native Python tests cover framework
adapters, persistence, CLI wiring, provider boundaries, and migration behavior.

Answer-program compilation, canonicalization, patching, instantiation,
invocation, question lifecycle, and projection are language-neutral boundaries
and belong in conformance cases. Live goldsets prove model-assisted composition
against real host APIs; repeated full-case runs are stability tests for model
stability, not deterministic `QuestionRun` reruns.

Prompt-rule tests should check required sections, schema fields, IDs, and
contract wording in one place per turn. They should not assert incidental prose.

## What Must Stay Out

The runtime must not contain:

- host-product-specific rules;
- endpoint-name heuristics masquerading as semantic meaning;
- generated Python for factual derivation;
- interface events as audit truth;
- duplicated proof stores;
- hidden compatibility shims;
- a second executor for initial answers or continuations beside answer-program
  invocation;
- freehand integrated-question rewrites used as downstream contracts;
- text heuristics or edit taxonomies that make backend code a conversation
  semantic oracle;
- UI-shaped data as canonical persistence;
- prompt contracts that require changing many tests for one turn-shape change.

When a concept becomes important enough to inspect, retry, explain, or audit, it
needs one first-class home. If two modules can both answer "what happened?", one
of them is wrong.
