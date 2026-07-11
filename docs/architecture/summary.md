# Fervis Architecture Summary

Fervis is a runtime for deterministic LLM answers over REST APIs. It turns a
natural-language factual question into bounded model decisions, framework-native
API reads, deterministic computation, and auditable proof.

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
2. the configured API catalog;
3. the selected source reads;
4. the deterministic operation plan;
5. the proof graph and final answer outputs.

If Fervis cannot prove the answer, it must ask a clarification, return a factual
terminal outcome, or fail visibly.

## Hard Rules

1. Fail closed. Do not answer from unverified facts.
2. Keep model authority bounded. Models may author bounded text inside strict,
   parsed contracts. When a model refers to backend-owned entities, evidence,
   or legal alternatives, it may use only IDs and choices shown in the current
   prompt surface.
3. Execute deterministically. Backend code performs API reads, relation
   operations, time resolution, proof projection, and canonical rendering.
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
  response shape, and relation metadata.
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
-> terminalization
-> interface projection
```

`questions/` owns question and run lifecycle. `run_work/` owns queued execution,
worker attempts, and terminalization. `lookup/` owns factual interpretation,
planning, execution, and outcomes. Interfaces translate transport; they do not
create a second lifecycle or lookup path.

Within `LookupService`, the factual runtime executes this flow:

```text
question + conversation context
-> conversation resolution
-> question contract
-> query enrichment
-> resolver catalog selection
-> grounding
-> answer-read catalog selection
-> read eligibility
-> plan selection
-> source binding
-> fact planning
-> verification
-> compilation
-> source reads and relation operations
-> factual outcome classification
-> canonical answer outputs
```

Model turns stop at interpretation and bounded selection. Verification,
compilation, execution, classification, proof construction, and canonical
rendering are deterministic backend work.

Lineage recording is interleaved throughout the run, not appended after the
answer. Model turns, endpoint snapshots, source reads, compile steps, execution,
facts, clarification, and terminal results record their canonical artifacts as
they occur. Required lineage is fail-closed: a successful factual answer cannot
outlive failure to persist its audit truth.

## Model Turn Boundaries

| Turn | Model chooses | Backend owns |
| --- | --- | --- |
| Conversation resolution | Clause-level dependencies, context-frame choices, and the contextual meaning needed to interpret the current utterance. | Memory validation, visible context-frame projection, and activation boundaries. |
| Question contract | Requested facts, answer shapes, outputs, populations, and literal or row-set question inputs. | Stable IDs, catalog blindness, contract parsing, and preservation of uncompiled time text. |
| Query enrichment | Resource and entity search terms grounded in the current question contract. | Catalog and resolver recall. |
| Grounding | A shown resolver route or candidate when semantic selection is required. | Exact-literal verification, deterministic time resolution, resolver execution, canonical values, and ambiguity detection. |
| Read eligibility | Candidate-local retain/drop decisions plus relevant row paths and field evidence hints. | Read-card construction, candidate identity, structural filtering, and retained-read caps. |
| Plan selection | Which source strategy to evaluate. | Strategy construction from retained reads. |
| Source binding | Source invocations, answer population, params and omission, finite-choice effects, fulfillment evidence, metric fit, and role-qualified targets. | Review-scope construction, ID and param validation, target compatibility, deterministic bindings, and executable consistency. |
| Fact planning | A typed operation plan. | Parsing, verification, compilation, execution. |

`requested_facts` are immutable inside a run. Downstream turns satisfy,
clarify, or block them; they do not rewrite what the user asked.

## Operations

Fact planning emits typed plans. It does not emit Python.

Operation-family code under `lookup/operation_families/`,
`lookup/fact_planning/`, `lookup/fact_plan/`, and `lookup/plan_execution/` owns
operation support construction, schemas, parsing, verification hooks, compile
behavior, and execution behavior.

Adding a new operation should add one operation-family implementation and
portable outcome tests. It should not require changing unrelated model turns.

## Lineage And Proof

Lineage is the canonical audit spine:

```text
Conversation
  -> Question
    -> QuestionRun
      -> RunStep
      -> ModelCall / ModelCallUsage / RunArtifact
      -> CatalogEndpoint / SourceRead
      -> RequestedFact
        -> FactResult
          -> ExecutionProofGraph
          -> ClarificationRequest
          -> AnswerOutput
      -> RunResult
        -> Answer
          -> AnswerOutput
          -> AnswerPresentation
      -> MemoryArtifact
      -> RuntimeErrorDetail
```

`QuestionRun` identity and authored intent are immutable once written.
Clarification continuation, user-visible rerun or replay, and future
review/correction flows create a new run. Retriable worker or provider execution
attempts may remain within the same run and are distinguished by attempt metadata;
they do not rewrite the run's question or lineage identity.

`ExecutionProofGraph` is the proof truth. Views may project it by answer output,
step, source read, or input contribution, but they do not store another proof
truth.

## Clarification And Memory

Clarification is a continuation of the same question. The runtime records a
`needs_clarification` fact result, structured clarification details, a
`ClarificationRequest`, and then a new run when the user answers.
Clarification question text, options, and client-facing display are projections
of the structured clarification payload; lineage and storage must preserve the
payload rather than reducing it to prose.

Memory is not a shortcut around proof. Conversation resolution may activate
prior context, but any memory contribution to a new answer must enter the new
run as typed evidence with proof refs.

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
- UI-shaped data as canonical persistence;
- prompt contracts that require changing many tests for one turn-shape change.

When a concept becomes important enough to inspect, retry, explain, or audit, it
needs one first-class home. If two modules can both answer "what happened?", one
of them is wrong.
