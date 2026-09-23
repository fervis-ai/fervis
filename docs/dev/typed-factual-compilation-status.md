# Typed factual compilation status

Fervis now authors a catalog-independent `RequestedFact` before choosing REST
sources. Source realization binds its sets, properties, associations and inputs
to current API contracts; deterministic operations produce the answer. The
normal runtime uses this path. Model-authored SQL is no longer its planning
contract.

The typed boundary preserves the requested count unit, output references,
grouping, qualification and ordering. Anonymous API records retain occurrence
identity without becoming nominal entities. Singular references are selected
from current data and guarded for uniqueness. A one-row settings value can be a
reference proxy without becoming an entity row. A required path parameter can
realize an addressed reference when the original supplied literal is bound to
that exact read. API pagination must establish complete traversal, or the read
is unavailable for complete-population claims.

Before source realization, the canonical input ledger is checked against the
original supplied input and its exact use sites. A raw reference name or member
set cannot be substituted under an existing input ID.

## Evidence so far

- Sentry WMS: named and categorical warehouse counts, duplicate-name
  population, and warehouse ranking passed through the ordinary CLI.
- Independent FastAPI: paginated scan and unpaginated measurement counts,
  default-site role and reporting-site proxy counts passed.
- After the staged-access change, the independent FastAPI pagination controls
  passed again through the ordinary CLI: four scans across numbered pages and
  three unpaginated measurements.
- Independent Django/DRF and Flask: standard user and Todo counts passed.
- Current CLI `doctor` passed on isolated FastAPI and Flask samples; a stale
  DRF quickstart mount was repaired by `fervis init --yes`, then `doctor` passed.
- Older isolated Ozana audit checkout, `supplies_01`: ordinary goldset CLI
  answered 5, matching the
  independent oracle. The persisted run compiled and executed a typed plan in
  104 seconds with nine model calls, zero source-access turns and $0.079 of
  model spend. The host JWT was carried through the encrypted delegated-read
  contract and reauthenticated as the same Django principal.
- The private Ozana suite's 281 oracles were computed offline against a fresh
  seeded database on the merged current-main host checkout, with each case's
  fixture rolled back before the next and no exceptions or `oracle_failed`
  results. The guarded independence command is tracked in the private suite.
  This checks fixture/oracle readiness, not assistant answers.
- The merged host checkout passed 3,683 distinct Ozana/accounts/brands Docker
  tests with no failures, and Fervis catalog inspection found 202 endpoints,
  payment-request offset pagination and sales-summary response-shape controls.
  All 262 goldset cases that name an expected GET endpoint have one in this
  catalog; the other 19 do not fix a source. Missing declared supply-balance
  fields can be inspected under current caller authority, while nested
  merch-balance fields are present in the catalog. This is structural source
  coverage, not a model-source-selection result.
  A separate local evaluation worktree passes `fervis doctor` and an
  authenticated Fervis read; it has no paid model receipt yet.
- Address-only FastAPI: integer, UUID and decimal path-addressed facility
  counts passed without a district-list endpoint.
- Anonymous collection control: two named district alternatives counted all
  five matching facilities; the two-reference Grounding boundary passed 10/10.
  Typed replay also covers aliases for the same record and a missing member.
- When a named collection member has multiple declared entity-key matches, a
  clarification choice now narrows only that member's current candidates. The
  typed guard rechecks the selected key and the original name on replay;
  mismatched key authorities fail before execution.
- A typed REST-binding regression proves that ranking related records by their
  average amount selects a different winner than ranking by sum, while returning
  the associated count. It also exercises duplicated observed parent rows.
- Serialized typed programs traverse dependent REST reads from keyed or
  anonymous complete parent rows, bind observed path values, avoid duplicate
  child calls, and count normal or empty populations without a SQL operation.
- Schema-free GET responses with a JSON scalar or scalar array now expose
  observed value rows, including nested arrays whose parent context has a
  colliding `value` field. Mixed object/scalar arrays and a changed response
  shape fail closed. No identity key is inferred from these values. The full
  typed count path now re-inspects a schema-free primitive array on replay and
  executes without SQL. The full repository verifier passed with 2,726 Python tests, Mypy over 633 source
  files, Ruff, installed-package checks, 93 desktop tests and desktop build.
- Framework mount regressions now prove that Fervis routes remain reachable
  ahead of host catch-all routes in Django, FastAPI and Flask. Django `doctor` rejects
  an existing shadowed mount, and `init` repairs its position.
- When an unannotated required path key admits many same-type parent fields,
  source-access discovery reviews parents with matching declared path, field
  and resource names first. Every compatible parent remains available and the
  model still must certify complete traversal; name similarity grants no key
  authority. The full verifier passed with 2,727 Python tests after this change.
- A selected schema-free GET with required parameters can now inspect the
  authorized current response using original, type-compatible question inputs
  used by the selected fact. A model turn maps meanings to parameters; the
  parser rejects invented or unrelated inputs. The compiled typed program must
  bind those same addresses, and saved-program replay re-inspects through the
  validated binding set. UUID, integer, number and decimal addressed facility
  counts pass typed compilation, execution and replay; an address-mismatch
  regression fails closed without SQL. The
  full repository verifier passed with 2,734 Python tests, Mypy over 634 source
  files, Ruff, installed distribution checks, 93 desktop tests and build.
- `fervis init` now replaces a retired `from fervis.flask import
  configured_fervis` in an existing Flask factory rather than adding a second,
  conflicting import. In an isolated MarketplaceOS copy with current Fervis,
  `init`, migration and `doctor --probe-read-context-key eval-principal` pass;
  doctor finds 18 readable GET endpoints.
  The sample's own suite has 51 passes and one review-count failure, reproduced
  unchanged in a baseline copy with Fervis removed. This is integration/setup
  evidence, not a live factual-answer receipt. A Fervis Flask adapter GET under
  `flask_principal:eval-principal` also returned HTTP 200 JSON with six
  categories from `api.list_categories`. Full repository verification
  after the import migration passed with 2,736 Python tests, Mypy over 634
  source files, Ruff, distribution checks, 93 desktop tests and build.
- Typed SUM, AVG and arithmetic no longer inherit the caller's Decimal context.
  A cancelling 38-digit amount previously produced a large wrong total under
  precision 6; exact cancellation, wide multiplication and stable division
  now pass at caller precisions 6, 28 and 50. Unsupported extreme precision
  fails explicitly. Full verification passed with 2,738 Python tests, Mypy
  over 635 source files, Ruff, distribution checks, 93 desktop tests and build.
- The corrected arithmetic advances the function-semantics compatibility
  version from 3 to 4. Previously saved programs with version 3 reject reuse
  before source reads and must be recompiled; their historical answer evidence
  remains readable. Updated conformance fixtures and a stale-version regression
  pass. Full verification passed with 2,739 Python tests, Mypy over 635 source
  files, Ruff, distribution checks, 93 desktop tests and build.
- Numeric question bindings no longer call context-sensitive
  `Decimal.normalize()`. A 38-digit supplied value previously became a
  rounded 6-digit value before planning when the caller's Decimal precision
  was 6. Exact text canonicalization now preserves all digits, retains
  equivalent-form normalization, and rejects an excessive decimal span.
  Full verification passed with 2,754 Python tests, Mypy over 635 source
  files, Ruff, distribution checks, 93 desktop tests and build.
- Provider token-cost and cached-input cost accounting now use an isolated
  decimal context. At caller precision 6, the old calculation raised
  `decimal.InvalidOperation` while quantizing a valid usage receipt; the new
  result equals a high-precision reference at precisions 6, 28 and 50. Full
  verification passed with 2,755 Python tests, Mypy over 635 source files,
  Ruff, distribution checks, 93 desktop tests and build.
- Complete duplicate anonymous reference rows can now offer a user choice
  based on observed scalar properties without promoting a sampled `id` or a
  query-local occurrence number to an entity key. The choice shows a minimal
  distinguishing property set but pins every present nonnullable scalar
  property. Typed replay rechecks those properties and the original literal;
  changed names, other properties, missing rows, forged source-field lineage,
  field types or display labels, and indistinguishable records fail closed. The
  answer-program schema advances
  to revision 27 so older saved programs recompile. Full
  verification passed with 2,761 Python tests, Mypy over 635 source files,
  Ruff, distribution checks, 93 desktop tests and build.
- A within-fact scalar aggregate can now feed a row predicate that filters a
  second aggregate. Typed controls compute AVG of values below or above the
  complete population AVG as 2 or 10, respectively. A plan omitting the scoped
  Boolean realization is rejected; singleton aggregate producers execute in
  dependency order, and independent one-row outputs combine without SQL.
  Compiler compatibility advances to `@55` to reject older saved plans that
  could have silently treated the filter as true. Full verification passed
  with 2,764 Python tests, Mypy over 635 source files, Ruff, distribution
  checks, 93 desktop tests and build.
- A typed local-day regression now covers the New York spring DST transition:
  two instants within March 8 and the first instant of March 9 fall into their
  correct local-day buckets. This preserves a SQL-era factual obligation in
  the typed runtime. Full verification passed with 2,740 Python tests, Mypy
  over 635 source files, Ruff, distribution checks, 93 desktop tests and build.
- Typed schema-free count/replay also covers three complete empty-object rows.
  The observed source has no scalar fields, yet the three row occurrences count
  correctly; no synthetic public property is introduced. Full verification
  passed with 2,741 Python tests, Mypy over 635 source files, Ruff,
  distribution checks, 93 desktop tests and build.
- Schema-free representation evidence is now tied to a fingerprint of the
  complete inspected argument map, without storing raw argument values in the
  catalog. Compilation rejects any required or optional argument mismatch;
  saved-program replay re-inspects with the full validated binding set. A
  no-required-argument read with a bound optional representation choice passes
  the replay test, while a plan inspected under the default representation and
  executed under that choice fails closed. Full verification passed with 2,742
  Python tests, Mypy over 635 source files, Ruff, distribution checks, 93
  desktop tests and build.
- The inspection-input grounding schema can also bind a fact-owned original
  input to an optional query argument, or explicitly omit that argument.
  It also offers declared finite response-shape or Boolean choices, rejecting
  any value outside the API's typed choice contract. Numeric choices written
  as catalog text parse to their declared scalar type; an internally
  inconsistent choice list cannot authorize a required probe.
  Selected schema-free reads with optional parameters now wait until that
  grounding opportunity before default-shape inspection; resolver-only reads
  retain their preflight path. Focused parser and preflight tests pass, and
  full verification passed with 2,748 Python tests, Mypy over 635 source files,
  Ruff, distribution checks, 93 desktop tests and build. The model-boundary
  stability and affected live-case gates remain outstanding.
- A captured-turn semantic assertion is prepared at
  `scripts/experiments/source_access/inspection_input_assertion.py`. Local
  controls reject a wrong original-input or declared-choice mapping without
  reconstructing the production prompt. Captured production invocations and
  paid 9/10 replay remain outstanding. Full verification after this assertion
  passed with 2,762 Python tests, Mypy over 635 source files, Ruff,
  distribution checks, 93 desktop tests and build.
- The repository verifier passed after typed dependent-read replay coverage:
  2,721
  Python tests, Mypy over 633 source files, Ruff, installed-package checks,
  93 desktop tests and the desktop build.
- A pull-request workflow now runs that full repository verifier with a
  placeholder model key and browser downloads disabled. Check the latest PR
  head's remote result separately from these local results.

## Release gates still open

The first isolated Ozana `supplies_01` run was interrupted after 24
source-access turns and $0.372 of new provider spend, before answer
compilation. A staged rerun reached execution with no source-access turns but
received HTTP 403 from Ozana's scope-aware JWT permission. A later run with
delegated authentication spent $0.079 and passed. The
source-access prompt previously carried every field of every parent candidate;
it now carries only fields structurally capable of supplying a required
argument, along with target identity fields. Access for unselected positive
reads is deferred until the first typed realization fails; a dependent-source
fallback is covered deterministically. Read Eligibility reviewed four positive
batches in the passing Ozana receipt. The current code attempts typed
realization after each batch and stops on a verified success. Representation
inspection and pagination interpretation are also deferred to each visited
batch. If a later read has no provably complete traversal, it is removed from
the current selection without dropping other candidates or crashing. The
reduced live cost of this last staging change is not yet measured. The authorization
handoff is proved for this host and case; other APIs and question classes still
need live coverage.

The 281-case live matrix and required repetitions have not passed on this
revision. Truly indistinguishable anonymous records remain ambiguous;
descriptive subqueries, cross-API
authority, and arbitrary REST completeness declarations need further proof.
Schema-free dependent reads whose required address comes only from an observed
parent row still lack structural inspection before source realization.
The new question-dependent inspection-input grounding turn has deterministic
parser and execution coverage but has not met the harness's 9/10 model-boundary
and affected-live-case promotion gates.
Schema-free reads whose optional argument changes response shape stop at the
inspection-address guard if an early resolver inspection or later source
binding uses different arguments. The optional pre-inspection binding path
needs captured model-boundary and end-to-end evidence before release.
Direct SQL-operation support remains in the program runtime even though the
model-SQL authoring route is retired; remove it after typed behavior parity.
The conservative local reconciliation of the original provider-credit budget
estimates about $0.046 remaining, so additional paid
matrix runs require a fresh budget check. Keep the PR draft and the README's
alpha designation until the complete Quality Bar and release gates pass.
