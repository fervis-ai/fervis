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
- Isolated Ozana `supplies_01`: ordinary goldset CLI answered 5, matching the
  independent oracle. The persisted run compiled and executed a typed plan in
  104 seconds with nine model calls, zero source-access turns and $0.079 of
  model spend. The host JWT was carried through the encrypted delegated-read
  contract and reauthenticated as the same Django principal.
- The private Ozana suite's 281 oracles were computed offline against the
  isolated fixture database with no exceptions or `oracle_failed` results.
  This checks fixture/oracle readiness, not assistant answers.
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
- Framework mount regressions now prove that Fervis routes remain reachable
  ahead of host catch-all routes in Django and FastAPI. Django `doctor` rejects
  an existing shadowed mount, and `init` repairs its position.
- The repository verifier passed after delegated Django reads and goldset
  credential preparation: 2,715
  Python tests, Mypy over 633 source files, Ruff, installed-package checks,
  93 desktop tests and the desktop build.
- A pull-request workflow now runs that full repository verifier with a
  placeholder model key and browser downloads disabled. Its remote check has
  not yet reported on this PR revision.

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
revision. Anonymous-record ambiguity without a stable declared key,
descriptive subqueries, cross-API
authority, and arbitrary REST completeness declarations need further proof.
Direct SQL-operation support remains in the program runtime even though the
model-SQL authoring route is retired; remove it after typed behavior parity.
The conservative local reconciliation of the original provider-credit budget
estimates about $0.046 remaining, so additional paid
matrix runs require a fresh budget check. Keep the PR draft and the README's
alpha designation until the complete Quality Bar and release gates pass.
