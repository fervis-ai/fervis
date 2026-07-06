# Deterministic, Auditable LLM Answers over REST APIs

Fervis uses LLMs to interpret user questions through typed contracts, then
answers with framework-native API calls and deterministic computation, while
retaining full lineage for explainability.

Fervis is alpha software. It is useful for evaluation and early integration
work, but not yet recommended for production.

## Install

Fervis requires Python 3.11 or newer.

```bash
uv add fervis
```

Install the extra for your framework:

```bash
uv add "fervis[django]"
uv add "fervis[fastapi]"
uv add "fervis[flask]"
```

## Start

Follow the framework guide:

- [Django / DRF](python/Django.md)
- [FastAPI](python/FastAPI.md)
- [Flask](python/Flask.md)

The normal setup flow is:

```bash
fervis init --yes
fervis migrate
fervis doctor
```

Run `fervis runtime ask` only after `fervis doctor` passes.

## Models

Configure allowed providers and model keys in `config/fervis.json`. API keys
stay in environment variables.

OpenAI currently has the most reliable structured outputs for Fervis. We
recommend `gpt-5.4-mini` for testing.

## Goldsets

Goldsets let host teams run their own evaluation questions over their own API
data. A goldset is a directory containing `fervis_goldset.py`.

```python
from fervis.evaluation.goldsets import GoldsetCase, GoldsetMatch, GoldsetSuite


def load_suite():
    return GoldsetSuite(
        name="orders",
        cases=(
            GoldsetCase(
                case_id="orders_this_month",
                question="How many orders happened this month?",
            ),
        ),
        match_answer=match_answer,
    )


def match_answer(case, result):
    if result.status == "COMPLETED" and result.answer == "42":
        return GoldsetMatch(passed=True, message="matched answer")
    return GoldsetMatch(passed=False, message="expected answer 42")
```

Run it from the host API project root after `fervis doctor` passes. The suite
can be a directory containing `fervis_goldset.py` or an import entrypoint such
as `package.suite:load_suite`.

```bash
fervis goldset run \
  --suite ./goldsets/orders \
  --tenant-id <tenant-id> \
  --principal-id <principal-id> \
  --model openai:gpt-5.4-mini \
  --ledger-file .goldset-runs/orders.jsonl
```

Use `--case-ids case_a,case_b` to run specific cases. Use `setup_questions` on
`GoldsetCase` when a case needs prior conversation context; setup questions run
in the same conversation before the evaluated question.

For repeated local runs, put stable values in environment variables instead of
retyping them:

```bash
export FERVIS_GOLDSET_SUITE=package.suite:load_suite
export FERVIS_GOLDSET_CASE_IDS=case_a,case_b
export FERVIS_GOLDSET_TENANT_ID=<tenant-id>
export FERVIS_GOLDSET_PRINCIPAL_ID=<principal-id>

fervis goldset run --ledger-file .goldset-runs/orders.jsonl
```

Suites may define `preflight` on `GoldsetSuite` for setup checks that must pass
before any model call runs, such as host API reachability or oracle database
connectivity.

## Development

```bash
cd python
uv sync --extra all --extra dev
uv run fervis --help
```

## How to run tests

Run the full repo check from the repository root:

```bash
scripts/verify-fervis-repo.sh
```

Run only the Python package tests:

```bash
cd python
uv sync --extra dev
uv run ruff check src
uv run pytest
```

Run local host goldset cases against this repo checkout, with import-shadowing
checks and per-case progress:

```bash
scripts/run-local-goldset.sh \
  --project-root /path/to/host-api \
  --suite package.suite:load_suite \
  --case-ids case_a,case_b \
  --tenant-id <tenant-id> \
  --principal-id <principal-id>
```

The script also reads `FERVIS_GOLDSET_SUITE`, `FERVIS_GOLDSET_CASE_IDS`,
`FERVIS_GOLDSET_TENANT_ID`, and `FERVIS_GOLDSET_PRINCIPAL_ID`, so a configured
shell can run `scripts/run-local-goldset.sh` with no repeated identifiers.

Run only the desktop app tests:

```bash
cd desktop-app
npm ci
npm test
```

## Project Contributions

Fervis is an architecture/runtime, not a typical application codebase. Changes
must preserve deterministic execution, typed contracts, current-run proof, and
clear ownership across parser, grounding, source binding, planning, runtime, and
lineage boundaries.

### Bug Reports

Report bugs in the independent review format:

- **Severity:** High, Medium, or Low.
- **Category:** Bug, regression, architecture boundary, proof/lineage, test gap,
  or maintainability.
- **Description:** State the incorrect behavior precisely.
- **Impact:** Explain how the bug can produce a wrong answer, wrong proof,
  silent fallback, lost clarification, or hard-to-maintain code path.
- **Evidence:** Link concrete files, line numbers, tests, fixtures, or runtime
  output.
- **How to reproduce:** Provide the smallest question, contract payload,
  fixture, or command that shows the issue.
- **Structural cause:** Identify the owning boundary where the issue belongs.
  Do not stop at a downstream symptom.
- **Recommended solution:** Describe the root fix at the correct layer.
- **Required tests:** Name the conformance or behavior coverage that should
  prove the fix.
- **Checks run:** List commands run and results.

### Changes And PRs

Before opening a PR, make sure the change satisfies this bar:

- Replace obsolete behavior; do not add legacy support, parallel
  implementations, or compatibility shims.
- Keep the backend deterministic. It should parse declared contracts, verify
  ownership and lineage, ground values through explicit authority, and bind only
  current-run grounded values.
- Do not make the parser or backend a semantic oracle. Do not infer business
  meaning, catalog fields, params, endpoints, resolver choices, or language
  intent from raw text outside the owning contract/model boundary.
- Avoid brittle heuristics. If a rule is needed, make it bounded, typed, and
  owned by the correct layer.
- Keep public/private package boundaries clear. Prefer a few stable public
  interfaces over broad exposed internals.
- Use proper encapsulation. A future change to one concept should not require
  edits across many unrelated files.
- Follow **One logic, one home**. Do not duplicate business rules across parser,
  grounding, source binding, planning, runtime, testkit, or lineage.
- Avoid overengineering and bloat. Use the smallest implementation that is easy
  to read, prove, and maintain.
- Cover non-trivial behavior with tests that assert outcomes at the right
  boundary. Prefer conformance tests for architecture-level behavior and focused
  behavior tests for runtime boundaries. Tests should prove what Fervis must do,
  not lock in incidental implementation details.
- Do not rely on negative assertions as broad compatibility tests. Use exact
  expected contracts for migration-critical behavior.
- Run the relevant focused tests and the full repository verification before
  merge.

## License

Fervis is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
