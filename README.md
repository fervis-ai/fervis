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

Run it from the host API project root after `fervis doctor` passes:

```bash
fervis goldset run \
  --suite-path ./goldsets/orders \
  --tenant-id <tenant-id> \
  --principal-id <principal-id> \
  --model openai:gpt-5.4-mini \
  --ledger-file .goldset-runs/orders.jsonl
```

Use `--case-ids case_a,case_b` to run specific cases. Use `setup_questions` on
`GoldsetCase` when a case needs prior conversation context; setup questions run
in the same conversation before the evaluated question.

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

Run only the desktop app tests:

```bash
cd desktop-app
npm ci
npm test
```

## License

Fervis is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
