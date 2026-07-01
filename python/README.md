# Deterministic, Auditable LLM Answers over REST APIs

This is the Python package for Fervis. It provides framework adapters for
Django + DRF, FastAPI, and Flask.

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

From the host API project root:

```bash
fervis init --yes
fervis migrate
fervis doctor
```

Follow every `doctor` next action until it passes. Then ask a factual question:

```bash
fervis runtime ask "How many orders happened this month?"
```

Framework-specific guides:

- [Django / DRF](Django.md)
- [FastAPI](FastAPI.md)
- [Flask](Flask.md)

## Goldsets

Goldsets run internal evaluation questions through Fervis over the host API and
host data. Create a suite directory with `fervis_goldset.py`:

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

Run the suite from the host API project root:

```bash
fervis goldset run \
  --suite-path ./goldsets/orders \
  --tenant-id <tenant-id> \
  --principal-id <principal-id> \
  --model openai:gpt-5.4-mini \
  --ledger-file .goldset-runs/orders.jsonl
```

Use `--case-ids case_a,case_b` for a subset. Add `setup_questions` to a
`GoldsetCase` when a test needs prior conversation context.

## Development

```bash
uv sync --extra all --extra dev
uv run fervis --help
```

## How to run tests

From this `python/` directory:

```bash
uv sync --extra dev
uv run ruff check src
uv run pytest
```

## License

Fervis is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
