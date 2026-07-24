# Model-call stability harness

The model-call stability harness replays one model boundary without rerunning
the rest of Fervis. Use it to understand a failure, test one controlled contract
change, and establish stability before changing production code.

The harness is an experiment boundary, not a substitute for focused tests or
end-to-end cases. A production change is admissible only after all three agree:

1. the isolated boundary passes its semantic assertion;
2. focused production tests pass; and
3. live cases pass through the complete pipeline.

## Inputs

`scripts/run-model-step-stability.py` accepts either:

- `--index`: an `index.json` captured by `fervis debug prompts`, plus `--step`;
- `--boundary-file`: a standalone boundary produced by a builder in
  `scripts/experiments/`.

A boundary contains the exact system prompt, user prompt, tool specifications,
provider, model key, and assertion context for one model turn. Replaying it does
not regenerate upstream model outputs.

An assertion module exposes:

```python
def validate(arguments: dict, context: dict) -> list[str]:
    ...
```

It returns an empty list for PASS and one or more precise semantic failures for
FAIL. Provider and strict-schema errors fail before the assertion runs.

## Replaying a captured turn

First capture prompts for a completed run:

```bash
scripts/local-fervis.sh debug prompts --run-id <run-id> --output-dir /tmp/prompts
```

Then replay one purpose ten times:

```bash
scripts/run-local-model-step-stability.sh \
  --index /tmp/prompts/index.json \
  --step <model-turn-purpose> \
  --runs 10 \
  --workers 1 \
  --assertion-file scripts/experiments/<boundary>_assertion.py \
  --output-jsonl /tmp/<boundary>-results.jsonl \
  --label <experiment-name>
```

Use `--sequence` when one run contains more than one turn with the same purpose.
Keep `--workers 1` for diagnosis unless concurrency is itself under test. Serial
calls make provider behavior, rate limits, and result ordering easier to
attribute.

## Standalone boundary wrappers

The checked-in wrappers build a production-equivalent standalone boundary and
run the shared harness. Examples include:

```bash
scripts/run-local-semantic-query-enrichment-stability.sh --runs 10
scripts/run-local-semantic-grounding-stability.sh --runs 10
scripts/run-local-semantic-read-eligibility-stability.sh --runs 10
scripts/run-local-semantic-plan-selection-stability.sh --runs 10
scripts/run-local-semantic-source-binding-stability.sh --runs 10
```

Question Contract has a unified-frame matrix wrapper:

```bash
scripts/run-local-semantic-unified-question-frame-stability.sh \
  --runs 1 \
  --max-failures 5
```

Builders must call the same production prompt, schema, and projection code as
the real step. Hardcoded semantic hints, reduced catalogs, or hand-authored
fields absent from production invalidate the experiment.

Use one run per case for broad fault discovery. Stop after five failures when
the matrix still has structural defects; additional repetitions spend provider
capacity without adding a new failure class. After every case is individually
green, repeat the unchanged matrix with `--runs 10` for the model-facing
promotion gate.

## One controlled change

Use a patch file to change one variable while preserving the captured boundary:

```json
{
  "prompt_replacements": [
    {
      "old": "exact existing text",
      "new": "replacement text",
      "expected_count": 1
    }
  ]
}
```

The harness also supports `system_prompt_replacements`, `tool_spec_patches`, and
`schema_patches`. Repeat `--patch-file` to layer independent patches only when
the earlier layer has already been measured. If several prompt edits have not
stabilized the result, investigate the contract or presented evidence instead
of adding more prose.

## Controls

Every experiment needs controls that can falsify the proposed rule.

- A positive control must exercise the intended behavior.
- A negative control must be structurally similar but require the opposite
  decision.
- A regression control covers a previously stable, adjacent behavior.
- A provider control reruns an unchanged boundary when failures may be caused by
  credentials, quota, transport, or unsupported schema.

Controls should vary meaning, cardinality, and shape. A rule that recognizes a
named identity should also be tested with a non-identity scalar. A grouping rule
should cover one and multiple operands. A multi-request inventory assertion
should use three or four requests rather than a trivial single request.

An assertion should test the semantic outcome, not require incidental wording,
local IDs, or byte-identical free-form rationale. Use
`--enforce-structured-stability` only when the entire structured result is
intentionally canonical.

## Reading results

Each call prints:

```text
STABILITY RUN 3/10 PASS
STABILITY RUN 4/10 FAIL: <semantic or provider error>
```

The summary reports semantic passes and the number of distinct structured
outputs. `--output-jsonl` preserves every submitted argument object and error.
Inspect all failures, including the rationale or basis fields authored before a
decision. A score alone does not establish root cause.

Classify failures before changing code:

- provider failure: no model output was generated;
- schema failure: output was generated but violated the strict contract;
- semantic failure: output passed the schema but failed the assertion;
- harness failure: the standalone boundary differs from production.

Never count provider or harness failures as evidence about model behavior.

## Promotion gates

Use the threshold owned by the affected boundary:

- ordinary model boundary: at least 9/10 across the positive case and controls;
- identity Grounding and Source Binding: 10/10;
- deterministic parser/compiler/runtime behavior: focused tests must pass
  100%; model sampling is not its proof.

Before promotion:

1. reproduce RED for the expected reason;
2. change one variable;
3. obtain the required score on the target and controls;
4. apply exactly the tested prompt/schema/projection change to production;
5. run focused conformance and framework-specific tests;
6. run the affected live cases end to end and inspect raw artifacts; and
7. rerun the combined case gate so later fixes cannot hide regressions.

A 10/10 wrong result is strong evidence of a coherent contract defect. A mixed
result may indicate ambiguity, missing evidence, or unstable structure. Neither
justifies backend semantic inference or a case-shaped prompt rule.

For live-case discovery, run every affected case once and stop after five
failures:

```bash
scripts/run-local-goldset.sh \
  --case-ids <comma-separated-case-ids> \
  --stable-runs 1 \
  --max-failures 5
```

Diagnose the first false pipeline turn from its complete raw prompt, schema,
output, and downstream artifact. Once every case passes individually, rerun the
entire case set together at the required stable-run count. The combined run is
the final gate because a later correction can regress an earlier case.

## Repository rules

Keep one reusable builder and assertion per model boundary. Case files contain
data; builders own production-equivalent projection; assertions own semantic
acceptance. Do not duplicate prompt construction, schema construction, catalog
meaning, or parser logic in experiments.

Experiment artifacts may be retained for evidence, but production code must not
import them. Once a change is promoted, record the command, controls, score, raw
artifact location, focused tests, and live run IDs in the relevant memo or
review.
