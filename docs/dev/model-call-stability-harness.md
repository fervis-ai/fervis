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
- `--boundary-file`: an isolated Question Frame or Question Contract invocation
  produced through the production prompt class.

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
  --assertion-file scripts/experiments/<model_step>/assertion.py \
  --assertion-context /tmp/<model-step>-expectations.json \
  --output-jsonl /tmp/<boundary>-results.jsonl \
  --label <experiment-name>
```

Use `--sequence` when one run contains more than one turn with the same purpose.
Keep `--workers 1` for diagnosis unless concurrency is itself under test. Serial
calls make provider behavior, rate limits, and result ordering easier to
attribute.

## Isolated entry boundaries

Question Frame and Question Contract can build an exact production invocation
without replaying an upstream model decision. Their adapters accept the
production prompt class as a dependency and serialize the invocation it
produces. The Question Frame matrix wrapper is:

```bash
scripts/run-local-semantic-unified-question-frame-stability.sh \
  --runs 1 \
  --max-failures 5
```

Query Enrichment, Grounding, Read Eligibility, Source Realization, and Source
Binding must use a turn captured from a real run. Synthetic request builders for
those steps duplicate upstream projection and are not valid evidence.

Use one run per case for broad fault discovery. Stop after five failures when
the matrix still has structural defects; additional repetitions spend provider
capacity without adding a new failure class. After every case is individually
green, repeat the unchanged matrix with `--runs 10` for the model-facing
promotion gate.

## One controlled change

For Question Frame or Question Contract, define a temporary class implementing
the production prompt interface and inject it into the boundary adapter:

```python
boundary = build_boundary_payload(
    question=question,
    expected_request_count=1,
    prompt_type=ExperimentalQuestionFramePrompt,
)
```

The injected class uses the production request, invocation assembly, model
adapter, and assertion path. It owns only the proposed prompt/schema change.
Keep it outside production until the gate passes, then promote the exact tested
implementation and delete the experiment class.

Captured later-stage turns may use one in-memory patch to falsify a narrowly
stated hypothesis. A serialized patch is diagnostic evidence, not a parallel
contract and not sufficient promotion evidence. Do not commit patch variants.
If several prompt edits have not stabilized the result, investigate the
contract or presented evidence instead of adding more prose.

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
- harness failure: the captured invocation or injected boundary differs from
  production.

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

Keep one semantic assertion per model step. Question Frame and Question Contract
also own one thin production-invocation adapter. Later steps replay captured
production invocations. Assertion-context files contain expected outcomes only.

Experiments never own prompt construction, schema construction, catalog
projection, upstream contract parsing, or a second implementation of production
logic. Temporary injected prompt classes implement the production interface and
are deleted after their result is accepted or rejected.

Experiment artifacts may be retained for evidence, but production code must not
import them. Once a change is promoted, record the command, controls, score, raw
artifact location, focused tests, and live run IDs in the relevant memo or
review.


For lower-priority evaluation runs, the production OpenAI Responses adapter accepts
`FERVIS_OPENAI_SERVICE_TIER=flex`. The default remains the project/provider default
when this variable is unset. Flex can take longer; set
`FERVIS_PROVIDER_TIMEOUT_SECONDS=900` for those runs. There is no automatic
fallback to standard processing. Returned Flex usage is priced at half of the
standard models.dev rates; explicitly configured effective prices are preserved.
The reported service tier is retained in usage metadata. Token-cache discounts
are not yet included, so these estimates remain conservative.

References: [Flex processing](https://developers.openai.com/api/docs/guides/flex-processing)
and [Batch pricing policy](https://developers.openai.com/api/docs/guides/batch).
