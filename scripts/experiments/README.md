# Model-boundary experiments

This package contains semantic assertions and the two isolated entry boundaries
that can be constructed without replaying upstream model decisions:
`question_frame` and `question_contract`.

Every later model step is tested from a captured production invocation. Its
system prompt, user prompt, schema, catalog projection, and upstream evidence
therefore come from the real pipeline—not an experiment-owned reconstruction.

## Structure

```text
question_frame/       production invocation adapter, cases, assertion, matrix
question_contract/    production invocation adapter, cases, assertion
query_enrichment/     captured-turn assertion
grounding/            captured-turn assertion
read_eligibility/     captured-turn assertion
source_realization/   captured-turn assertion
source_binding/       captured-turn assertion
shared/               production-invocation serialization
```

The two boundary adapters receive the production prompt class as a dependency.
An experiment may inject another implementation of that same interface without
copying production prompt or schema construction. Once a change is promoted or
rejected, its temporary implementation is deleted and its evidence is recorded
in the governing memo.

Downstream experiments use:

```bash
scripts/run-local-model-step-stability.sh \
  --index /tmp/prompts/index.json \
  --step source_binding \
  --assertion-file scripts/experiments/source_binding/assertion.py \
  --assertion-context /tmp/source-binding-expectations.json \
  --runs 10
```

The context file contains expected outcomes only. It cannot redefine the
production request, catalog, prompt, or schema.

See
[`docs/dev/model-call-stability-harness.md`](../../../docs/dev/model-call-stability-harness.md).
