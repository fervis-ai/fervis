# Literal Question Input Contract - Phase 1 Ledger

Authority memo: `docs/memo/2026-07-03-literal-question-input-contract.md`

Supporting architecture authority: `docs/architecture-summary.md`

## Scope

Phase 1 covers contract and test plumbing:

- expose `literal_text` roles in the question-contract schema/model/parser
- preserve literal reference/time values through question-contract parsing
- reject declared literal inputs that no requested fact owns
- separate model-visible conversation-resolution annotations from backend-only
  canonical identity handoff
- validate `source=conversation_resolution` question inputs against the
  conversation-resolution overlay authority
- add grounding certification payload support for imported prior identities
- register the new conformance algorithms in the shared YAML harness

The question-contract prompt rules are grouped into smaller instruction blocks
for readability. The review should treat that as presentation structure; the
Phase 1 semantic changes are the literal roles, CR-source authority, copied-span
rules, ownership checks, and time-requirement linkage described below.

## Test Implementation Ledger

| Planned behavior | Conformance case path | Harness/runner | Status | Notes |
| --- | --- | --- | --- | --- |
| Schema exposes `literal_text` with `reference_value`, `time_value`, and `result_limit` only | `python/tests/conformance/cases/algorithms/question_contract/schema_exposes_literal_text_roles.yaml` | `question_contract.schema` | GREEN | Covered by full conformance run: 532 passed. |
| Parser preserves `literal_text/reference_value` with copied `field_label_text` and `literal_text/time_value` requirement ownership | `python/tests/conformance/cases/algorithms/question_contract/reference_value_with_field_label_parses_exactly.yaml` | `question_contract.parse` | RED -> GREEN | RED confirmed before parser changes: rejected literal fields as unsupported. GREEN confirmed in focused and full conformance runs. |
| Parser rejects declared literal inputs that no requested fact owns | `python/tests/conformance/cases/algorithms/question_contract/unowned_literal_input_fails_closed.yaml` | `question_contract.parse` | GREEN | Covers fail-closed ownership for model-declared literals that would otherwise leak as global context. |
| Parser rejects CR-sourced literal inputs whose `resolved_input_ref` does not match the overlay | `python/tests/conformance/cases/algorithms/question_contract/conversation_resolution_literal_ref_must_match_overlay.yaml` | `question_contract.parse` | RED -> GREEN | RED confirmed before overlay authority validation: invented CR ref was accepted. GREEN after parser received and checked the overlay. |
| Prompt instructs the model to use `literal_text` roles instead of old question-input taxonomy | `python/tests/conformance/cases/algorithms/question_contract/prompt_uses_literal_text_roles.yaml` | `question_contract.prompt` | GREEN | Guards the model-facing contract text for reference, time, and result-limit roles. Prompt rules are grouped into smaller overview, inventory, source, CR handoff, and time-requirement blocks. |
| Conversation-resolution overlay carries model-visible resolved annotation separately from backend canonical identity/proof payload | `python/tests/conformance/cases/algorithms/conversation_resolution/prior_grounded_identity_handoff_carries_canonical_identity.yaml` | `conversation_resolution.overlay` | GREEN | Exact-match coverage verifies prompt/backend split. No separate pre-fix RED was captured for this tightened boundary case in the current ledger. |
| Conversation-resolution overlay boundary uses concrete kind-owned payload types | `python/tests/conformance/cases/algorithms/conversation_resolution/prior_grounded_identity_handoff_carries_canonical_identity.yaml` | `conversation_resolution.overlay` | GREEN | The same case now exercises `LiteralQuestionInputOverlay`; invalid legacy lookup fields are not representable on the literal overlay constructor. |
| Grounding contract can represent importing a prior grounded identity into current-run certification | `python/tests/conformance/cases/algorithms/grounding/current_run_certification_payload_supports_prior_identity_import.yaml` | `grounding.contract` | GREEN | Provides typed payload support for `imported_prior_identity` certification. |
| Shared conformance harness can run the new boundary cases | `python/tests/conformance/schemas/case.schema.json`, `python/tests/testkit/runner.py` | shared case loader/runner | GREEN | New algorithms registered: `conversation_resolution.overlay`, `grounding.contract`. |

## Verification

Commands run from `python/` unless noted:

```text
.venv/bin/pytest tests/test_conformance.py -q
532 passed in 3.94s
```

```text
.venv/bin/pytest tests/lookup/question_contract tests/lookup/grounding/test_grounding.py tests/lookup/conversation_resolution/test_context_frame_contract.py tests/lookup/test_prompt_rule_contract.py::test_model_turn_instruction_blocks_keep_required_headings tests/lookup/test_prompting_invocation.py::test_model_turn_invocations_match_approved_prompt_chars -q
53 passed in 0.42s
```

```text
.venv/bin/pytest tests/test_conformance.py -q -k 'prompt_uses_literal_text_roles or prompt_contains_current_question_context or prompt_includes_conversation_value_frame_annotations'
3 passed, 529 deselected in 1.00s
```

```text
.venv/bin/pytest tests/lookup/question_contract/test_prompt_contract.py -q
4 passed in 0.10s
```

```text
.venv/bin/ruff check src/fervis/lookup/question_contract/prompt.py tests/lookup/question_contract/test_prompt_contract.py
All checks passed!
```

```text
.venv/bin/ruff check src/fervis/lookup/conversation_resolution/__init__.py src/fervis/lookup/conversation_resolution/overlay.py src/fervis/lookup/question_contract/__init__.py src/fervis/lookup/question_contract/model.py src/fervis/lookup/question_contract/parser.py src/fervis/lookup/question_contract/prompt.py src/fervis/lookup/question_contract/turn.py tests/testkit/algorithms/conversation_resolution.py tests/testkit/algorithms/question_contract.py tests/testkit/algorithms/lookup_runtime.py tests/testkit/algorithms/grounding.py tests/lookup/question_contract/test_prompt_contract.py tests/lookup/grounding/test_grounding.py tests/testkit/prompt_surfaces.py tests/lookup/test_prompting_invocation.py
All checks passed!
```

## Strict Self-Review

| Quality Bar item | Assessment |
| --- | --- |
| No legacy support | Phase 1 still leaves legacy input kinds in place because this phase is contract/test plumbing. The memo allows migration sequencing, but later phases must remove old taxonomy support after callers/tests migrate. |
| No parallel implementation | Phase 1 still has old and new input kinds because this phase is contract/test plumbing, but the overlay handoff is no longer one loose parallel shape. Later phases must remove old taxonomy support after callers/tests migrate. |
| No compatibility shims | No translator shim was added. Parser accepts the new declared shape directly and validates CR-sourced inputs against the overlay authority. |
| No backend/parser semantic-oracle behavior | Parser validates copied spans, ownership, requirement linkage, and CR overlay authority. It does not infer whether `completed`, `BBS Mall`, or `top 5` is semantically correct. |
| No brittle heuristics | No catalog guessing, endpoint guessing, resolver guessing, or text-to-field inference was added. |
| Clear public/private package boundaries | `LiteralInputRole`, the concrete conversation-resolution question-input overlays, and `ResolvedCanonicalIdentityOverlay` are exported through package boundaries because they are now typed boundary objects. Test runners use public imports. |
| Few stable public surfaces/interfaces | Added small typed surfaces only: literal roles, canonical overlay value payload, and concrete overlay payload types. |
| Proper encapsulation | Literal role normalization and positive model invariants live in `RequestedFactKnownInput`; parser ownership/authority rules live in `question_contract.parser`; prompt/backend payload split and kind-owned overlay payloads live in `conversation_resolution.overlay`. Schema owns provider object field shape. |
| No 10+ file ripple for one concept | The literal role concept is centralized in `LiteralInputRole`; parser/schema/test actualizers consume it. Further phases should keep source binding/grounding ownership similarly localized. |
| One logic, one home | Ownership rejection is in parser/canonicalizer flow, not duplicated downstream. Canonical identity handoff serialization is owned by the overlay object. |
| No DRY violations | No repeated semantic mapping tables were introduced. The conformance actualizers expose fields for testing only. |
| Avoid overengineering | The implementation uses small dataclasses/enums and direct parser branches. No registry, adapter layer, compatibility translator, or kind-specific parser field-policing layer was introduced. |
| Avoid bloat | The code changes are localized and direct. Comment-driven cleanup removed duplicate literal shape validation from the model/parser and replaced the loose overlay object with small concrete payload types. The remaining broadness is from old and new input kinds coexisting during Phase 1, which must be reduced in later phases. |

## Independent Review

Final independent review verdict:

```text
No blocking Phase 1 issues remain.
```

Residual risks assigned to later phases:

- legacy question-input kinds still exist in model/parser/schema paths and must be
  removed in Phase 2
- conversation resolution still has a legacy named-reference overlay production
  path that must migrate to `literal_text/reference_value` or be removed in Phase 3
- grounding certification is payload support only; executable prior-identity import
  belongs to Phase 4
- source binding still needs later enforcement so raw literals cannot bind without
  grounding proof

## Review Authority

The independent review used these authority documents:

```text
docs/memo/2026-07-03-literal-question-input-contract.md
docs/architecture-summary.md
```
