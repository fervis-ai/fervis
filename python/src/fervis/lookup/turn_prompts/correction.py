"""A bounded correction turn using the same semantic contract and diagnostics."""
from .turn_prompt import TurnPromptBase


class CorrectionTurnPrompt(TurnPromptBase):
    def __init__(self, original, failure):
        self.original=original
        self.failure=failure
        self.turn_name=original.turn_name
        self.turn_task=original.turn_task
        self.include_current_question=original.include_current_question

    def system_prompt(self,context):return self.original.system_prompt(context)
    def provider_metadata(self):return self.original.provider_metadata()

    def prompt_sections(self,builder):
        return (*self.original.prompt_sections(builder),
            builder.json_section('Previous rejected submission:',self.failure.artifact.submitted_payload,indent=2),
            builder.json_section('Compiler diagnostics:',self.failure.error_context,indent=2),
            builder.instruction_block('Correct the declaration',(
            'Correct the rejected submission using the compiler diagnostic and the declared contract.',
            'Preserve the original task, its stated authority, scope and outputs. Do not weaken or replace the task to avoid a validation error.',
            'If the contract provides an unavailable or clarification outcome and a faithful declaration is impossible, use that outcome.',
        )))

    def response_contract(self):return self.original.response_contract()
    def tool_contract(self):return self.original.tool_contract()


def run_with_correction(prompt, generate, on_failure=None):
    """Retry one rejected declaration; preserve each attempt through its observer."""
    from fervis.lookup.model_turn import LookupModelTurnError
    from fervis.model_io.structured_output.errors import ModelValidationKind
    original = prompt
    for attempt in range(2):
        try:
            return generate(prompt)
        except LookupModelTurnError as exc:
            if attempt or exc.validation_kind not in {ModelValidationKind.SCHEMA, ModelValidationKind.SEMANTIC}:
                raise
            if on_failure is not None:
                on_failure(exc)
            prompt = CorrectionTurnPrompt(original, exc)
    raise AssertionError('unreachable correction attempt')
