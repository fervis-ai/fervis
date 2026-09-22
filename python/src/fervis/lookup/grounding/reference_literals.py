"""Certify supplied reference text without asserting a resolved entity identity."""
from fervis.lookup.answer_program.values import FactValue
from .semantic import CanonicalInputValue


def reference_input_values(partitions, *, inputs):
    """Certify the supplied text; entity identity remains an execution result."""
    result = []
    for partition in partitions:
        if not partition.requires_identity_resolution:
            continue
        term = inputs[partition.input_ref]
        arguments = dict(id=f'reference_text:{term.id}', known_input_id=term.id,
                         proof_refs=(f'question_input:{term.id}',))
        value = (FactValue.string_set(values=term.operand, **arguments)
                 if isinstance(term.operand, tuple)
                 else FactValue.named(text=term.operand, **arguments))
        result.append(CanonicalInputValue(value.id, term.id, partition.use_refs, value, value.proof_refs))
    return tuple(result)
