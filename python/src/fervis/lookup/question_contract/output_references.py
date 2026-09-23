"""Resolve question-owned output references before physical realization."""


def requested_value_output_index(request, value_ref: str) -> int:
    """Locate the one output owned by a stable requested-value identifier."""
    values = request.requested_value_refs
    if len(values) != len(set(values)) or value_ref not in values:
        raise ValueError('Ordering must reference a declared requested value')
    index = request.result_key_count + values.index(value_ref)
    if index >= len(request.output_origins):
        raise ValueError('Requested value has no declared output')
    return index
