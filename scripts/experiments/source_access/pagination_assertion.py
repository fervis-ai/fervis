def validate(arguments, context):
    actual = arguments.get('reads', {})
    expected = context['reads']
    errors = []
    if set(actual) != set(expected):
        errors.append('Pagination decisions do not cover the expected API scope')
    for read, values in expected.items():
        for field, value in values.items():
            if actual.get(read, {}).get(field) != value:
                errors.append(f'{read}.{field}: expected {value!r}, got {actual.get(read, {}).get(field)!r}')
    return errors
