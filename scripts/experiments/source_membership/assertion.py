"""Check ordinary membership independently of explicit qualifications."""

def validate(arguments, context):
    expected = context.get('expected_choice_reviews', ())
    if not expected:
        return ['source membership requires expected choice outcomes']
    reviews = {(branch, owner, surface_ref, choice): review
               for branch, scopes in arguments.items()
               for owner, surfaces in scopes.items()
               for surface_ref, surface in surfaces.items()
               for choice, review in surface['choice_reviews'].items()}
    errors = []
    for item in expected:
        if not item.get('branch_id') or not item.get('owner_set_ref'):
            errors.append('membership expectations must identify their branch and logical set')
            continue
        key = (item['branch_id'], item['owner_set_ref'], item['surface_ref'], item['choice'])
        actual = reviews.get(key, {}).get('baseline_decision')
        if actual != item['baseline_decision']:
            errors.append(f'{key}: baseline is {actual}; expected {item["baseline_decision"]}')
    return errors
