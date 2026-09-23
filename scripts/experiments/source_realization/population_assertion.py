"""Independent finite truth-table checks for logical set membership."""


def _evaluate(node, fields, values):
    kind = node["kind"]
    if kind == "boolean_field":
        value = fields[node["field_ref"]]
        return None if value is None else value is node["expected_value"]
    if kind == "field":
        return fields[node["field_ref"]]
    if kind == "value":
        return values[node["value_ref"]]
    if kind == "unary":
        value = _evaluate(node["operand"], fields, values)
        if node["operator"] == "not":
            return None if value is None else not value
        if node["operator"] == "is_null":
            return value is None
        if node["operator"] == "not_null":
            return value is not None
        raise ValueError("Unsupported unary operator")
    if kind == "binary":
        left = _evaluate(node["left"], fields, values)
        right = _evaluate(node["right"], fields, values)
        operation = node["operator"]
        if operation == "and":
            return (
                False
                if left is False or right is False
                else None
                if left is None or right is None
                else left and right
            )
        if operation == "or":
            return (
                True
                if left is True or right is True
                else None
                if left is None or right is None
                else left or right
            )
        if left is None or right is None:
            return None
        if operation == "equals":
            return left == right
        if operation == "not_equals":
            return left != right
        if operation == "lt":
            return left < right
        if operation == "lte":
            return left <= right
        if operation == "gt":
            return left > right
        if operation == "gte":
            return left >= right
        raise ValueError("Unsupported binary operator")
    raise ValueError("Unknown condition")


def validate(arguments, context):
    decisions = arguments.get("populations", {})
    if set(decisions) != set(context["sets"]):
        return ["Population decisions must cover the expected logical sets."]
    errors = []
    for ref, expected in context["sets"].items():
        values = decisions[ref]
        if len(values) != 1:
            errors.append(ref + ": expected one selected branch")
            continue
        population = values[0].get("population", {})
        if population.get("kind") != expected["kind"]:
            errors.append(ref + ": incorrect exact/restricted population decision")
            continue
        for case in expected.get("truth_cases", []):
            try:
                observed = _evaluate(
                    population["condition"], case["fields"], context.get("values", {})
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    ref
                    + ": membership expression cannot be independently evaluated: "
                    + str(exc)
                )
                break
            if observed is not None and not isinstance(observed, bool):
                errors.append(ref + ": membership is not Boolean")
            elif (observed is True) is not case["expected"]:
                errors.append(ref + ": incorrect membership truth set")
    return errors
