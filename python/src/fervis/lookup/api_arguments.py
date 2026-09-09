"""Shared scalar and declared-identity contracts for REST argument bindings."""

def compatible_argument(parameter, description, *, literal_lookup=False):
    if description.get("kind") == "definition":
        return False
    identity = description.get("identity")
    target = parameter.get("entity_target")
    if description.get('kind') == 'reference_literal' and not literal_lookup and not (
        parameter.get('source') == 'path' or parameter.get('type') == 'uuid'
    ):
        return False
    if target and not identity and not literal_lookup:
        return False
    if identity and target:
        if (
            identity["entity_kind"],
            identity["key_id"],
            description["projection"],
        ) != (
            target["entity_kind"],
            target["key_id"],
            "key_component:" + target["component_id"],
        ):
            return False
    if identity and '_projected_values' in description:
        from fervis.lookup.relation_catalog.parameter_values import catalog_parameter_wire_value, parse_catalog_parameter_value
        kind = str(parameter.get('type') or 'unknown')
        values = description['_projected_values']
        try:
            if kind in {'array', 'list'}:
                parse_catalog_parameter_value(tuple(catalog_parameter_wire_value(value) for value in values), type_name=kind)
            else:
                for value in values:
                    parse_catalog_parameter_value(catalog_parameter_wire_value(value, type_name=kind), type_name=kind)
        except (ValueError, TypeError):
            return False
        return True
    source_type = description.get("value_type", description.get("type"))
    if source_type == "array" and identity:
        source_type = description.get("element_type")
        if source_type in {None, "unknown", "any"}:
            return False
    target_type = parameter.get("type")
    aliases = {
        "choice": "string",
        "enum": "string",
        "int": "integer",
        "long": "integer",
        "numeric": "number",
        "decimal": "number",
        "float": "number",
        "double": "number",
        "bool": "boolean",
    }
    source_type = aliases.get(source_type, source_type)
    target_type = aliases.get(target_type, target_type)
    if source_type == 'string' and target_type in {'integer', 'number', 'boolean', 'date', 'datetime', 'time', 'uuid'} and (
        literal_lookup or (not target and (description.get('kind') == 'reference_literal' or not identity and 'value' in description))
    ):
        from fervis.lookup.plan_execution.declared_values import parse_declared_value
        try:
            parse_declared_value(description.get('value', description.get('label')), target_type)
        except (ValueError, TypeError):
            return False
        source_type = target_type
    source_type = aliases.get(source_type, source_type)
    target_type = aliases.get(target_type, target_type)
    if source_type in {None, "unknown", "any"} or target_type in {
        None,
        "unknown",
        "any",
    }:
        return True
    return source_type == target_type or (source_type == "uuid" and target_type == "string") or (
        source_type == "string" and target_type == "uuid" and description.get('kind') in {'field_projection', 'reference_argument'}
    ) or (
        {source_type, target_type} <= {"integer", "number"}
    )



def field_argument_description(value_type, identities, target=None):
    """Describe observed field provenance without inventing destination authority."""
    matches = list(identities)
    if target is not None:
        matches = [item for item in matches if item == (target.entity_kind, target.key_id, target.component_id)]
    description = {'kind': 'field_projection', 'value_type': value_type}
    if matches:
        kind, key, component = matches[0]
        description.update(identity={'entity_kind':kind, 'key_id':key}, projection='key_component:'+component)
    return description


def row_source_argument_description(source, field_ref, target=None):
    field = next(field for field in (*source.fields, *source.request_argument_fields) if field.field_ref == field_ref)
    identities = [(key.entity_kind, key.id, component.id) for key in source.candidate_keys
                  for component in key.components if component.field_id == field.id]
    identities += [(key.target_entity_kind, key.target_key_id, component.target_component_id)
                   for key in source.entity_references for component in key.components if component.local_field_id == field.id]
    return field_argument_description(field.type.value, identities, target)


def relation_argument_description(contract, field_id, target=None):
    identities = [(key.entity_kind, key.key_id, component.component_id) for key in contract.entity_keys
                  for component in key.components if component.field_id == field_id]
    return field_argument_description(contract.field_types[field_id], identities, target)
