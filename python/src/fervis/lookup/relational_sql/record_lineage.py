"""Observed record fields cannot absorb computed measures from a SQL answer."""
from .column_lineage import output_column_lineage, unchanged_field_origins
from fervis.lookup.plan_execution.errors import VerificationError


def record_field_origins(query, columns_by_view, fields, *, preserve_occurrences=False):
    nodes = output_column_lineage(query, columns_by_view)
    result = {}
    for field in fields:
        node = nodes.get(field.casefold())
        if node is None:
            raise VerificationError('Record field must be an observed SQL column')
        try:
            origins = unchanged_field_origins(node, preserve_occurrences=preserve_occurrences)
        except VerificationError as exc:
            raise VerificationError('Record fields must preserve observed properties; computed measures require separate value outputs') from exc
        if not origins:
            raise VerificationError('Record field has no observed source')
        result[field] = {(view.casefold(), column.casefold()) for view,column,_ in origins}
    return result


def verify_record_projection(program, projection, *, preserve_occurrences=False):
    """Follow field-preserving operations back to observed input records."""
    from fervis.lookup.answer_program.operations import SqlQuerySpec, OrderSpec, FilterSpec, ProjectSpec, UnionSpec
    from fervis.lookup.answer_program.expressions import FieldRef
    from fervis.lookup.answer_program.relations import SourceKind
    relations = {relation.id:relation for relation in program.relations}
    producers = {operation.output_relation:operation.spec for operation in program.operations if operation.output_relation}
    checked, visiting = set(), set()

    def verify(relation_id, field):
        target = (relation_id, field)
        if target in checked:
            return
        if target in visiting:
            raise VerificationError('Observed record provenance contains a cycle')
        visiting.add(target)
        if relation_id in relations:
            relation = relations[relation_id]
            if relation.source.kind not in {SourceKind.API_READ, SourceKind.MEMORY_READ} or field not in {item.field_id for item in relation.fields}:
                raise VerificationError('Record field requires an observed input property')
        else:
            spec = producers.get(relation_id)
            if isinstance(spec, SqlQuerySpec):
                origins = record_field_origins(spec.query, {item.name:{column.name for column in item.columns}
                    for item in spec.inputs}, (field,), preserve_occurrences=preserve_occurrences)[field]
                for view, column in origins:
                    if preserve_occurrences and column != field.casefold():
                        raise VerificationError('Observed reference field must preserve its selected carrier property')
                    source = next(item for item in spec.inputs if item.name.casefold() == view)
                    binding = next(item for item in source.columns if item.name.casefold() == column)
                    verify(source.relation_id, binding.field_id)
            elif isinstance(spec, (OrderSpec, FilterSpec)):
                verify(spec.input_relation, field)
            elif isinstance(spec, ProjectSpec):
                expression = next((item.expression for item in spec.outputs if item.output_field == field), None)
                if not isinstance(expression, FieldRef):
                    raise VerificationError('Record field must preserve an observed property without computation')
                if preserve_occurrences and expression.field_id != field:
                    raise VerificationError('Observed reference field must preserve its selected carrier property')
                verify(spec.input_relation, expression.field_id)
            elif isinstance(spec, UnionSpec):
                if preserve_occurrences and spec.identity_fields:
                    raise VerificationError('Observed references cannot deduplicate uncertified candidate records')
                for source in spec.inputs:
                    verify(source, field)
            else:
                raise VerificationError('Record field has no verified observed source')
        visiting.remove(target)
        checked.add(target)

    for field in projection.record_fields.values():
        verify(projection.relation_id, field)
