"""Verify observed record properties through typed operation ancestry."""
from fervis.lookup.plan_execution.errors import VerificationError


def verify_record_projection(program, projection, *, preserve_occurrences=False, relation_contracts=None, preserve_property_names=True):
    """Follow field-preserving operations back to observed input records."""
    from fervis.lookup.answer_program.operations import SqlQuerySpec, OrderSpec, FilterSpec, ProjectSpec, ProjectToKeySpec, AggregateSpec, UnionSpec, ReferenceGuardSpec, JoinSpec, CrossJoinSpec, AntiJoinSpec, UniversalConditionSpec
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
                from fervis.lookup.relational_sql.record_lineage import record_field_origins
                origins = record_field_origins(spec.query, {item.name:{column.name for column in item.columns}
                    for item in spec.inputs}, (field,), preserve_occurrences=preserve_occurrences)[field]
                for view, column in origins:
                    if preserve_occurrences and preserve_property_names and column != field.casefold():
                        raise VerificationError('Observed reference field must preserve its selected carrier property')
                    source = next(item for item in spec.inputs if item.name.casefold() == view)
                    binding = next(item for item in source.columns if item.name.casefold() == column)
                    verify(source.relation_id, binding.field_id)
            elif isinstance(spec, (OrderSpec, FilterSpec, ReferenceGuardSpec)):
                verify(spec.input_relation, field)
            elif isinstance(spec, (JoinSpec, CrossJoinSpec)):
                if preserve_occurrences:
                    raise VerificationError('Observed reference provenance cannot multiply candidate occurrences')
                contracts = relation_contracts or {}
                owners = [ref for ref in (spec.left,spec.right) if ref in contracts and field in contracts[ref].fields]
                if len(owners) != 1:
                    raise VerificationError('Joined record property requires one verified owning input')
                verify(owners[0], field)
            elif isinstance(spec, (AntiJoinSpec, UniversalConditionSpec)):
                expression = next((item.expression for item in spec.output_fields if item.output_field == field), None)
                if preserve_occurrences or not isinstance(expression, FieldRef):
                    raise VerificationError('Record field must preserve its observed candidate property')
                source = spec.candidate.relation_id if isinstance(spec, AntiJoinSpec) else spec.candidate_subject.relation_id
                verify(source, expression.field_id)
            elif isinstance(spec, ProjectToKeySpec):
                if preserve_occurrences and not any(occurrence_number_origin(program, spec.input_relation, key) is not None for key in spec.key_fields):
                    raise VerificationError('Observed reference provenance cannot deduplicate candidate occurrences')
                if field not in (*spec.key_fields, *spec.carry_fields):
                    raise VerificationError('Record field is absent from its observed projection')
                verify(spec.input_relation, field)
            elif isinstance(spec, AggregateSpec):
                if preserve_occurrences or field not in spec.group_by:
                    raise VerificationError('Record fields must preserve observed properties, not computed aggregates')
                verify(spec.input_relation, field)
            elif isinstance(spec, ProjectSpec):
                expression = next((item.expression for item in spec.outputs if item.output_field == field), None)
                if not isinstance(expression, FieldRef):
                    raise VerificationError('Record field must preserve an observed property without computation')
                if preserve_occurrences and preserve_property_names and expression.field_id != field:
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


def occurrence_number_origin(program, relation_id, field):
    """Return the shared producer of a carried query-local row occurrence key."""
    from fervis.lookup.answer_program.operations import ProjectSpec, ProjectToKeySpec, FilterSpec, OrderSpec
    from fervis.lookup.answer_program.expressions import FieldRef, FunctionExpression, ExpressionFunction
    producers = {operation.output_relation: operation.spec for operation in program.operations if operation.output_relation}
    seen = set()
    def origin(relation_id, field):
        target = (relation_id, field)
        if target in seen:
            return None
        seen.add(target)
        spec = producers.get(relation_id)
        if isinstance(spec, ProjectSpec):
            expression = next((item.expression for item in spec.outputs if item.output_field == field), None)
            if isinstance(expression, FunctionExpression) and expression.function is ExpressionFunction.ROW_NUMBER:
                return target
            return origin(spec.input_relation, expression.field_id) if isinstance(expression, FieldRef) else None
        if isinstance(spec, (FilterSpec, OrderSpec)):
            return origin(spec.input_relation, field)
        if isinstance(spec, ProjectToKeySpec) and field in spec.key_fields:
            return origin(spec.input_relation, field)
        return None
    return origin(relation_id, field)
