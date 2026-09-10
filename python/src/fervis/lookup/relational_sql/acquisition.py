"""Materialize SQL views through the canonical, audited REST read kernel."""
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from fervis.lookup.plan_execution.declared_values import parse_declared_value
from typing import Any, Mapping
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.relations import Relation, RelationSource, RelationField, FieldBindingRole, SourceKind, EndpointParamBinding
from fervis.lookup.answer_program.expressions import Expression, expression_references
from fervis.lookup.plan_execution.relations import CompletenessStatus
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_compilation import expand_read_access
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from fervis.lookup.relational_sql.execution import SqlTable, QueryValidationError


@dataclass(frozen=True)
class ApiView:
    name: str
    row_source_id: str
    columns: Mapping[str, str]
    arguments: Mapping[str, Expression]
    argument_relation_id: str = ""


@dataclass(frozen=True)
class RelationView:
    name: str
    relation_id: str
    columns: Mapping[str, str]


@dataclass(frozen=True)
class MaterializedViews:
    tables: Mapping[str, SqlTable]
    proof_refs: Mapping[str, tuple[str, ...]]


def compile_api_views(views: tuple[ApiView, ...], *, catalog,
                      access: ReadAccessCatalog = ReadAccessCatalog()) -> RelationProgram:
    sources = {source.id: source for source in build_api_row_source_catalog(catalog).sources}
    if len({view.name for view in views}) != len(views):
        raise QueryValidationError('API view names must be unique')
    relations = []
    for view in views:
        source = sources[view.row_source_id]
        params = {param.param_ref: param for param in source.params}
        bindings = []
        for ref, value in view.arguments.items():
            if ref not in params:
                raise QueryValidationError('API view references an unknown parameter')
            try:
                references = expression_references(value)
            except (TypeError, AssertionError):
                raise QueryValidationError('API arguments require grounded program expressions') from None
            if (references.fields and not view.argument_relation_id) or any(not item.value.proof_refs for item in references.constants):
                raise QueryValidationError('API arguments require grounded values with proof')
            bindings.append(EndpointParamBinding(params[ref].id, value))
        fields = []
        for name, field_id in view.columns.items():
            field = source.field(field_id)
            role = FieldBindingRole.OUTPUT if FieldBindingRole.OUTPUT in field.allowed_roles else field.allowed_roles[0]
            fields.append(RelationField(field.id, (role,)))
        relations.append(Relation(view.name, RelationSource(SourceKind.API_READ,
            read_id=source.read_id, row_source_id=source.id, param_bindings=tuple(bindings),argument_relation_id=view.argument_relation_id), tuple(fields)))
    program = expand_read_access(RelationProgram(relations=tuple(relations)), access)
    return program


def materialize_views(views: tuple[ApiView, ...], *, catalog, read_session: ApiReadSession,
                      access: ReadAccessCatalog = ReadAccessCatalog()) -> MaterializedViews:
    program = compile_api_views(views,catalog=catalog,access=access)
    sources = {source.id:source for source in build_api_row_source_catalog(catalog).sources}
    executed = execute_access_program(program, catalog=catalog, read_session=read_session)
    if executed.engine_output.issue is not None:
        raise QueryValidationError('API views could not be completely materialized')
    tables, proofs = {}, {}
    for view in views:
        relation = executed.engine_output.relation(view.name)
        if relation.completeness.status is not CompletenessStatus.COMPLETE:
            raise QueryValidationError('Query requires complete API observations')
        rows = tuple({name: row[field] for name, field in view.columns.items()}
                     for row in relation.rows)
        column_types = {name: _sql_type(sources[view.row_source_id].field(field_id).type.value,
                                      values=tuple(row[name] for row in rows))
                        for name,field_id in view.columns.items()}
        tables[view.name] = SqlTable(column_types, rows)
        proofs[view.name] = tuple(dict.fromkeys((*relation.evidence.proof_refs, *relation.completeness.proof_refs)))
    return MaterializedViews(tables, proofs)


def _sql_type(kind: str, *, values: tuple[Any, ...] = ()) -> str:
    if kind == 'uuid':
        return 'UUID'
    if kind in {'integer', 'int', 'long'}:
        return 'BIGINT'
    if kind in {'number', 'numeric', 'decimal', 'float', 'double'}:
        numbers = [Decimal(str(value)) for value in values if value is not None]
        if not numbers:
            return 'DECIMAL(38,18)'
        if any(not value.is_finite() for value in numbers):
            raise QueryValidationError('API numeric values must be finite')
        exponents = [value.as_tuple().exponent for value in numbers]
        if any(not isinstance(exponent, int) for exponent in exponents):
            raise QueryValidationError('API numeric values must be finite')
        scale = max(max(-exponent, 0) for exponent in exponents if isinstance(exponent, int))
        integer_digits = max(max(value.adjusted() + 1, 0) for value in numbers)
        precision = max(integer_digits + scale, 1)
        if precision > 38:
            raise QueryValidationError('API numeric values exceed fixed-point precision')
        return f'DECIMAL({precision},{scale})'
    if kind in {'boolean', 'bool'}:
        return 'BOOLEAN'
    if kind == 'date':
        return 'DATE'
    if kind in {'datetime', 'timestamp'}:
        datetimes = [parse_declared_value(value, 'datetime') for value in values if value is not None]
        if any(not isinstance(value, datetime) for value in datetimes):
            raise QueryValidationError('Invalid API datetime value')
        awareness = {value.utcoffset() is not None for value in datetimes if isinstance(value, datetime)}
        if len(awareness) > 1:
            raise QueryValidationError('API datetime values mix local and offset-based time')
        return 'TIMESTAMPTZ' if awareness == {True} else 'TIMESTAMP'
    return 'TEXT'


def sql_value_type(kind: str) -> str:
    """Describe the result value domain, preserving typed identity values."""
    if kind == "uuid":
        return "uuid"
    physical = _sql_type(kind, values=())
    if physical.startswith('DECIMAL('):
        return 'number'
    return {'BIGINT':'integer','BOOLEAN':'boolean','DATE':'date',
            'TIMESTAMP':'datetime','TIMESTAMPTZ':'datetime','TEXT':'string'}[physical]
