"""Public scalar/identity projection, independent of physical SQL columns."""
from dataclasses import dataclass, field
from typing import Mapping

from fervis.lookup.answer_program.result_projection import EntityKeyProjection, EntityKeyProjectionComponent, ResultProjectionError, verify_entity_display_type, verify_record_fields
from .execution import QueryValidationError


@dataclass(frozen=True)
class QueryOutput:
    label: str
    column: str = ''
    identity: EntityKeyProjection | None = None
    display_column: str = ''
    record_fields: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        if self.record_fields:
            return tuple(self.record_fields.values())
        return (self.column,) if self.identity is None else (*tuple(item.field_id for item in self.identity.components),
            *((self.display_column,) if self.display_column else ()))


def identity_carriers(tables):
    authorities={}
    for view,table in tables.items():
        for index,key in enumerate(table.get('candidate_keys',())):
            authorities[f'{view}:key:{index}']={'view':view,'entity_kind':key['entity_kind'],
                'key_id':key['key_id'],'components':key['components']}
        for index,key in enumerate(table.get('entity_references',())):
            authorities[f'{view}:reference:{index}']={'view':view,'entity_kind':key['target_entity_kind'],
                'key_id':key['target_key_id'],'components':key['components']}
    return authorities


def identity_authorities(tables):
    """Declare logical output types; SQL lineage owns their physical evidence."""
    from urllib.parse import quote
    signatures = sorted({(item['entity_kind'], item['key_id'], tuple(sorted(item['components'])))
                         for item in identity_carriers(tables).values()})
    return {quote(kind, safe='')+'/'+quote(key, safe='')+'('+','.join(quote(c, safe='') for c in components)+')':
            {'entity_kind': kind, 'key_id': key, 'components': components}
            for kind, key, components in signatures}


def query_output_shapes(payload, *, tables) -> tuple[QueryOutput,...]:
    authorities=identity_authorities(tables)
    outputs=[]
    for item in payload:
        if item['kind']=='value':
            output=QueryOutput(item['label'],column=item['column'])
        elif item['kind']=='record':
            fields={field['name']:field['column'] for field in item['fields']}
            if len(fields)!=len(item['fields']):
                raise QueryValidationError('Record output requires unique nonempty field names and columns')
            output=QueryOutput(item['label'],record_fields=fields)
        elif item['kind']=='identity':
            authority=authorities.get(item['authority'])
            if authority is None or set(item['components'])!=set(authority['components']):
                raise QueryValidationError('Identity output must use one complete declared authority')
            output=QueryOutput(item['label'],identity=EntityKeyProjection(authority['entity_kind'],authority['key_id'],
                tuple(EntityKeyProjectionComponent(component,column) for component,column in item['components'].items())),
                display_column=item.get('display_column') or '')
        else:
            raise QueryValidationError('Unknown public query output kind')
        outputs.append(output)
    if not outputs:
        raise QueryValidationError('Query must declare its requested public outputs')
    return tuple(outputs)


def parse_query_outputs(payload, *, columns: Mapping[str,str], tables, query: str) -> tuple[QueryOutput,...]:
    outputs = query_output_shapes(payload, tables=tables)
    for output in outputs:
        if not set(output.columns)<=set(columns):
            raise QueryValidationError('Public output references an undeclared SQL column')
        try:
            if output.record_fields:
                verify_record_fields(output.record_fields)
            verify_entity_display_type(output.display_column, columns)
        except ResultProjectionError as exc:
            raise QueryValidationError(str(exc)) from exc
    record_columns = {column for output in outputs for column in output.record_fields.values()}
    if record_columns:
        from .record_lineage import record_field_origins
        from fervis.lookup.plan_execution.errors import VerificationError
        try:
            record_field_origins(query, {name:table['columns'] for name,table in tables.items()}, record_columns)
        except VerificationError as exc:
            raise QueryValidationError(str(exc)) from exc
    _verify_authored_identity_outputs(query, columns, tables, tuple(outputs))
    return tuple(outputs)



def _verify_authored_identity_outputs(query, columns, tables, outputs):
    """Run the canonical identity checks inside the model correction boundary."""
    from .identity_lineage import sql_identity_keys
    from fervis.lookup.answer_program.operations import SqlQuerySpec, SqlRelationInput, SqlColumnBinding, SqlOutputField
    from fervis.lookup.plan_execution.verification.contract_types import RelationContract, RelationEntityKey, RelationEntityKeyComponent
    from fervis.lookup.plan_execution.errors import VerificationError

    keys=tuple(output.identity for output in outputs if output.identity is not None)
    if not keys:
        return
    authorities=identity_carriers(tables)
    contracts={name:RelationContract(
        {column:frozenset() for column in table['columns']},(),{},
        field_types={column:definition['type'] for column,definition in table['columns'].items()},
        entity_keys=tuple(RelationEntityKey(authority['entity_kind'],authority['key_id'],
            tuple(RelationEntityKeyComponent(component,column) for component,column in authority['components'].items()))
            for authority in authorities.values() if authority['view']==name),
    ) for name,table in tables.items()}
    spec=SqlQuerySpec(query,
        tuple(SqlRelationInput(name,name,tuple(SqlColumnBinding(column,column) for column in table['columns'])) for name,table in tables.items()),
        tuple(SqlOutputField(column,kind) for column,kind in columns.items()),entity_keys=keys)
    try:
        sql_identity_keys(spec,contracts)
    except VerificationError as exc:
        raise QueryValidationError(str(exc)) from exc


def annotate_identity_columns(tables):
    """Put declared nominal key roles beside their scalar columns for authoring."""
    authorities = identity_authorities(tables)
    refs = {(value['entity_kind'], value['key_id'], tuple(value['components'])):ref
            for ref,value in authorities.items()}
    roles = {}
    for carrier in identity_carriers(tables).values():
        authority = refs[(carrier['entity_kind'], carrier['key_id'], tuple(sorted(carrier['components'])))]
        for component,column in carrier['components'].items():
            role = {'authority':authority, 'component':component}
            values = roles.setdefault((carrier['view'], column), [])
            if role not in values:
                values.append(role)
    return {name:{**table, 'columns':{column:{**definition,
        **({'identity_roles':roles[(name,column)]} if (name,column) in roles else {})}
        for column,definition in table.get('columns', {}).items()}} for name,table in tables.items()}
