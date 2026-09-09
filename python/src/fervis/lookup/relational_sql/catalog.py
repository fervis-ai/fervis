"""Project declared API row sources into SQL view names and typed metadata."""
from dataclasses import asdict, dataclass
import re
from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.acquisition import ApiView, sql_value_type
from fervis.lookup.source_reads.access_model import ReadAccessCatalog


@dataclass(frozen=True)
class QueryViewCatalog:
    views: tuple[ApiView, ...]
    tables: dict[str, dict[str, Any]]


def build_query_view_catalog(catalog: RelationCatalog, *, access: ReadAccessCatalog = ReadAccessCatalog()) -> QueryViewCatalog:
    views = []
    tables = {}
    for source in build_api_row_source_catalog(catalog).sources:
        fields = tuple(field for field in (*source.fields, *source.request_argument_fields)
                       if not field.declared_entity_kind
                       and field.type.value not in {'array', 'list', 'json', 'object', 'any', 'unknown'})
        columns: dict[str, str] = {}
        definitions = {}
        for field in fields:
            stem = re.sub(r'[^a-zA-Z0-9_]', '_', field.path or field.label or field.id).strip('_') or 'value'
            if stem[0].isdigit():
                stem = 'column_' + stem
            name = stem
            suffix = 2
            while name.casefold() in {item.casefold() for item in columns}:
                name = f'{stem}_{suffix}'
                suffix += 1
            columns[name] = field.id
            definitions[name] = {'type': sql_value_type(field.type.value), 'description': field.description,
                                 'label': field.label, 'nullable': field.nullable,
                                 'choices': list(field.choices),
                                 'request_parameter_ref': field.request_parameter_ref}
        aliases = {field_id: name for name,field_id in columns.items()}
        read = catalog.read(source.read_id)
        read_name = re.sub(r'[^a-zA-Z0-9_]', '_', read.id).strip('_') or 'read'
        row_name = re.sub(r'[^a-zA-Z0-9_]', '_', source.row_path).strip('_') or 'root'
        name = f'api_{read_name}__{row_name}__{source.id}'.lower()
        views.append(ApiView(name, source.id, columns, {}))
        tables[name] = {
            'read_id': source.read_id, 'path': read.path,
            'description': source.description, 'row_path': source.row_path,
            'columns': definitions,
            'request_parameters': [asdict(param) for param in source.params if param.param_ref not in access.supplied_parameters(source)],
            'automatic_request_parameters': sorted(access.supplied_parameters(source)),
            'candidate_keys': [
                {'entity_kind': key.entity_kind, 'key_id': key.id,
                 'components': {component.id: aliases[component.field_id] for component in key.components},
                 'context_columns': [aliases[field_id] for field_id in key.context_field_ids]}
                for key in source.candidate_keys
                if all(field_id in aliases for field_id in
                       (*key.context_field_ids, *(component.field_id for component in key.components)))
            ],
            'entity_references': [
                {'target_entity_kind': reference.target_entity_kind,
                 'target_key_id': reference.target_key_id,
                 'components': {component.target_component_id: aliases[component.local_field_id]
                                for component in reference.components},
                 'context_columns': [aliases[field_id] for field_id in reference.context_field_ids]}
                for reference in source.entity_references
                if all(field_id in aliases for field_id in
                       (*reference.context_field_ids, *(component.local_field_id for component in reference.components)))
            ],
        }
    return QueryViewCatalog(tuple(views), tables)
