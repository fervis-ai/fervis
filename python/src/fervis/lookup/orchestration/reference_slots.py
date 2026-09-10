"""Declared reference identity expectations before query compilation."""
from dataclasses import dataclass, replace

from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.relational_sql.acquisition import RelationView
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relational_sql.outputs import identity_carriers, identity_authorities


@dataclass(frozen=True)
class ReferenceContract:
    kind: str
    authority: str = ''
    view: str = ''
    fields: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'fields', tuple(self.fields))
        valid = (self.kind == 'identity' and bool(self.authority) and not self.view and not self.fields
                 or self.kind == 'record' and bool(self.view) and not self.authority and bool(self.fields) and len(set(self.fields)) == len(self.fields)
                 or self.kind == 'address' and not self.authority and not self.view and not self.fields)
        if not valid:
            raise ValueError('Reference contract must select one identity, record carrier, or address')

    def to_payload(self):
        return {'kind':self.kind, **({'authority':self.authority} if self.authority else {}),
                **({'view':self.view,'fields':list(self.fields)} if self.view else {})}


@dataclass(frozen=True)
class ReferenceSlot:
    view: RelationView
    table: dict
    input_refs: tuple[str, ...]
    reference_id: str

    @property
    def key(self):
        return self.table['candidate_keys'][0] if self.table['candidate_keys'] else None

    @property
    def record_source(self):
        return self.table.get('record_source')


def reference_slots(*, fact, inputs, denotations, tables, selected_contracts):
    """Lower declared logical demands to one stable relation per resolved input."""
    authorities = identity_authorities(tables)
    carriers = tuple(identity_carriers(tables).values())
    slots = []
    for input_ref in fact.input_refs:
        if denotations[input_ref].kind is not InputDenotationKind.IDENTITY_REFERENCE:
            continue
        contract = selected_contracts[input_ref]
        if contract.kind == 'address':
            continue
        name = f'{fact.requested_fact_id}__reference_{input_ref}'
        if contract.kind == 'record':
            if contract.view not in tables or not tables[contract.view].get('columns'):
                raise QueryValidationError('Observed reference requires a declared record carrier')
            if not set(contract.fields) <= set(tables[contract.view]['columns']):
                raise QueryValidationError('Observed reference fields are not declared by its carrier')
            columns = {field:tables[contract.view]['columns'][field] for field in contract.fields}
            slots.append(ReferenceSlot(RelationView(input_ref, name+'.rows', {column:column for column in columns}), {
                'kind':'reference_slot', 'input_ref':input_ref, 'input_refs':[input_ref],
                'record_source':contract.view, 'candidate_keys':[], 'entity_references':[],
                'request_parameters':[], 'automatic_request_parameters':[], 'columns':dict(columns),
                'supplied_reference':inputs[input_ref].operand, 'operand_meaning':denotations[input_ref].operand_meaning,
            }, (input_ref,), name))
            continue
        authority_ref = contract.authority
        if authority_ref not in authorities:
            raise QueryValidationError('Reference demand selects an undeclared identity authority')
        authority = authorities[authority_ref]
        matching = tuple(carrier for carrier in carriers if
            (carrier['entity_kind'], carrier['key_id'], set(carrier['components'])) ==
            (authority['entity_kind'], authority['key_id'], set(authority['components'])))
        components = {}
        for component in authority['components']:
            kinds = {tables[carrier['view']]['columns'][carrier['components'][component]]['type']
                     for carrier in matching}
            if len(kinds) != 1:
                raise QueryValidationError('Reference identity component has inconsistent declared scalar types')
            components[component] = next(iter(kinds))
        name = f'{fact.requested_fact_id}__reference_{input_ref}'
        key = {'entity_kind': authority['entity_kind'], 'key_id': authority['key_id'],
               'components': {component: component for component in components}, 'context_columns': []}
        slots.append(ReferenceSlot(RelationView(input_ref, name+'.rows', key['components']), {
            'kind': 'reference_slot', 'input_ref': input_ref, 'input_refs': [input_ref],
            'request_parameters': [], 'entity_references': [], 'automatic_request_parameters': [],
            'supplied_reference': inputs[input_ref].operand, 'operand_meaning': denotations[input_ref].operand_meaning,
            'columns': {component: {'type': kind, 'nullable': False, 'description': 'Required identity key component'}
                        for component, kind in components.items()},
            'candidate_keys': [key],
        }, (input_ref,), name))
    return tuple(slots)


def reference_input_menu(menu, denotations):
    expressions, descriptions = dict(menu.expressions), dict(menu.descriptions)
    for name, description in menu.descriptions.items():
        denotation = denotations.get(description.get('input_ref'))
        if denotation is None or denotation.kind is not InputDenotationKind.IDENTITY_REFERENCE:
            continue
        if denotation.reference_descriptions:
            expressions.pop(name)
            descriptions.pop(name)
        else:
            descriptions[name] = {**description, 'kind':'reference_literal', 'may_interpret':False}
    return replace(menu, expressions=expressions, descriptions=descriptions)
