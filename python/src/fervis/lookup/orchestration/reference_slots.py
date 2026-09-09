"""Typed reference demands selected by factual SQL before identity resolution."""
from dataclasses import dataclass, replace

from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.relational_sql.acquisition import RelationView
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relational_sql.outputs import identity_carriers, identity_authorities


@dataclass(frozen=True)
class ReferenceSlot:
    view: RelationView
    table: dict
    input_refs: tuple[str, ...]

    @property
    def key(self):
        return self.table['candidate_keys'][0]


def reference_slots(*, fact, inputs, denotations, tables, selected_authorities):
    """Lower declared logical demands to one stable relation per resolved input."""
    authorities = identity_authorities(tables)
    carriers = tuple(identity_carriers(tables).values())
    slots = []
    for input_ref in fact.input_refs:
        if denotations[input_ref].kind is not InputDenotationKind.IDENTITY_REFERENCE:
            continue
        authority_ref = selected_authorities.get(input_ref)
        if authority_ref is None:
            continue
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
        slots.append(ReferenceSlot(RelationView(name, name+'.rows', key['components']), {
            'kind': 'reference_slot', 'input_ref': input_ref, 'input_refs': [input_ref],
            'request_parameters': [], 'entity_references': [], 'automatic_request_parameters': [],
            'supplied_reference': inputs[input_ref].operand, 'operand_meaning': denotations[input_ref].operand_meaning,
            'columns': {component: {'type': kind, 'nullable': False, 'description': 'Required identity key component'}
                        for component, kind in components.items()},
            'candidate_keys': [key],
        }, (input_ref,)))
    return tuple(slots)


def selected_reference_slots(authored, slots, menu):
    used_views = set(authored.referenced_views)
    used_views.update(menu.descriptions[argument.binding]['view'] for argument in authored.request_arguments
        if menu.descriptions[argument.binding].get('kind') == 'reference_argument')
    literals = {menu.descriptions[argument.binding]['input_ref'] for argument in authored.request_arguments
                if menu.descriptions[argument.binding].get('kind') == 'reference_literal'}
    selected = {}
    for slot in slots:
        if slot.view.name not in used_views:
            continue
        input_ref = slot.table['input_ref']
        if input_ref in selected:
            raise QueryValidationError('One supplied reference must have one initial identity authority; use a declared relationship for other domains')
        selected[input_ref] = slot
    expected = {slot.table['input_ref'] for slot in slots}
    if set(selected) & literals:
        raise QueryValidationError('A reference cannot be both a literal resource address and a resolved identity in one answer')
    if set(selected) != expected - literals:
        raise QueryValidationError('Factual SQL must select the required identity authority for every reference input')
    return selected


def bind_reference_slots(menu, references):
    actual = {reference.view.name: reference for reference in references}
    expressions = dict(menu.expressions)
    descriptions = dict(menu.descriptions)
    for name, description in menu.descriptions.items():
        reference = actual.get(description.get('view'))
        if description.get('kind') != 'reference_argument' or reference is None:
            continue
        key = reference.table['candidate_keys'][0]
        if (description['identity']['entity_kind'], description['identity']['key_id'],
            set(description['identity']['components'])) != (key['entity_kind'], key['key_id'], set(key['components'])):
            raise QueryValidationError('Resolved reference differs from its consuming identity demand')
        column = description['column']
        expressions[name] = FieldRef(reference.view.columns[column])
        descriptions[name] = {**description, 'relation_id': reference.view.relation_id,
                              'value_type': reference.table['columns'][column]['type']}
    return replace(menu, expressions=expressions, descriptions=descriptions)


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
