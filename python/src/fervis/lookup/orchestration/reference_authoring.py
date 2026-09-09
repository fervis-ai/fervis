"""Explicit reference meaning declarations at the factual SQL authoring boundary."""
from dataclasses import dataclass

from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt, QueryUnavailable, AuthoredQueryAnswer, parse_query_answer, _object, _array
from fervis.lookup.relational_sql.outputs import identity_authorities, identity_carriers
from fervis.lookup.relational_sql.parameters import with_reference_arguments
from fervis.lookup.api_arguments import compatible_argument
from fervis.lookup.relational_sql.execution import QueryValidationError
from .reference_slots import reference_slots, selected_reference_slots


@dataclass(frozen=True)
class ReferenceDemand:
    input_ref: str
    authority: str | None


@dataclass(frozen=True)
class AuthoredFactualQuery:
    authored: AuthoredQueryAnswer
    demands: tuple[ReferenceDemand, ...]


class FactualQueryPrompt(QueryAnswerPrompt):
    def __init__(self, *, inputs, denotations, reference_tables, **kwargs):
        super().__init__(**kwargs)
        self.inputs, self.denotations, self.reference_tables = inputs, denotations, reference_tables
        self.reference_inputs = {
            ref: {'relation': f'{self.meaning.requested_fact_id}__reference_{ref}',
                  'supplied_reference': inputs[ref].operand,
                  'meaning': denotations[ref].operand_meaning,
                  'allows_literal_address': not denotations[ref].reference_descriptions and isinstance(inputs[ref].operand, str)}
            for ref in self.meaning.input_refs if denotations[ref].kind is InputDenotationKind.IDENTITY_REFERENCE}

    def output_identity_authorities(self):
        return {**identity_authorities(self.reference_tables), **super().output_identity_authorities()}

    def data_sections(self, builder):
        return (*super().data_sections(builder), *((builder.json_section('Supplied entity references:', self.reference_inputs, indent=None),) if self.reference_inputs else ()))

    def instruction_sections(self, builder):
        return (*super().instruction_sections(builder), *((builder.instruction_block('Reference demands', (
            'Declare one reference_demands entry per supplied entity reference. Choose its logical identity authority explicitly; this is a semantic decision, not a SQL source selection.',
            'For a resolved reference, use its stable relation from Supplied entity references as a SQL table. Its columns are the component IDs of the selected identity authority. The compiler resolves and guards that relation before executing the answer. Keep it as the population when references with no matching observations must be returned.',
            'Bind a resolved reference into REST with {reference_input: input_ref}. A declared identity target fixes its component. For an opaque parameter, also supply component_id (or null when exactly one component is type-compatible); this is your explicit semantic mapping according to the API documentation and does not create a destination identity relationship. Do not invent reference argument symbols.',
            'Use authority=null only for an allowed original literal resource address. Such an input must be used through its original parameter-menu binding in REST arguments; it does not establish an output identity. Do not mix literal-address and resolved-identity uses of one input.',
        )),) if self.reference_inputs else ()))

    def reference_usage_instructions(self):
        return (
            "Supplied entity references declare stable SQL relation names. For a resolved demand, FROM/JOIN that input's relation using the selected identity's component columns. These relations are valid SQL sources even though they are listed separately from API views. Do not reimplement reference matching against raw API rows.",
            "An original reference_literal parameter-menu binding is available only when that input declares authority=null. A resolved demand uses its reference relation or a structured reference_input REST binding; declaring an authority without consuming its relation or binding is incomplete.",
        )

    def _schema(self):
        schema = super()._schema()
        if self.reference_inputs:
            choices = list(identity_authorities(self.reference_tables))
            variants = []
            for ref, description in self.reference_inputs.items():
                authorities = [*choices, *([None] if description['allows_literal_address'] else [])]
                if not authorities:
                    continue
                variants.append(_object({'input_ref': {'type': 'string', 'enum': [ref]},
                    'authority': {'type': ['string', 'null'] if description['allows_literal_address'] else 'string', 'enum': authorities}}))
            schema['properties']['reference_demands'] = {**_array({'anyOf': variants} if variants else _object({})),
                'minItems': len(self.reference_inputs), 'maxItems': len(self.reference_inputs)}
            schema['required'].append('reference_demands')
        return schema

    def reference_argument_schema(self, parameter):
        if not self.reference_inputs:
            return None
        components = set()
        for carrier in identity_carriers(self.reference_tables).values():
            table = self.reference_tables[carrier['view']]
            for component, column in carrier['components'].items():
                description = {'kind':'reference_argument', 'identity':{'entity_kind':carrier['entity_kind'], 'key_id':carrier['key_id']},
                    'projection':'key_component:'+component, 'value_type':table['columns'][column]['type']}
                if compatible_argument(parameter, description):
                    components.add(component)
        if not components:
            return None
        properties = {'reference_input': {'type':'string', 'enum': list(self.reference_inputs)}}
        if not parameter.get('entity_target'):
            properties['component_id'] = {'type':['string','null'], 'enum':[*sorted(components), None]}
        return _object(properties)


def reference_contracts(prompt, demands, menu):
    slots = reference_slots(fact=prompt.meaning, inputs=prompt.inputs, denotations=prompt.denotations,
        tables=prompt.reference_tables, selected_authorities={item.input_ref:item.authority for item in demands})
    return slots, {**prompt.tables, **{slot.view.name:slot.table for slot in slots}}, with_reference_arguments(menu, slots)


def parse_factual_query(payload, *, prompt, menu, selection_limit):
    if 'unavailable' in payload:
        return parse_query_answer(payload, table_names=set(prompt.tables), parameter_names=set(menu.expressions))
    raw_demands = payload.get('reference_demands', ())
    if any(not isinstance(item, dict) or set(item) != {'input_ref', 'authority'} or
           not isinstance(item['input_ref'], str) or item['authority'] is not None and not isinstance(item['authority'], str) for item in raw_demands):
        raise QueryValidationError('Reference demand requires its input and logical authority')
    demands = tuple(ReferenceDemand(**item) for item in raw_demands)
    if len({item.input_ref for item in demands}) != len(demands) or {item.input_ref for item in demands} != set(prompt.reference_inputs):
        raise QueryValidationError('Reference demands must declare each supplied entity reference exactly once')
    for demand in demands:
        if demand.authority is None and not prompt.reference_inputs[demand.input_ref]['allows_literal_address']:
            raise QueryValidationError('This supplied reference requires resolved identity authority')
    slots, tables, lowered_menu = reference_contracts(prompt, demands, menu)
    arguments = []
    for argument in payload['request_arguments']:
        binding = argument['binding']
        if isinstance(binding, dict):
            if 'reference_input' not in binding or set(binding) - {'reference_input', 'component_id'}:
                raise QueryValidationError('Reference REST binding requires one supplied input and an optional key component')
            parameter = next((item for item in prompt.tables.get(argument['view'], {}).get('request_parameters', ())
                if item['param_ref'] == argument['parameter_ref']), None)
            names = [name for name, description in lowered_menu.descriptions.items()
                     if description.get('kind') == 'reference_argument' and description.get('input_ref') == binding['reference_input']
                     and (binding.get('component_id') is None or description.get('projection') == 'key_component:'+str(binding['component_id']))
                     and parameter is not None and compatible_argument(parameter, description)]
            if len(names) != 1:
                ref = binding['reference_input']
                demand = next((item for item in demands if item.input_ref == ref), None)
                available = {description['projection']:description['value_type'] for description in lowered_menu.descriptions.values()
                             if description.get('kind') == 'reference_argument' and description.get('input_ref') == ref}
                if demand is not None and demand.authority is None:
                    raise QueryValidationError(f'Reference input {ref} declares literal-address mode and has no resolved key. A reference_input binding requires a resolved authority; original address values use parameter-menu bindings.')
                raise QueryValidationError(f'Reference binding for {ref} does not supply exactly one compatible component to {argument["parameter_ref"]}. '
                    f'Parameter type/authority: {parameter}. Available key components: {available}. '
                    f'Use a compatible identifier parameter or consume {prompt.reference_inputs.get(ref, {}).get("relation")} directly in SQL.')
            argument = {**argument, 'binding': names[0]}
        arguments.append(argument)
    body = {key:value for key,value in payload.items() if key != 'reference_demands'}
    body['request_arguments'] = arguments
    authored = parse_query_answer(body, table_names=set(tables), parameter_names=set(lowered_menu.expressions),
        meaning=prompt.meaning, selection_limit=selection_limit, expected_input_refs=prompt.meaning.input_refs,
        parameter_descriptions=lowered_menu.descriptions, tables=tables,
        request_parameters={name:{param['param_ref'] for param in table.get('request_parameters', ())} for name,table in tables.items()})
    if isinstance(authored, QueryUnavailable):
        return authored
    selected = selected_reference_slots(authored, slots, lowered_menu)
    if set(selected) != {item.input_ref for item in demands if item.authority is not None}:
        raise QueryValidationError('Reference use differs from its declared resolution mode')
    return AuthoredFactualQuery(authored, demands)
