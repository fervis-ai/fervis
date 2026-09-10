"""Select reference contracts before their factual SQL consumer is authored."""
from dataclasses import asdict, dataclass

from fervis.lookup.api_arguments import compatible_argument
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.relational_sql.authoring import parse_query_answer, _object
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relational_sql.outputs import identity_authorities
from fervis.lookup.turn_prompts import TurnPromptBase, ProviderResponseContract, ProviderToolContract
from fervis.model_io.structured_output.specs import required_tool_spec
from .reference_slots import ReferenceSlot, ReferenceContract, reference_slots


@dataclass(frozen=True)
class ReferenceSelection:
    contracts: dict[str, ReferenceContract]
    slots: tuple[ReferenceSlot, ...]


class ReferenceContractPrompt(TurnPromptBase):
    turn_name = 'reference contract selection'
    turn_task = 'select supported reference contracts for the supplied references'

    def __init__(self, *, question, meaning, inputs, denotations, tables, parameters, reference_tables, failed_reference_plans=()):
        self.question, self.meaning = question, meaning
        self.inputs, self.denotations = inputs, denotations
        self.tables, self.parameters, self.reference_tables = tables, parameters, reference_tables
        self.failed_reference_plans = tuple(failed_reference_plans)
        self.reference_inputs = {
            ref: {'supplied_reference':inputs[ref].operand, 'meaning':denotations[ref].operand_meaning,
                  'instance_kind':denotations[ref].denoted_instance_kind,
                  'reference_descriptions':list(denotations[ref].reference_descriptions),
                  'allows_literal_address':not denotations[ref].reference_descriptions and isinstance(inputs[ref].operand,str)
                      and any(description.get('input_ref')==ref and compatible_argument(parameter,description)
                              for description in parameters.values() for table in tables.values()
                              for parameter in table.get('request_parameters',()))}
            for ref in meaning.input_refs if denotations[ref].kind is InputDenotationKind.IDENTITY_REFERENCE}
        self.identities = identity_authorities(reference_tables)
        self.record_views = {name:table for name,table in reference_tables.items() if table.get('columns')}
        self.choices = {ref:self._contract_variants(info) for ref,info in self.reference_inputs.items()}

    def _contract_variants(self, info):
        choices = []
        if self.identities:
            choices.append(_object({'kind':{'type':'string','enum':['identity']},
                'authority':{'type':'string','enum':list(self.identities)}}))
        for view,table in self.record_views.items():
            choices.append(_object({'kind':{'type':'string','enum':['record']},
                'view':{'type':'string','enum':[view]},
                'fields':{'type':'array','items':{'type':'string','enum':list(table['columns'])},'minItems':1,'maxItems':len(table['columns'])}}))
        if info['allows_literal_address']:
            choices.append(_object({'kind':{'type':'string','enum':['address']}}))
        return choices

    def data_sections(self, builder):
        consumers = {table.get('read_id') for table in self.tables.values()}
        return (
            builder.text_section('Question:',self.question),
            builder.json_section('Assigned requested answer:',asdict(self.meaning),indent=None),
            builder.json_section('Supplied references:',self.reference_inputs,indent=None),
            builder.json_section('Declared identity contracts:',identity_authorities(self.reference_tables),indent=None),
            builder.json_section('API relationship contracts:',{
                name:{'path':table.get('path',''),'description':table.get('description',''),
                      'possible_consumer':table.get('read_id') in consumers,
                      'fields':{field:{'type':meta['type'],'description':meta.get('description','')}
                                for field,meta in table.get('columns',{}).items()},
                      'candidate_keys':table.get('candidate_keys',[]),'entity_references':table.get('entity_references',[]),
                      'request_parameters':table.get('request_parameters',[])}
                for name,table in self.reference_tables.items()},indent=None),
            *((builder.json_section('Unavailable reference plans:',self.failed_reference_plans,indent=None),)
              if self.failed_reference_plans else ()),
        )

    def instruction_sections(self,builder):
        return (builder.instruction_block('Reference contract selection',(
            'Select one reference contract for each supplied reference, preserving its meaning in the assigned requested answer. This chooses a result type, not a known value. Current-data existence and uniqueness are checked by runtime guards, not by this contract-selection step.',
            'Prefer identity when a declared nominal authority supports the reference. Otherwise record selects a declared API row carrier whose observed properties can resolve the reference and support the requested relationship. Choose only record fields needed to identify the observed record and support the consuming relationship; do not require unrelated optional properties. A record contract does not invent an entity namespace, key, or foreign-key authority. Missing nominal annotations alone do not make an observed name lookup unavailable.',
            'The selected contract must both be establishable from the supplied literal or description and support the requested relationship through the declared API contracts. An endpoint that requires an identifier does not itself establish a supplied name.',
            'Use address only for an allowed original resource address. That strategy sends the original scalar to a compatible API argument; it does not certify an identity output.',
            'Reference queries will be compiled and guarded before factual SQL authoring. The consumer then receives concrete reference tables and typed key arguments.',
            'If a prior reference or consumer plan was unavailable, reconsider its authority or address strategy without changing the requested fact. Report unavailable only when no supported strategy preserves the question.',
        )),)

    def _schema(self):
        import json
        definitions, names, properties = {}, {}, {}
        for ref, choices in self.choices.items():
            variants = []
            for choice in choices:
                signature = json.dumps(choice, sort_keys=True)
                name = names.setdefault(signature, 'reference_contract_'+str(len(names)+1))
                definitions[name] = choice
                variants.append({'$ref':'#/$defs/'+name})
            properties[ref] = {'anyOf':variants}
        schema = _object({'reference_contracts':_object(properties)})
        schema['$defs'] = definitions
        return schema

    @staticmethod
    def _unavailable_schema():
        return _object({'unavailable':{'type':'boolean','enum':[True]},'reason':{'type':'string','minLength':1}})

    def response_contract(self):
        schemas = {}
        if all(self.choices.values()):
            schemas['submit_reference_contracts']=self._schema()
        schemas['report_query_unavailable']=self._unavailable_schema()
        return ProviderResponseContract(provider_schema=schemas)

    def tool_contract(self):
        return ProviderToolContract(tool_specs=tuple(required_tool_spec(tool_name=name,
            tool_description='Select reference contracts or explain why they are unavailable.',input_schema=schema)
            for name,schema in self.response_contract().provider_schema.items()))


def parse_reference_contracts(payload, *, prompt):
    if 'unavailable' in payload:
        return parse_query_answer(payload,table_names=set(),parameter_names=set())
    if set(payload)!={'reference_contracts'} or not isinstance(payload['reference_contracts'],dict):
        raise QueryValidationError('Reference selection requires a complete contract map')
    selected=payload['reference_contracts']
    if set(selected)!=set(prompt.reference_inputs):
        raise QueryValidationError('Select one supported contract for every reference')
    contracts = {}
    for ref, item in selected.items():
        if not isinstance(item,dict) or not all(isinstance(value,str) for key,value in item.items() if key != 'fields'):
            raise QueryValidationError('Reference contract must be a declared closed variant')
        kind = item.get('kind')
        expected = {'identity':{'kind','authority'}, 'record':{'kind','view','fields'}, 'address':{'kind'}}.get(kind)
        if set(item) != expected:
            raise QueryValidationError('Reference contract must be a declared closed variant')
        if (kind == 'identity' and item['authority'] not in prompt.identities
            or kind == 'record' and item['view'] not in prompt.record_views
            or kind == 'address' and not prompt.reference_inputs[ref]['allows_literal_address']):
            raise QueryValidationError('Reference contract is not supported by the declared sources')
        if kind == 'record' and (not isinstance(item['fields'],list) or not item['fields'] or not all(isinstance(field,str) for field in item['fields']) or len(set(item['fields'])) != len(item['fields']) or not set(item['fields']) <= set(prompt.record_views[item['view']]['columns'])):
            raise QueryValidationError('Observed reference fields must be a nonempty unique carrier projection')
        contracts[ref] = ReferenceContract(**item)
    slots=reference_slots(fact=prompt.meaning,inputs=prompt.inputs,denotations=prompt.denotations,
        tables=prompt.reference_tables,selected_contracts=contracts)
    return ReferenceSelection(contracts,slots)
