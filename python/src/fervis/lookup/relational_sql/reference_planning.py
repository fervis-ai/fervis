"""Model-facing reference queries use the canonical relational authoring path."""

from dataclasses import dataclass, replace

from fervis.lookup.semantic_types import SourceOrigin
from .authoring import QueryAnswerPrompt, QueryUnavailable, parse_query_answer
from .binding import bind_query_answer
from .compiler import compile_query_answer
from .reference_compilation import compile_reference_result


@dataclass(frozen=True)
class ReferenceMeaning:
    requested_fact_id: str
    reference_input_ref: str
    denoted_instance_kind: str
    return_request_basis: str
    output_origins: tuple[SourceOrigin, ...]
    input_refs: tuple[str, ...]
    reference_text: str
    reference_is_collection_member: bool = False
    reference_kind: str = "literal"
    timezone: str = "UTC"
    result_kind: str = "qualifying_instances"
    output_kinds: tuple[str, ...] = ("identity",)
    ordering_origins: tuple[SourceOrigin, ...] = ()
    selection_kind: str = "all_results"
    selection_limit_input_ref: str | None = None

    def __post_init__(self):
        if self.reference_kind not in {"literal", "description"}:
            raise ValueError("Reference syntax must be literal or description")


class ReferenceQueryPrompt(QueryAnswerPrompt):
    turn_name = "reference query"
    turn_task = "identify the entity denoted by the assigned reference"

    def __init__(self, *, question, meaning, tables, parameters, consumer_context=None):
        self.consumer_context = consumer_context
        super().__init__(question=question, meaning=meaning, tables=tables, timezone=meaning.timezone,
            parameters={name:{**{key:value for key,value in description.items() if key != 'may_interpret'},
                              **({'kind':'definition' if meaning.reference_kind == 'description' else 'input'} if description.get('input_ref') else {})}
                        for name,description in parameters.items()})

    def compilation_scope(self):
        return {
            "unit": "reference_resolution",
            "timezone": self.meaning.timezone,
            "input_ref": self.meaning.reference_input_ref,
            "requested_fact_id": self.meaning.requested_fact_id,
            "reference_text": self.meaning.reference_text,
            "reference_is_collection_member": self.meaning.reference_is_collection_member,
        }

    def _result_modes(self):
        return ["rows"]

    def _schema(self):
        schema = super()._schema()
        schema['properties']['reference_binding'] = {'anyOf':[
            {'type':'object','properties':{'kind':{'type':'string','enum':['literal']},'match_column':{'type':'string','minLength':1}},
             'required':['kind','match_column'],'additionalProperties':False},
            {'type':'object','properties':{'kind':{'type':'string','enum':['description']},'basis':{'type':'string'}},
             'required':['kind','basis'],'additionalProperties':False},
        ]}
        schema['properties'] = {'reference_binding': schema['properties'].pop('reference_binding'), **schema['properties']}
        schema['properties']['reference_binding']['anyOf'] = [branch for branch in schema['properties']['reference_binding']['anyOf'] if branch['properties']['kind']['enum'] == [self.meaning.reference_kind]]
        schema['required'] = ['reference_binding', *schema['required']]
        return schema

    def _input_usage_instruction(self):
        if self.meaning.reference_kind == 'description':
            return ("Implement the complete description with SQL predicates, relationships, or extrema over observed API fields. "
                    "The query must select the described role or relationship; merely projecting role fields or returning all entities does not resolve it. "
                    "Fervis checks uniqueness but does not add the description's missing predicates or joins. Preserve extrema ties. "
                    "Description inputs are fixed definitions, not SQL placeholders or REST argument values. Use catalog value symbols where needed.")
        return ("Project candidate keys and an observed match_column for the supplied literal. Do not author WHERE, joins, grouping, ordering, "
                "auxiliary queries, or truncation; UNION of candidate projections is allowed. The match_column must be an observed name/code "
                "field or null-preserving concatenation of observed fields with punctuation or whitespace separators. Never invent missing values. "
                "Fervis adds equality to the original input and then checks identity uniqueness. The original literal may bind a declared lookup "
                "parameter to retrieve candidates; the returned observed match is still required.")

    def data_sections(self, builder):
        return (*super().data_sections(builder), *((builder.json_section('Consuming answer and API identity contracts:',
            self.consumer_context, indent=2),) if self.consumer_context is not None else ()))

    def _interpretation_schema(self):
        return {'type':'array','maxItems':0,'items':{'type':'object','properties':{},'required':[],'additionalProperties':False}}

    def _identity_display_schema(self):
        return {"type": "null"}

    def _identity_display_instruction(self):
        return "Return complete canonical key columns and, for literal references, the observed match_column. The matching column is internal, not a public output. Set display_column to null; the factual answer owns presentation."

    def instruction_sections(self, builder):
        return (builder.instruction_block("Reference query contract", (
            *self.sql_surface_instructions(),
            "Resolve only reference_text from Compilation scope. Other collection members and the surrounding factual answer belong to separate queries.",
            "Copy Requested answer.reference_kind into reference_binding. The frame has already fixed whether this is a literal name/code or a descriptive role; do not reinterpret that choice.",
            "Use the consuming answer and API identity contracts to disambiguate the API domain of the reference. The question's instance-kind wording is not an API namespace. Do not transfer the factual answer's filters or measures into reference resolution.",
            self._input_usage_instruction(),
            "Use one declared key or entity-reference authority. Project all its components unchanged from that authority's view and map their SQL aliases in outputs. Matching column names or UUID types do not make different identity domains interchangeable; use declared relationships when necessary.",
            "Declare every selected SQL alias and its scalar type in columns. For a literal, match_column must name a selected alias, not an unselected source field. Return exactly one identity output, mode rows, ordering empty, interpretations empty.",
            self._identity_display_instruction(),
            "Preserve all candidate keys and ties. Do not choose an arbitrary candidate or use LIMIT/OFFSET. Fervis checks missing and ambiguous identities at execution.",
            "Submit one submit_query_answer call, or report_query_unavailable if the declared views cannot establish this reference. Do not invent unavailable fields or change the requested identity.",
        )),)


def parse_reference_query(payload, *, prompt, menu):
    from .execution import QueryValidationError

    if 'unavailable' in payload:
        return parse_query_answer(payload,table_names=set(prompt.tables),parameter_names=set(menu.expressions))
    binding = payload.get('reference_binding', {})
    kind = binding.get('kind')
    if kind != prompt.meaning.reference_kind:
        raise QueryValidationError('Reference binding changes the declared reference syntax')
    if (kind == 'literal' and (set(binding) != {'kind','match_column'} or not isinstance(binding.get('match_column'), str) or not binding['match_column'].strip())
            or kind == 'description' and (set(binding) != {'kind','basis'} or not str(binding['basis']).strip())
            or kind not in {'literal','description'}):
        raise QueryValidationError('Reference query requires a literal or documented description binding')
    from .reference_matching import reject_reference_truncation
    reject_reference_truncation(payload['query'])
    body = {key:value for key,value in payload.items() if key != 'reference_binding'}
    value_names = {name for name, description in menu.descriptions.items()
                   if kind == 'literal' or description.get('input_ref') != prompt.meaning.reference_input_ref}
    authored = parse_query_answer(
        body, table_names=set(prompt.tables), parameter_names=value_names,
        meaning=prompt.meaning,
        expected_input_refs=None,
        tables=prompt.tables, parameter_descriptions=menu.descriptions,
        request_parameters={name:{param['param_ref'] for param in table.get('request_parameters',())}
                            for name,table in prompt.tables.items()},
    )
    if isinstance(authored, QueryUnavailable):
        return authored
    if authored.interpretations:
        raise QueryValidationError('Reference text is matched literally or defines the query; it is not replaced by a catalog value')
    if any(output.display_column for output in authored.outputs):
        raise QueryValidationError('Reference resolution returns keys without a display projection')
    names=tuple(name for name,description in menu.descriptions.items()
                if description.get('input_ref') == prompt.meaning.reference_input_ref)
    if len(names) != 1:
        raise QueryValidationError('Reference binding requires one original input projection')
    if kind == 'literal':
        from .reference_matching import literal_match_query
        if binding['match_column'] not in authored.output_types:
            raise QueryValidationError('Literal matching requires a declared observed column')
        body['query'] = literal_match_query(authored.query, column=binding['match_column'],
            parameter=names[0], tables=prompt.tables)
        authored = parse_query_answer(body, table_names=set(prompt.tables),
            parameter_names=set(menu.expressions), meaning=prompt.meaning,
            expected_input_refs=prompt.meaning.input_refs, tables=prompt.tables,
            parameter_descriptions=menu.descriptions,
            request_parameters={name:{param['param_ref'] for param in table.get('request_parameters',())}
                                for name,table in prompt.tables.items()})
    if kind == 'description':
        authored=replace(authored,definition_inputs=names)
    return authored


def compile_reference_plan(
    authored,
    *,
    meaning,
    menu,
    views,
    catalog,
    inputs,
    input_denotations,
    access,
    reference_id=None,
    operand="",
    selected_key=None,
    selection_proof_ref="",
):
    if isinstance(authored, QueryUnavailable):
        return authored
    subject=next((item for item in inputs if item.id == meaning.reference_input_ref),None)
    if (subject is None or meaning.reference_text != (operand or subject.operand)
            or meaning.reference_is_collection_member != isinstance(subject.operand,tuple)):
        from .execution import QueryValidationError
        raise QueryValidationError('Reference query scope differs from its original input or member')
    denotation = next((item for item in input_denotations if item.input_ref == meaning.reference_input_ref), None)
    if denotation is None or meaning.reference_kind != (
        "description" if meaning.reference_text in denotation.reference_descriptions else "literal"
    ):
        from .execution import QueryValidationError
        raise QueryValidationError('Reference query syntax differs from its original input')
    bound = bind_query_answer(authored, menu, views)
    compiled = compile_query_answer(
        question=meaning.return_request_basis,
        timezone=meaning.timezone,
        lookup_input_ref=meaning.reference_input_ref if meaning.reference_kind == "literal" else "",
        query=authored.query,
        views=bound.views,
        output_types=authored.output_types,
        result_contract=authored.result,
        catalog=catalog,
        query_parameters=bound.query_parameters,
        parameters=bound.parameters,
        bindings=bound.bindings,
        inputs=inputs,
        input_denotations=input_denotations,
        access=access,
        public_outputs=authored.outputs,
        output_origins=meaning.output_origins,
        meaning_inputs=bound.meaning_inputs,
        fact_id=meaning.requested_fact_id,
        namespace=(reference_id or f"lookup_{meaning.reference_input_ref}") + ".",
        expected_input_refs=meaning.input_refs,
    )
    return compile_reference_result(
        compiled,
        input_ref=meaning.reference_input_ref,
        output_types=authored.output_types,
        reference_id=reference_id,
        operand=operand,
        selected_key=selected_key,
        selection_proof_ref=selection_proof_ref,
    )
