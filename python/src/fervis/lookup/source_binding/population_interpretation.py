"""Interpret API row-admission semantics independently of a user's question."""

from dataclasses import dataclass

from fervis.host_api.contracts import ParameterSemantics
from fervis.host_api.contracts.population import ParameterPopulation, ParameterRowValues
from fervis.lookup.relation_catalog.row_sources import RowSourceKind
from fervis.lookup.provider_contract import ProviderOutput, ProviderObject
from fervis.lookup.source_binding.model import SourcePopulationInterpretation
from fervis.lookup.turn_prompts import TurnPromptBase, ProviderResponseContract, ProviderToolContract
from fervis.model_io.structured_output.specs import required_tool_spec


@dataclass(frozen=True)
class PopulationInterpretationOutput(ProviderOutput):
    parameters: dict[str, ProviderObject]


@dataclass(frozen=True)
class UnknownPopulationOutput(ProviderOutput):
    kind: str
    mapping_basis: str


@dataclass(frozen=True)
class UnfilteredPopulationOutput(ProviderOutput):
    kind: str
    mapping_basis: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class ArgumentRowsOutput(ProviderOutput):
    argument: str
    row_values: tuple[str, ...]


@dataclass(frozen=True)
class CategoricalPopulationOutput(ProviderOutput):
    kind: str
    mapping_basis: str
    field_ref: str
    arguments: tuple[ArgumentRowsOutput, ...]


@dataclass(frozen=True)
class ComparisonPopulationOutput(ProviderOutput):
    kind: str
    mapping_basis: str
    field_ref: str
    operator: str


def _comparison_options(source, param):
    from fervis.lookup.source_binding.population_values import parameter_comparison_field
    if param.finite_choices:
        return ()
    options = []
    for field in source.fields:
        if field.declared_entity_kind:
            continue
        for operator in ("equals", "gt", "gte", "lt", "lte"):
            effect = ParameterPopulation(field_path=field.field_ref, comparison_operator=operator)
            try:
                parameter_comparison_field(source, param, effect)
            except ValueError:
                continue
            options.append((field.field_ref, operator))
    return tuple(options)


def population_interpretation_targets(request, *, finite_cover=False):
    canonical_refs = {item.canonical_value_id for item in request.canonical_values}
    usable_targets = {(item.source_ref, item.target_ref) for item in request.invocation_projection_options if item.value_ref in canonical_refs}
    interpreted = {(item.source_ref, item.parameter_ref) for item in request.population_interpretations
                   if not finite_cover or item.population is not None}
    return {
        f"{source.id}:{param.param_ref}": (source, param)
        for source in request.source_catalog.sources
        for param in source.params
        if source.kind is RowSourceKind.API_READ
        and param.source == "query"
        and param.semantics is ParameterSemantics.OPAQUE_QUERY_PARAM
        and param.population is None
        and (source.id, param.param_ref) not in interpreted
        and (param.default is not None or not param.default_is_known or param.required or (source.id, param.param_ref) in usable_targets)
        and (not finite_cover or param.finite_choices and _domain_fields(source))
    }


def _domain_fields(source):
    return tuple(field for field in source.fields
                 if field.finite_choices and not field.nullable and field.declared_value_domain)


def _closed(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def population_interpretation_schema(request, *, finite_cover=False):
    from .population_values import supports_population_argument
    parameters = {}
    for key, (source, param) in population_interpretation_targets(request, finite_cover=finite_cover).items():
        basic = {'kind': {'enum': ['unknown'] if finite_cover else ['unknown', 'preserves_all_rows_for_every_argument', *(['unfiltered_default'] if param.default is not None else [])]}, 'mapping_basis': {'type': 'string', 'minLength': 1}}
        variants = [UnknownPopulationOutput.schema(basic)]
        if not finite_cover and (param.finite_choices or supports_population_argument(param)):
            variants.append(UnfilteredPopulationOutput.schema({
                'kind': {'enum': ['unfiltered_values']}, 'mapping_basis': basic['mapping_basis'],
                'values': {'type': 'array', 'minItems': 1, 'items': {'enum': list(param.finite_choices)} if param.finite_choices else {'type': 'string'}},
            }))
        if not finite_cover:
            for field_ref, operator in _comparison_options(source, param):
                variants.append(ComparisonPopulationOutput.schema({
                    "kind": {"enum": ["row_comparison"]}, "mapping_basis": basic["mapping_basis"],
                    "field_ref": {"enum": [field_ref]}, "operator": {"enum": [operator]}}))
        if finite_cover:
            for field in _domain_fields(source):
                variants.append(CategoricalPopulationOutput.schema({
                    'kind': {'enum': ['finite_domain_cover']}, 'mapping_basis': basic['mapping_basis'],
                    'field_ref': {'enum': [field.field_ref]},
                    'arguments': {'type': 'array', 'minItems': 1, 'maxItems': len(param.finite_choices),
                        'items': ArgumentRowsOutput.schema({'argument': {'enum': list(param.finite_choices)},
                            'row_values': {'type': 'array', 'minItems': 1, 'items': {'enum': list(field.finite_choices)}}})},
                }))
        parameters[key] = {'anyOf': variants}
    return PopulationInterpretationOutput.schema({'parameters': _closed(parameters)})


def parse_population_interpretations(payload, *, request, finite_cover=False, on_failure=None):
    parsed = PopulationInterpretationOutput.parse(payload)
    targets = population_interpretation_targets(request, finite_cover=finite_cover)
    if set(parsed.parameters) != set(targets):
        raise ValueError('population interpretation must cover the exact source parameter scope')
    interpretations = []
    for key, raw in parsed.parameters.items():
        source, param = targets[key]
        try:
            result = _parse_parameter_interpretation(raw, source=source, param=param, finite_cover=finite_cover)
        except ValueError as exc:
            if on_failure is None:
                raise
            message = f"Invalid population evidence for {source.id}:{param.param_ref}: {exc}"
            on_failure(ValueError(message))
            result = SourcePopulationInterpretation(source.id, param.param_ref, None, message, validation_error=str(exc))
        interpretations.append(result)
    return tuple(interpretations)


def _parse_parameter_interpretation(raw, *, source, param, finite_cover):
    kind = raw.discriminator('kind')
    if finite_cover and kind not in {'unknown', 'finite_domain_cover'} or not finite_cover and kind == 'finite_domain_cover':
        raise ValueError('population decision does not belong to this interpretation boundary')
    if kind in {'unknown', 'preserves_all_rows_for_every_argument', 'unfiltered_default'}:
        item = raw.parse_as(UnknownPopulationOutput)
        if kind == 'unfiltered_default' and param.default is None:
            raise ValueError('unfiltered default requires a declared default')
        effect = (ParameterPopulation(preserves_population=True) if kind == 'preserves_all_rows_for_every_argument'
                  else ParameterPopulation(preserves_default=True) if kind == 'unfiltered_default' else None)
    elif kind == 'unfiltered_values':
        item = raw.parse_as(UnfilteredPopulationOutput)
        if not item.values or len(set(item.values)) != len(item.values):
            raise ValueError('unfiltered interpretation uses undeclared or duplicate argument values')
        from fervis.lookup.source_binding.population_values import validate_population_argument
        for value in item.values:
            validate_population_argument(param, value)
        effect = ParameterPopulation(unfiltered_values=tuple(sorted(item.values)))
    elif kind == 'row_comparison':
        item = raw.parse_as(ComparisonPopulationOutput)
        if finite_cover or (item.field_ref, item.operator) not in _comparison_options(source, param):
            raise ValueError("parameter comparison lacks compatible declared operands")
        effect = ParameterPopulation(field_path=item.field_ref, comparison_operator=item.operator)
    elif kind == 'finite_domain_cover':
        item = raw.parse_as(CategoricalPopulationOutput)
        field = next((field for field in _domain_fields(source) if field.field_ref == item.field_ref), None)
        if field is None or not item.arguments:
            raise ValueError('finite coverage requires an exact declared field domain')
        arguments = tuple(entry.argument for entry in item.arguments)
        if len(set(arguments)) != len(arguments) or not set(arguments) <= set(param.finite_choices):
            raise ValueError('finite coverage uses undeclared or duplicate arguments')
        domains = tuple(set(entry.row_values) for entry in item.arguments)
        domain = set(field.finite_choices)
        if any(not values or not values <= domain or len(values) != len(entry.row_values)
               for values, entry in zip(domains, item.arguments)):
            raise ValueError('finite coverage uses undeclared or duplicate row values')
        if set().union(*domains) != domain or any(
            set().union(*(values for j, values in enumerate(domains) if j != i)) == domain
            for i in range(len(domains))
        ):
            raise ValueError('finite coverage must be complete and irredundant')
        effect = ParameterPopulation(field_path=field.field_ref, value_mapping=tuple(
            ParameterRowValues(entry.argument, tuple(sorted(entry.row_values)))
            for entry in sorted(item.arguments, key=lambda entry: entry.argument)
        ))
    else:
        raise ValueError('unknown population interpretation kind')
    if not item.mapping_basis.strip():
        raise ValueError('population interpretation requires its source-semantic basis')
    if effect is not None:
        from dataclasses import replace
        from fervis.lookup.source_binding.population_values import population_values
        population_values(source, replace(param, population=effect))
    return SourcePopulationInterpretation(source.id, param.param_ref, effect, item.mapping_basis)

def population_interpretation_payload(items):
    return {"parameters": {
        f"{item.source_ref}:{item.parameter_ref}": {
            "population": item.population.to_public_dict() if item.population is not None else None,
            "mapping_basis": item.basis,
            "validation_error": item.validation_error,
        }
        for item in items
    }}


class SourcePopulationTurnPrompt(TurnPromptBase):
    turn_name = 'source population'
    turn_task = 'interpret source row admission and complete-read controls'
    include_current_question = False

    def __init__(self, request, *, read_catalog=None, finite_cover=False):
        self.request = request
        self.finite_cover = finite_cover
        self.read_catalog = read_catalog

    def _response_structure(self, source):
        read = next((read for read in self.read_catalog.reads if read.id == source.read_id), None) if self.read_catalog is not None else None
        return [] if read is None else [
            {'path': field.path, 'type': field.type, 'nullable': field.nullable,
             'description': (field.metadata or {}).get('description', ''),
             'requirements': [{'parameter_ref': item.param_ref, 'value': item.value} for item in field.requirements]}
            for field in read.fields
        ]

    def data_sections(self, builder):
        return (builder.json_section('API parameter semantics:', {
            key: {'source': {'label': source.label, 'description': source.description, 'row_path': source.row_path},
                  'complete_response_structure': self._response_structure(source),
                  'parameter': {'name': param.name, 'description': param.description, 'type': param.type.value,
                                'default': param.default, 'default_is_known': param.default_is_known, 'choices': list(param.finite_choices), 'choice_labels': param.choice_labels},
                  'returned_fields': [{'field_ref': field.field_ref, 'label': field.label, 'description': field.description,
                                       'type': field.type.value, 'choices': list(field.finite_choices),
                                       'nullable': field.nullable, 'declared_value_domain': field.declared_value_domain}
                                      for field in source.fields]}
            for key, (source, param) in population_interpretation_targets(self.request, finite_cover=self.finite_cover).items()
        }, indent=2),)

    def instruction_sections(self, builder):
        common = (
            "Interpret the supplied API contract, not hypothetical undocumented backend behavior. The contract's descriptions are the authority for the meanings they state.",
            "Judge only additional row restrictions attributable to this parameter, on the target source.row_path. Other filters and caller authorization are separate obligations.",
            "A change to nested response content is not a change to its parent row population. Use the complete response structure to distinguish the target rows from their descendants.",
            "Do not decide which business states a user normally wants.",
            "Use unknown when the required row-admission relationship cannot be established. Legal request values alone never prove row coverage.",
            "Use source names, descriptions, types, and value labels together. Sampled values do not establish a closed row domain.",
        )
        if self.finite_cover:
            rules = (
                "Direct complete-read controls were not established. Try an exact finite-domain cover, or return unknown.",
                "Use finite_domain_cover only when chosen arguments jointly cover a declared non-nullable field domain.",
                "Each chosen argument must admit exactly all source rows with its mapped values, with no additional restriction imposed by that argument. A necessary property of returned rows is insufficient.",
                "Supply only arguments needed for that cover. Do not classify other arguments. Redundant or incomplete covers are invalid.",
                "Do not treat numeric thresholds as equality partitions.",
            )
        else:
            rules = (
                "Find a directly supported complete-read control. Do not derive field partitions in this step.",
                "Use preserves_all_rows_for_every_argument only when every argument changes presentation, ordering, or computation without removing resource rows.",
                "Use unfiltered_values for arguments whose documented meaning disables this parameter's row restriction. For open scalar parameters, supply a literal stated by the API documentation, serialized as a string matching its declared type. Do not classify remaining argument meanings.",
                "Use unfiltered_default when this parameter can filter rows but its shown default disables the restriction. This says nothing about other argument values.",
                "Shown parameters are logical read inputs; runtime-owned pagination arguments have already been removed. Limits, offsets, search, thresholds, visibility, and state filters can remove rows.",
                "If no complete-read control is established, row_comparison may describe an exact scalar admission relation: a row is admitted if and only if its field satisfies the operator against the supplied argument. This is an argument-dependent predicate, not evidence that the source is unfiltered. Do not use it for limits, offsets, ranking, search, or conditional/sentinel behavior that is not this scalar comparison. Otherwise use unknown; a later step may examine finite covers.",
            )
        return (builder.instruction_block("Complete-read controls", (*common, *rules)),)

    def tool_contract(self):
        return ProviderToolContract(tool_specs=(required_tool_spec(
            tool_name='submit_finite_population_cover' if self.finite_cover else 'submit_source_population', tool_description='Record source parameter row-admission semantics.',
            input_schema=population_interpretation_schema(self.request, finite_cover=self.finite_cover),
        ),))

    def response_contract(self):
        return ProviderResponseContract(provider_schema={spec.name: spec.input_schema for spec in self.tool_contract().tool_specs})
