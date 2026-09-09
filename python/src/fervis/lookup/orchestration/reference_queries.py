"""Compile fact-owned reference inputs into live, guarded query relations."""
from dataclasses import asdict, replace

from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.values import BindingSet, FactValue
from fervis.lookup.available_sources import snapshot_source_catalog
from fervis.lookup.clarification.model import GroundingIdentityResponse
from fervis.lookup.clarification.context import clarification_response_ref
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.authoring import QueryUnavailable
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.compiler import _merge_parameter_declarations
from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_catalog_choices
from fervis.lookup.relational_sql.reference_compilation import combine_reference_members
from fervis.lookup.relational_sql.reference_planning import (
    ReferenceMeaning, ReferenceQueryPrompt, parse_reference_query, compile_reference_plan,
)
from fervis.model_io.turns import ModelTurnPurpose


def reference_input_values(partitions, *, inputs):
    """Certify the supplied text; entity identity remains an execution result."""
    result = []
    for partition in partitions:
        if not partition.requires_identity_resolution:
            continue
        term = inputs[partition.input_ref]
        arguments = dict(id=f'reference_text:{term.id}', known_input_id=term.id,
                         proof_refs=(f'question_input:{term.id}',))
        value = (FactValue.string_set(values=term.operand, **arguments)
                 if isinstance(term.operand, tuple)
                 else FactValue.named(text=term.operand, **arguments))
        result.append(CanonicalInputValue(value.id, term.id, partition.use_refs, value, value.proof_refs))
    return tuple(result)


def plan_fact_references(*, fact, inputs, denotations, values, catalog, access,
                         question, responses, turn, reference_catalog=None, consumer_catalog=None, discover_access=None, timezone="UTC"):
    query_catalog = catalog if reference_catalog is None else reference_catalog
    if consumer_catalog is not None:
        from fervis.lookup.relation_catalog import RelationCatalog
        reads = {read.id: read for read in (*query_catalog.reads, *consumer_catalog.reads)}
        query_catalog = RelationCatalog(reads=tuple(reads.values()))
    views = build_query_view_catalog(query_catalog, access=access)
    consumer_ids = {read.id for read in consumer_catalog.reads} if consumer_catalog is not None else set()
    consumer_context = {'requested_answer': asdict(fact), 'views': {
        name: {key: table[key] for key in ('path', 'description', 'request_parameters', 'candidate_keys', 'entity_references') if key in table}
        for name, table in views.tables.items() if table.get('read_id') in consumer_ids}}
    sources = snapshot_source_catalog(build_api_row_source_catalog(query_catalog).sources, read_access=access)
    results = []
    for value in values:
        if denotations[value.input_ref].kind is not InputDenotationKind.IDENTITY_REFERENCE:
            continue
        term = inputs[value.input_ref]
        denotation = denotations[value.input_ref]
        menu = with_catalog_choices(query_parameter_menu((value,)), source_catalog=sources,
                                    source_refs={view.row_source_id for view in views.views})
        collection = isinstance(term.operand, tuple)
        operands = value.typed_value.payload.values if collection else ('',)
        members = []
        for position, operand in enumerate(operands):
            member_menu = menu
            if collection:
                expressions = dict(menu.expressions)
                descriptions = dict(menu.descriptions)
                for name, description in menu.descriptions.items():
                    if description.get('input_ref') == term.id:
                        expressions[name] = replace(expressions[name], item_index=position)
                        descriptions[name] = {**description, 'value_type':'string',
                                              'label':operand, 'may_interpret':True}
                member_menu = replace(menu, expressions=expressions, descriptions=descriptions)
            reference_id = f'{fact.requested_fact_id}__reference_{term.id}'
            meaning = ReferenceMeaning(fact.requested_fact_id, term.id,
                denotation.denoted_instance_kind or '',
                f'{denotation.operand_meaning}: {operand or term.operand}',
                (term.origin,), (term.id,), reference_text=operand or term.operand,
                reference_is_collection_member=collection, timezone=timezone,
                reference_kind="description" if (operand or term.operand) in denotation.reference_descriptions else "literal")
            prompt = ReferenceQueryPrompt(question=question, meaning=meaning,
                                          tables=views.tables, parameters=member_menu.descriptions, consumer_context=consumer_context)
            authored = turn(ModelTurnPurpose.GROUNDING, prompt,
                            lambda payload:parse_reference_query(payload, prompt=prompt, menu=member_menu))
            if isinstance(authored, QueryUnavailable):
                return authored
            if discover_access is not None:
                from fervis.lookup.relational_sql.binding import reads_requiring_access_discovery
                access = discover_access(reads_requiring_access_discovery(authored, views.views, catalog=catalog))
            choices = [response for response in responses
                       if isinstance(response, GroundingIdentityResponse)
                       and response.requested_fact_id == fact.requested_fact_id
                       and response.known_input_id == term.id
                       and response.reference_operand == operand]
            if len(choices) > 1:
                raise ValueError('Reference member has conflicting clarification responses')
            choice = choices[0] if choices else None
            if choice is not None and choice.option.key is None:
                raise ValueError('Reference clarification requires a complete identity key')
            members.append(compile_reference_plan(authored, meaning=meaning, menu=member_menu,
                views=views.views, catalog=catalog, inputs=tuple(inputs.values()),
                input_denotations=tuple(denotations.values()), access=access,
                reference_id=reference_id + (f'_member_{position}' if collection else ''),
                operand=operand, selected_key=choice.option.key if choice else None,
                selection_proof_ref=clarification_response_ref(choice.response_id) if choice else ''))
        reference = (combine_reference_members(term, tuple(members), reference_id=reference_id)
                     if collection else members[0])
        results.append(replace(reference, table={**reference.table,
            'supplied_reference':term.operand, 'operand_meaning':denotation.operand_meaning}))
    return tuple(results)


def reference_prerequisites(references, bindings, *, argument_operations=()):
    """Combine the fact's disjoint reference programs and shared input bindings."""
    bound = {item.parameter_id:item for item in bindings.bindings}
    for reference in references:
        for item in reference.bindings.bindings:
            if item.parameter_id in bound and bound[item.parameter_id] != item:
                raise ValueError('Reference query bindings conflict with answer inputs')
            bound[item.parameter_id] = item
    return RelationProgram(
        parameters=_merge_parameter_declarations(tuple(p for r in references for p in r.program.parameters)),
        relations=tuple(item for r in references for item in r.program.relations),
        operations=(*tuple(item for r in references for item in r.program.operations), *argument_operations),
    ), BindingSet.from_bindings(tuple(bound.values()))
