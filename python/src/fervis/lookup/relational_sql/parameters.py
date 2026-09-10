"""Expose scalar projections of certified inputs without copying their values."""

from dataclasses import dataclass, replace
from typing import Any, Mapping

from fervis.lookup.answer_program.value_projections import identity_contract, projection_description
from fervis.lookup.answer_program.compiler_inputs import (
    grounded_program_inputs,
    compiler_input_context_from_program_inputs,
)
from fervis.lookup.answer_program.contracts import ProgramInputs
from fervis.lookup.answer_program.expressions import Expression
from fervis.lookup.answer_program.values import (
    TimeValuePayload,
    LiteralValuePayload,
    LiteralType,
    NamedValuePayload,
    ConstantRef,
)


@dataclass(frozen=True)
class QueryParameterMenu:
    program_inputs: ProgramInputs
    expressions: Mapping[str, Expression]
    descriptions: Mapping[str, Any]


def query_parameter_menu(values) -> QueryParameterMenu:
    program_inputs = grounded_program_inputs(values)
    context = compiler_input_context_from_program_inputs(program_inputs)
    expressions, descriptions = {}, {}
    for index, value in enumerate(values, start=1):
        payload = value.typed_value.payload
        identity = identity_contract(payload)
        if identity:
            components = tuple(
                f"key_component:{component}" for component in identity["components"]
            )
        elif isinstance(payload, TimeValuePayload):
            components = ("start", "end")
        else:
            components = ("value",)
        for component_index, component in enumerate(components, start=1):
            name = f"p{index}_{component_index}"
            expressions[name] = context.expression_for_value(
                value.canonical_value_id, component=component
            )
            descriptions[name] = {
                "input_ref": value.input_ref,
                **projection_description(value.typed_value, component),
                "label": (
                    f"Canonical {identity['entity_kind']} {component}"
                    if identity
                    else value.typed_value.label
                ),
                **(
                    {"resolved_entity_label": value.typed_value.label}
                    if identity
                    else {}
                ),
                **(
                    {
                        "value_type": "datetime"
                        if payload.granularity == "hour"
                        else "date",
                        "boundary": "inclusive",
                        "granularity": payload.granularity,
                    }
                    if isinstance(payload, TimeValuePayload)
                    else {}
                ),
                "may_interpret": isinstance(payload, NamedValuePayload)
                or (
                    isinstance(payload, LiteralValuePayload)
                    and payload.literal_type is LiteralType.STRING
                ),
            }
    return QueryParameterMenu(program_inputs, expressions, descriptions)


def with_catalog_choices(
    menu: QueryParameterMenu, *, source_catalog, source_refs
) -> QueryParameterMenu:
    from fervis.lookup.available_sources import source_choice_literal

    expressions, descriptions = dict(menu.expressions), dict(menu.descriptions)
    for index, choice in enumerate(
        sorted(source_catalog.choice_values, key=lambda item: item.value_ref), start=1
    ):
        if choice.source_ref not in source_refs:
            continue
        surface = source_catalog.choice_surface(choice.surface_ref)
        if surface.declared_entity_kind:
            # Entity types describe authority, not observed values to which a
            # lexical question operand can be translated.
            continue
        name = f"c{index}"
        value = source_choice_literal(
            choice, snapshot_ref=source_catalog.contract_snapshot.ref
        )
        expressions[name] = ConstantRef(value.id, "catalog_choice@1", value)
        descriptions[name] = {
            "kind": "catalog_choice",
            "source_ref": choice.source_ref,
            "surface_ref": choice.surface_ref,
            "target_ref": surface.target_ref,
            "surface_kind": surface.kind.value,
            "description": surface.description,
            "value": choice.value,
            "label": choice.label,
            "type": choice.declared_type.value,
        }
    return replace(menu, expressions=expressions, descriptions=descriptions)


def with_reference_arguments(menu, references):
    """Expose current-run guarded fields for REST bindings and scoped SQL use."""
    from fervis.lookup.answer_program.expressions import FieldRef
    from sqlglot import exp
    expressions, descriptions = dict(menu.expressions), dict(menu.descriptions)
    for position, reference in enumerate(references, start=1):
        key = next(iter(reference.table['candidate_keys']), None)
        components = {column:component for component,column in key['components'].items()} if key else {}
        for index, column in enumerate(reference.view.columns, start=1):
            name = f'r{position}_{index}'
            if name in expressions:
                raise ValueError('Reference argument symbol collides with an input')
            expressions[name] = FieldRef(reference.view.columns[column])
            component = components.get(column)
            sql = exp.select(exp.column(column, quoted=True)).from_(exp.Table(this=exp.to_identifier(reference.view.name, quoted=True)))
            descriptions[name] = {
                'kind':'reference_argument', 'input_ref':reference.table['input_ref'],
                'input_refs':list(reference.input_refs),
                'relation_id':reference.view.relation_id, 'view':reference.view.name,
                'column':column, 'value_type':reference.table['columns'][column]['type'],
                'sql_expression':'('+sql.sql(dialect='duckdb')+')',
                'reference_is_collection':isinstance(reference.table.get('supplied_reference'), (tuple,list)),
                **({'identity':{'entity_kind':key['entity_kind'], 'key_id':key['key_id'],
                                'components':list(key['components'])},
                    'projection':'key_component:'+component,
                    'label':f"Resolved {key['entity_kind']}/{key['key_id']} key component {component}"}
                   if component is not None else {'projection':'field:'+column,'label':'Observed reference property '+column}),
            }
    return replace(menu, expressions=expressions, descriptions=descriptions)


def without_input_parameters(menu, input_refs):
    """Keep compiler-owned inputs bound without offering them for SQL authoring."""
    names={name for name,description in menu.descriptions.items() if description.get('input_ref') not in input_refs}
    return replace(menu, expressions={name:expression for name,expression in menu.expressions.items() if name in names},
                   descriptions={name:description for name,description in menu.descriptions.items() if name in names})
