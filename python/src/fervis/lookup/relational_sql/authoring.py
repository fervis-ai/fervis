"""Native query authoring declarations before canonical program compilation."""

from fervis.lookup.answer_program.operations import SQL_VALUE_TYPES
from fervis.lookup.api_arguments import compatible_argument
from .parameter_usage import validate_sql_parameter_uses
from dataclasses import dataclass, asdict, replace
from typing import Any, Mapping

from sqlglot import exp
from fervis.lookup.turn_prompts import (
    TurnPromptBase,
    ProviderResponseContract,
    ProviderToolContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec
from .execution import _validate, SqlTable, QueryValidationError
from .results import ResultContract, ResultOrder
from .outputs import QueryOutput, parse_query_outputs, identity_authorities


@dataclass(frozen=True)
class QueryUnavailable:
    reason: str


@dataclass(frozen=True)
class QueryArgument:
    view: str
    parameter_ref: str
    binding: str
    instance: str | None = None

    @property
    def sql_view(self) -> str:
        return self.instance if self.instance is not None else self.view


@dataclass(frozen=True)
class QueryInterpretation:
    input: str
    choice: str
    basis: str


@dataclass(frozen=True)
class AuthoredQueryAnswer:
    query: str
    output_types: Mapping[str, str]
    output_labels: Mapping[str, str]
    result: ResultContract
    request_arguments: tuple[QueryArgument, ...]
    parameter_names: tuple[str, ...]
    referenced_views: tuple[str, ...]
    interpretations: tuple[QueryInterpretation, ...] = ()
    outputs: tuple[QueryOutput, ...] = ()
    definition_inputs: tuple[str, ...] = ()


def query_instances(arguments, table_names):
    """Validate named invocations once before expanding their declared contracts."""
    registered = {name.casefold() for name in table_names}
    instances = {}
    folded = {}
    for argument in arguments:
        if argument.view not in table_names:
            raise QueryValidationError('Request argument references an undeclared API view')
        if argument.instance is None:
            continue
        name = argument.instance
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise QueryValidationError('API invocation instance must be a nonempty SQL identifier')
        if name.casefold() in registered:
            raise QueryValidationError('API invocation instance shadows a registered view')
        if name.casefold() in folded and folded[name.casefold()] != name:
            raise QueryValidationError('API invocation instance names must be unique ignoring case')
        if name in instances and instances[name] != argument.view:
            raise QueryValidationError('API invocation instance mixes API definitions')
        folded[name.casefold()] = name
        instances[name] = argument.view
    return instances


def invocation_tables(arguments, tables):
    instances = query_instances(arguments, tables)
    return {**tables, **{name: tables[source] for name, source in instances.items()}}


def query_output_roles(meaning):
    """Lower question-level entity roles to the public SQL value contract."""
    roles = {"identity":"identity", "related_entity":"identity", "value":"value"}
    return tuple(roles[kind] for kind in meaning.output_kinds)


def parse_query_answer(
    payload,
    *,
    table_names,
    parameter_names,
    meaning=None,
    selection_limit=None,
    request_parameters=None,
    parameter_descriptions=None,
    tables=None,
    expected_input_refs=None,
) -> AuthoredQueryAnswer | QueryUnavailable:
    if "unavailable" in payload:
        if (
            set(payload) != {"unavailable", "reason"}
            or payload["unavailable"] is not True
            or not str(payload["reason"]).strip()
        ):
            raise QueryValidationError(
                "Unavailable query outcome requires one explicit reason"
            )
        return QueryUnavailable(str(payload["reason"]).strip())
    arguments = tuple(QueryArgument(**item) for item in payload["request_arguments"])
    instances = query_instances(arguments, table_names)
    table_names = set(table_names) | set(instances)
    if tables is not None:
        tables = invocation_tables(arguments, tables)
    statement, referenced_views = _validate(
        payload["query"], {name: SqlTable({}, ()) for name in table_names}
    )
    if instances:
        from sqlglot.optimizer.scope import traverse_scope
        changed = False
        for scope in traverse_scope(statement):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table) and instances.get(source.alias) == source.name:
                    source.set('this', exp.to_identifier(source.alias, quoted=True))
                    changed = True
        if changed:
            payload = {**payload, 'query': statement.sql(dialect='duckdb')}
            statement, referenced_views = _validate(payload['query'], {name: SqlTable({}, ()) for name in table_names})
    if tables:
        from .column_usage import project_query
        project_query(payload['query'], {name: table['columns'] for name, table in tables.items()})
    names = tuple(sorted({node.name for node in statement.find_all(exp.Placeholder)}))
    if not set(names) <= parameter_names:
        raise QueryValidationError("Query references an undeclared grounded parameter")
    columns = payload["columns"]
    output_types = {item["name"]: item["value_type"] for item in columns}
    if (
        not columns
        or len(output_types) != len(columns)
        or any(not name for name in output_types)
    ):
        raise QueryValidationError(
            "Query output declarations must be nonempty and unique"
        )
    if not set(output_types.values()) <= set(SQL_VALUE_TYPES):
        raise QueryValidationError("Query output has an unsupported value type")
    mode = payload["mode"]
    outputs = (
        ()
        if mode == "existence"
        else parse_query_outputs(
            payload["outputs"],
            columns=output_types,
            tables=tables or {},
            query=payload["query"],
        )
    )
    if mode == "existence" and payload["outputs"]:
        raise QueryValidationError(
            "Existence output is owned by the deterministic result contract"
        )
    public = tuple(
        dict.fromkeys(column for output in outputs for column in output.columns)
    )
    ordering = tuple(
        ResultOrder(item["column"], item["descending"]) for item in payload["ordering"]
    )
    if any(item.column not in output_types for item in ordering):
        raise QueryValidationError("Ordering references an undeclared query output")
    mode = payload["mode"]
    if mode == "scalar" and len(outputs) != 1:
        raise QueryValidationError("Scalar answers require exactly one public output")
    if "selection" in payload or "limit" in payload:
        raise QueryValidationError(
            "Result selection belongs to the grounded question request"
        )
    selection = (
        {
            "all_results": "all",
            "first_rank_with_ties": "first_with_ties",
            "take_with_boundary_ties": "take_with_ties",
            "position_with_ties": "position_with_ties",
        }[meaning.selection_kind]
        if meaning is not None
        else "all"
    )
    if mode != "rows" and (
        ordering or selection != "all" or selection_limit is not None
    ):
        raise QueryValidationError("Only row answers may select or order results")
    result = ResultContract(
        mode, public if mode == "rows" else (), ordering, selection, selection_limit
    )
    if meaning is not None:
        if mode == "existence" and meaning.result_kind != "scalar":
            raise QueryValidationError(
                "An existence output requires a population value request"
            )
        effective_kinds = (
            ("value",)
            if mode == "existence"
            else tuple(
                "identity" if output.identity is not None else "value"
                for output in outputs
            )
        )
        if len(effective_kinds) != len(meaning.output_origins):
            raise QueryValidationError(
                "Query public outputs differ from the requested output inventory"
            )
        if len(ordering) != len(meaning.ordering_origins):
            raise QueryValidationError(
                "Query ordering differs from the requested ranking keys"
            )
        if effective_kinds != query_output_roles(meaning):
            raise QueryValidationError(
                "Query output identity/value roles differ from the request"
            )
    if len({(item.sql_view, item.parameter_ref) for item in arguments}) != len(arguments):
        raise QueryValidationError("Request parameter binding is repeated")
    if any(
        item.view not in table_names or item.binding not in parameter_names
        for item in arguments
    ):
        raise QueryValidationError(
            "Request argument references an undeclared view or grounded binding"
        )
    if any(item.sql_view not in referenced_views for item in arguments):
        raise QueryValidationError("Request argument belongs to an unused query view")
    if request_parameters is not None and any(
        item.parameter_ref not in request_parameters.get(item.view, ())
        for item in arguments
    ):
        raise QueryValidationError("Request argument is not owned by query authoring")
    interpretations = tuple(
        QueryInterpretation(**item) for item in payload.get("interpretations", ())
    )
    descriptions = parameter_descriptions or {}
    if any(descriptions.get(name, {}).get('kind') == 'reference_argument' for name in names):
        raise QueryValidationError('Reference argument symbols are for REST bindings; use their declared relation columns in SQL')
    used_names = set(names) | {item.binding for item in arguments}
    if len({(item.input, item.choice) for item in interpretations}) != len(
        interpretations
    ):
        raise QueryValidationError("An input interpretation is repeated")
    alternatives = {
        name: {item.choice for item in interpretations if item.input == name}
        for name in {item.input for item in interpretations}
    }
    for input_name, choices in alternatives.items():
        if input_name in used_names:
            if len(choices) != 1:
                raise QueryValidationError(
                    "An interpreted input has multiple representations; use explicit choice symbols"
                )
            choice = next(iter(choices))
            transformed = statement.transform(
                lambda node, source=input_name, target=choice: (
                    exp.Placeholder(this=target)
                    if isinstance(node, exp.Placeholder) and node.name == source
                    else node
                )
            )
            assert isinstance(transformed, exp.Query)
            statement = transformed
            arguments = tuple(
                replace(argument, binding=choice)
                if argument.binding == input_name
                else argument
                for argument in arguments
            )
    names = tuple(sorted({node.name for node in statement.find_all(exp.Placeholder)}))
    used_names = set(names) | {item.binding for item in arguments}
    for item in interpretations:
        if (
            not item.basis.strip()
            or not descriptions.get(item.input, {}).get("may_interpret")
            or descriptions.get(item.choice, {}).get("kind") != "catalog_choice"
        ):
            raise QueryValidationError(
                "Interpretation must connect a lexical input to a used catalog choice"
            )
        if item.choice not in used_names:
            statement = _intern_choice_literal(
                statement, item.choice, descriptions[item.choice]
            )
            names = tuple(
                sorted({node.name for node in statement.find_all(exp.Placeholder)})
            )
            used_names.update(names)
        if item.choice not in used_names:
            raise QueryValidationError(
                "Interpreted catalog value is not used by the query"
            )
    if expected_input_refs is not None:
        input_names = used_names | {item.input for item in interpretations}
        used_inputs = {
            descriptions[name]["input_ref"]
            for name in input_names
            if "input_ref" in descriptions.get(name, {})
        }
        for view_name in referenced_views:
            view = (tables or {}).get(view_name, {})
            if view.get("kind") in {"resolved_reference", "reference_slot"}:
                used_inputs.update(view["input_refs"])
        if meaning.selection_limit_input_ref is not None:
            used_inputs.add(meaning.selection_limit_input_ref)
        if used_inputs != set(expected_input_refs):
            missing = set(expected_input_refs) - used_inputs
            channels = {ref: {
                'relations': {name: list(table.get('columns', {})) for name, table in (tables or {}).items()
                              if ref in table.get('input_refs', ())},
                'parameters': [name for name, description in descriptions.items() if description.get('input_ref') == ref],
            } for ref in sorted(missing)}
            raise QueryValidationError(
                "Query input operands differ from this requested fact ownership. "
                f"Missing input consumption: {channels}. Unowned inputs: {sorted(used_inputs - set(expected_input_refs))}. "
                "A reference demand declaration alone does not consume its input; use the declared reference relation or compatible REST binding."
            )
    query = statement.sql(dialect="duckdb") if interpretations else payload["query"]
    for argument in arguments:
        parameter = next(
            (
                p
                for p in (tables or {})
                .get(argument.view, {})
                .get("request_parameters", ())
                if p["param_ref"] == argument.parameter_ref
            ),
            None,
        )
        if parameter is not None and not compatible_argument(
            parameter, descriptions.get(argument.binding, {}), literal_lookup=(
                getattr(meaning, 'reference_kind', None) == 'literal' and
                descriptions.get(argument.binding, {}).get('input_ref') == getattr(meaning, 'reference_input_ref', None))
        ):
            raise QueryValidationError(
                "Request argument type or identity authority does not match the bound parameter"
            )
    validate_sql_parameter_uses(query, tables or {}, descriptions)
    return AuthoredQueryAnswer(
        query,
        output_types,
        {output.column: output.label for output in outputs if output.column},
        result,
        arguments,
        names,
        referenced_views,
        interpretations,
        outputs,
    )


def _intern_choice_literal(statement, name, description):
    """Intern an explicitly selected, exactly equal typed constant; infer no meaning."""
    from decimal import Decimal
    from fervis.lookup.plan_execution.declared_values import parse_declared_value

    kind = description["type"]
    value = parse_declared_value(description["value"], kind)

    def replace_literal(node):
        matches = False
        if type(value) is bool:
            matches = isinstance(node, exp.Boolean) and node.this is value
        elif isinstance(value, str):
            matches = (
                isinstance(node, exp.Literal) and node.is_string and node.this == value
            )
        elif isinstance(value, (int, Decimal)):
            matches = (
                isinstance(node, exp.Literal)
                and node.is_number
                and Decimal(node.this) == Decimal(value)
            )
        return exp.Placeholder(this=name) if matches else node

    return statement.transform(replace_literal)


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


class QueryAnswerPrompt(TurnPromptBase):
    turn_name = "relational answer"
    turn_task = "author the assigned answer request over authorized API views"

    def __init__(
        self, *, question: str, meaning: Any, tables: Mapping, parameters: Mapping, timezone: str = "UTC"
    ):
        self.timezone = timezone
        self.question, self.meaning, self.tables, self.parameters = (
            question,
            meaning,
            tables,
            parameters,
        )

    def compilation_scope(self):
        return {
            "unit": "one_requested_fact",
            "timezone": self.timezone,
            "requested_fact_id": self.meaning.requested_fact_id
            if self.meaning is not None
            else None,
        }

    def data_sections(self, builder):
        return (
            builder.json_section(
                "Compilation scope:",
                self.compilation_scope(),
            ),
            builder.text_section("Question:", self.question),
            builder.json_section("Requested answer:", asdict(self.meaning), indent=None),
            builder.json_section("View invocation requirements:", {
                name: {"required_parameters": [parameter["param_ref"] for parameter in table.get("request_parameters", ()) if parameter.get("required")],
                       "has_no_required_parameters": not any(parameter.get("required") for parameter in table.get("request_parameters", ()))}
                for name, table in self.tables.items()
            }, indent=None),
            builder.json_section("Declared API views:", {
                name: {key: value for key, value in table.items() if key != "read_id"}
                for name, table in self.tables.items()
            }, indent=None),
            builder.json_section(
                "Declared output identity types:",
                self.output_identity_authorities(),
                indent=None,
            ),
            builder.json_section(
                "Grounded parameter menu:", {
                    name: {**{key:value for key,value in description.items() if not key.startswith("_")}, **({"sql_expression": "$" + name}
                        if description.get("kind") not in {"reference_argument", "reference_literal", "definition"} else {})}
                    for name, description in self.parameters.items()
                }, indent=None
            ),
        )

    def sql_surface_instructions(self):
        return (
            "Author an executable query, not evaluated rows or a known identity value. Fervis reads the declared API views and evaluates SQL after compilation. A value read from another view can be joined or queried at execution; its absence from this prompt does not make the query unavailable.",
            "Use DuckDB SQL over only the declared views. Quote identifiers when necessary. No files, network, external tables or database access.",
            "Copy SQL parameters from the menu sql_expression exactly. Use the menu key for request_arguments bindings. FROM and JOIN use declared API view keys or declared reference relation names as SQL tables, never endpoint paths or resource labels. These views are not SQL functions: do not put parentheses or request parameters after a view name. REST arguments belong exclusively in request_arguments. Do not invent values or inline grounded operands. SQL literals may express arithmetic constants and documented catalog enum values.",
            "SQL calendar operations and date-to-timestamp conversions use the timezone in Compilation scope.",
            "request_arguments binds declared view request parameter refs to the same grounded menu. Supply only arguments needed for the question; automatic_request_parameters are already supplied by complete traversal; never bind them. To invoke the same API view with different bindings in one query, assign a distinct instance name to each invocation. Use FROM declared_view AS instance, or refer to the named invocation directly as a SQL table. All arguments sharing an instance must name the same declared view. Use instance=null for the default view. Physical prerequisite enumeration and pagination belong to Fervis.",
        )

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Relational answer contract",
                (
                    *self.sql_surface_instructions(),
                    "This invocation compiles only the assigned Requested answer. Other answer requests in the question are compiled separately and combined by Fervis. Use the complete question to interpret this assigned request and its scoped operands.",
                    *self.reference_usage_instructions(),
                    "Preserve the original question, requested row or group grain, relationships and conditions. Do not introduce filters absent from the requested meaning, change the requested measure or assume a single API invocation covers its whole parent population. A stored record and a qualifying business occurrence are not necessarily the same population: use source contracts and observed state or timestamps to preserve the question's population qualifiers, rather than treating a resource label as proof that every returned row qualifies.",
                    "A literal menu value is the canonical scalar, not its original label. Use that value directly; numeric percentages have already been converted to ratios.",
                    "An aggregate can be computed from a complete row view. Missing inputs for a summary endpoint do not make the question unavailable when another declared view can supply the observations. Required parent traversal remains the compiler's responsibility.",
                    "Temporal parameter metadata declares the boundary convention and scalar type. Inclusive calendar end dates include that whole date; do not treat an inclusive end as the first excluded date.",
                    self._input_usage_instruction(),
                    "Declare every returned SQL alias and its scalar type in columns. Separately declare exactly the requested public outputs. A value output selects one column. An identity output selects a declared candidate-key or entity-reference authority and maps all its key components to unchanged SQL key-column aliases. Keep grouping and ties based on identity and the requested ranking keys, not display labels. Unrequested ordering columns stay out of outputs.",
                    self._identity_display_instruction(),
                    "For a yes/no existence question use mode existence and return witness rows. Fervis determines true or false even when no witness exists. Do not count the whole population or return a Boolean in this mode.",
                    "For a scalar question use mode scalar and return exactly one public value and one row, including a directly observed property. An aggregate is required only when the question requests one.",
                    "For row or grouped answers use mode rows. Return all candidates, including ordering columns. Declare only the ordering keys requested in the question; do not add name or identifier tie breakers. Fervis owns ordering, first rank, top-N with boundary ties and ordinal selection. Do not LIMIT or OFFSET the SQL query.",
                    "Use EXISTS and NOT EXISTS for presence and absence; preserve shared-row correlation. Aggregate independent child measures before joining them to avoid fanout. DISTINCT on a measure is not a substitute for preserving observation identity.",
                    "No observations may imply a zero count, but a missing measured value is NULL unless the question or documented measure defines zero. Do not invent observations.",
                    "Return one submit_query_answer tool call, or report_query_unavailable when the available source contracts cannot support the requested fact. Do not substitute another measure, source population or question.",
                ),
            ),
        )

    def _schema(self):
        string = {"type": "string"}
        binding_groups = {}
        argument_schema = self._argument_schema(binding_groups)
        schema = _object(
            {
                "query": string,
                "mode": {"type": "string", "enum": self._result_modes()},
                "columns": _array(
                    _object(
                        {
                            "name": string,
                            "value_type": {
                                "type": "string",
                                "enum": list(SQL_VALUE_TYPES),
                            },
                        }
                    )
                ),
                "outputs": self._output_schema(),
                "ordering": {
                    **_array(
                        _object({"column": string, "descending": {"type": "boolean"}})
                    ),
                    "minItems": len(self.meaning.ordering_origins)
                    if self.meaning is not None
                    else 0,
                    "maxItems": len(self.meaning.ordering_origins)
                    if self.meaning is not None
                    else 0,
                },
                "request_arguments": argument_schema,
                "interpretations": self._interpretation_schema(),
            }
        )
        if binding_groups:
            schema["$defs"] = {
                name: {"type": "string", "enum": list(values)}
                for values, name in binding_groups.items()
            }
        return schema

    def _input_usage_instruction(self):
        return "Consume every supplied input assigned in Requested answer.input_refs. Catalog choices are optional unless needed for this request. For a lexical category whose API representation is a documented catalog choice, add an interpretations entry connecting the original input symbol to that choice, with its contract basis. The interpreted input symbol resolves to that choice in SQL and request arguments; explicit choice symbols are also valid. This fixes that interpretation for this compiled program. Inputs supplied in the parameter menu remain direct parameter references unless explicitly interpreted; inputs supplied as resolved_reference views are consumed through those relations."

    def _identity_display_schema(self):
        return {"type": ["string", "null"]}

    def reference_usage_instructions(self):
        return (
            "A reference_literal is an original resource-address value, not a SQL identity. It may be used only in compatible REST arguments.",
            "A reference_argument menu symbol projects an established reference relation field into a REST argument. Use its declared relation and column in SQL. Resolved reference relations preserve missing and ambiguous outcomes and must be consumed as declared, rather than resolving the input again from raw text.",
        )

    def _identity_display_instruction(self):
        return "Choose one logical identity authority by entity kind, key and component IDs. Its producing API view is established by SQL lineage, not by choosing a physical carrier. For an identity, provide display_column from an observed human-readable name when one is available; this is presentation metadata for that identity, not an additional requested output."

    def output_identity_authorities(self):
        return identity_authorities(self.tables)

    def _output_schema(self):
        roles = (
            set(query_output_roles(self.meaning))
            if self.meaning is not None
            else {"value", "identity"}
        )
        variants = (
            [
                _object(
                    {
                        "kind": {"type": "string", "enum": ["value"]},
                        "column": {"type": "string"},
                        "label": {"type": "string"},
                    }
                )
            ]
            if "value" in roles
            else []
        )
        for ref, authority in (
            self.output_identity_authorities().items() if "identity" in roles else ()
        ):
            variants.append(
                _object(
                    {
                        "kind": {"type": "string", "enum": ["identity"]},
                        "authority": {"type": "string", "enum": [ref]},
                        "components": _object(
                            {
                                component: {"type": "string"}
                                for component in authority["components"]
                            }
                        ),
                        "label": {"type": "string"},
                        "display_column": self._identity_display_schema(),
                    }
                )
            )
        if not variants:
            return {"type": "array", "maxItems": 0, "items": _object({})}
        schema = _array(variants[0] if len(variants) == 1 else {"anyOf": variants})
        if self.meaning is not None:
            schema.update(
                minItems=0
                if "existence" in self._result_modes()
                else len(self.meaning.output_origins),
                maxItems=len(self.meaning.output_origins),
            )
        return schema

    def _result_modes(self):
        if self.meaning is None:
            return ["scalar", "existence", "rows"]
        modes = ["rows"]
        if len(self.meaning.output_origins) == 1 and not self.meaning.ordering_origins:
            modes.append("scalar")
            if self.meaning.result_kind == "scalar" and self.meaning.output_kinds == (
                "value",
            ):
                modes.append("existence")
        return modes

    def _interpretation_schema(self):
        inputs = [
            name for name, item in self.parameters.items() if item.get("may_interpret")
        ]
        choices = [
            name
            for name, item in self.parameters.items()
            if item.get("kind") == "catalog_choice"
        ]
        if not inputs or not choices:
            return {"type": "array", "maxItems": 0, "items": _object({})}
        return _array(
            _object(
                {
                    "input": {"type": "string", "enum": inputs},
                    "choice": {"type": "string", "enum": choices},
                    "basis": {"type": "string"},
                }
            )
        )

    def _argument_schema(self, binding_groups):
        variants = []
        for view, table in self.tables.items():
            for parameter in table.get("request_parameters", ()):
                names = tuple(
                    name
                    for name, description in self.parameters.items()
                    if compatible_argument(parameter, description, literal_lookup=(
                        getattr(self.meaning, 'reference_kind', None) == 'literal' and
                        description.get('input_ref') == getattr(self.meaning, 'reference_input_ref', None)))
                )
                binding_options = []
                if names:
                    group = binding_groups.setdefault(names, "bound_value_" + str(len(binding_groups) + 1))
                    binding_options.append({"$ref": "#/$defs/" + group})
                if reference_binding := self.reference_argument_schema(parameter):
                    binding_options.append(reference_binding)
                if not binding_options:
                    continue
                binding_schema = binding_options[0] if len(binding_options) == 1 else {"anyOf": binding_options}
                variants.append(
                    _object(
                        {
                            "view": {"type": "string", "enum": [view]},
                            "instance": {"type": ["string", "null"], "minLength": 1},
                            "parameter_ref": {
                                "type": "string",
                                "enum": [parameter["param_ref"]],
                            },
                            "binding": binding_schema,
                        }
                    )
                )
        if not variants:
            return {"type": "array", "maxItems": 0, "items": _object({})}
        return _array(variants[0] if len(variants) == 1 else {"anyOf": variants})

    def reference_argument_schema(self, parameter):
        return None

    def _unavailable_schema(self):
        return _object(
            {
                "unavailable": {"type": "boolean", "enum": [True]},
                "reason": {"type": "string", "minLength": 1},
            }
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={
                "submit_query_answer": self._schema(),
                "report_query_unavailable": self._unavailable_schema(),
            }
        )

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_query_answer",
                    tool_description="Submit a typed relational query answer.",
                    input_schema=self._schema(),
                ),
                required_tool_spec(
                    tool_name="report_query_unavailable",
                    tool_description="Explain why available contracts cannot support the requested fact.",
                    input_schema=self._unavailable_schema(),
                ),
            )
        )
