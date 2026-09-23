"""SQL output-to-input field lineage, independent of public output roles."""
from sqlglot import exp, parse_one
from sqlglot.lineage import lineage
from sqlglot.errors import SqlglotError
from fervis.lookup.plan_execution.errors import VerificationError


def output_column_lineage(query, columns_by_view):
    try:
        return lineage(None, query, schema={name:{column:'TEXT' for column in columns}
            for name,columns in columns_by_view.items() if columns}, dialect='duckdb', trim_selects=False)
    except (SqlglotError, ValueError) as exc:
        raise VerificationError('SQL output has no field lineage') from exc


def unchanged_field_origins(node, path=(), *, preserve_occurrences=False):
    if preserve_occurrences and isinstance(node.source, exp.Select) and any(node.source.args.get(key) for key in ('joins','distinct','group','having','qualify','limit','offset')):
        raise VerificationError('Observed reference provenance must preserve candidate occurrences')
    if preserve_occurrences and (len(node.downstream) > 1 or any(node.source.find_all(exp.SetOperation))):
        raise VerificationError('Observed reference provenance cannot change occurrences through set operations')
    expression = node.expression
    while isinstance(expression, exp.Alias):
        expression = expression.this
    if isinstance(expression, exp.Table):
        column = parse_one(node.name, into=exp.Column, dialect="duckdb").name
        return {(expression.name, column, (*path, expression.alias_or_name))}
    if not isinstance(expression, exp.Column):
        raise VerificationError(
            "SQL outputs must preserve observed field values without computation"
        )
    next_path = (*path, expression.table) if expression.table else path
    # A projected column can have alternative producers without an explicit
    # set-operation node: SQLGlot flattens UNIONs behind CTE references.
    return {
        origin
        for index, child in enumerate(node.downstream)
        for origin in unchanged_field_origins(
            child,
            (*next_path, ("set_branch", index))
            if len(node.downstream) > 1
            else next_path,
            preserve_occurrences=preserve_occurrences,
        )
    }
