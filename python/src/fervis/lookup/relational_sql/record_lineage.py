"""Observed record fields cannot absorb computed measures from a SQL answer."""
from .column_lineage import output_column_lineage, unchanged_field_origins
from fervis.lookup.plan_execution.errors import VerificationError


def record_field_origins(query, columns_by_view, fields, *, preserve_occurrences=False):
    nodes = output_column_lineage(query, columns_by_view)
    result = {}
    for field in fields:
        node = nodes.get(field.casefold())
        if node is None:
            raise VerificationError('Record field must be an observed SQL column')
        try:
            origins = unchanged_field_origins(node, preserve_occurrences=preserve_occurrences)
        except VerificationError as exc:
            raise VerificationError('Record fields must preserve observed properties; computed measures require separate value outputs') from exc
        if not origins:
            raise VerificationError('Record field has no observed source')
        result[field] = {(view.casefold(), column.casefold()) for view,column,_ in origins}
    return result
