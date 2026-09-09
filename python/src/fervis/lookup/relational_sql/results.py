"""Deterministic answer shape, projection and selection over query results."""
from dataclasses import dataclass
from fervis.lookup.relational_sql.execution import QueryValidationError


@dataclass(frozen=True)
class ResultOrder:
    column: str
    descending: bool = False


@dataclass(frozen=True)
class ResultContract:
    mode: str
    output_columns: tuple[str, ...] = ()
    ordering: tuple[ResultOrder, ...] = ()
    selection: str = 'all'
    limit: int | None = None

    def __post_init__(self):
        if self.mode not in {'existence', 'scalar', 'rows'}:
            raise QueryValidationError('Unknown result mode')
        if self.mode != 'rows':
            if self.output_columns or self.ordering or self.selection != 'all' or self.limit is not None:
                raise QueryValidationError('Scalar and existence contracts cannot select rows')
            return
        if not self.output_columns or len(set(self.output_columns)) != len(self.output_columns):
            raise QueryValidationError('Row output columns must be nonempty and unique')
        if any(not name for name in self.output_columns):
            raise QueryValidationError('Row output columns must be named')
        if len({item.column for item in self.ordering}) != len(self.ordering):
            raise QueryValidationError('Ordering repeats a column')
        if self.selection not in {'all', 'first_with_ties', 'take_with_ties', 'position_with_ties'}:
            raise QueryValidationError('Unknown result selection')
        if self.selection != 'all' and not self.ordering:
            raise QueryValidationError('Bounded result selection requires ordering')
        if self.selection in {'take_with_ties', 'position_with_ties'}:
            if type(self.limit) is not int or self.limit < 1:
                raise QueryValidationError('Result selection requires a positive boundary')
        elif self.limit is not None:
            raise QueryValidationError('This result selection has no explicit limit')
