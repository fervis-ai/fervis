"""Source-owned evidence relating request values to admitted row populations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterRowValues:
    argument: str
    row_values: tuple[str, ...]


@dataclass(frozen=True)
class ParameterPopulation:
    """A categorical row predicate, or request values that disable filtering.

    This is semantic source authority, not an inference from legal arguments.
    Row-value mappings are usable for exhaustiveness only against a declared,
    non-nullable finite field domain.
    """

    preserves_population: bool = False
    preserves_default: bool = False
    field_path: str = ""
    value_mapping: tuple[ParameterRowValues, ...] = ()
    unfiltered_values: tuple[str, ...] = ()
    comparison_operator: str = ""

    def __post_init__(self):
        if self.comparison_operator and (not self.field_path or self.value_mapping or self.comparison_operator not in {"equals", "gt", "gte", "lt", "lte"}):
            raise ValueError("parameter comparison requires one returned field and a supported scalar operator")
        if self.preserves_population and (self.field_path or self.value_mapping or self.unfiltered_values):
            raise ValueError("a non-filtering parameter cannot also declare row restrictions")
        arguments = tuple(item.argument for item in self.value_mapping)
        if len(set(arguments)) != len(arguments):
            raise ValueError("population mapping repeats an argument")
        if self.value_mapping and not self.field_path:
            raise ValueError("population mapping requires a returned field")
        if len(set(self.unfiltered_values)) != len(self.unfiltered_values):
            raise ValueError("population mapping repeats an unfiltered argument")
        if any(
            len(set(item.row_values)) != len(item.row_values)
            for item in self.value_mapping
        ):
            raise ValueError("population mapping repeats a returned value")

    def to_public_dict(self):
        return {
            "preservesPopulation": self.preserves_population,
            "preservesDefault": self.preserves_default,
            "fieldPath": self.field_path,
            "valueMapping": [
                {"argument": item.argument, "rowValues": list(item.row_values)}
                for item in self.value_mapping
            ],
            "unfilteredValues": list(self.unfiltered_values),
            **({"comparisonOperator": self.comparison_operator} if self.comparison_operator else {}),
        }
