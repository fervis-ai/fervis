"""Catalog-backed ways to acquire required request values from other reads."""

from dataclasses import dataclass
from fervis.lookup.relation_catalog.row_sources import RowSource
from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
from fervis.lookup.plan_execution.declared_values import (
    declared_comparison_types_compatible,
)


@dataclass(frozen=True)
class AccessArgument:
    parameter_ref: str
    parent_field_ref: str


@dataclass(frozen=True)
class ReadDependency:
    source_ref: str
    parent_source_ref: str
    arguments: tuple[AccessArgument, ...]
    coverage_basis: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadAccessCatalog:
    sources: tuple[RowSource, ...] = ()
    dependencies: tuple[ReadDependency, ...] = ()

    def source(self, ref: str) -> RowSource:
        return next(source for source in self.sources if source.id == ref)

    def alternatives(self, ref: str) -> tuple[ReadDependency, ...]:
        return tuple(item for item in self.dependencies if item.source_ref == ref)

    def can_enumerate(
        self, source: RowSource, *, visiting: frozenset[str] = frozenset()
    ) -> bool:
        if source.id in visiting:
            return False
        # A callable list with an unexplained default restriction is not a full key domain.
        if any(
            param.source == "query"
            and (param.default is not None or not param.default_is_known)
            and param.semantics != "response_shape"
            and not (
                param.population
                and (
                    param.population.preserves_population
                    or param.population.preserves_default
                )
            )
            for param in source.params
        ):
            return False
        required = {
            param.param_ref
            for param in source.params
            if requires_caller_supplied_input(param)
        }
        if not required:
            return True
        return any(
            required <= {binding.parameter_ref for binding in dependency.arguments}
            and self.can_enumerate(
                self.source(dependency.parent_source_ref),
                visiting=visiting | {source.id},
            )
            for dependency in self.alternatives(source.id)
        )

    def supplied_parameters(self, source: RowSource) -> frozenset[str]:
        required = {
            param.param_ref
            for param in source.params
            if requires_caller_supplied_input(param)
        }
        if any(
            required <= {arg.parameter_ref for arg in dependency.arguments}
            and self.can_enumerate(self.source(dependency.parent_source_ref))
            for dependency in self.alternatives(source.id)
        ):
            return frozenset(required)
        return frozenset()

    def validate(self) -> None:
        sources = {source.id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("access catalog repeats a source")
        for dependency in self.dependencies:
            if (
                not dependency.coverage_basis.strip()
                or dependency.source_ref == dependency.parent_source_ref
            ):
                raise ValueError(
                    "read dependency requires an independent parent and coverage basis"
                )
            source = sources[dependency.source_ref]
            parent = sources[dependency.parent_source_ref]
            fields = {
                field.field_ref: field
                for field in (*parent.fields, *parent.request_argument_fields)
            }
            params = {param.param_ref: param for param in source.params}
            refs = [binding.parameter_ref for binding in dependency.arguments]
            if not refs or len(refs) != len(set(refs)):
                raise ValueError("read dependency repeats or omits argument mappings")
            for binding in dependency.arguments:
                if not declared_comparison_types_compatible(
                    fields[binding.parent_field_ref].type.value,
                    params[binding.parameter_ref].type.value,
                ):
                    raise ValueError(
                        "read dependency field cannot supply its argument type"
                    )
