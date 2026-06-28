"""Catalog fact availability helpers."""

from __future__ import annotations

from fervis.lookup.relation_catalog.model import (
    CatalogFact,
    CatalogFactAvailability,
    RelationCatalog,
)


def catalog_facts(catalog: RelationCatalog) -> tuple[CatalogFact, ...]:
    facts: list[CatalogFact] = []
    facts.extend(catalog.facts)
    for read in catalog.reads:
        facts.extend(
            fact if fact.read_id else _fact_with_read_id(fact, read_id=read.id)
            for fact in read.facts
        )
        facts.extend(_field_fact(read.id, field.ref) for field in read.fields)
    return tuple(_dedupe_facts(facts))


def catalog_fact_by_ref(catalog: RelationCatalog) -> dict[str, CatalogFact]:
    return {fact.ref: fact for fact in catalog_facts(catalog)}


def blocked_catalog_fact(fact: CatalogFact) -> bool:
    return fact.availability != CatalogFactAvailability.AVAILABLE


def _field_fact(read_id: str, field_ref: str) -> CatalogFact:
    return CatalogFact(
        ref=field_ref,
        availability=CatalogFactAvailability.AVAILABLE,
        field_ref=field_ref,
        read_id=read_id,
        proof_refs=(f"catalog:{read_id}:{field_ref}",),
    )


def _fact_with_read_id(fact: CatalogFact, *, read_id: str) -> CatalogFact:
    return CatalogFact(
        ref=fact.ref,
        availability=fact.availability,
        field_ref=fact.field_ref,
        read_id=read_id,
        proof_refs=fact.proof_refs,
    )


def _dedupe_facts(facts: list[CatalogFact]) -> tuple[CatalogFact, ...]:
    by_ref: dict[tuple[str, str], CatalogFact] = {}
    for fact in facts:
        by_ref.setdefault((fact.read_id, fact.ref), fact)
    return tuple(by_ref.values())
