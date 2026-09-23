"""Named projections of a physical relation for composed programs."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RelationView:
    name: str
    relation_id: str
    columns: Mapping[str, str]
