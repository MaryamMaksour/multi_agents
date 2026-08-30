from __future__ import annotations
from dataclasses import dataclass     

from enum import Enum



class FilterKind(Enum):
    """
        The retrieval mode a column supports.
    """

    SEMANTIC       = "semantic_search"
    OPERATOR       = "operator_search"
    DATETIME       = "date_time_search"
    ENUM           = "enum_search"
    TEXT           = "text_search"
    VECTOR_STORAGE = "vector_storage"


@dataclass(frozen=True)
class ColumnFilter:
    """
    A column's filter kind plus the guidance the model reads.

    The kind is for us; the guidance is for the model. `SEMANTIC` tells the
    SQL builder which predicate shape applies, while the guidance is the
    sentence that gets handed over verbatim.
    """

    column:     str
    kind   :    FilterKind
    guidance:   str


ENUM_MAX_DISTINCT = 20

