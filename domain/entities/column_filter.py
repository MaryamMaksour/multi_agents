"""How a column can be filtered - derived, not hand-listed.

The previous codebase kept four hand-maintained lists (semantic, word,
operation, datetime) and rescanned all four on every get_filter call. This
replaces them with a classification computed once from the introspected
schema, because almost all of it is recoverable from the database:

    OPERATOR   numeric types                      -> from data_type
    DATETIME   date / timestamp types             -> from data_type
    SEMANTIC   text column with an embed_ partner -> from column pairing
    ENUM       text column with few distinct values -> cardinality probe
    TEXT       anything else

Only the ENUM case needs to touch data, and only once per agent at startup.

This is the piece that lowers the model-capability bar: the model never has
to reason about which retrieval strategy a column supports, because the tool
hands it the answer.
"""

from __future__ import annotations

from enum import Enum

# An Enum is correct here, unlike for agent keys: this IS a closed set. New
# filter kinds are a code change, not deployment data.


class FilterKind(Enum):
    """The retrieval mode a column supports.

    TODO: define members - OPERATOR, DATETIME, SEMANTIC, ENUM, TEXT.
    """

    # TODO: implement
    ...


class ColumnFilter:
    """A column's filter kind plus the guidance string handed to the model.

    Fields to define:

        column     str
        kind       FilterKind
        guidance   str  - the sentence the model reads, e.g. how to write the
                          vector predicate for a SEMANTIC column, or which
                          values exist for an ENUM one

    TODO: decide whether `guidance` is stored or generated on read. Generating
    keeps the entity small and the wording in one place; storing lets a
    deployment override the wording per column. Given that per-deployment
    annotation is a planned feature, storing is probably right - but then the
    default wording still has to live somewhere single.
    """

    # TODO: implement
    ...


# TODO: the ENUM cutoff. Measured on the development schema:
#   genre 10 distinct / 420 rows, language 3/420, tier 3/340, status 3/4200
#   isbn 420/420, shelf_code 399/420
# So `distinct <= 50 or distinct/total < 0.05` separates them cleanly there.
# Confirm the ratio arm earns its place before keeping both.
ENUM_MAX_DISTINCT = ...
ENUM_MAX_RATIO = ...
