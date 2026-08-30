# Deferred

Decisions and fixes that were raised, understood, and deliberately put off.
Kept here rather than in a conversation so they survive being forgotten.

Each entry says what the thing is, why it was parked, and what has to be true
before picking it up.

---

## Embedding cost, and who pays

**Parked because** the embedding model in use is currently free, so no
decision is being forced yet.

**Comes back when** a deployment brings its own key. At that point the person
running it pays per embedded row, and a backfill over a large table is a real
bill they should see before pressing Run, not after.

Things that will need answering then: what a table's backfill would cost
before it starts, whether to embed the whole table or only a sample first,
and whether re-embedding after a model change is a decision or a default.

**Not deferred with it:** even while free, rate limits mean a large backfill
takes time. Run is therefore a long-running operation regardless of money -
it belongs in the background with visible progress, not in a request. That
part is a design constraint now, not a cost question for later.

---

## A "before" development database

**The problem.** `seeds/001_schema.sql` already has `embed_*` columns. That
models the state *after* provisioning, so it tests the query path but not the
path that creates those columns. A client's database will not have them.

**Comes back when** the provisioning path is built - it needs a schema with
ordinary text columns and no vectors, so the column-selection heuristic and
the backfill can be exercised on something realistic.

---

## `_get_filter` is case-sensitive on column names

**The bug.** In `SqlToolAdapter._get_filter`:

```python
filters[col] = self._filters[table_key].get(col, "column not found")
```

`table_key` is lowercased; `col` is not. A model that writes `Genre` when the
classifier keyed `genre` gets "column not found" for a column that exists.

**Parked because** it was found mid-way through another change and is not
worth mixing in. **Comes back with** feature 2, which owns that dictionary.

---

## `get_lsit_values` is misspelled

**The constraint.** It is a dispatch key in `_handlers`, a name in
`get_tool_schemas`, and a method name. Renaming one without the others turns
every call into an `UnknownToolError` at runtime rather than a failure at
startup, so all three move together or none do.

**Comes back when** something else is already touching that adapter. There is
a test asserting the current spelling; it changes with the rename.

---

## Listing a column's values in the filter guidance

**The idea.** For an ENUM column, telling the model `one of: novel, poetry,
drama` is far more useful than telling it the column has few distinct values -
it matches what is stored rather than what it imagines.

**Parked because** it needs a second call per ENUM column at startup and a cap
on how many values to include, and the classifier is worth getting working
first.

**Comes back after** feature 2 runs end to end and the prompts can be judged.

---

## The ENUM cutoff

**Still unset.** `ENUM_MAX_DISTINCT` and `ENUM_MAX_RATIO` in
`libs/agent_core/filter_classifier.py` are placeholders.

Measured on the development database:

| column | distinct | rows | ratio | avg words |
|---|---|---|---|---|
| `books.language` | 3 | 420 | 0.007 | 1.0 |
| `books.genre` | 10 | 420 | 0.024 | 1.1 |
| `books.summary` | 97 | 420 | 0.231 | 14.1 |
| `authors.bio` | 10 | 28 | 0.357 | 9.8 |
| `books.shelf_code` | 399 | 420 | 0.950 | 1.0 |
| `books.isbn` | 420 | 420 | 1.000 | 1.0 |
| `members.email` | 340 | 340 | 1.000 | 1.0 |
| `authors.name_en` | 28 | 28 | 1.000 | 2.2 |

Two things the numbers already settle: average word count separates every
identifier (1.0) from every semantic column (>2) with no overlap, and ratio
alone is unreliable on small tables - `city` reads as 0.015 in a 340-row
table and 0.833 in a 12-row one.

---

## Login

**Parked deliberately**, and left absent rather than faked, so nothing in the
console can be mistaken for authentication that is not there.

**Comes back last.** It is users, hashed passwords and sessions - a feature of
its own, not a field on a form.
