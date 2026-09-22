# 02b1 — Do not identify telescope time as KIDs data

## Reproduction and decision

All four actual telescope fixtures have a `time` dimension. The KIDs mapper's
structural fallback used that dimension alone to report RawTimeStream. Passing
a correctly guessed LmtTel source through `identify_source` therefore changed
its kind to RawTimeStream; `run_source` calls that identifier even for no-op
execution. An unknown-name telescope input could also acquire this false kind.

Require the mapped raw I/Q pair before structural sweep/timestream inference.
Use the existing mapper membership checks, which resolve fields without reading
their values. Explicit `Header.Kids.kind` / observation-type classification is
unchanged, including header-only inputs and processed-kind handling.

This is intentionally narrower than a general reader registry or catalog
opening policy. It prevents a false classification without inventing telescope
metadata or inserting telescope-specific conditions into the KIDs mapper.
Header-backed telescope identity/fallback and typed catalog projection remain
step 02b work.

## Behavior change

Datasets with only time/sweep-size information and no explicit type header or
mapped raw I/Q pair now return Unknown instead of a KIDs kind. Valid raw KIDs
structures and explicit headers retain their classification. A filename-known
telescope remains LmtTel through the no-op runner; an anonymous telescope is not
automatically identified by this KIDs-specific function. No values are invented
for it. There is no new exception type or eager array read.

## Tests and self-review

- Before the fix, the telescope regression failed with RawTimeStream instead of
  Unknown. Four real KIDs fallback cases (type header deliberately removed)
  already passed and still pass afterward.
- Tests forbid all NetCDF variable reads during structural identification. They
  cover all four telescope payloads, preserving known filename kind, and the
  anonymous-source case. The no-op regression checks the live serialized source
  record is still LmtTel/tel/145609 without reading telescope values.
- 119 focused tests passed before adding the final no-op regression; the final
  focused identification/no-op run has 21 passes. Full-suite checkpoint results
  are recorded in the index.
- Scoped typing passes. The changed tests pass Ruff/formatting. The raw core
  retains its six existing Ruff findings and an existing formatting difference
  in `_get_frequency`; do not rewrite unrelated lines. An extra early return
  initially introduced a seventh lint finding, removed by nesting the fallback
  under the positive I/Q condition instead of adding a suppression.
- Independent diff review: only structural fallback, its public contract,
  regression tests and this record changed. No metadata model/default, filename
  registry, acquisition type mapping, selector syntax, or numerical code changed.
