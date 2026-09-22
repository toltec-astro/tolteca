# 02b9 — Require recorded scalar/string settings for raw metadata

V2's `_nc_node_mapper_defs['meta']` reads attenuation, calibration identifiers,
maximum tone count, original path and repeat level unconditionally for raw KIDs
data. All four reference acquisitions contain them. Returning None after silently
omitting one was an unapproved weakening of that contract, not compatibility
needed by the reference acquisition format.

## Change and preserved boundaries

Full raw metadata now requires those fields. The existing mapped getters report
missing values: numeric getters raise ValueError and `get_str` raises KeyError.
No new exception layer, fallback value, header name registry or validation cache
is introduced. The already computed data kind supplies one `is_raw` predicate.
Repeat uses the existing scalar loop rather than its own duplicate branch.

Processed metadata can still omit the same fields. Its broader model boundary
remains a separate review item; this commit does not silently tighten that API.
Valid acquisition values, source identification, catalog columns, numerical
reduction and array laziness are unchanged. Metadata models retain nullable fields
for now because they are also used for processed data. This is a reader-boundary
correction, not a claim that strict header-only model composition is complete.

Remaining raw omissions include design/sweep dimensions, the ADC variable
reference, repetitions and LO center frequency. Those have different shape or
model consequences and are not hidden inside this scalar/string correction.

## Tests and self-review

- Before implementation, 32 real-file omission tests fail because no exception
  is raised; one deliberately synthetic processed encoding passes. Each negative
  case drops a header from an in-memory view, never edits the fixture.
- Existing toy numerical tests obtain their extra readout settings from the real
  target-sweep header through a shared explicit fixture. Their toy geometry and
  identities remain labelled numerical tests, not evidence of acquisition format.
  No autouse patch, fixture mutation or fabricated calibration fallback is used.
- The first full run had 999 passing tests and five failures caused by expecting
  ValueError from the string getter. Its documented KeyError is retained and the
  test expectations corrected, rather than changing Tollan to satisfy the tests.
- **1,004 full Tolteca tests pass in 55.66 s**, with the actual v2 interpreter and
  benchmarks disabled. Log: `/tmp/tolteca-20260922-required-headers-final.log`.
- A fresh four-case end-to-end comparison passes exact values/units for VNA,
  target, first tune block and last tune block, through direct/planned no-op,
  reduction, Zarr, local SQLite/Parquet and Web traces/Dash callbacks. Artifacts:
  `run/overnight-slice-20260922-required-headers/` under the development location.
- Scoped Ruff, formatting and ty pass. Existing unrelated core/test-accessor lint
  debt is not reformatted. `git diff --check` passes; `just review` is regenerated.
  Reference data, DB and Web checkouts remain clean.

Self-review confirms valid-file reads are not increased: all fields were already
read when present. The changed failure path is the intended observable behavior.
No production service, dependency change or fixture write is involved.
