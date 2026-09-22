# 02b15 — Typed source metadata and vectorized catalog projection

This step adds composition and storage/query support. Automatic file opening
still uses the previous identity-only KIDs reader; wiring header enrichment into
CLI preparation is the next bounded change. Do not present this checkpoint as a
fix for `build_input_catalog(...).query('atten_drive ...')` yet.

## Contract and implementation

`SourceInfoModel.metadata` is optional typed acquisition metadata. A filename
guess has None, not a partial observation model. Supported records are telescope
metadata, raw sweep header metadata and raw timestream metadata. `data_kind`
selects the model when restoring dictionaries; no additional serialized type tag
is needed. A cached table constructs one TypeAdapter per model, reused by its
atomic and family kinds. Imports are deferred to avoid the existing root-file /
telescope-metadata dependency cycle. This is static adapter caching, not cached
source values or file contents.

`SerializeAsAny` retains domain fields under the common metadata base annotation.
Restored dictionaries use normal (non-strict Python-instance) dataclass
validation. The reader's strict construction and raw requiredness are unchanged.
An incompatible metadata *instance* rejects; dictionary extras retain the domain
dataclasses' existing ignore behavior. This is not a new cross-field integrity
validator checking kind, source identifiers and all header values. The public
schema describes the shared metadata base, not a closed union of every reader.

The catalog stores metadata in an Arrow struct inferred from supplied records.
Filename-only/empty catalogs have an Arrow null metadata column. Missing fields
in a mixed telescope/KIDs catalog are null. Only present model fields are exposed:
there is no huge nullable cross-instrument model or hardcoded scientific-column
list. JSON, Python and mixed Parquet restoration reconstruct the concrete models.

Queries support `metadata.atten_drive` and scalar shorthand `atten_drive`.
Shorthands are Arrow field projections, resolved only when referenced. Existing
source columns and filename aliases take precedence: `master` stays a source
label, `metadata.master` is the recorded code. The existing alias resolver now
serves both filename and metadata fields. It stays query-local because pandas
may overwrite resolved entries during expression evaluation.

`model_dump(context={'flat': True})` provides the same scalar aliases to
conditional configuration. `round_trip=True` omits those added aliases so the
result remains valid model input. Quantities and coordinates keep their existing
parseable strings; observation times are numeric Unix seconds after 02b14.
No implicit unit stripping or arithmetic on unit-bearing strings is introduced.
Nested expressions remain a catalog feature; Tollan's condition engine receives
the scalar aliases and is unchanged.

## Rejected alternatives and remaining boundaries

- Arbitrary DataFrame-only scalar extras would disappear on source restoration
  and would not reliably reach `enable_if`.
- A second model tag or an instrument-wide nullable DTO duplicates information
  already provided by `data_kind` and the domain models.
- Python object-column traversal would repeat per-row work on every query;
  Arrow struct projection avoids that.
- Calling full tune metadata would scan the LO vector needlessly; use 02b13's
  header-only reader when automatic enrichment is connected.

The inferred Arrow schema is appropriate for this in-memory catalog. A fixed
cross-batch Parquet dataset schema/union policy remains a DB-layer design issue;
do not assume independent catalogs can always be concatenated without schema
alignment. Source identifiers are still stored independently of metadata and
must be populated consistently by the reader. No broad DB/Web redesign here.

## Tests and self-review

Five focused tests use the four actual KIDs files plus the actual telescope
timestream. They verify concrete types and values through JSON and mixed Parquet,
exact Time/exposure and target/reference coordinates, vectorized dotted/short
queries without model serialization/restoration during queries, matching flat
`enable_if` context, invalid kinds/instance types, and absent metadata.

These tests exposed two pre-existing losses, fixed separately in 02b14 and
Tollan 02b14a. Telescope timestamp ISO rounding reached 0.424 ms; coordinate
formatting also rounded native doubles. Neither test was replaced by a tolerance.
An initial test wrongly assumed a tel2 fixture existed; no example was invented.

The initial full run had one catalog assertion failure because the newly added
metadata column is Arrow null rather than Python object. The scalar-column test
now excludes both struct fields; their types/roundtrips have dedicated coverage.
The final checkpoint passes **1,056 full tests in 56.42 s**, actual-v2 parity
enabled, benchmarks disabled. Log:
`/tmp/tolteca-20260922-source-metadata-checkpoint.log`. Scoped Ruff (existing
CPY001 excluded), formatting and ty pass;
the model wrapper intentionally omits its return annotation so Pydantic retains
the existing model JSON schema rather than replacing it with a generic dict.

Fresh four-case exact v2 integration through direct/planned no-op, Zarr, local
SQLite/Parquet, Web traces and Dash callbacks passes at
`run/overnight-slice-20260922-source-metadata/`. That checks regression safety;
the automatic catalog path does not yet attach the new metadata. No fixture
writes, tolerance changes, external services or dependency upgrades.

Next-step review must distinguish header-backed selection from array-derived
facts such as actual tune block count. This source field accepts the header
record, not a full sweep metadata instance. Do not imply those array-derived
fields are now available. Re-identification must replace/clear an existing
metadata snapshot when it refreshes header identifiers; otherwise it could
combine fresh identifiers with stale scalar settings.
