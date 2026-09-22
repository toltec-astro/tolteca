# 02b16 — Attach typed headers during catalog preparation

## Scope / observable changes

The shared `filestore.identify_source` now attaches typed acquisition metadata.
Its telescope branch retains the strict model it already constructed. Its KIDs
branch requests the existing header-only reader through a new optional
`read_metadata=True` keyword on the KIDs identifier. The domain helper defaults
to its previous inexpensive identifier-only behavior; it remains useful for
partial/processed data. There is one identifier correction/consistency policy,
not a separate metadata-to-source projection with different precedence.

As a result, the default `build_input_catalog`, CLI `--select`, `--show_inputs`,
`--show_config`, no-op execution and actual reduction have typed header snapshots.
Scalar headers such as `atten_drive` reach both vectorized catalog queries and
per-input `enable_if`. The CLI itself needs no new projection logic. Manifest
input records now retain this same typed metadata through serialization.

Raw acquisition preparation is stricter: missing required headers fail even
for no-op/inspection, using the reader's established errors. The low-level
identifier still supports partial records by default. Filename-only and fallback
opening policies are unchanged. Fallback opens only unrecognized filenames,
not every record with absent metadata. Processed and unsupported sources retain
identifier handling and have no acquisition metadata snapshot.

Each identification replaces the previous snapshot; the identifier-only path
clears it. A fresh accessor is used for each call, so an existing cached accessor
snapshot cannot hide a deliberate in-memory source mutation. This is not a file
lock or transactional guarantee across separate discovery/execution opens.

For a telescope with unrecognized original/current filenames, the interface is
still known but the tel/tel2 stream kind is not. The metadata adapter now accepts
the `tel` interface when kind is Unknown, without inventing a stream kind. Normal
kind-directed restoration remains unchanged. The separate 02b15a commit handles
the nullable parsed original path needed for serialization of that case.

## Read cost and performance review

The repeatable `design/probes/catalog_metadata.py` trace reports:

| Reference input | Previous identity-only catalog | Typed catalog | Full metadata |
| --- | --- | --- | --- |
| VNA / target | 7 reads, 536 bytes | 20 reads, 604 bytes | 20 reads, 604 bytes |
| Tune | 7 reads, 536 bytes | 20 reads, 604 bytes | 21 reads, 14,764 bytes |
| Timestream | 7 reads, 536 bytes | 18 reads, 596 bytes | 18 reads, 596 bytes |

These are logical backend-returned bytes, not filesystem traffic. Header arrays
such as the filename are included. No LO, I/Q, detector-position or ADC payload
is read. Dimension sizes and the ADC variable's existence/name are checked.
Actual tune block/sweep counts are deliberately absent from the header record,
not fabricated from allocated capacity. Full reader metadata remains available
when those array-derived facts are needed.

Self-review caught a cache-dependent inefficiency in the initial composition:
calling the identifier and then the header reader produced 30 backend reads for
tune with `cache=False`, versus 22 for the header reader itself. The final helper
projects identifiers from the typed snapshot, requiring 23: the extra read is
kind dispatch. The underlying header reader repeats ObsType and RoachIndex for
separate fields with caching disabled; that existing behavior is recorded, not
hidden by an assertion about all configurations. Default xarray caching gives
20 unique reads. Tests cover both cases. No long-lived source/value cache or
extra read policy was introduced. One C901 suppression keeps the two read modes
with their shared correction policy instead of extracting a many-argument helper
solely for the branch-count threshold.

## Tests / evidence

- Four real KIDs files: concrete metadata types, native master vs source label,
  scalar/dotted queries, no signal scans, and exact read counts.
- All eight reference KIDs/telescope records: mixed catalog queries and model
  restoration; header opening modes preserve absent/present metadata correctly.
- Snapshot replacement after deliberate in-memory mutation, clearing on a
  processed/identity-only read, and unknown telescope-path serialization.
- Removed Master, DriveAtten and ObsType from views of actual raw data: catalog
  preparation raises and closes the source; no fixture is changed.
- CLI selection, inspection, no-op and conditional configuration use actual
  attenuation and observation values. Direct/Deploy equivalence now includes
  a header-scalar condition.
- File-based integration tests that previously used one-header fake acquisitions
  now use reference paths or read-only symlinks. Numerical standalone tests keep
  their small arrays. Real VNA/target or tune/timestream observations replace
  invented ROACH variants in per-input configuration tests.
- Failure propagation tests assert the intended failure message as well as the
  exit code. They initially still passed for the wrong reason (missing headers)
  after strict preparation was added; this is corrected.
- All 12 Python guide blocks and four CLI examples execute. The verifier now
  checks header-query selections and the metadata Arrow struct after Parquet.

All **1,068 Tolteca tests pass in 58.54 s**, actual-v2 parity enabled and
benchmarks disabled. Log: `/tmp/tolteca-auto-metadata-checkpoint.log`. Scoped
Ruff/format/ty and Git whitespace checks pass. The existing whole-file formatting
debt in `test_core.py` is not rewritten; its scoped lint and changed hunks pass.
The first targeted run had 23 expected identity-projection/partial-fixture
failures; the first full run exposed seven additional CLI fixture failures.
Those were resolved with recorded inputs, not relaxed raw requiredness.
Fresh four-case exact v2 integration passes at
`run/overnight-slice-20260922-catalog-preparation/`, covering direct/planned no-op,
reduction, Zarr, local SQLite/Parquet, Web traces and Dash callbacks. No numerical
tolerance, fixture, production service, dependency or shared Deploy checkout
was changed.

## Remaining review boundaries

Metadata aliases are scalar serializations: unit-bearing values remain parseable
strings, not implicitly unit-stripped numbers. Telescope/processed/custom model
families beyond the supported acquisitions need explicit type adapters. Separate
catalog batches can have different inferred Arrow schemas; cross-batch DB schema
alignment remains deferred. Browser visual verification and broader DB/Web
architecture remain outside this checkpoint. Finish the final end-to-end review
and consolidate the handoff index; do not describe every historical pending
statement in earlier step records as still current.
