# 02b13 — Typed acquisition headers without tune-array scans

The catalog gap measured in 02b4 needs typed reader metadata, but full tune
metadata counts acquired sweeps by scanning the LO vector. Programmed step
count, repetitions and allocated capacity do not need that scan. Separate these
concepts instead of constructing an incomplete full sweep record.

## Scope and observable behavior

- Add `ToltecSweepMetadataHeader` for sample metadata and programmed settings.
  Existing `ToltecSweepMetadata` extends it with actual sweep/block counts. Every
  existing full-model field and value remains; serialized field order may change.
- Add mapper `get_metadata_header` and cached accessor `meta_header` for
  unpartitioned raw acquisitions. Timestreams return their existing complete
  metadata type; sweeps return a header type without `n_sweeps`/`n_blocks`.
- Share reader-field, sample and geometry builders. There is no independent
  catalog-to-header-name map, hidden cache or new numerical computation. Full
  tune detection and selected-block semantics are unchanged. Missing raw capacity
  now fails before a tune LO scan.
- Processed and explicitly block-indexed inputs reject the new API; their
  existing `meta` behavior remains. The first negative test caught a scalar-type
  read preceding the block check. The explicit dimension guard now runs first.
- Existing nullable model fields still support processed records. Reader-level
  raw requiredness is enforced; this is not globally strict raw dataclass
  construction. Catalog composition remains a separate step.

This follows Pydantic's documented inherited-dataclass validation and
`TypeAdapter` serialization ([documentation](https://docs.pydantic.dev/latest/concepts/dataclasses/)).
No partial construction or duplicate scientific-field definitions are used.

## Verification and self-review

Eleven new tests compare every header field against full metadata on four real
inputs, reject missing fields and unsupported layouts, test cached access and
JSON roundtrips, and prohibit `Data.*`, detector-position and ADC backend reads.
**1,050 tests pass in 55.07 s**, actual-v2 parity enabled, benchmarks disabled.
Log: `/tmp/tolteca-20260922-header-only-full.log`.

Fresh four-case exact integration passes through CLI/planned no-op, reducer,
Zarr, local SQLite/Parquet, Web traces and Dash callbacks. Artifacts:
`run/overnight-slice-20260922-header-only/` under the development location.
No fixture writes, tolerance changes or production services.

`catalog_metadata.py` now measures header-only reads independently. Output:
`/tmp/tolteca-20260922-header-only-reads.json`. VNA/target use 20 reads/604 logical
bytes in both paths. Tune falls from 21 reads/14,764 bytes to 20/604, avoiding
the 3,540-element float32 LO vector. Timestream uses 18 reads/596 bytes in both.
These are returned backend values, not physical disk/network traffic. Requesting
both independently cached APIs may read scalar headers twice.

Scoped lint (excluding repository-wide CPY001), formatting and ty pass. Existing
core style debt is untouched. Public docstrings describe accepted inputs and
read costs, not history. Next: typed source composition and vectorized catalog
selection, without object-column traversal or another scientific-field map.
