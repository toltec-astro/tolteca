# 02b10 — LO-center metadata for both timestreams and sweeps

The v2 `KidsData` metadata group reads `Header.Toltec.LoCenterFreq` for every
KIDs acquisition. V3 exposed it only on sweep metadata, and silently accepted a
missing raw value. This is an omission in the port, not an acquisition-format
difference.

## Changes and boundaries

- Move `f_lo_center` to `ToltecTimeStreamMetadata`, inherited by sweep metadata.
  Its value remains a float in Hz, so existing sweep serialization is unchanged;
  timestream serialization now includes the field.
- Read the recorded center for timestreams too, through the existing schema and
  unit-aware scalar helper. Raw metadata requires it. Processed metadata can
  still omit it and obtain None. Do not substitute a mean of the LO data vector.
- Preserve explicitly block-indexed centers: select the block before reading
  its scalar. The existing private builders receive the already determined kind
  or raw predicate, rather than independently guessing requiredness. Passing the
  kind also removes a repeated identification call in the sweep builder.
- No changes to reducer calculations, public method signatures, source catalog
  columns, native LO arrays or acquisition timing.

The extra read for a raw timestream is **one float64 scalar (8 logical bytes)**.
Sweep center reads remain one scalar per selected block. I/Q and ADC payloads
remain lazy. These are logical backend values, not physical disk-byte claims.

## Evidence and QA

Twelve new cases cover all four recorded acquisitions, missing centers, equivalent
MHz representation on raw/processed timestreams, nullable processed centers,
JSON round-trip and backend read counts. Before the patch, eight fail and four
pass. Existing explicitly block-indexed tests continue to check different centers
per block. The missing-center regression leaves a sample LO array present and
still requires the header, ruling out a hidden data-derived fallback.

The actual v2 and v3 interpreters independently report **675,190,000 Hz** for all
four files. Reports: `/tmp/tolteca-20260922-lo-{v2,v3}.json`, produced with the
committed `design/probes/catalog_metadata.py`. The v3 trace records exactly one
8-byte LO-center read for each acquisition.

**1,016 full Tolteca tests pass in 53.81 s**, with actual-v2 parity enabled and
benchmarks disabled. Log: `/tmp/tolteca-20260922-lo-metadata-full.log`. Scoped
Ruff/format/ty and `git diff --check` pass; `just review` is regenerated. The
previous 02b9 full CLI/Zarr/DB/Parquet/Web integration is retained; this step does
not claim a separate full product integration run beyond the full test suite.

Self-review: no frequency arithmetic is changed for sweeps; the timestream path
reuses the same conversion, and omission errors retain the existing mapped scalar
exception. No data-kind change, new cache or model-copy layer is introduced.
Remaining required design/sweep dimensions and ADC-reference presence are not
silently included in this correction.
