# 04b1 — Consolidate completed product metadata

## Decision and scope

The 04b probe demonstrated hundreds of small metadata reads on opening the
write-once product hierarchy. Enable xarray's native consolidated-metadata writer
for Zarr format 2 and require that index during the writer's verification reopen.
Only after verification does the existing completion manifest get published.

The public reader uses xarray's `consolidated=None` policy: prefer the index when
present; fall back to individual metadata with xarray's performance warning when
absent. Consolidation is an optional native Zarr index, not a new product payload
schema. Requiring it in the reader would needlessly reject an otherwise valid
store. No compatibility classes, alternate serializers or custom fallback logic
are introduced. Existing local review products remain inspectable.

The user-visible change is an additional `.zmetadata` file and much fewer opening
reads. Products remain read-only after publication; modifying a completed store's
metadata without refreshing its index can produce a stale view. The public guide
instructs callers to publish a new result rather than mutate a completed product.
No changes to scientific arrays, chunk shapes, codecs, Zarr format, manifest
format, raw-file reads, DB models or Web APIs.

## Evidence / official guidance

This matches [Zarr's consolidation guidance](https://zarr.readthedocs.io/en/latest/user-guide/consolidated_metadata/)
for read-heavy, slowly changing hierarchies. Inspected installed xarray's native
writer and reader implementation as well: DataTree writing consolidates the full
hierarchy, and its automatic reader warns on fallback. New code uses those public
options rather than accessing metadata internals.

On freshly generated reference products with the public reader's automatic
policy, VNA/target opening uses six LocalStore gets, and tune uses seven. There
are no directory listings or signal chunk reads during opening. Compared with
the previous 285/297/304 gets, the improvement is metadata operations, not a
claim of equivalent network requests or measured remote latency. Coordinate
index reads remain unchanged. See 04b for the original instrumented comparison.

## Verification and self-review

Four regressions fail before the patch: metadata index absent, no failure at the
consolidation boundary, per-child metadata reads in the reader, and no indexed
store for the fallback test. All pass afterward. Tests also verify exact native
Dataset round-trip, no completion marker on consolidation failure, and readable
nonconsolidated data with the documented warning.

- 951 full Tolteca tests pass (57.13 s), actual-v2 enabled, benchmarks disabled:
  `/tmp/tolteca-20260922-consolidation-pytest.log`.
- 18 DB product tests and 10 Web tests pass; scoped Ruff/format/ty pass.
- Fresh four-case exact CLI/Zarr/DB/Parquet/Web integration passes at
  `toltec_astro_dev/run/overnight-slice-20260922-consolidated/report.json`.
  Arrays and units match actual v2 without tolerance changes.

No existing output directory or reference input was modified. The read-only
performance probe remains separate from runtime implementation. Self-review also
confirmed that consolidation does not resolve sparse-channel chunk amplification
or repeated `f_chan` loads. A separate experiment computing only selected I/Q
and shared coordinates together reduced four frequency-coordinate chunk reads
to one, with exact plotted values; that is the next bounded Web change.
