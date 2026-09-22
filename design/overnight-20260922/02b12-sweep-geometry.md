# 02b12 — Recorded raw sweep geometry, not missing-field fallbacks

V2 reads raw sweep repetitions, sweep capacity and the acquired time dimension.
V3 had allowed missing repetitions to become None, missing capacity to become
the actual block count, and missing time to become the number of sweep steps.
None of those fallbacks is needed for the reference raw acquisition format.

## Change and scope

Require these three fields when the already determined kind is raw. Use the
same schema/mapped getters and keep processed-data fallbacks unchanged. Valid
counts, tune LO-reset detection, selected-block counts and exposure calculations
are unchanged. No new I/O or schema names are introduced.

The sample-dimension requirement applies to explicitly block-indexed raw inputs
too. Toy tests with one value per step and no acquisition time axis now explicitly
declare `processed_sweep`, matching their layout. Their numerical arrays and
per-block LO centers are unchanged. Toy raw datasets with a time axis declare
their small allocated capacity explicitly. Neither category is used as evidence
for a real filename or acquisition convention; actual-file tests establish the
raw contract independently.

No metadata class is renamed or made globally strict in this step. The existing
dataclasses still serve processed inputs as well as raw inputs. Their nullable
annotations and the header-only catalog model boundary remain distinct work.
The existing mapped candidate order for header/dimension representations is not
changed. This is not an integrity validator comparing every header with arrays.

## Regressions and verification

- Twelve real-file cases cover the three sweep acquisitions with missing step
  header, repetition header, capacity or mapped time dimension. Before the patch,
  nine incorrectly succeed; the existing missing-step error was already KeyError.
  The initial test run also reported three expectation failures for that existing
  KeyError (the test had expected ValueError). The getter is unchanged; the test
  now preserves its actual exception.
- One synthetic processed layout explicitly verifies that raw repetitions,
  capacity and a time axis are not required there.
- **1,039 full Tolteca tests pass in 53.43 s**, actual-v2 parity enabled,
  benchmarks disabled. Log: `/tmp/tolteca-20260922-sweep-geometry-final.log`.
- Fresh four-case exact v2 integration passes through direct/planned no-op,
  reducer, Zarr, local SQLite/Parquet, Web traces and Dash callbacks. Artifacts:
  `run/overnight-slice-20260922-raw-metadata/` under the development location.
- The repeated omission diagnostic now rejects **all 47 tested raw header
  omissions**, versus silent acceptance at the initial audit. Result:
  `/tmp/tolteca-20260922-raw-contract-final.json`, generated with the committed
  `design/probes/kids_metadata_contract.py`. Dimension omissions are covered by
  the additional tests, not included in that 47-case count.
- On saved reduced children, the raw accessor no longer invents VNA/target sample
  counts/exposure: it reports missing `numSweeps` before reaching the time check.
  Tune children still reject their block-promoted scalar header. The supported
  `ReducedSweepView` path passes exact integration; raw `.meta` on a product child
  remains an unsupported route, not an acquisition-metadata source.
- Scoped lint/format/ty, diff whitespace checks and `just review` pass. Unrelated
  legacy core/accessor-test style debt is preserved. Reference, DB and Web trees
  remain clean. No fixture writes, tolerance changes or production services.

Self-review confirms that `is_raw` only gates existing getter/fallback choices;
it does not change kind identification or trigger a second scan. The remaining
catalog work should reuse a typed header/dimension-only builder, not call full
tune metadata (which needs the LO vector) just to select on attenuation. A
coherent raw/processed metadata model boundary must be chosen explicitly rather
than hiding a partial full model in the source record.
