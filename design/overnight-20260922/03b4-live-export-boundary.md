# 03b4 — Keep the live context store out of persisted products

## Evidence and chosen boundary

On the actual VNA fixture, NoOp followed by SweepReduce copies the root attrs into
the reduced dataset. Those attrs contain Tollan's `StepContextDict`. The previous
writer passed it to Zarr and failed with `Invalid attribute in Dataset.attrs`.
Fresh CLI executions missed this because they choose no-op or reduction, not
both on one working tree.

The writer now removes only `PIPELINE_CONTEXT_KEY` from its export dataset. The
existing `drop_encoding()` already returns new variables and a fresh attrs dict;
the live tree, contexts, raw buffers and ordinary metadata are not mutated. The
standalone reducer does not gain knowledge of pipeline context storage.

The explicit SweepReduce JSON context remains in the product. Arbitrary opaque
attributes are not silently discarded, stringified, or pickled; they still fail
serialization and no completion manifest is published. This implements the
existing product contract: numerical results and selected serializable metadata,
not a dump of the complete runtime. No new general serializer or output schema.

## Behavior changes / verification

The previously failing composed workflow now writes and reopens successfully.
The regression compares every saved variable and ordinary attribute, verifies
live context identity and untouched raw attrs, and confirms that unknown opaque
metadata still fails without a manifest. A repeat on the real VNA fixture uses
`xr.testing.assert_identical` against the expected dataset minus the reserved
live store and verifies the working tree still contains that store.

- All 947 Tolteca tests pass in 55.38 s with actual-v2 parity enabled and
  benchmarks disabled: `/tmp/tolteca-20260922-lifecycle-full-pytest.log`.
- Scoped Ruff, formatting and ty pass; Tollan's full typing and 1,289-test suite
  passed at the preceding alias checkpoint.
- Fresh complete integration passes at
  `toltec_astro_dev/run/overnight-slice-20260922-lifecycle/report.json`.
  Four cases retain exact I/Q, uncertainties, channel/sweep frequencies and units
  against actual v2; direct/planned CLI no-op, Zarr, SQLite registration, Parquet,
  Web traces and Dash test-client callbacks all pass.
- The reference checkout remains clean. No production server, browser session,
  remote publish, dependency upgrade or fixture write was used.

## Self-review and remaining boundaries

Do not claim generic object serialization: only Tollan's reserved live store is
excluded. Nor does this make products a debug-session restore mechanism. Existing
exclusive destinations and failure/incomplete-directory policy remain unchanged.

The three 03b lifecycle/alias findings are now fixed. Next review the remaining
DB/Web discovery, resource lifetime and request/chunk amplification boundaries.
Broader typed scalar metadata projection for catalog selection is still open;
it is not solved by stripping a context store or by passing exact science tests.
