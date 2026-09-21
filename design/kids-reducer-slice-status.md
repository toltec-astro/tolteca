# Reducer-only slice: implementation and observed friction

Status: local implementation and integration probe; no-op verification updated
2026-09-16. Not committed.

## Current review boundary: ObsSpec and no-op CLI

### Standalone CLI checkpoint (2026-09-17)

The current entrypoint is `tolteca_kids [OPTIONS] -- OBSSPEC...`, not the
earlier `tolteca kids reduce` prototype shown in the historical probes below.
Paths and catalog selection are config-backed (`kids.data_lmt_path`,
`kids.dataprod_toltec_path`, `kids.select`) with explicit CLI shorthands.
Kids does not derive these paths from environment variables or deployment layout.

The agreed launcher boundary follows the v2 shell/Python dispatcher separation:
Deploy consumes deployment configuration/environment and constructs the explicit
Kids argv/config inputs. The current generic `tolteca_deploy run` does not yet
perform that translation; it exports environment values and forwards supplied
argv. Tests of that forwarding must not be reported as completion of the
deployment-aware Kids launcher. The launcher adapter remains to be designed.

ObsSpec narrows filesystem discovery using glob patterns, then filters a
filename-metadata catalog with its full constraints. Scalar header enrichment
precedes the general catalog `select` expression. Per-input conditions are
evaluated afterward. Discovery/selection configuration must be unconditional.

One discovered input issue is fixed: a selector rooted through a resolved
deployment symlink and an explicit file path could refer to the same file with
different strings. Deduplication now compares resolved paths and retains the
first path spelling for metadata.

### Earlier prototype evidence

The next review is deliberately about input/configuration plumbing. The earlier
reducer/output prototype remains in the uncommitted diff, but its view/layout,
metadata accessor, reducer configuration, and persistence choices still await
the requested follow-up. A no-op is not evidence that those choices are settled.

`tolteca --log_level DEBUG kids reduce '<selector>' --data_root <root> --dry_run`
resolves a catalog, applies optional row selection, constructs per-input Kids
handlers, wraps each source, and executes a real NoOp Step/StepContext through
SequentialPipeline. No reduction or product output runs, even with output
enabled. Scalar headers may be read. The four reference inputs (VNA, target,
tune, timestream) completed this path successfully.

The 2026-09-16 pass adds regression coverage and a user-facing `kids-inputs`
guide, without changing production behavior. A backend read guard on each of
the four reference kinds permits scalar headers but rejects reads of `Data.*`
variables. It also checks selector/path deduplication, live/completed step
context, absence of reducer/output calls, unchanged input size/mtime, and closure
of both catalog and execution file handles. Separate failure tests check closure
after invalid configuration or a failing step. Inspection-only commands and
empty selections are tested without executing the step.

The installed entrypoint also completed all four inputs with:

```console
toltec_astro_dev/.venv/bin/tolteca kids --log_level INFO reduce \
  'tcs-{145605,145606,145609}/0' \
  --data_root toltec_astro_dev/run/data_lmt --dry_run
```

This verification does not approve the pending reducer/output implementation
or Tollan's proposed `set_context` hook. Saved-run replay remains follow-up work.

The shared typed ObsSpec now lives in datamodels. Both DB query entrypoints use
it; the URL-based API translates its constraints to SQL and the existing
session API filters loaded metadata. There is no second parser or identity
compatibility class. The documented obs/scan/roach shortcut is right-aligned,
invalid syntax raises, and half-open identifier ranges have no arbitrary cap.
Catalog and SQLite query tests compare results for the same parsed selectors.
DB DTOs, location-aware URI resolution, and publication architecture remain
separate follow-up work. This pass does not add a DB-backed CLI option.

Debug logs cover discovery, normalized selectors, row counts, scalar input
metadata, conditional config evaluation, effective configuration, selected
steps, start/completion/failure, and timings. Array contents are never formatted.
The effective-model dump is lazy so disabled debug logging avoids serialization.

Parser tests moved to the shared module. DB tests now cover actual SQL/query
behavior. Old quartet UID/from-path/ordering assertions were removed because
ObsSpec is a selector, not a row identity or filename parser; shared filename
discovery tests retain the latter responsibility.

## Implemented boundary

The live DataTree remains the stable working container. Tollan's Boolean
`run(data, context)` contract is retained, with a final `set_context` hook.
`SweepReduce` calls the independent `SweepReducer`, replaces its result node,
and stores small JSON context metadata on that node. `Kids` owns configured
steps and selects an explicit `SequentialPipeline` by data kind; the runner
owns input-file lifetimes. `DataProdOutput` is the final configured step and
owns publication policy, delegating array serialization to the product writer.
No duplicate array-bearing StepData container is introduced.

The CLI uses the same runner as Python callers. The product has a reduced
Zarr subtree and a manifest written last. Reopening gives the same views and
typed context, without recreating runtime/config-source resources.

Shared `tolteca_datamodels.toltec.file` source records and filename guessing
feed a metadata-only pandas catalog. The local resolver accepts files, globs,
and the shared observation selector DSL. The runner enriches
scalar headers without I/Q reads; catalog row selection precedes per-input
conditional config. `ensure_datatree` only wraps. Source metadata is attached
separately on an execution child, not mixed into the raw dataset.

Timestream and processed kinds are recognized but explicitly unsupported for
execution in this slice. They never fall through to sweep averaging. The DB
scanner uses the shared filename parser via its existing ingestion DTO.
No production query/ORM redesign has been folded into this change.

## Executed evidence

- `test_v2_parity.py` runs the actual v2 `NcFileIO.read()` in a separate
  interpreter and compares persisted v3 I/Q, standard deviations, channel
  frequencies, sweep offsets, and complete frequency grids with exact array
  equality. Target and VNA sweeps plus first and last tune blocks pass.
- The normal slice tests exercise conditional recipes, CLI overrides, metadata
  preparation without I/Q reads, disabled steps, result reuse, selected blocks,
  raw-input immutability, sibling preservation, coordinate inheritance,
  exclusive output creation, incomplete product rejection, and context reopen.
- `probes/kids_vertical_slice.py` ingests three reference files using the actual
  DB `DataIngestor`, invokes the public CLI as a subprocess, registers completed
  products and reduced-to-raw edges using existing ORM models, projects committed
  rows to Parquet, queries the actual Web catalog, and builds Web Plotly figures
  from the shared saved view. Figure I/Q and frequency arrays are checked exactly.
- Successful probe output: `/tmp/tolteca-kids-slice-20260915-4` (SQLite database,
  Parquet catalog, three products, and three self-contained HTML previews).

The probe is deliberately **not** a production DB registration API, a migration
of the existing Web sweep page, or an installed Dagster asset. It tests the
package boundary with explicit local glue and no existing database. SQLite is
tested here; DuckDB/concurrent registration/remote storage are not established.

Run it from the workspace root with a fresh output directory:

```console
toltec_astro_dev/.venv/bin/python tolteca/design/probes/kids_vertical_slice.py \
  toltec_astro_dev/run/data_lmt /tmp/my-new-sweep-slice
```

Cross-version tests:

```console
cd tolteca
TOLTECA_V2_PYTHON=/Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  ../toltec_astro_dev/.venv/bin/python -m pytest tests/tolteca_kids/test_v2_parity.py
```

These checks establish reducer-level parity for these fixtures, not final
SweepCheck/KidsFind/Tlaloc pipeline parity, every instrument, or every raw format.

## Friction to guide subsequent design

1. **DB ingestion bug, fixed narrowly:** `DataIngestor.__init__` locally imported
   `select` only for string locations, breaking integer-location construction.
   Removed that shadowing import; regression tests cover both forms.
2. **Product publication API absent:** there is no existing reduced-sweep
   registration API coordinating a completed manifest, product/source records,
   and lineage. The probe uses explicit ORM glue rather than pretending the
   refs' proposed exporter is implemented. Define this boundary in the DB review.
3. **Location identity:** source URI alone is the DB primary key, although
   ingestion stores paths relative to a separate Location. Identical relative
   paths in different locations can collide. The probe uses disjoint paths.
4. **Catalog product identity:** existing Web lookup uses observation quartet
   plus network and takes the first match. It cannot distinguish processing
   attempts, algorithm/config variants, or blocks. The probe exports a
   reduced-only catalog and keeps `product_pk` and `product_path` explicitly.
   Do not mix this projection into the existing raw catalog without a selector.
5. **Web schema duplication:** the current sweep page/data service expects flat
   root arrays and its own tone/frequency conventions; it cannot consume this
   hierarchical product unchanged. The new focused figure consumer uses the
   shared view. Wiring that view into the page belongs in the Web review.
6. **Remote access:** current product open/manifest handling is local Path-based,
   and the existing Web service tests local existence before remote lookup.
   Remote manifests/locations and publication atomicity remain future work.
7. **DataTree inheritance:** local result datasets can report unused parent
   dimensions. Export must preserve inherited coordinates used by the result
   while excluding unrelated raw axes; `ds[list(ds.data_vars)]` performs that
   metadata-only projection. The regression includes non-default channel IDs.
8. **Zarr metadata:** copying netCDF fixed-byte headers into Zarr3 requests an
   unstable dtype extension. The initial format2 writer preserves bytes exactly.
   Decide a deliberate metadata projection/encoding before standardizing format3;
   do not suppress the warning or silently decode arbitrary bytes.
9. **Legacy dispatcher success:** it requires KidsFind context even for recipes
   disabling that stage. The new reducer command reports success for its own
   requested workflow rather than requiring unrelated later outputs.
10. **Existing step alias back-link:** aliases mutate the shared context class's
    `step_cls` back-link. Separate result namespaces work, but generic
    `context.make_step()` reconstruction after alias creation needs focused
    follow-up; this slice does not establish persisted alias reconstruction.
11. **Header inventory:** the reducer copies all non-time variables, including
    some large/irrelevant headers. This is retained to avoid discarding needed
    downstream metadata before the next stages are ported. Projection into a
    stable scientific product schema should be driven by those actual consumers.

12. **Unit labels:** the reducer previously invented `units="ADU"`, which is
    not an Astropy-recognized unit string. Raw fixtures have no units, and the
    v2 reader returns dimensionless I/Q and uncertainties. The reducer now
    retains input units if present and otherwise leaves signals bare. Tests
    cover bare, adu, and V round-trips through the unit accessor, and compare
    v2/v3 unit meaning as well as numerical arrays.

13. **Source discovery ownership:** DB filename scanning delegates to the shared
    datamodel parser, and both query APIs now consume the same ObsSpec DSL.
    DB API/repository DTO consolidation remains DB-review work. Product/association
    selectors and implicit local latest-observation policy are not implemented.
14. **Enum ownership:** shared datamodel and DB data-kind flags have distinct
    bit assignments and coverage. The scanner maps by name, not integer;
    ancillary hardware streams absent from DB flags remain unclassified there.
    In particular HWPR is no longer mislabeled as a KIDs RawTimeStream.
15. **Input scope:** local file catalogs are supported; URI sources and completed
    product directories are not yet run inputs. Processed netCDF files are
    identified but rejected for execution. There is no claim that output-only
    replay or downstream scientific stages already work on reopened products.

No array arithmetic was replaced to obtain parity. New frequency properties
reuse existing header interpretation and align the two names used for the
ordered tone/signal channel axes.

## QA

Current no-op verification (2026-09-16):

- Tolteca: 590 tests passed, including four v2 parity cases using the legacy
  interpreter.
- Focused no-op, input-source, and shared ObsSpec tests: 90 passed.
- Scoped pre-commit passed for the changed tests and public documentation.
  Its global typecheck hook was skipped; explicit `ty check` passed for the
  no-op/CLI/runner/core, shared input modules, and no-op tests.
- The installed public CLI completed all four read-only reference inputs.
- No production code changed in this verification pass. This is not a claim
  that repository-wide `just qa` is clean; the baseline issues below remain.

Earlier ObsSpec/no-op pass:

- Tolteca: 577 tests passed, including four v2 parity cases.
- Tollan pipeline: 58 tests passed.
- DB selector/discovery/constructor/ORM/metadata: 87 tests passed. The selector
  tests also exercise the session-based API against the same catalog results.
- The public no-op CLI completed all four read-only reference inputs.
- Scoped pre-commit and explicit typechecking are used for this pass; this is
  not a claim that repository-wide `just qa` is clean. Existing failures below
  remain outside this work.
- Sphinx HTML built successfully with the existing missing-toctree and offline
  inventory warnings; no new ObsSpec or no-op page warnings.

Earlier reducer-slice evidence (before the current pass):

- Tollan `just qa`: passed in the preceding slice iteration, including 1,035
  tests and all hooks. No further Tollan changes in the source/output iteration.
- Tolteca full test suite: 531 passed, including the four cross-environment v2
  comparisons. The source/output and reducer slice now has 64 focused tests.
- Explicit typecheck of the slice code, modified reducer, and new tests passed.
- Scoped pre-commit for the new Tolteca code/docs passed, with the global
  typecheck hook skipped in favor of an explicit check of all slice code/tests.
  Including the existing reducer file reports its 15 existing naming/style/
  complexity diagnostics; those are not hidden by this scoped result.
- DB discovery/constructor/ORM/metadata tests: 50 passed. Web consumer: 2 passed.
- The preceding Tolteca `just qa` stopped at repository-wide pre-commit: 228 remaining
  lint errors after its automatic fixes and 16 pre-existing type diagnostics.
  Unrelated hook-only edits were removed; the scoped slice patch was preserved.
- Sphinx HTML build completed again; external inventory network warnings and existing
  missing-toctree warnings remain. No new reduction-page warning was reported.
