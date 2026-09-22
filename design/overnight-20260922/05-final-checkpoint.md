# 05 — Reviewed reducer-only vertical slice: final handoff

The second overnight review/rework is complete for the agreed no-op and
reducer-only slice, with the verification limits below. It is not the full
legacy KIDs science pipeline or a general DB/Web redesign. All work stays local
on temporary branches, with the pending daytime baseline and previous overnight
commits preserved. Automation status is recorded in the main ledger.

## Start the review here

1. Workspace `review/overnight-20260922/README.md` gives actual Git ranges and
   focused diff files. Items 1–2 separate the preserved daytime work from the
   approved direct-registry/source-kind refactor. Do not use an old `/tmp` diff
   as evidence of the current tree.
2. Review the file/reader/catalog decisions (02a, 02b1–02b18), then lifecycle
   fixes (03a, 03b1–03b4), then bounded product/Web changes (04a, 04b1–04b5).
   Each design record states its observable changes and its evidence.
3. `docs/guide.md` is the focused telescope → metadata → catalog → no-op → Deploy
   walkthrough. `docs/kids-reduction.md` covers computation and output. Public
   documentation contains current usage, not the overnight discussion trail.
4. The first night's implementation remains in
   `design/overnight-20260921/`; this round's findings and fixes supersede its
   pending-work statements. No commit history was rewritten.

## What holds end to end

| Layer | Current responsibility and verified outcome |
| --- | --- |
| File parsing | Root registry assembles domain registrations. Path models eagerly store filename-derived fields; source records own data kind. 132 recorded DB/LMTMC path projections remain covered. Prefix, interface and UID are not conflated. |
| Raw readers | Telescope names/times/coordinates follow the reviewed contract. KIDs metadata keeps native master/repeat codes, original path and recorded settings. All 47 tested raw omissions reject instead of becoming optional defaults. |
| Catalog | Explicit never/fallback/always opening policies. Shared identification attaches typed headers and closes files. Arrow structs support dotted queries/scalar aliases without per-row model traversal. Header selection and conditional config work through the CLI. |
| CLI and Deploy | Tollan owns runtime/CLI mechanics. Kids receives explicit paths and does not derive them from environment/deployment state. Isolated Deploy planning/launch forwards recipes and translates location settings. Direct/planned no-op behavior agrees. |
| Computation | Standalone SweepReducer owns the numerical work. Kids composes configured steps; no orchestration dependency enters the reducer. The result owns its channel-frequency coordinate and is usable without the raw source. |
| Live context/views | Each invocation attaches its live context before running. Failed reruns remain incomplete; aliases retain independent context-class bindings and reuse their own unforced results. Arrays stay in xarray containers, not a duplicate StepData payload. |
| Output | A configured final step publishes write-once Zarr-v2 products and writes the typed completion manifest last. Live context stores are omitted from the export copy; the working tree is unchanged. Metadata is consolidated. |
| DB | Existing raw ingestion and completed-product registration create lineage in caller-owned transactions. A separate session publishes Parquet discovery records. No production DB or ORM redesign was needed. |
| Web | Product IDs distinguish reruns/blocks. Selection precedes frequency broadcasting; selected I/Q and shared coordinates compute together. Reads close on success/failure. Exact trace arrays and Dash HTTP callbacks pass. |

## Exact scientific acceptance

Final artifacts are under the workspace development location:
`toltec_astro_dev/run/overnight-slice-20260922-handoff/`.
They include `report.json`, four completed products, v2 reference arrays,
command plans/logs, a local SQLite database, the Parquet catalog and Plotly HTML.
They are generated test state, not fixture inputs or source-control artifacts.

The probe runs the actual v2 interpreter, not a reimplemented oracle. Reopened
v3 products use `assert_array_equal` for I, Q, unc_I, unc_Q, f_chan, sweep and
frequency; units and selected Web traces agree exactly. No tolerance was relaxed.

| Source | Selected block | Selected signal shape |
| --- | ---: | ---: |
| ROACH 0 VNA, 145605/0/1 | 0 (requested -1) | 1000 × 491 |
| ROACH 0 target, 145606/0/1 | 0 (requested -1) | 668 × 177 |
| ROACH 0 tune, 145609/0/1 | 0 | 632 × 177 |
| Same tune | 1 (requested -1) | 632 × 177 |

Tune products retain both blocks; the table describes the selected view. To
repeat, choose a new output directory; the probe refuses to overwrite one:

```sh
./toltec_astro_dev/.venv/bin/python tolteca/design/probes/kids_vertical_slice.py \
  toltec_astro_dev/run/data_lmt toltec_astro_dev/run/NEW-REVIEW-RUN \
  --v2_python /Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  --deploy_source worktrees/tolteca_deploy-overnight/src
```

## Final QA and self-review

| Check | Observed result |
| --- | --- |
| Tolteca full pytest, actual-v2 tests enabled | **1,071 passed**, 57.64 s |
| Tollan full pytest / full ty | **1,293 passed** / pass |
| DB default pytest selection | **192 passed**, 54 existing deselections |
| Web full pytest | Known collection blocker: legacy `test_raw_obs_db.py` imports removed `tollan.db` |
| Web excluding that one legacy module | **13 passed** |
| Isolated Deploy launcher/CLI, real reference enabled | **57 passed** |
| Observation guide | **12 Python blocks + 4 CLI examples pass** |
| Reduction guide | **8 Python blocks pass** |
| All Python src/tests changed this round in Tolteca, DB, Web | ty passes |
| New/changed focused patches | Ruff/format/whitespace checks as recorded per step |

Tests use the existing shared environment and `--benchmark-disable`.
Tolteca sets `TOLTECA_V2_PYTHON` to the actual v2 interpreter. DB uses temporary
SQLite, not a production/PostgreSQL integration certification. Deploy uses the
isolated checkout with `PYTHONPATH=src` and `TOLTECA_TEST_DATA_LMT` set to the
read-only reference path. No dependencies were upgraded.

Final logs: `/tmp/tolteca-20260922-final-pytest.log`,
`/tmp/tollan-20260922-final.log`, `/tmp/tolteca-db-20260922-final-fixed.log`,
`/tmp/tolteca-web-20260922-runnable.log`,
`/tmp/tolteca-deploy-20260922-final.log`, and
`/tmp/tolteca-20260922-handoff-integration.log`.
Durable acceptance assertions live in the committed tests/probes; temporary logs
are supporting evidence, not the only record of how to reproduce it.

Final self-review caught and fixed three QA gaps: DB's incomplete synthetic raw
fixture, missing narrowing in two new metadata tests, and a six-argument metadata
helper. The DB suite now uses a read-only reference link and explicitly skips
those 18 integration tests when the reference deployment is unavailable.
Do not count such skips as successful integration. The complete changed-file
type check passes after the assertion fixes. The helper no longer passes a
redundant flag, with zero/one/two-block regression tests and unchanged read counts.

Global `just qa` is **not claimed green**. Its all-file mutation/sync workflow
would include unrelated debt. A whole-changed-file Ruff check (excluding existing
CPY001 policy) retains **28 diagnostics** in four existing files; their code/message
signatures match the preserved baseline. Two newly introduced argument warnings
were fixed rather than classed as baseline debt. Formatting still flags one
existing `_get_frequency` expression in core.py; 51 other changed Python files
pass. No unrelated mass-format or naming rewrite was made merely to hide these
results. DB/Web changed-file typing and lint pass.

## Measured computation / I/O consequences

- Typed raw catalog preparation intentionally reads more headers: VNA/target/
  tune use 20 backend reads / 604 logical bytes versus the earlier seven / 536.
  Timestream uses 18 / 596. No I/Q, ADC or detector-position payload is read.
  Header-only tune metadata avoids the full reader's 14,160-byte LO scan.
- With xarray caching disabled, identification uses 23 reads, versus 22 for the
  header reader alone. That remaining kind-dispatch read is recorded, not hidden
  behind a new persistent source cache. Cached property snapshots are local to
  an accessor; fresh identification creates a fresh metadata snapshot.
- Consolidated product opening falls from 285–304 local store gets to 6–7 in
  measured cases. These are local operations, not measured network requests.
- Web frequency-coordinate reads fall from four to one; channel selection
  precedes frequency-grid allocation. Numerical equality remains exact.
- Sixty chunk-layout/selection comparisons preserve values. Automatic writer
  chunks remain unchanged: sparse-vs-bulk request/byte tradeoffs need a serving
  workload, not a universal policy inferred from a two-channel local plot.

## Items for user review and later work

1. Review the larger observable choices explicitly: source-owned kind, typed
   header snapshots in catalog/manifest, strict raw preparation even for no-op,
   timestamp wire format, coordinate precision, and optional parsing of unknown
   telescope original names. Each has a focused rationale/test record. Passing
   tests do not substitute for approval of the public design.
2. Full raw-file reading is strict, but shared raw/processed metadata dataclasses
   still allow nullable fields for processed records. This is not a globally
   strict raw-model constructor or a general source/metadata integrity validator.
   Native header values are not invented to fill that model flexibility.
3. Catalog schemas are inferred per batch; separate batches can have different
   Arrow struct fields. Cross-batch schema policy belongs with the DB/catalog
   design. Serialized unit-bearing metadata remains strings, not implicit bare
   numbers. Source columns win alias collisions (notably source master label vs
   numeric `metadata.master`). ObsSpec's registered-master policy versus free
   source labels such as `simu` still needs an explicit user decision.
4. DB/Web friction remains: global URI uniqueness vs per-location sources,
   canonical interface vs legacy prefix identity, filename UT stored in ORM
   `obs_datetime`, remote mirrors/product access, durable cross-catalog product
   identity, and cache lifecycle. The narrow adapters are exercised; the broader
   architectures are intentionally not redesigned here.
5. **Browser rendering is not verified.** The earlier local-server/browser
   permission denial remains unresolved; no alternate access route was used.
   HTTP layout/callbacks and exact plot arrays are verified. A live visual check
   needs authorization to bind a local server and use the browser. The old Web
   `tollan.db` collection blocker also remains separate work.
6. Later KIDs checking/detection/fitting and timestream reduction remain outside
   this reducer-only slice. Deploy's additive `run_kids` prototype still lives
   only in its isolated checkout; the shared C++ Deploy repository was untouched.

## Repository safety

Runtime checkpoint: Tolteca `e9c3951`, Tollan `1c65394`, DB `c29e3b6`, Web
`04f0c73`. The final ledger commit follows these without changing runtime code.
All four repositories use `codex/overnight-kids-slice-20260922`. Isolated Deploy
remains `117ba21` on the first overnight branch; shared Deploy remains `21cc971`.
Reference data is clean at `99711bd`. Unrelated `.DS_Store` files are untouched.
No pushes, merges, resets, production/instrument actions, fixture writes,
dependency upgrades or agent delegation occurred in this round.
