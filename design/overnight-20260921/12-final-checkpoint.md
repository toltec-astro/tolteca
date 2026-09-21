# 12 — Reducer-only vertical slice: final checkpoint

Implementation and executable integration verified on 2026-09-21 after the
user's explicit continuation. This is the no-op and reducer-only slice, not the
complete v2 KIDs analysis pipeline. The expired overnight automation is paused.

## End-to-end evidence

The updated `design/probes/kids_vertical_slice.py` invokes the actual APIs rather
than manually constructing derived ORM records or catalog rows:

1. Deploy `plan_kids` translates roots to explicit Kids CLI arguments. The probe
   executes that plan; a separate Deploy test executes `run_kids` itself and
   proves conditional configuration replay without deployment environment values.
2. Direct and planned no-op CLI stdout agree. The normal CLI discovers/reads
   typed input metadata and constructs per-input runtime configuration.
3. Kids prepares the working tree and executes SweepReduce and configured output.
   SweepReducer remains independent. StepContext and numerical results are
   reopened through the result view, not a separate array-bearing StepData model.
4. The typed completion manifest retains input and effective configuration;
   incomplete writes are not discoverable as completed products.
5. Existing raw ingestion plus `register_sweep_product` populate a fresh SQLite
   database. A new session exports `write_sweep_catalog` to typed Parquet.
6. SweepProductService selects each distinct product ID and plots saved data.
   Actual Dash layout and catalog callbacks run through the Flask test client.

The probe compares the actual v2 interpreter's output with each reopened product
using `assert_array_equal`, not tolerances: I, Q, unc_I, unc_Q, f_chan, sweep and
frequency; quantity units also agree. Selected Web trace values and frequencies
are compared directly with those v2 arrays.

| Reference | Observation | Requested block | Actual block | Selected I/Q shape |
| --- | --- | --- | --- | --- |
| VNA sweep | 145605/000/0001, toltec0 | -1 | 0 | 1000 × 491 |
| target sweep | 145606/000/0001, toltec0 | -1 | 0 | 668 × 177 |
| tune | 145609/000/0001, toltec0 | 0 | 0 | 632 × 177 |
| tune | 145609/000/0001, toltec0 | -1 | 1 | 632 × 177 |

Reproduce from the Python workspace, choosing a new output directory:

```sh
toltec_astro_dev/.venv/bin/python tolteca/design/probes/kids_vertical_slice.py \
  toltec_astro_dev/run/data_lmt \
  toltec_astro_dev/run/overnight-slice-20260921-new \
  --v2_python /Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  --deploy_source worktrees/tolteca_deploy-overnight/src
```

Final generated evidence is in workspace
`toltec_astro_dev/run/overnight-slice-20260921-c/`: report.json, four product
directories, command plans, stdout/stderr logs, v2 comparison NPZs, SQLite,
Parquet and self-contained Plotly HTML files. The probe leaves failed output
directories for investigation and refuses reuse. Run `a` exposed the pending
ORM relationship defect fixed in step 10; `b` passed before channel batching;
`c` passed afterward. These generated artifacts are not committed or included
in review diffs. No fixture inputs were modified.

## Final QA

All use the existing shared v3 environment, without dependency upgrades:

| Repository / scope | Result |
| --- | --- |
| Tollan full pytest | 1,286 passed |
| Tolteca full pytest, actual v2 comparison enabled | 1,260 passed |
| DB default pytest selection | 191 passed, 54 deselected by existing integration selection |
| Web full pytest | collection fails in unchanged `test_raw_obs_db.py`: removed `tollan.db` import |
| Web excluding that legacy collection blocker | 13 passed |
| Isolated Deploy launcher and CLI, reference test enabled | 57 passed |
| Isolated Deploy earlier full pytest | 327 passed, 10 existing worktree-layout failures; see step 03 |

Pytest commands: `python -m pytest -q --benchmark-disable --color=no` for the
Python libraries, with `TOLTECA_V2_PYTHON` pointing to the v2 interpreter for
Tolteca. Web's follow-up adds `--ignore=tests/test_raw_obs_db.py`. Deploy uses
`PYTHONPATH=src`, `TOLTECA_TEST_DATA_LMT` set to the reference root, and
`tests/test_launch.py tests/test_cli.py` from the isolated worktree. No production
database or instrument service is used. The default DB run is not a PostgreSQL
integration certification.

Focused pinned pre-commit hooks and type checks were run per step, with explicit
`ty check --python <shared v3 Python>`. The final Web batching check passed both.
The final check of every changed Tolteca Python application/test file passed.
A broader Kids directory check also included unchanged KidsFind modules/tests
and the v2-only extraction helper, producing 98 diagnostics; those are not
silently excluded from a claimed whole-package pass. A pre-existing optional
enum dereference in the touched DB discovery test was made explicit with a
non-null assertion; all changed DB Python files then passed. Web's new consumer
and tests pass; the legacy catalog module's five-line reload addition retains
its earlier scoped/manual review rather than claiming that module is debt-free.
The integration probe was formatted, import-sorted, compiled and executed; it is
not presented as passing every application-module lint rule. `git diff --check`
passes. Global `just qa` is not claimed: it runs mutating hooks on all files and
has known unrelated formatting/type debt. Full tests and scoped checks are
reported separately rather than silently rewriting legacy modules to make it
green.

## Visual verification boundary

Opening the local Dash app through the browser connector was denied by the
permission check. Explicit permission was requested; no alternate browser,
raw browser automation or screenshot workaround was used. Plot arrays, layout
JSON and HTTP callbacks are verified. Pixel rendering and interactive behavior
in a browser are not yet verified. This is the only remaining verification
permission request, not a reason to extend the scientific scope.
The attempted local development server also exited with `Operation not
permitted` in this environment; no server is left running. A live visual check
will require permission to bind the local server as well as browser access.

## Decisions that need human review

- `files.*` owns shared roots and selection; Kids configuration owns stages.
- The additive Deploy command is `run_kids`; generic `run` remains unchanged.
  Config paths retain conditional-source semantics; environment roots become
  explicit CLI options. This is a launcher choice for review, not a generalized
  framework for every application.
- Output defaults remain opt-in. Products use a typed `tolteca.reduced-sweep.v2`
  completion manifest and Zarr format 2; the version number identifies the
  prototype storage format, not the legacy scientific implementation.
- Registration uses existing derived-product/association classes with explicit
  provenance, not a new ORM hierarchy. A caller owns its transaction.
- The completed-product browser is a focused app factory usable by the existing
  Web CLI, not a replacement for the older raw/KidsFind portal.

## Remaining friction, intentionally not redesigned

1. DB: source URI is globally unique rather than unique per location; raw and
   derived identity, mirrors and product-level versus source-level provenance
   need a unified policy. Existing metadata dataclasses/defaults and the shared
   Pydantic model are not yet one model. See DB steps 07b/08/10.
2. Web: the old raw cache and quartet-first lookup do not represent reruns or
   independently selected blocks. A unified portal, remote product access and
   lifecycle-aware caches remain follow-up work. The new browser uses product
   IDs and closes each opened tree.
3. Performance: plotted I/Q channels are now batched. Automatic Zarr chunks can
   still amplify sparse reads, and the frequency view broadcasts a full grid
   before selection. Measure local/remote access patterns before selecting a
   chunk policy or adding a selection-capable shared view. No floating-value
   or cross-source cache was introduced. See Web step 11 for measured shapes.
4. The old Web `tollan.db` import and Deploy worktree test assumptions remain
   visible QA debt. They do not invalidate the independently tested slice.
5. Later scientific stages, generalized telescope/raw-reader modernization,
   production ingest scheduling and distributed publication are not completed
   by this reducer-only slice. No claims are made about their parity.

## Review trail and repository safety

Review exports live at workspace `review/overnight-20260921/`. Start with its
README, then repository diffs and per-commit diffs. New-work ranges begin at the
baseline checkpoint; separate original-branch-to-checkpoint diffs expose the
pre-existing work without mislabeling it as overnight implementation.

All independent repositories remain on `codex/overnight-kids-slice-20260921`;
the shared Deploy checkout and original v3.x branches retain their original
commits. Deploy changes exist only in the isolated worktree. Reference data
remains clean at 99711bde64f96d0df042e3c44a1e174ab68dcb4c. The unrelated Tollan
`.DS_Store` is left untouched. Nothing was pushed, merged or reset.
