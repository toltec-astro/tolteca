# 01 — Shared file-access configuration

Status: implemented and verified; commit recorded in the review index.

## Evidence

Read the current CLI, runner, Kids config/handler, output step, shared ObsSpec and
filesystem resolver, no-op/launcher tests, the no-op plan, reducer status and
phase-A acceptance plan. The starting CLI correctly uses Tollan infrastructure,
restricts user dotted overrides to `kids\..+`, and keeps process env out of Kids.
But `KidsConfig` inherits `KidsInputConfig` with raw/product paths and selection,
contrary to the agreed shared ownership. `dataprod_toltec_path` is validated and
displayed but is never used by the output step. Existing launcher tests only
exercise generic `deploy run -- <argv>` forwarding, not deployment translation.
The older integration probe still invokes the obsolete `tolteca kids reduce`
CLI. These last two issues are recorded for later focused steps.

## Decision and alternatives

Use a shared `FileAccessConfig` and standard `ConfigHandler` named `FileAccess`,
with the runtime section `files`, next to the existing shared file discovery.
`files` is concise and distinguishes discovery settings from `kids` science
settings. No ORM imports, environment lookup, parser patch or new config engine.
Plain discovery functions still accept Python paths and do not require a handler.
Keeping these fields on Kids would perpetuate the known ownership problem;
adding a generic multi-backend store hierarchy now is unnecessary for this step.

## Observable changes

- YAML uses `files: {data_lmt_path, dataprod_toltec_path, select}`. These are
  removed from KidsConfig, which continues rejecting unsupported stage keys.
- Explicit `--data_lmt_path`, `--dataprod_toltec_path`, `--select` stay available
  and override `files` YAML. Arbitrary user dotted options remain `kids\..+` only.
- A configured product root supplies the default `kids.output.path`, while
  explicit per-stage recipe/CLI values win. It never enables output by itself.
- `--show_config` includes both the shared `files` and per-input `kids` sections,
  so the effective configuration has the same root shape as its YAML input.
- No changes to selection expressions, ObsSpec, env consumption, numerical
  reduction, file-opening policy, or step lifecycle in this commit.

## Verification / self-review

- `python -m pytest -q --benchmark-disable --color=no` on
  `tests/tolteca_datamodels/test_filestore.py` and the KIDs CLI-launch, no-op and
  core tests: **67 passed** in 40.46 seconds. These include real installed direct
  and generic Deploy-forwarded invocations on the reference files.
- Scoped pre-commit hooks passed (global typecheck skipped); explicit `ty check`
  passed for all six changed Python source/test files. `git diff --check` passed.
- Added tests for eager independent handler snapshots, unknown file settings,
  removal of file settings from KidsConfig, shared-root/recipe/dotted output
  precedence, no implicit output enabling or directory creation, and an actual
  enabled-output run publishing beneath the shared product root.
- Reviewed public documentation separately: only the changed config shape,
  precedence and show-config behavior are described, not design history.
- Confirmed the source-identification, array-reading, env helper and step
  lifecycle implementations are unchanged. No source-name compatibility aliases
  were added. `src/tolteca_kids/config.py` is removed because its sole model moved
  into the shared module; it remains recoverable in the baseline commit.

Self-review follow-up: the shared root is translated into stage defaults at the
CLI composition boundary. A caller using `run_source` directly must currently
provide `kids.output.path`; step 02 should centralize application preparation so
the CLI and the Python runner cannot disagree on this default. This is a local
application integration issue, not a reason for Kids or SweepReducer to depend
on deployment settings. Also, metadata catalog enrichment still lives in the
Kids runner; move that existing behavior into the shared discovery layer before
extending it. The no-op tests must continue guarding payload reads and lifetimes.
