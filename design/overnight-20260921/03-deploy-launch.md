# 03 — Deployment-to-KIDs command translation

Status: focused verification complete; commit recorded in the index.

## Evidence / alternatives

The v2 shell dispatcher translates site environment values into explicit data
and product paths, then delegates conditional configuration to its Python
dispatcher. Current Deploy `run` only constructs an environment. Its existing
tests intentionally require all child arguments to remain untouched and non-env
configuration not to be forwarded. Changing that command by detecting executable
names would make a previously generic operation application-dependent.

Add an explicit `run_kids` command instead. This is an additive CLI choice for
review, not an agreed universal launcher framework. Keep generic `run` unchanged.
A small pure `plan_kids()` function produces a typed `CommandPlan(argv, cwd)`;
the CLI either prints it (`--show_command`) or executes precisely those fields.
No shell quoting/interpolation, config temporary files, input opening, Dagster
dependency, service setup or implicit deployment mutation.

Use existing deployed Toltecpipe variable names, not a new competing env schema:
`TOLTECPIPE_DATA_LMT_ROOT`, `TOLTECPIPE_DATAPROD_TOLTEC_ROOT`, and
`TOLTECA_DEPLOY_ROOT`. A private Pydantic model types these values. RuntimeContext
and its EnvHandler load configured environment; the Kids application does not
interpret it. A location is optional for standalone explicit file processing.

## Observable behavior / review points

- `--location` overrides the environment's location. Per-path environment values
  override the location's conventional `run/` paths. Configured env overrides the
  inherited process environment, without modifying it.
- Derived roots are explicit application CLI options, as in v2. Therefore they
  override `files.*` recipe roots. Later child CLI path options override them.
  Stage-specific `kids.output.path` remains a separate, higher-priority setting
  inside the application. The root alone does not enable output.
- Config paths are forwarded intact/in order, preserving conditional sources.
  Deploy also reads them to obtain env settings; it does not freeze or flatten
  input-dependent Kids configuration. Env paths are not forwarded as CLI options.
- The current interpreter and working directory are captured. The plan has no
  environment dump (potential credentials), and is a command plan, not a hermetic
  snapshot of all recipe content or shared libraries.
- A caller still must place the separator before child arguments. No special
  rewriting of ObsSpecs or Kids dotted arguments happens here.

## Isolation

Worktree: `worktrees/tolteca_deploy-overnight`, branch
`codex/overnight-kids-slice-20260921`, baseline `21cc971`. Shared C++ checkout stays
on `main`. Tests use `PYTHONPATH=src`; no editable reinstall. The ignored generated
`_version.py` is linked read-only from the existing checkout for test imports.

## Verification / self-review

- Focused CLI/launcher tests: 57 passed, including a real subprocess check using
  reference VNA observation 145605/0 and a conditional YAML source list. A printed
  command plan, replayed without deployment env variables, produced byte-for-byte
  identical `--show_config` stdout to execution through `run_kids`.
- Shared generic `run` tests remain unchanged and passed.
- Scoped pre-commit and `ty check --python <v3 interpreter>` passed. The initial
  typecheck without `--python` searched Homebrew's global Python, not the shared
  venv; corrected the verification command rather than changing package imports.
- Full isolated-worktree suite: 327 passed, 10 failed. Failures are in existing
  config/location/repo tests assuming the checkout basename is `tolteca_deploy`
  and a sibling `refs/tollan` exists. The worktree is named
  `tolteca_deploy-overnight`, so `source_name(file://...)` legitimately differs,
  and its sibling fixture directory is absent. No failures involve the launcher.
  These are recorded as a portability limitation, not hidden or fixed by changing
  unrelated deployment behavior. Repeat full-suite QA in the canonical checkout
  layout when reviewing/integrating this branch.
- Reviewed docs independently: examples and precedence describe user-visible
  behavior; implementation rationale stays in this step record.

Next: reducer/step/view self-review and exact persisted-value parity, including
schema-based access, unit handling and output replay metadata.
