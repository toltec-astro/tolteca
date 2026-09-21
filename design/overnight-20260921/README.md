# Overnight KIDs vertical slice — review trail

Status: in progress. Start: 2026-09-21 04:53:49 UTC. Hard stop:
2026-09-21 12:53:49 UTC (08:53:49 America/New_York).

The user authorized autonomous implementation, self-review, adjustment and small
local commits for up to eight hours. The scope is the existing reducer-only
vertical slice, including the previously agreed no-op input/CLI checkpoint,
not the complete SweepCheck/KidsFind/Tlaloc scientific pipeline.

## Resume here

1. Read `WORKSPACE.md` and this index. Inspect Git status before editing.
2. Read the latest step entry and its explicit next action. Re-read the relevant
   source and tests; do not treat old design status as evidence of completion.
3. Finish one bounded step: evidence, decision, focused implementation, tests,
   independent diff review, and a commit. Record observable behavior changes.
4. Append self-review corrections in subsequent focused commits. Do not rewrite
   the review trail, silently accept scientific differences, or hide failures.
5. At completion or the deadline, finish a safe checkpoint, write the handoff,
   generate range-based review diffs, and pause the task heartbeat
   `kids-overnight-vertical-slice`. Stop sooner if meaningful scoped work is done.

Next action: step 07 — implement completed-product registration and catalog
projection in DB, using the typed manifest and existing ORM. Then integrate an
actual Web consumer path; replace the old hand-written integration probe glue.

## Branches and baseline

Use `codex/overnight-kids-slice-20260921` in each independent repository.
The workspace root is not a Git repository. Original branches are not advanced.
The initial checkpoint per repository records previously uncommitted work;
it is not presented as a newly designed or newly reviewed overnight change.

| Repository | Original branch | Original HEAD | Baseline checkpoint |
| --- | --- | --- | --- |
| tollan | v3.x | de75e826ad243ffedf3b08a1933a3df42eb40787 | 1fea73b |
| tolteca | v3.x | 6ef2c1617ccb071b74f1142aac6e9c5d6ed7e56c | fe3b722 |
| tolteca_db | v3.x | 4ea880436368dbcfbf99ee59d2c4c7854a7e83ee | c01f1fa |
| tolteca_web | v3.x | 3bd8c0b14d03c7bfbc4bab237109a9ead8880a66 | 739c963 |

The shared `tolteca_deploy` checkout begins at
`21cc97110280f1f3f5f9bfa11a8cb4c7d93071ad` on `main`. It is shared with the C++
workspace; do not switch or edit that checkout silently. If changes are needed,
use an isolated worktree inside the Python workspace and record its location.
Reference data begins at `99711bde64f96d0df042e3c44a1e174ab68dcb4c`, clean.
Do not stage unrelated `.DS_Store` files.

Baseline evidence: the preceding reviewed Tolteca state passed 1,239 tests with
the actual v2 interpreter enabled, plus scoped lint/type checks. This is not
a claim that global QA in every repository is clean. Initial checkpoint commits
disable Git hooks to preserve the starting state exactly; subsequent changes
must have explicit scoped verification, with pre-existing QA failures recorded.

## Agreed boundaries

- Retain schema → mapper → accessor/view. Algorithms such as SweepReducer and
  D21Analysis stay independently callable and unaware of runtime/orchestration.
- Kids owns configured stages. Pipeline prepares a live DataTree; stages use
  Tollan Step/StepContext and expose data-backed views. Avoid a second array-
  bearing StepData container. Output is a configured final step.
- `tolteca_kids [OPTIONS] -- OBSSPEC...` remains the standalone entrypoint.
  Use Tollan CLI/runtime infrastructure, Python-style option names, and allow
  arbitrary dotted overrides only under `kids\..+`.
- Kids does not derive settings from deployment/environment variables. Deploy
  translates them to explicit application arguments/config, analogous to v2
  dispatch.sh versus dispatch.py. Manual and orchestrated invocation agree.
- Raw paths, product paths, ObsSpec, discovery and catalog selection are shared
  file-access concerns, not Kids-specific scientific configuration. Preserve
  pandas expression semantics and avoid metadata/payload reads when unnecessary.
- Filename guesses are not raw-header truth. Preserve numeric acquisition
  codes, nullable rather than fabricated information, and source-location versus
  recorded-path identity. Path-derived master names can include `simu`.
- Reference fixtures are read-only. All outputs/DBs/caches belong in temporary
  directories or a dedicated generated run directory, never in the fixtures.
- First establish no-op input/selection/config/lifetime behavior. Then validate
  raw sweep → reducer → saved Zarr/Parquet view → DB discovery → Web figure.
  Compare v2 scientific arrays exactly, including NaNs, units, ordering and
  selected tune blocks. No relaxed tolerances or invented metadata.
- DB/Web changes should make this bounded integration real and testable. Record
  broader architecture friction rather than launching an ORM/UI redesign.
- No pushes, merges, history rewriting, destructive resets, CI, production
  services, instrument actions, unrelated cleanup, or unnecessary dependency
  upgrades. No delegated agents are authorized by this request.
- Public docs/docstrings describe current usage, not this design process.
  Rationale, rejected options and review corrections belong in this directory.

## Milestones and acceptance

1. Preserve baseline; audit current state and convert discrepancies into bounded
   steps. Retain readable Git-per-step review commands.
2. Complete shared file-access/config ownership and the no-op CLI. Test direct
   files, ObsSpec, selection, conditional config, dry-run, handle closure and
   absence of signal-array reads/output side effects.
3. Complete deployment-to-Kids argument translation with a reproducible command
   plan and direct-versus-launched equivalence, without changing production state.
4. Review reducer/output prototype against established architecture and requested
   corrections; prove exact v2 parity and saved-view reopening on all reference
   sweep types/selected tune blocks. Preserve input and unrelated tree nodes.
5. Exercise completed-product registration/discovery and Web plotting through
   existing DB/Web infrastructure. Add focused integration code/tests where
   required, and record remaining non-slice limitations honestly.
6. Run final scoped/full QA as applicable, an actual end-to-end local demonstration,
   self-review and a handoff organized by small commits with remaining questions.

## Step ledger

| Step | Scope | Status | Commits / evidence |
| --- | --- | --- | --- |
| 00 | Preserve baseline and define overnight review discipline | complete | baseline table above |
| 01 | Shared file-access configuration, independent of Kids stages | committed | 389e104; [step record](01-file-config.md); 67 tests, scoped lint/type checks |
| 02 | Shared catalog preparation and Python/CLI config parity | committed | b1fd2ab; [step record](02-catalog-and-runtime.md); 232 tests, scoped lint/type checks |
| 03 | Explicit deployment launcher and executable command plan | committed | Deploy 117ba21; [step record](03-deploy-launch.md); 57 focused tests including real reference replay; full-suite worktree limitations recorded |
| 04 | Fixed pipeline reduction policy and current-use guide | committed | a29494e; [step record](04-fixed-reduction-stage.md); 250 tests including exact v2 values; scoped lint/type checks |
| 05 | Mapper-based reducer reads and one LO evaluation | committed | 74ad500; [step record](05-reducer-mappings.md); 279 tests, exact v2 values, scoped typecheck; pre-existing numerical-module lint debt recorded |
| 06a | Configuration model import boundary | committed | 552e86d; [step record](06a-config-model-boundary.md); 33 pipeline tests, scoped lint/type checks |
| 06b | Typed completion manifest and effective-config replay | verified | [step record](06b-product-manifest.md); 56 pipeline/no-op/v2 tests, 2 Web regressions, scoped lint/type checks |

Each entry should contain: evidence inspected; exact intended behavior and
invariants; options and chosen rationale; changed files/APIs; tests and command
outputs; self-review findings and corrections; commit IDs; next step.

## Working environment

- v3 Python: `toltec_astro_dev/.venv/bin/python` (workspace-relative).
- v2 Python: `/Users/ma/.pyenv/versions/tolteca_v2/bin/python`.
- v2 source: `/Users/ma/Codes/toltec/toltec_astro/tolteca/src`.
- v2 dispatcher: `/Users/ma/Codes/toltec/toltec_astro/taco_recipes/scripts`.
- Reference inputs: `toltec_astro_dev/run/data_lmt`.
- Existing design/evidence: `design/file-access-noop-plan.md`,
  `design/kids-reducer-slice-status.md`, `design/kids-v3-rewrite-plan.md`,
  `design/reader-v2-v3-comparison.md`, and `design/probes/kids_vertical_slice.py`.
- April references and SPIE paper: workspace `refs/`; read pertinent evidence,
  but current code and user decisions override stale implementation claims.

`just review` only shows uncommitted changes. For committed overnight work use
`git show <commit>` or `git diff <baseline>..HEAD` in each repo. The final handoff
must include committed-range text diffs, not an empty uncommitted review.
