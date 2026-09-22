# Vertical-slice review and rework, 2026-09-22

## Authorization and deadline

The user renewed eight hours of local work after approving the root filename
registry / source data-kind separation. Window: 2026-09-22 05:06:38–13:06:38 UTC
(01:06:38–09:06:38 America/New_York). The existing heartbeat
`kids-overnight-vertical-slice` is active for this window. Pause it at the final
checkpoint, or sooner if the scoped review is complete. A scheduled wakeup is
not evidence that work ran: only record commands and outcomes actually observed.

Read `WORKSPACE.md` before resuming. The previous implementation and its evidence
are in [the prior ledger](../overnight-20260921/README.md). This round reviews and
reworks that implementation against the subsequent user decisions, rather than
repeating it or assuming its tests prove the design correct.

## Starting state and preservation

- Tolteca starts at `94b87a03a631ab7ac1a5d075d0db026aff7434a3`, on
  `codex/overnight-kids-slice-20260921`, with the daytime file-layer changes pending.
- DB and Web contain pending imports/tests for the shared root file API.
- `/tmp/toltec-python-source-kind-before.diff` captures the pending work before
  this round's registry change. `/tmp/toltec-source-kind-baseline.json` captures
  all 132 recorded-path source projections for comparison.
- Preserve the pending baseline and all previous commits. Work on local branches
  `codex/overnight-kids-slice-20260922`; identify baseline checkpoint commits as
  such, not as newly authored or fully reviewed overnight work.
- Leave unrelated `.DS_Store` files and the shared Deploy checkout alone.
  Changes to Deploy, if needed, require an isolated checkout, not a silent branch
  switch in the shared C++ workspace.

## Scope and invariants

1. Finish direct domain registration in root `tolteca_datamodels.file` and remove
   data-kind ownership from `FilepathInfo`. Preserve parsing and header precedence.
2. Review the no-op CLI, runtime/config, source catalog, selection, and deployment
   invocation end to end using the revised file/metadata design.
3. Review reducer-only raw sweep → live DataTree/StepContext → reduced view →
   configured Zarr/Parquet output → DB discovery → Web visualization.
4. Repeat exact v2 reference comparisons and integration checks; inspect failure,
   resource lifetime, and avoidable computation paths. Do not relax tolerances.
5. Prepare a focused end-to-end user guide and a separately reviewable design
   trail. Record DB/Web architectural friction; do not broadly redesign them.

Use actual v2/taco code and reference records as evidence. The standalone
SweepReducer owns computation; pipeline steps/config own execution and product
policy. Kids does not consume deployment environment variables. Reuse Tollan
runtime/CLI and schema/mapper/accessor/view machinery. Do not invent metadata,
change acquisition conventions, or silently broaden behavior. No fixture writes,
pushes, merges, destructive resets, CI, production services, instrument actions,
unnecessary dependency upgrades, or agent delegation.

## Step ledger

| Step | Scope | Status / evidence |
| --- | --- | --- |
| 00 | Preserve pending baseline; finish registry and source-kind refactor | Implemented and verified; see [step record](00-file-registry.md). Baseline: Tolteca `269d1db`, DB `317f951`, Web `3bc81fe`. |
| 01 | Reconcile previous overnight work with current decisions | Pending |
| 02 | No-op invocation and catalog review | Pending |
| 03 | Reducer / live context / output review | Pending |
| 04 | DB discovery / Web consumption review | Pending |
| 05 | Exact reference verification, guide, final handoff | Pending |

Every completed step must record rationale, explicit observable changes,
alternatives considered when material, tests, self-review, commit IDs, and next
actions. Keep the index current before yielding. End with a safe checkpoint and
remaining issues, not an unsupported claim of completeness.

## Resume here

Finish the step-00 commit/review export if still pending, then start step 01:
read the prior ledger's per-step records and inspect the current implementation
against the revised metadata/file rules. Build a concise review matrix with
actual findings before changing the pipeline. The previous final checkpoint
already identifies DB/Web identity drift, product/chunking limits and unverified
live browser rendering; verify current state rather than assuming those notes
are still complete. The requested focused telescope→catalog→no-op user guide is
still pending consolidation; existing pages are not the final guide.
