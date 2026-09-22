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
| 00 | Preserve pending baseline; finish registry and source-kind refactor | Complete: Tolteca `62602a2`; see [step record](00-file-registry.md). Baseline: Tolteca `269d1db`, DB `317f951`, Web `3bc81fe`. |
| 01 | Reconcile previous overnight work with current decisions | Initial boundary review and fresh exact integration complete; see [review matrix](01-review-matrix.md). This is a findings inventory, not a claim all layers are redesigned. |
| 02a | Explicit canonical telescope-interface discovery | Complete: `34f6b4e`; 102 focused tests pass. See [step record](02a-telescope-discovery.md). |
| 02b | Metadata opening/enrichment and catalog review | [Usage and projection gaps](02b-catalog-review.md) recorded; implementation pending. |
| 02b1 | Prevent false KIDs classification of telescope files | Complete: `51e0728`; see [regression record](02b1-kids-identification.md). General telescope header identification remains pending. |
| 02b2 | Shared source identification through existing readers | Complete: `4f19abe`; see [step record](02b2-shared-identification.md). |
| 02b3 | Explicit catalog opening policy and shared runner identity | Complete: `673a2ec`; see [step record](02b3-catalog-opening.md). Broader typed scalar catalog projection remains open. |
| 02b4 | Measure the scalar projection gap and model boundary | Diagnostic complete: `970c7ef`; [step record](02b4-scalar-projection-boundary.md). Actual v2/v3 reads and missing `atten_drive` catalog selection reproduced; implementation remains open. |
| 02b5 | Preserve native KIDs master codes | Complete; [step record](02b5-native-master.md). 956 full tests and fresh four-case exact integration pass; source labels and UIDs are unchanged. |
| 02c | Focused telescope/catalog/no-op guide | Complete: `ef546a3`; see [step record](02c-observation-guide.md). Python examples, direct CLI and isolated Deploy invocation verified. |
| 03a | Reduced channel-frequency coordinate | Complete: Tolteca `aabb582`; 67 tests including four exact v2 cases pass. See [step record](03a-reduced-frequency-coordinate.md). |
| 03b | Remaining reducer / live context / output review | [Three reproduced lifecycle/alias gaps](03b-lifecycle-review.md); fixed in 03b1–03b3. Output boundary review continues. |
| 03b1 | Attach live context before execution | Complete; see [step record](03b1-live-context.md). 228 KIDs tests pass, including actual-v2 parity. |
| 03b2 | Keep alias context back-links independent | Complete: Tollan `62e3a5c` and dependent KIDs adapters; see [step record](03b2-context-binding.md). 1,289 Tollan / 230 KIDs tests pass. |
| 03b3 | Reuse alias-local reduction results | Complete; see [step record](03b3-alias-cache.md). 234 KIDs tests pass; the real-reference lifecycle probe now matches all three intended behaviors. |
| 03b4 | Exclude the live store at the product boundary | Complete; see [step record](03b4-live-export-boundary.md). 947 full Tolteca tests and fresh four-case exact integration pass. |
| 04a | Web selection before frequency broadcast | Complete: Web `928bf28`; 10 tests and fresh exact integration pass; [step record](../../../tolteca_web/design/overnight-20260922/04a-frequency-selection.md). Depends on 03a. |
| 04b | Remaining DB discovery / Web consumption review | [Measured storage review](04b-storage-review.md): metadata consolidation and repeated frequency-coordinate reads are the next bounded steps; broader DB/Web redesign stays deferred. |
| 04b1 | Consolidate immutable product metadata | Complete; see [step record](04b1-consolidated-metadata.md). 951 full Tolteca, 18 DB and 10 Web tests pass; fresh exact integration passes. |
| 04b2 | Read selected I/Q and shared coordinates together | Complete: Web `04f0c73`; [step record](../../../tolteca_web/design/overnight-20260922/04b2-shared-coordinate-reads.md). 10 tests and four actual-v2 exact trace checks pass; `f_chan` reads fall 4 → 1. |
| 04b3 | Recorded DB product-test acquisition identity | Complete: DB `412fe6e`; [step record](../../../tolteca_db/design/overnight-20260922/04b3-recorded-fixture-identity.md). All 18 focused tests pass; no runtime change. |
| 04b4 | Measure sparse/full sweep chunk policy | Complete: `42e03cd`; [step record](04b4-chunk-policy.md). 60 exact layout/selection comparisons; writer default unchanged pending serving-workload evidence. |
| 05 | Exact reference verification, guide, final handoff | Pending |

Every completed step must record rationale, explicit observable changes,
alternatives considered when material, tests, self-review, commit IDs, and next
actions. Keep the index current before yielding. End with a safe checkpoint and
remaining issues, not an unsupported claim of completeness.

## Resume here

Resume the KIDs header-metadata boundary in [02b4](02b4-scalar-projection-boundary.md),
then final verification. Do not bolt scalar extras onto the catalog: align the
existing header model first, preserve source identity and numerical behavior,
then evaluate typed header-only composition. The bounded [04b](04b-storage-review.md)
consolidation, repeated-coordinate-read and unit-fixture findings are fixed in
04b1–04b3. Fresh products open with 6–7 LocalStore gets instead of 285–304;
Web reads `f_chan` once instead of four times, preserving exact plot values.
Chunk tradeoffs are measured in 04b4; no universal layout was chosen from local
measurements. All 60 candidate-layout/selection comparisons preserve values.
The export boundary finding is fixed in 03b4.
All three earlier lifecycle/alias gaps are fixed in 03b1–03b3. The repeatable
read-only probe is saved at `design/probes/check_step_lifecycle.py`.
Tollan is now on the new overnight branch; its unrelated `.DS_Store` is untouched.

The guide is consolidated and its examples executed; see 02c. The shared identifier serves catalog preparation
and execution. Python catalog opening policies are implemented; the CLI retains
all-header preparation and now enriches telescope records too. See 02b2/02b3 for
measured reads, behavior changes and self-review. Full scalar catalog projection
remains unresolved.

KIDs raw master codes are now retained in 02b5, with cached `master_str` and
unchanged identifiers. Original-path naming and the remaining header-model
boundary are the next focused checks, not implicit parts of that correction.

Steps 02a, 02b1, 03a
and 04a are committed and exported in `review/overnight-20260922/`. Explicit
`/tel` now searches the actual telescope directory; unqualified/ROACH selection
is preserved and no master is guessed from a telescope filename. Review opening/
enrichment separately: v2 catalog `.open()` merges scalar metadata, whereas the
current KIDs projection exposes only identity/kind. Existing KIDs metadata has
not received all telescope-model refinements. Do not paper over that with another
untyped metadata dictionary or copy of scientific fields.

Fresh four-case integration at Tolteca `aabb582`, DB `317f951`, Web `928bf28`:
`toltec_astro_dev/run/overnight-slice-20260922-coordinates/report.json`. The prior
baseline report remains alongside it. Do not reuse either output directory.
The complete pre-existing 20260921 step records in Tolteca, DB and Web have been
read. The standalone reducer and Step call implementation have now been read
fully. Frequency coordinate ownership/selection is corrected; other numerical,
context failure/lifetime and output-design findings still need a focused review.
Exact integration is evidence, not a substitute for that review.

The focused telescope→catalog→no-op guide replaces the three overlapping pages;
its 11 Python and four KIDs CLI examples pass, as do isolated Deploy planning and
no-op launch. No production services or browser server were launched. Prior
live-browser permission/verification limits remain open.

## Latest QA checkpoint

After 04b1, 951 Tolteca tests pass (57.13 s), with
`TOLTECA_V2_PYTHON=/Users/ma/.pyenv/versions/tolteca_v2/bin/python` and
`--benchmark-disable`. Fresh four-case exact integration through CLI, Zarr,
SQLite, Parquet, Web traces and Dash HTTP callbacks passes; outputs are in
`toltec_astro_dev/run/overnight-slice-20260922-consolidated/`. All 18 DB
sweep-product and 10 Web tests pass against this checkpoint. Scoped typing passes;
the raw core/reducer retain their documented pre-existing lint/format findings.
Reference checkout, DB and Web are clean. No production service or fixture write
was needed. All 1,289 Tollan tests and its full type check pass. Broader scalar
catalog projection and remaining DB/Web review are open; the reproduced lifecycle
issues are fixed and tested. See 03b1–03b4 for the focused behavior changes.

After 04b2/04b3, 10 focused Web and 18 DB tests pass. A fresh full four-case exact
integration passes at `run/overnight-slice-20260922-web-reads/` under the development
location, including direct/planned no-op, actual-v2 values/units, Zarr, SQLite,
Parquet, Web traces and Dash callbacks. No browser server or remote benchmark
was started. Production changes in this checkpoint are only Web's shared
coordinate loading; DB changes are unit-fixture identity, and the remaining
commits are diagnostic/design records. Automation remains active before the
13:06:38 UTC deadline.

After 02b5, 956 Tolteca tests pass (52.87 s) and fresh exact integration passes
in `run/overnight-slice-20260922-master/`. This verifies the native master-code
change without changing scientific outputs or source-catalog labels.
