# Vertical-slice review and rework, 2026-09-22

## Authorization and deadline

The user renewed eight hours of local work after approving the root filename
registry / source data-kind separation. Window: 2026-09-22 05:06:38–13:06:38 UTC
(01:06:38–09:06:38 America/New_York). The heartbeat
`kids-overnight-vertical-slice` was confirmed **PAUSED at 10:20:26 UTC** after
the scoped verification completed, before the deadline. A scheduled wakeup is
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
| 02b | Metadata opening/enrichment and catalog review | Findings addressed through 02b16; see the individual records for the sequence. |
| 02b1 | Prevent false KIDs classification of telescope files | Complete: `51e0728`; see [regression record](02b1-kids-identification.md). Shared identification follows in 02b2. |
| 02b2 | Shared source identification through existing readers | Complete: `4f19abe`; see [step record](02b2-shared-identification.md). |
| 02b3 | Explicit catalog opening policy and shared runner identity | Complete: `673a2ec`; see [step record](02b3-catalog-opening.md). Typed scalar projection follows in 02b15/02b16. |
| 02b4 | Measure the scalar projection gap and model boundary | Diagnostic complete: `970c7ef`; [step record](02b4-scalar-projection-boundary.md). Actual v2/v3 reads and missing `atten_drive` catalog selection reproduced; implementation remains open. |
| 02b5 | Preserve native KIDs master codes | Complete; [step record](02b5-native-master.md). 956 full tests and fresh four-case exact integration pass; source labels and UIDs are unchanged. |
| 02b6 | Consistent KIDs original-path name | Complete; [step record](02b6-original-path.md). 961 full tests pass. Rename to `filepath_orig`, preserving values and current missing-field behavior. |
| 02b7 | Require raw observation type; audit remaining omissions | Complete; [step record](02b7-observation-type.md). 966 full tests and fresh exact integration pass. Missing type no longer silently changes tune block count. |
| 02b8 | Native repeat and registered instrument component | Complete; [step record](02b8-native-repeat-component.md). 971 full tests pass; integer `repeat` and `toltec`/`roach` identity, with no extra reads. |
| 02b9 | Require raw scalar/string readout settings | Complete; [step record](02b9-required-readout-headers.md). 1,004 tests and fresh exact integration pass. Missing recorded settings no longer silently become None; processed omissions remain supported. |
| 02b10 | Shared recorded LO-center metadata | Complete; [step record](02b10-lo-center-metadata.md). 1,016 tests pass. Timestreams now expose the same recorded center as v2; raw missing centers reject and processed omissions remain nullable. |
| 02b11 | Required raw design dimension and lazy ADC reference | Complete; [step record](02b11-design-metadata.md). 1,026 tests pass; missing raw fields reject without loading their payloads. |
| 02b12 | Required recorded raw sweep geometry | Complete; [step record](02b12-sweep-geometry.md). 1,039 tests and fresh exact integration pass; raw repetitions, capacity and sample dimension cannot silently use processed fallbacks. |
| 02b13 | Typed header-only acquisition metadata | Complete; [step record](02b13-header-metadata.md). 1,050 tests and fresh exact integration pass; tune catalog metadata can be read without an LO-vector scan. Catalog composition remains next. |
| 02b14 | Exact native acquisition-time serialization | Complete; [step record](02b14-timestamp-serialization.md). Use numeric Unix seconds for t0/t1; exact roundtrips on all four telescope files. Separate catalog prototype remains in progress. |
| 02b14a | Exact ICRS coordinate serialization | Complete: Tollan `1c65394`; [step record](../../../tollan/design/overnight-20260922/02b14a-coordinate-precision.md). 1,293 Tollan tests pass; named-source behavior unchanged. |
| 02b15 | Typed source metadata and Arrow projection | Complete; [step record](02b15-source-metadata.md). Source/JSON/Parquet/query/condition support is implemented. Automatic header attachment remains next. |
| 02b15a | Serializable unknown telescope original names | Complete; [step record](02b15a-unrecognized-telescope-path.md). Parsed path is None for unrecognized names; explicit integrity checks remain strict. 120 focused LMT tests pass. |
| 02b16 | Automatic typed catalog preparation | Complete; [step record](02b16-catalog-preparation.md). Header selection and conditional config work through CLI and Deploy; 1,068 tests and fresh exact integration pass. |
| 02c | Focused telescope/catalog/no-op guide | Complete: `ef546a3`; see [step record](02c-observation-guide.md). Python examples, direct CLI and isolated Deploy invocation verified. |
| 03a | Reduced channel-frequency coordinate | Complete: Tolteca `aabb582`; 67 tests including four exact v2 cases pass. See [step record](03a-reduced-frequency-coordinate.md). |
| 03b | Remaining reducer / live context / output review | [Three reproduced lifecycle/alias gaps](03b-lifecycle-review.md); fixed in 03b1–03b3. Output boundary addressed in 03b4. |
| 03b1 | Attach live context before execution | Complete; see [step record](03b1-live-context.md). 228 KIDs tests pass, including actual-v2 parity. |
| 03b2 | Keep alias context back-links independent | Complete: Tollan `62e3a5c` and dependent KIDs adapters; see [step record](03b2-context-binding.md). 1,289 Tollan / 230 KIDs tests pass. |
| 03b3 | Reuse alias-local reduction results | Complete; see [step record](03b3-alias-cache.md). 234 KIDs tests pass; the real-reference lifecycle probe now matches all three intended behaviors. |
| 03b4 | Exclude the live store at the product boundary | Complete; see [step record](03b4-live-export-boundary.md). 947 full Tolteca tests and fresh four-case exact integration pass. |
| 04a | Web selection before frequency broadcast | Complete: Web `928bf28`; 10 tests and fresh exact integration pass; [step record](../../../tolteca_web/design/overnight-20260922/04a-frequency-selection.md). Depends on 03a. |
| 04b | Remaining DB discovery / Web consumption review | [Measured storage review](04b-storage-review.md): findings addressed in 04b1–04b5; broader DB/Web redesign stays deferred. |
| 04b1 | Consolidate immutable product metadata | Complete; see [step record](04b1-consolidated-metadata.md). 951 full Tolteca, 18 DB and 10 Web tests pass; fresh exact integration passes. |
| 04b2 | Read selected I/Q and shared coordinates together | Complete: Web `04f0c73`; [step record](../../../tolteca_web/design/overnight-20260922/04b2-shared-coordinate-reads.md). 10 tests and four actual-v2 exact trace checks pass; `f_chan` reads fall 4 → 1. |
| 04b3 | Recorded DB product-test acquisition identity | Complete: DB `412fe6e`; [step record](../../../tolteca_db/design/overnight-20260922/04b3-recorded-fixture-identity.md). All 18 focused tests pass; no runtime change. |
| 04b4 | Measure sparse/full sweep chunk policy | Complete: `42e03cd`; [step record](04b4-chunk-policy.md). 60 exact layout/selection comparisons; writer default unchanged pending serving-workload evidence. |
| 04b5 | Complete raw fixture for DB integration | Complete: DB `c29e3b6`; [record](../../../tolteca_db/design/overnight-20260922/04b5-complete-product-fixture.md). 192 default-selection tests pass. |
| 02b17 | Recorded identity-only test filenames | Complete: `9625c23`; [record](02b17-identifier-fixture-evidence.md). 35 focused tests pass; no runtime change. |
| 02b18 | Final changed-file type narrowing | Complete: `2affe0c`; [record](02b18-test-type-narrowing.md). All changed source/tests pass ty. |
| 05a | Executed reduction guide | Complete: `cf0307f`; [record](05a-reduction-guide.md). Eight Python examples pass with prepared headers. |
| 05b | Remove redundant metadata-helper state | Complete: `e9c3951`; [record](05b-metadata-helper-state.md). Explicit zero/one/two-block tests; no extra reads. |
| 05 | Exact reference verification and handoff | Complete with stated verification limits: [final checkpoint](05-final-checkpoint.md). |

## Current status and next user review

The agreed no-op and reducer-only review/rework is complete. **1,071 Tolteca,
1,293 Tollan, 192 DB, 13 runnable Web and 57 focused Deploy tests pass.**
Fresh exact-v2 integration is in
`toltec_astro_dev/run/overnight-slice-20260922-handoff/` at the workspace root.
Guide verification covers 12 observation-guide Python blocks, four CLI examples,
and eight reduction-guide Python blocks.

Start with [the final checkpoint](05-final-checkpoint.md) and workspace
`review/overnight-20260922/README.md`. Review the focused commits before merging
or extending to later scientific steps. Remaining decisions, measured costs,
baseline lint debt, the legacy Web collection blocker and the unverified browser
rendering are explicit in that checkpoint. Nothing was pushed or merged.

Earlier per-step records intentionally retain their contemporary findings and
pending follow-ups. They describe the sequence, not current blockers. Superseded
resume/QA paragraphs were removed from this index; they remain in Git history.
Do not restart already completed work from an older checkpoint's “next” section.
