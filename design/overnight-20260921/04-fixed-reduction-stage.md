# 04 — Configured reduction policy and current CLI guide

Status: verified; commit recorded in the index.

## Evidence / decision

The user's established requirement was that the pipeline always detect blocks
and compute uncertainties. The prototype still exposed a nested `reducer`
configuration and used it in tests to turn uncertainties off. This was an
unreviewed extra capability, not required for the reducer-only slice.

Remove that field from `SweepReduceConfig`. Retain `enabled`, `block`, and
`force`; the stage uses one immutable, source-free default `SweepReducer`.
The independently callable numerical object's API and numerical implementation
are unchanged. Unknown nested reducer options now fail validation instead of
silently changing the scientific result. Alias/context and conditional-config
tests use valid stage options and continue to check independent invocations.

The current reduction guide also retained obsolete `tolteca kids reduce`,
`--output`, and `--data_root` examples and described superseded conflict behavior.
Correct these to the actual standalone CLI and header-authoritative metadata
contract. Do not add the implementation history to public docs.

## Verification / self-review

- 250 tests passed: all KIDs tests plus reducer/view tests, including the actual
  v2 cross-environment comparisons for target, VNA, last tune block and tune
  block 0. Array equality remains exact, not tolerance-based.
- Scoped pre-commit and typecheck passed. Final diff is limited to the stage,
  its tests, current-use guide, and this review trail.
- The first test run exposed two old conditional fixtures still using the
  removed configuration. Updated both to exercise `force` and reran the full
  focused set; no numerical regression or failure was waived.

Next: reduce literal backend-name access in the numerical reducer through the
existing schema/mapper boundary, preserving arithmetic order and unit values.
