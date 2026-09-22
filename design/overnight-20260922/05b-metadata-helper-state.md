# 05b — Remove redundant block-dimension state from the metadata helper

The final whole-changed-file Ruff check found 30 diagnostics. Comparison against
the same files at the preserved `269d1db` checkpoint found 28 matching baseline
diagnostics and two new argument-count warnings in `_get_sweep_metadata`.
Passing the already-read data kind avoids another header read, but had expanded
the helper to six arguments alongside redundant `is_multi_block` / `n_blocks`.

Replace that pair with `n_blocks_dim: int | None`: None means no explicit block
dimension; an integer is its existing size. The helper reconstructs the same
Boolean and flat-input default. In particular, zero is not treated as missing.
Keep the already-read kind, so no additional dimension/header lookup is added.
Annotate the private helper's inputs and return type rather than weakening lint
rules or introducing a grouping class for this one call.

The public `get_metadata` signature, validation order, empty-dimension behavior,
flat tune LO-reset detection, fields, units and numerical results are unchanged.
Add three explicit layout tests (zero/one/two blocks), using the existing
deliberately synthetic geometry fixture. They check list shape, selected-block
equality and out-of-range errors, not acquisition filename validity.

Verification: 1,068 full tests pass after the helper edit; the subsequent focused
45-test run includes the three new layout cases, header-only read boundaries
and catalog read counts. All changed Python source/test files pass ty. This
removes the two newly introduced lint diagnostics; the six remaining findings
in core.py match baseline signatures. Existing core formatting debt is left
unchanged. The final checkpoint records the full run including the three new
tests. No tolerance, scientific algorithm or source cache changes.
