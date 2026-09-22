# 03a — Store the reduced channel-frequency axis

## Evidence and boundary

The prior result view computed `f_chan` from copied raw tone/center headers,
then renamed the raw tone dimension to the I/Q channel dimension on each view.
Those dimensions differ in real data (`toneFreqLen` versus `iqlen`). Native
selection of the reduced I/Q dimension therefore left the tone header unselected
and made the view fail its channel-count check. Web worked around the structure
by constructing a full frequency grid before selecting channels.

The user's established direction is that the reducer computes the result axes,
the view exposes them, and consumers use the schema/mapper/view. Store `f_chan`
as a reduced coordinate and map it explicitly in `ToltecSweepSchema`. Move the
existing calculation/alignment to reducer execution; retain its exact arithmetic
and the existing raw accessor for tone/center interpretation. The view's getter
becomes a mapped array read. No new selection abstraction or Web-only frequency
formula is needed. Xarray's native Dataset indexing selects arrays sharing a
dimension: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.isel.html.

## Observable changes

- New results carry `f_chan` in Hz on the I/Q channel dimension (also block for
  tune). It survives Zarr serialization and does not depend on raw headers when
  reopened. Its array name is now explicitly `f_chan`.
- Native channel selection can precede `ReducedSweepView.frequency`; order and
  duplicate positions are retained. No source is mutated by this selection.
- The reducer now reads the small tone/center metadata to construct this axis,
  rather than deferring that work to the first frequency-view access. Invalid
  tone/signal channel counts now fail reduction, not just a later view access.
- Standalone inputs without tone/center metadata still reduce I/Q; their
  `f_chan`/`frequency` properties remain None. No assumed frequencies are added.
- Unreleased products made before this change do not acquire the coordinate
  through a compatibility getter. Regenerate them from raw inputs. Likewise,
  a pre-existing in-memory reduced tree must be regenerated with `force=True`.

Mean/std loops, masks, dtype, NaN behavior, block detection/order, center fallback,
and unit interpretation are unchanged. Raw header copying is deliberately not
redesigned in this patch. The additional coordinate has only block × channel
values; the larger channel × sweep grid is still constructed only on request.

## Verification and self-review

- Three new recorded-input cases failed before implementation because there was
  no mapped stored frequency coordinate. They pass for VNA, target and tune,
  including selection `[3, 0, 3]` after all raw `Header.*` variables are removed.
- 67 focused tests pass: standalone sweep tests, mapper/read boundaries, pipeline
  lifecycle/products and four actual-v2 parity cases. Exact array equality covers
  I, Q, both uncertainties, f_chan, sweep and frequency, with units unchanged.
- Scoped typing and formatting pass. Ruff has the same 14 pre-existing reducer
  naming/complexity/style findings as HEAD; an introduced line-length issue was
  fixed. The changed test file passes Ruff. No blanket suppression or unrelated
  numerical cleanup was added.
- Diff review confirms the numerical `_reduce` and `_reduce_single_block`
  implementations are untouched. Only coordinate assembly moved into the
  callable, with schema/view/docs/tests updated.

The Web follow-up selects native dimensions before requesting the grid and
measures its shape. Its consumer change remains a separate commit. The fresh
four-case exact CLI/Zarr/SQLite/Parquet/Web probe passed at 05:41:47 UTC, with
artifacts under `toltec_astro_dev/run/overnight-slice-20260922-coordinates/`.
No browser/server was started. Commit IDs are recorded in the shared index.
