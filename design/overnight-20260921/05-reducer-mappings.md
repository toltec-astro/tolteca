# 05 — Reducer mappings and repeated LO reads

Status: verified; commit recorded in the index.

## Evidence / decision

The numerical reducer independently spelled raw I/Q, LO, center-header, and
`numSweeps` names although the raw schema already defines them. Use the existing
raw accessor's resolved mapper for these reads. Keep explicit `sweep_axis` and
`time_axis` arguments, generic `I`/`Q`/`f_lo` input support, and the existing
median-center fallback when no center is available. Do not request full metadata
or turn computation into an orchestration/config-handler concern.

Reduced coordinate/dimension lookup belongs to its schema. Add a dimension
mapping `n_blocks` distinct from the optional block coordinate; views work even
without a coordinate variable. Declare uncertainty variables and config/center
attributes at explicit locations. Use the output mapper for output coordinate
names, and the raw mapper for block-dependent metadata dimensions.

Use `.u.unit_str` to propagate I/Q units without constructing quantities or
triggering conversion. The existing NumPy means/std, sample masks, ordering,
allocation dtype and NaN behavior remain unchanged.

The LO vector was read for block detection, then read again inside each block.
Pass the already loaded block slice into the existing numerical routine instead.
This is invocation-local data flow, not a cache keyed by floating-point data or
a cache persisted on the source. Add a Dask read-counter test for two blocks.

## Behavior boundary / self-review

Mapper resolution now enforces its established ambiguous-location checks for
raw fields. A same-name header attribute and variable no longer silently prefers
the attribute inside the reducer; such ambiguous standalone inputs are invalid.
Reference acquisition files have unique header locations. Center variable reads
follow the mapper's scalar contract, rather than accepting arbitrary one-element
numeric array shapes. No fixture requires those relaxed forms. These are explicit
review points; do not call them bit-for-bit preservation for malformed inputs.

Consulted current Xarray documentation on coordinate inheritance:
https://docs.xarray.dev/en/stable/generated/xarray.DataTree.to_dataset.html
https://docs.xarray.dev/en/stable/user-guide/data-structures.html
The current result projection intentionally keeps inherited channel coordinates
used by result variables and excludes unrelated time coordinates. Simply changing
it to `inherit=False` could drop the channel coordinate. Leave that behavior
unchanged in this step.

## Verification

- 279 tests passed: all KIDs tests plus the raw/reduced sweep datamodel tests,
  with actual v2 parity enabled. Exact values, units, tune blocks, no-op reads,
  saved products, and pre-existing standalone reducer options remain covered.
- The new Dask counter confirms one LO computation for two blocks. A direct
  diagnostic loaded the previous committed reducer (`a29494e`) and the new one
  against equivalent fresh lazy inputs: before = 3 evaluations, after = 1.
- Scoped typecheck passed. Pre-commit for the stage and new tests passed;
  numerical-module formatting and other hooks passed. Its Ruff hook still reports
  14 existing naming, complexity and style diagnostics; the separately installed
  newer Ruff also reports copyright (15 baseline diagnostics). No new diagnostic
  class was introduced. Complexity decreased, but algorithm-wide cleanup is not
  included. Do not rename scientific I/Q fields to satisfy generic naming lint.
- Independent diff review confirmed no changes to the mean/std loop, mask
  construction, output dtype or block ordering. Shape diagnostics now refer to
  arrays obtained by the raw mapper; their output attribute keys are unchanged.
- Test correction: center-location and center-fallback cases use separate source
  datasets. Mutating a source's schema after its accessor has resolved a reusable
  mapper is outside the established mapping-plan lifetime contract.

Next: typed completed-product interchange metadata, per-input configuration
replay, then DB registration and actual Web consumption.
