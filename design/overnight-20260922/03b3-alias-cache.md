# 03b3 — Present an alias's cached result to the standalone reducer

## Scope / choice

The numerical reducer knows its native result namespace. The step adapter now
presents an existing aliased result under that namespace in a shallow working
tree containing the raw root and that one child. This does not copy array buffers
or traverse unrelated subtrees. The reducer still owns configuration comparison,
force handling and numerical computation; it gains no pipeline or alias API.

A context-only node has no variables. On that first invocation the original tree
is passed unchanged, preserving the existing ability to reuse a primary result.
`DataTree.has_data` tests local variables, not inherited root coordinates, and is
not a substitute for the reducer's own cache-validity check.

## Behavior and tests

- Two unforced alias-only invocations reduce once rather than twice.
- An existing alias prefers its own cache when the primary result is replaced.
- First-alias reuse of a primary result is retained; force still recomputes.
- A cached result from a different numerical configuration is still recomputed.
- Alias-only work does not add a canonical result child. Primary results,
  unrelated siblings and shared input buffers remain unchanged.

The regression tests failed before the adapter change. All 234 KIDs tests pass
(41.68 s), including actual-v2 parity; scoped Ruff, format and ty pass. Log:
`/tmp/tolteca-20260922-alias-cache-pytest.log`.

The read-only lifecycle probe on the actual ROACH-0 VNA fixture now reports all
three intended results: an attached incomplete live invocation on failure,
correct original-step reconstruction after alias creation, and one numerical
call for two unforced alias invocations. This counts `_reduce` calls, not a
claimed disk benchmark. Buffer-identity tests establish that repeat calls reuse
the numerical arrays. Raw source mutation still requires `force=True`, as in the
standalone reducer; no new invalidation or source hashing policy was introduced.

## Self-review / next finding

Rejected copying the complete tree or adding pipeline-specific parameters to the
numerical object. Existing namespace and cache checks are reused rather than
duplicated. Public documentation specifies reuse and force behavior.

A separate real-reference probe found an output boundary gap: execute NoOp then
SweepReduce on the same tree and the copied raw attrs include Tollan's live
`__pipeline_context__` store. `write_sweep_product` then raises
`TypeError: Invalid attribute in Dataset.attrs`. The normal CLI chooses either
no-op or reduction, so its integration check did not expose this composition.
Handle the reserved live-context store at the export boundary in a separate
patch; do not change numerical reduction or generically serialize open objects.
