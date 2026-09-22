# 03b1 — Attach the current invocation before execution

## Scope and choice

Restore the established live-context contract in the two KIDs storage overrides.
Create a fresh context on the step's owned DataTree node before `run`, and retain
that same object when the reducer installs its result node. The existing final
`set_context` still records JSON-compatible metadata after success. No change to
Tollan, numerical computation, input arrays, or the completed-product schema.

An empty owned node before the first result is sufficient; no new context store,
StepData class, revision counter, or persistent cache is needed. This follows the
v2 attachment lifecycle and current default Tollan behavior while allowing native
arrays and serialized metadata to remain together.

## Observable changes

- `get_context` works during execution and returns the live invocation.
- A first failure leaves an incomplete context, not a missing context.
- A failed rerun replaces the old success metadata with the current incomplete
  attempt. Old array buffers may remain inspectable but cannot be published.
- A failed output attempt has no new product path. Existing products are neither
  overwritten nor removed. Failures still propagate.
- A failed alias does not change the primary result or its metadata.

## Verification and self-review

Initial regressions failed against the baseline. The final KIDs suite passes:
228 tests in 41.46 s, actual v2 interpreter enabled, benchmarks disabled.
Scoped Ruff, formatting and ty checks pass. Log:
`/tmp/tolteca-20260922-live-context-final-pytest.log`.

Tests cover first failure, failed rerun and successful retry, buffer identity,
publication rejection before filesystem creation, alias result isolation,
context identity both before and after the numerical call, final JSON metadata,
and failed repeated publication without changing its earlier manifest.

Self-review found that installing the result node replaces its attrs. A second
regression demonstrated loss of the live context inside `run`; reattaching that
same object after node assignment fixes the gap. The public guide describes the
observable lifecycle and required `completed` check, not the review history.

Alias back-links and alias-only repeated computation remain separate issues
from 03b. Full workspace integration will be repeated after those checkpoints;
the preceding 935-test full suite is not claimed for this new revision.
