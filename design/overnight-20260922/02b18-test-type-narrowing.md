# 02b18 — Verify narrowing across every changed source/test file

The final type check covered all Python source/test files changed since the
preserved daytime checkpoint `269d1db`, not only the most recent patch's selected
files. It found 12 diagnostics in two tests added/affected by 02b15a and 02b16:
dereferencing the now-nullable parsed telescope path, passing optional filename
guesses to identification, and accessing subtype-specific metadata without
narrowing the shared metadata base.

Add assertions for the tested successful parse and concrete header model. These
assertions also fail more locally if a regression violates the fixture's intended
preconditions. No casts, suppression, annotation weakening or runtime-library
changes. Include a non-null resolved mapping-name assertion before the test's
deliberate in-memory mutation.

All changed Python application/test files now pass:

```sh
git diff --name-only 269d1db -- src tests | rg '\.py$' | \
  xargs ../toltec_astro_dev/.venv/bin/ty check --python ../toltec_astro_dev/.venv
```

This corrects the earlier scope gap in QA; earlier statements of scoped typing
success did not establish that these two files were clean. The full Tolteca
suite passed 1,068 tests immediately before these assertion-only edits; the
final checkpoint records the targeted rerun afterward. Scoped Ruff/format and
Git whitespace checks pass. Scientific outputs and public behavior are unchanged.
