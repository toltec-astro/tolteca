# 06a — Configuration models independent of execution

Status: verified; commit recorded in the index.

## Evidence / decision

A typed product manifest should contain the effective `KidsConfig` without
importing handlers and output executors. Previously `KidsConfig` lived in
`core.py` and imported configs from both step modules; the output step imports
the product writer. Having the writer import `KidsConfig` there creates a cycle.

Move the three existing configuration model definitions, unchanged, into
`tolteca_kids.config`. The existing modules import them for execution and retain
their current exports. There are no new compatibility classes, duplicate model
definitions, validators, deferred model rebuilds or dynamic type registration.
This is an import-dependency boundary, not a new handler or config mechanism.

## Verification

All 33 configured-pipeline tests passed, covering config construction, generic
Step/Context bindings, conditional recipes, output and serialization. Scoped
pre-commit and typecheck passed. Diff review confirms model bodies/defaults
are unchanged; no numerical code changed. Next: typed manifest and replay.
