# 02 — Shared catalog preparation and Python/CLI configuration parity

Status: verified; commit recorded in the index.

## Evidence / decision

Step 01 exposed a real asymmetry: CLI composition supplied `kids.output.path`
from `files.dataprod_toltec_path`, but library callers of `run_source` did not.
Use one small application function, `create_kids(runtime, context_input=...)`,
for constructing a per-input handler with the shared default and context.
It returns the established Kids handler, uses an independent runtime variant,
does not open data, and does not change SweepReducer or generic analysis APIs.
Passing the already serialized context avoids computing it again in the runner.
Keep the base runtime unchanged; user recipe/CLI output paths outrank defaults.

Move the existing `build_input_catalog` function to the shared TolTEC filestore
module. It remains filesystem discovery followed by scalar KIDs header enrichment
and a metadata-only DataFrame projection. This is a responsibility move, not a
new opening policy or a claim that full telescope/ORM metadata is already unified.
The old Kids import is removed without a compatibility re-export (all consumers
are owned). Update actual callsites/tests, not only the public CLI.

## Invariants and deferred work

No new fields, env consumption, payload reads, selection language changes,
source caching or backend registry. Catalog and execution file handles remain
scoped. Existing no-op backend-read guards must still pass. Telescope catalog
enrichment, lazy/explicit opening policy and common ORM metadata projection
remain separate steps. Do not silently broaden this relocation to those changes.

## Verification / self-review

- 232 tests passed: all `tests/tolteca_kids` plus shared filestore tests, with
  the real v2 interpreter enabled. This includes backend-read guards, actual
  product writing/reopening, CLI precedence, runtime isolation and v2 parity.
- Scoped pre-commit passed (typecheck run separately); scoped `ty check` passed.
- Reviewed the diff: catalog function is moved without opening-policy changes;
  CLI no longer mutates its base runtime; Python and CLI share construction.
- Self-review correction to step 01: codespell changed an intentionally misspelled
  test key into the valid `select` key after the initial test run. The expanded
  run correctly failed. Replaced it with `unknown_setting`, then ran hooks before
  tests and confirmed both pass. No production extra-field validation bug was
  found. Earlier 67-test evidence referred to the pre-hook test state.

Next: bounded deployment command translation, preserving generic `run` behavior.
