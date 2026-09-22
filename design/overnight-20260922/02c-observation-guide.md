# 02c — Focused observation-file walkthrough

## Scope

Replace the overlapping `observation-data.md`, `raw-data.md` and
`file-catalogs.md` pages with one `docs/guide.md`. The sequence is telescope
metadata and lazy arrays → current versus recorded filenames → catalog creation,
opening and selection → KIDs CLI inspection/no-op → Deploy invocation/planning.
Examples use recorded observation 145609 and its actual ROACH-0/telescope files.
The three old pages are removed from the current tree, not Git history; they
remain recoverable in the preceding commit.

Retain practical metadata/array tables, filename aliases, Parquet and Pydantic
round-trip examples, and location/lifetime caveats in the guide. Move the useful
standalone S21 and tune-block metadata examples to `kids-reduction.md`. Keep the
ObsSpec language and detailed CLI/reduction references separate, with updated
links. Avoid registry architecture, review history and rejected choices in the
public prose. In particular, do not advertise full scalar-header catalog fields
that are not implemented, or no-op success as telescope/timestream reduction.

## Verification

`design/probes/check_observation_guide.py` executes all 11 Python blocks from the
actual guide, substituting only the raw-root placeholder and using a temporary
working directory for generated catalogs. It also executes all four KIDs console
examples, using the guide's YAML recipe and the current interpreter. All pass.
Assertions cover the real UID, counts, source-model round trips, and exact Arrow
struct values/types after Parquet. Raw inputs are only read.

The initial whole-frame dtype assertion in this verifier failed because
`read_parquet(dtype_backend='pyarrow')` also changes scalar strings from pandas
StringDtype to Arrow large_string. The guide intentionally requests that backend.
The verifier now compares full typed source records and exact struct columns;
no scientific tolerance, nested values, or declared struct types were relaxed.

The guide's standalone CLI inspection/no-op commands also pass directly. With
the isolated Deploy checkout in PYTHONPATH, both `run_kids --show_command` and
actual `run_kids --location toltec_astro_dev -- --dry_run -- 145609/0` pass.
The plan supplies explicit roots and the execution completes the two KIDs inputs
without product output. The shared C++ Deploy checkout was not changed.

Sphinx HTML builds from a temporary copy of docs/README, leaving generated files
outside the checkout. Fixed one new missing section link found in the first
build. Final offline build (Python Sphinx API with empty intersphinx mapping)
has only two existing warnings for absent `cli` and `modules` reference pages.
The new guide has no build warnings. Output/logs:
`/tmp/tolteca-guide-docs.9OUcYO/`. This validates rendering generation and links,
not a live-browser/pixel inspection. Probe Ruff/format and diff checks pass.

## Self-review and remaining scope

The guide distinguishes filename UT, telescope times and integration time; it
keeps canonical `alt`, `target_ref`, interface `tel`, native prefix and UIDs
separate. Opening recognition, strict metadata validation and optional original
path consistency are described at their actual call boundaries. It explains
bounded telescope identification reads and preserves no-op/output separation.

This is a documentation checkpoint, not completion of the larger overnight
review. Full scalar catalog projection, remaining reducer/context/output review,
and DB/Web friction still need follow-up. No scientific or runtime behavior is
changed in this commit.
