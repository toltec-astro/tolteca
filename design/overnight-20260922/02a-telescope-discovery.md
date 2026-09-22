# 02a — Explicit telescope-interface discovery

## Evidence and chosen change

The recorded telescope files are in `data_lmt/tel/`, unlike readout files in
`data_lmt/toltec/<master>/<interface>/`. The v2 `LmtFileStore.tel_path` and taco
`get_lmt_info_table_from_toltec_info` use this separate directory. The current
resolver nevertheless sent `145609/tel` into the TolTEC layout, returning no
matches for two files present in the reference root.

For an explicitly requested canonical `tel` interface, glob
`tel/tel_*_<obsnum>_*.nc`, then apply the same typed ObsSpec constraints to the
same filename catalog. Sets/ranges use a wildcard candidate observation and the
existing vectorized selector. Logging now includes the actual search pattern.

This is an explicit behavior addition, not a claim of unchanged discovery:
`145609/tel` and `145609/2/tel` now find the recorded telescope files. Existing
unqualified, ROACH and named-readout selectors retain their scope and ordering.
No filename regex, schema, array reader, catalog model, CLI override, DB identity
or default opening policy changes.

## Alternatives and boundaries

- Adding telescope files to every unqualified observation would change existing
  KIDs invocations into mixed inputs. Do not do that implicitly.
- Recursive scanning of the whole root would lose the narrow layout-based search
  and make performance depend on unrelated trees. Use the known directory.
- A new filesystem registry/class hierarchy is unnecessary for this two-layout
  correction. Keep this separate from any future common store architecture.
- `tel_toltec` and `tel_toltec2` remain filename prefixes, not new interfaces.
  Select streams through `file_prefix` after canonical `tel` discovery.
- A master-constrained telescope selector cannot match at the filename-only
  stage because that path does not supply master. Do not infer TCS from the
  association rules in taco, or open headers implicitly to satisfy it.

## Tests and self-review

New tests first reproduced **5 failures and 2 passes** against the prior resolver.
After the change, **102 tests pass** across the new filestore tests, ObsSpec,
source identification, and CLI/Deploy equivalence. Cases cover exact observation,
scan, set/range selection, no header opens, no invented master, unqualified/ROACH
preservation, and deduplication with a temporary symlink to the real fixture.

All positive acquisition files are recorded reference cases. The fixture is
read-only; no synthetic acquisition layout was substituted for this regression.
Ruff (existing CPY001 exclusion), formatting and scoped typecheck pass for both
changed Python files; `git diff --check` passes. Numerical code is untouched;
the step-01 four-case exact end-to-end result remains the baseline, not a test
rerun claimed for this directory-only change. No commit hooks are installed in
Tolteca; QA commands were run explicitly. No pushes, resets or dependency changes.

The public ObsSpec guide now states the new explicit telescope path and the
unchanged unqualified scope. It does not claim telescope header enrichment or
KIDs reduction of telescope data. Those remain separate boundaries.

## Next

Review opening/enrichment independently, using actual v2 metadata consumers and
the existing typed reader models. In parallel with that analysis (not delegated),
review the reducer/output and Web computation/lifetime paths. Do not introduce
a partial copy of scientific metadata merely to make a catalog example work.
