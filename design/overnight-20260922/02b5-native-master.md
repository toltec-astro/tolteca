# 02b5 — Preserve the KIDs acquisition master code

## Scope and evidence

The user's raw-metadata rule is to retain recorded integer codes and derive
readable names separately. Telescope metadata already does this; the older
KIDs model still replaced the code with a string and stored a redundant string
`mastervar`. This is the first bounded contract correction from 02b4, not the
broader scalar-catalog implementation.

The reference database's master table contains 0/TCS, 1/ICS, 2/CLIP and
3/REPEAT. All four available ROACH-0 acquisitions record 0. Existing
`ToltecDBRawObsMaster` supports 0–2; the established reader already rejects 3.
The header's code-3 description and database label are not equivalent, so this
change does not invent new code-3 semantics or broaden enum coverage.

## Observable change

- `ToltecRawObsMetadata.master` is the required `ToltecDBRawObsMaster` enum,
  retaining the acquisition integer; JSON round-trip emits that integer.
- `master_str` is a cached computed name using the existing enum mapping.
  Quartet and raw-observation identifiers therefore retain their values.
- Remove redundant `mastervar`. No compatibility alias is needed in this
  development-only v3 API. No production consumer was found in the workspace.

SourceInfo/catalog master labels, ORM models, file paths, UID conventions,
scientific arrays, header reads, field requiredness and failure policy are
unchanged. Generic LMT metadata still permits free names such as `simu`.
Other KIDs model issues (original-path naming, defaults, repeat level and
header-only composition) remain separate work; none are folded into this patch.

## Verification and self-review

- Five new tests fail on the prior implementation and pass after the change:
  real VNA/target/tune/timestream codes, cached names, identifiers, JSON
  round-trip, lazy I arrays and rejection of names as raw codes.
- Existing model tests use enum values. The database comparison now independently
  asserts both numeric code and readable database label. Missing-field tests use
  a valid master so they continue to test their intended missing fields.
- Full Tolteca suite: **956 passed in 52.87 s**, actual-v2 interpreter enabled,
  benchmarks disabled. Log: `/tmp/tolteca-20260922-master-pytest-final.log`.
- Fresh four-case exact integration passes through direct/planned CLI, Zarr,
  local SQLite, Parquet, Web traces and Dash callbacks. Report:
  `toltec_astro_dev/run/overnight-slice-20260922-master/report.json` at workspace
  root. No live browser or production service was started.
- Scoped typing and formatting pass. Scoped Ruff passes except two pre-existing
  FURB171 findings at the end of `test_toltecdb.py` and existing CPY001 policy
  findings; the unrelated membership assertions are left untouched. The large
  raw core file is not reformatted.

Diff review confirms one reader construction change and one model contract
change. Tests retain the existing broader synthetic model cases; those are not
new evidence of acquisition formats. No fixture changes, dependency changes,
DB/Web runtime changes or new caches are introduced.
