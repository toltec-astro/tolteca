# 02b7 — Require the recorded type for full raw KIDs metadata

## Reproduction and boundary

The read-only `design/probes/kids_metadata_contract.py` diagnostic removes one
mapped header at a time from in-memory views of each real ROACH-0 file. Before
this change, 47 cases returned metadata instead of rejecting the absent field.
These cover observation type, attenuation, calibration IDs, channel capacity,
original path, repeat level, ADC reference, LO center and sweep repetitions
where mapped. The timestream model does not expose an LO-center field.

Most importantly, deleting `Header.Toltec.ObsType` from the actual tune changes
`n_blocks` from **2 to 1**, while returning `ToltecSweepMetadata` with
`obs_type=None`. Shape identifies a raw sweep family, not its subtype. The user
requires established raw-file fields to retain their read semantics; treating
this absent control field as optional produces misleading full metadata.

Re-read v2 `toltec/ncfile.py`: raw identification uses the observation-type
header, and applicable metadata mappings read their declared fields directly.
Generic shape inference is a useful v3 numerical-input capability, but it is
not evidence that a complete acquisition record exists.

## Focused change

Full metadata reading now requires the mapped observation-type field whenever
the source is identified as raw KIDs data, including a structurally inferred
raw family. The existing mapper raises its normal ValueError if absent.
There is no new exception hierarchy, validation cache or extra native read.

Preserved:

- `data_kind` still performs structural inference. The standalone reducer may
  operate on appropriate numerical arrays without acquisition metadata.
- A processed source identified by `Header.Kids.kind` may still lack the raw
  observation-type field. This branch is explicitly tested; no real legacy
  processed file is present in the fixture set.
- Present type values, other header behavior, reduced arrays, source catalog
  identity, CLI/config and model field declarations are unchanged. Unknown
  recorded type codes are not given a new policy by this patch.

The metadata dataclass is still shared by raw/processed reader paths and retains
nullable fields. This is not a claim that model strictness or all 47 omissions
are fixed. Remaining fields require a separately scoped correction, not a
blanket weakening or a fake partial observation object for the catalog.

## Other reproduced findings, not changed here

- Actual raw files currently yield `instru_component=None` and `repeatvar='0'`;
  the registry says `toltec`/`roach`, and the native repeat level is an integer.
- Applying the raw accessor's full metadata reader to saved reduced VNA/target
  children uses their reduced sample counts (491/177), giving 1.005568/0.362496 s.
  These are not the raw acquisition integration times. Tune children have a
  two-element block dimension on copied `ObsType` and reject a scalar read.
  The error text says `(ndim=0)` as a requirement, not the actual ndim; this
  confusing message is existing Tollan behavior, not a Dask scalar failure.
- The established product consumers use reduced views and step context, not
  the raw child's `.meta`. Do not fix catalog selection by reading full metadata
  from a reduced child, nor flatten promoted header arrays arbitrarily.

The next raw-metadata step should address the remaining established mapped
fields against the v2 inventory, separating raw requirements from processed
representations. Reuse the schema and a common field-construction path for a
later header-only typed projection. Do not first add an untyped scalar dictionary
to SourceInfo or another nullable catch-all model.

## Regression and self-review

Four new real-file missing-type cases fail before the patch and pass afterward.
They retain structural family identification and verify I/Q/LO are not loaded
on rejection. A fifth test preserves processed-header behavior. The previous
minimal-raw positive test now explicitly tests its absent required header;
the cache test uses the actual target file. The existing toy time-series fixture
now declares the same Nominal code found in the real reference timestream; it
is not treated as evidence of an acquisition format.

Baseline/after JSON: `/tmp/tolteca-20260922-metadata-contract-{before,after}.json`.
The after report differs for the four missing-type cases; other recorded gaps
remain visible. The diagnostic asserts that I/Q and ADC payloads remain lazy.

QA:

- **966 full Tolteca tests pass in 55.40 s**, actual-v2 interpreter enabled,
  benchmarks disabled. `/tmp/tolteca-20260922-obs-type-final.log`.
- Fresh four-case exact integration passes at workspace
  `toltec_astro_dev/run/overnight-slice-20260922-header-contract/`, including
  CLI/Deploy no-op, Zarr, SQLite, Parquet, Web traces and Dash callbacks.
- A JSON comparison verifies that only the four missing-type cases change in
  the diagnostic; other omission cases and reduced-product observations match.
- Scoped typing and formatting pass. New test/probe and changed metadata tests
  pass Ruff excluding existing CPY001. `test_accessor.py` has the same six
  unrelated findings as its committed version (TC003, four E402, E741); these
  are not suppressed or swept into the change. Existing core lint debt remains.
- `just qa` would run all-file modifying pre-commit hooks and UV environment
  synchronization. It was not run; the explicit shared-environment checks above
  avoid unrelated formatting/dependency changes. `just review` is regenerated.

The final diff contains a single raw-metadata read-policy change. Reference
inputs and the other repositories are unmodified by this step. No tolerance,
scientific computation, model hierarchy, source-catalog shape or dependency
change is made.
