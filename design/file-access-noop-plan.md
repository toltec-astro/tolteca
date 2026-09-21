# Shared file access for the no-op slice

Status: proposal, 2026-09-17. The shared-layer changes below are not implemented.
The current Kids-owned input config is a prototype, not the intended ownership.

## Metadata-first core, shared by filesystem and ORM access

The eight-page `refs/zma_spie2026_v2.pdf` was read in full, including visual
inspection of the system overview and DPDB diagrams. Section 2 places validated,
serializable metadata in `tolteca_datamodels`, beneath DB and Web. Sections 2.2
and 3.3 distinguish logical products, physical source locations, and provenance.
This is the architectural direction, not evidence that the current implementation
already satisfies every claim in the paper.

The shared core must therefore be a metadata/query contract, not merely a set
of filesystem helpers that DB later emulates:

- Domain metadata definitions, field meanings, units, enum values, and validation
  belong in datamodels. Reuse and complete the existing Pydantic metadata types;
  do not add a separate Kids/catalog copy of those definitions.
- Source location and logical observation/product identity are separate from
  measured metadata. A DB key or a local path is not the common domain model.
- Filesystem access constructs typed metadata through schema/mapper/accessor
  readers. ORM access restores or projects the same domain metadata from stored
  records. ORM tables and persistence serialization stay in `tolteca_db`; the
  shared core has no SQLAlchemy dependency and no required database session.
- The in-memory catalog is a tabular projection of those typed records. Its
  scalar columns must have declared, consistent meanings, units, enum encodings,
  and missing-value behavior regardless of where records came from. It is not
  an independently maintained metadata schema.
- `--select` consumes that projection. Initially keep the existing pandas
  expression behavior. A DB adapter can push ObsSpec constraints into SQL and
  evaluate the remaining general expression on the common projection. Do not
  promise arbitrary pandas expressions can be translated directly to SQL.
- Missing metadata remains missing or fails validation when required. Unknown
  selection fields should fail explicitly. Never substitute plausible identities,
  times, or instrument settings just to complete a metadata model.
- Distinguish file-level metadata from block-level metadata and computed science
  results. Building a scalar selection catalog must not reduce arrays or expand
  one file into multiple block rows implicitly.

The paper uses ObsSpec as an observation address. Preserve that useful exact
quartet representation, while keeping broad DSL selectors (sets/ranges/wildcards)
distinct from identity. An ObsSpec does not uniquely identify every derived
product, processing attempt, or storage replica.

### Raw-reader checkpoint (2026-09-18)

Tollan's mapper foundation is committed as `658b701`: explicit field locations,
shared native/scalar/string/dimension getters, source-only ordinary reads, and
an explicit `get_value_or_default()` fallback operation. Xarray uses `get_arr()`
and `has_arr()` aliases; NetCDF keeps the shared `get_var()` and `has_var()` names.

The local, still-uncommitted KIDs and telescope reader work uses these hooks
without overriding backend lookup or adding a separate string decoder. KIDs
dimension counts use `n_*` schema fields, independently of coordinate values.
Tests cover counts represented as a scalar variable, attribute, or dimension;
same-name coordinates/attributes cannot replace a mapped dimension. Qualified
candidate lists define precedence; ambiguous unqualified headers raise.

The initial KIDs metadata implementation fabricated identifiers, timestamps,
and durations and could average the LO array for metadata. The pending reader
instead requires identifiers/acquisition headers, derives integration time from
sample count/rate, and leaves missing optional metadata unknown. It retains the
single-block metadata helper. The telescope reader exposes typed metadata and
lazy pointing arrays; shared string reads handle fixed-width and character
headers. Reference-file tests prohibit detector/pointing-array reads during
metadata access, except the documented telescope time endpoints and tune LO
scan needed for block counting.

This completes neither the shared selection catalog nor the no-op CLI's final
ownership split. Full tune metadata still reads the LO array, so catalog
enrichment must not blindly request every field from `.meta`. The scalar catalog
projection and its agreement with ORM records are the next boundary to settle.

### Remaining shared-layer gaps

`ToltecRawObsMetadata` already declares fields such as `atten_drive`, but the
current `identify_source()` projection exports only a small identity/kind subset.
V2's drivefit selection opens metadata before querying and can use fields beyond
filename identity. Thus the no-op catalog's current field set is not the final
shared selection contract.

DB's `RawObsMeta` / `RoachInterfaceMeta` and its two API source records contain
separate overlapping definitions and defaults. Unification requires explicit
reuse/projection of the domain models, not just returning a DataFrame with a few
matching column names. Product aggregation and persistence-only fields may
remain distinct; their overlapping scientific fields must agree.

The first cross-backend acceptance test should construct metadata from a real
reference file, store/reload its corresponding metadata in a temporary DB, and
compare the common catalog values and selector results. Include missing fields,
enums, units/time normalization, filename/header conflicts, and duplicate storage
locations. File-side tests must guard against science-array reads. Full remote
storage and DB schema migration remain separate work.

## Established usage

The v2 source tree inspected is
`/Users/ma/Codes/toltec/toltec_astro/tolteca`; its corresponding dispatcher
scripts are under `../taco_recipes/scripts`.

- `tolteca_datamodels/filestore.py`: `FileStoreBase` holds an explicit root and
  exposes managed paths and globbing. It does not interpret environment variables.
- `tolteca_datamodels/lmt/filestore.py`: `LmtFileStore` manages the telescope and
  instrument subtrees; its `toltec` property returns a `ToltecFileStore`.
- `tolteca_datamodels/toltec/filestore.py`: `ToltecFileStore` knows the master /
  interface directories and current-file symlinks. `ObsSpec` separately resolves
  selectors, explicit files, and current/latest observations into a metadata table.
- `taco_recipes/scripts/file_utils/toltec_file_utils.py`: `LmtToltecPathOption`
  adds shared path and ObsSpec arguments, creates stores, and resolves inputs.
  Product and Tlaloc roots are paths; their file-store implementations are TODOs.
  The same module locates telescope files corresponding to a TolTEC observation.
- `kids/dispatch_tolteca_kids.sh`: resolves environment-specific settings into
  `--config`, `--data_lmt_path`, `--dataprod_path`, and `--tlaloc_etc_path`.
  Its Python partner separates `^kids\..+` overrides, resolves the input table,
  translates the product root into stage-output defaults, then processes files.
- `kids/tone_match.py`: uses the shared helper with multiple ObsSpecs, sorts
  by ROACH/time, groups by ROACH, and separately selects reference data.
- `drivefit/dispatch_drivefit.py`: also accepts multiple ObsSpecs and `--select`.
  It opens the catalog to expose scalar metadata before evaluating selection,
  then groups by ROACH. Its output root also supplies product search paths.
- `kids/reduce_ql.py` and `reduce_noise_ql.py`: use the same discovery helper,
  followed by application-specific kind filtering and observation constraints.
- `tlaloc/dispatch_tlaloc.py`: uses discovery to expand one input into an
  observation/ROACH selector for a subsequent operation.

These uses require both per-file and grouped consumers. The shared layer must
not assume that a selected row always becomes one Kids pipeline invocation.

## What the v3 DB design actually supplies

`tolteca_db/design/toltec_data_file_and_data_flow.md` separates logical products,
per-interface sources, and storage locations. ObsSpec is a user-facing selector,
not an ORM identity. The architecture document proposes a DB-backed store
returning the same catalog interface as filesystem discovery.

The proposed `compat/file.py` / `ToltecDBFileStore` is not present. The current
implementation has two `ObsQuery` APIs (`api/obs.py` and
`repository/file_api.py`) and distinct source-record models. The API query uses
SQL predicates; the repository query filters loaded metadata. Both now use the
shared ObsSpec parser. `FileScanner` already delegates filename interpretation
to datamodels, but adapts it into another ingestion record.

Source location needs particular attention: ingestion can store relative source
URIs, while query conversion currently returns those strings without resolving
them against `Location.root_uri`. A relative string must not become a path
relative to the consumer's working directory. Multiple replicas also require a
location choice; local symlink deduplication is not a substitute for that choice.

## Proposed ownership

1. **Deploy:** interpret deployment configuration/environment, choose executable
   and working directory, and construct explicit application arguments/config.
2. **Shared datamodel core:** own typed metadata, source/identity semantics,
   ObsSpec, and the catalog projection/selection contract. Filesystem stores
   implement local discovery and resource-scoped metadata access using that core.
   No environment or deployment lookup.
3. **Application CLI:** compose shared file options with application options;
   only accept its allowed dotted override namespace. Kids uses `kids\..+`.
4. **Kids handler/pipeline:** validate per-input scientific configuration and
   execute the requested workflow. It does not own general discovery settings.
5. **DB adapter:** query indexed metadata and resolve storage locations into the
   shared catalog contract. Keep SQL, registration, and availability policy in DB.

Use a shared configuration section for roots and selection, separate from
`kids`; the section name is not settled. Explicit `--data_lmt_path`,
`--dataprod_toltec_path`, and `--select` should populate that shared configuration.
Do not broaden arbitrary dotted overrides to every runtime key just to support
these known options. Primitive stores should remain directly constructible with
plain Python paths, without requiring a runtime context.

Retain the useful `LmtFileStore` / `ToltecFileStore` distinction. A product root
is a different storage role, not part of the raw TolTEC tree. Keep it as an
explicit shared setting initially; product naming, writing, and publication
remain responsibilities of the configured output step. Do not port the v2
plot/pickle export mixin into the discovery API.

## Query and metadata behavior

- Parse ObsSpec once. Local discovery uses its exact constraints to narrow
  filesystem globs, then applies the full constraints to filename metadata.
  DB queries translate those same constraints to SQL. Neither requires loading
  the entire archive into a catalog first.
- Support explicit files/globs without requiring a conventional data tree.
- Keep filename discovery separate from scalar metadata enrichment and from
  array reads. Metadata enrichment must use the established accessors/mappers,
  not copy NetCDF header interpretation into a new scanner.
- Apply general `select` to the enriched metadata catalog. The v2 drivefit
  workflow needs more than the identity fields currently exposed by the no-op
  prototype. Define that scalar projection explicitly; do not blindly request
  all accessor metadata, which may perform block/array work.
- Keep grouping, reference selection, and scientific input validation with the
  consuming operation. Share catalog operations where their semantics match.
- Make latest/current selection explicit. V2's `None` selector chooses current
  symlinks/latest, whereas a DB `None` query currently means unconstrained.
  These meanings must not be treated as equivalent.
- Resolve file handles within an explicit lifetime. A reusable metadata catalog
  should not silently keep all files open or contain loaded science arrays.

## Focused next implementation

Agree on the shared typed metadata/catalog and configuration contracts first.
Identify the existing metadata definitions to reuse and the extraction gaps to
fix. Then move discovery settings out of Kids, restore the small local store
abstractions, and have the CLI consume their catalog. Exercise explicit files,
ObsSpecs, duplicate/symlink inputs, metadata selection, and no-op execution with
the reference fixtures.

Use existing DB query tests to validate selector and record agreement without
starting an ORM redesign. Add the deployment-to-Kids argument adapter after the
shared argument contract is settled. Generic `deploy run` forwarding alone does
not implement this adapter.

The runtime CLI helper still exposes both config and env path options. Kids
does not interpret environment values; whether to omit the env option from
this command should be handled in the shared option helper, not with another
Kids-specific parser patch.

## Reference acquisition codes

The read-only `tolteca_ref_data/test_fixtures/toltecdb_slice.sqlite` lookup
tables confirm masters `0=TCS`, `1=ICS`, `2=CLIP` and observation types
`0=Nominal`, `1=Timestream`, `2=VNA`, `3=TARG`, `4=TUNE`. These are external
acquisition codes, not the bit values of `ToltecDataKind`.

Master code 3 remains unresolved: the database labels it `REPEAT`, while the
reference NetCDF `Header.Toltec.Master.long_name` labels it `TCS_SIM`. The slice
contains no acquisition rows with that master, and all available reference
KIDs files use master 0. Do not add a canonical master or alias code 3 to one
without acquisition-system evidence. Tests compare supported constants to the
lookup tables and available file headers to the corresponding database rows.
