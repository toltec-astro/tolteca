# 02b — Header opening and typed catalog projection review

This is a findings record, not an implemented opening-policy design.

## Actual v2 usage

Re-read v2 `tolteca_datamodels/toltec/file.py` and taco
`scripts/drivefit/dispatch_drivefit.py` (the CLI block near line 1240). The
dispatcher builds the filename catalog, then uses `tbl.toltec_file.open()` before
an explicit `--select`. Its later drive-fit code selects on `atten_drive`.
V2's `.open()` owns an ExitStack of IO objects and merges scalar metadata into
the table; `.read()` is a separate operation. That scalar merge is broader than
the current v3 `identify_source`, which only projects identity and kind.

## Current gaps and constraints

1. `build_input_catalog` unconditionally opens known ROACH and unknown `.nc`
   inputs through the KIDs identifier, but does not open known telescope inputs.
   This is neither general format identification nor the agreed explicit
   filename-only / fallback / all-header policy.
2. KIDs `get_metadata()` is not a cheap drop-in for discovery. Tune metadata
   calculates block boundaries by reading the full LO vector. Catalog identity
   does not need it. I/Q remains lazy, but calling this "scalar-header-only"
   would be inaccurate.
3. Telescope `get_metadata()` is strict and typed, and reads six endpoint/target
   coordinate values besides scalar headers. It must not be turned into a
   nullable patch model to serve filename discovery.
4. The KIDs metadata model still has older optional/default fields and string
   master semantics. Its raw integer/header modernization is a separate task;
   copying these fields into another ad-hoc SourceInfo dictionary would cement
   that mismatch instead of resolving it.
5. The shared source record and catalog serializer are closed typed models.
   Simply stuffing `atten_drive`/`obs_goal` into a DataFrame would lose their
   typed origin, round-trip behavior, and per-input configuration projection.

## Next design check

Keep file identity/effective filename evidence distinct from strict observation
metadata. Determine how a catalog row can optionally carry a typed, header-backed
projection without requiring block/numerical metadata or inventing defaults.
Use existing acquisition metadata mixins and actual reader schema fields, not a
duplicate string-key list. Check the root file registry import/lifetime boundary
before registering header identifiers; it currently registers filename data,
not open readers. Validate an unknown-name telescope file as a fallback case,
and instrument variable reads and resource closure for each opening mode.

This review does not change CLI defaults, source model shape, or catalog field
availability yet. Discovery fix 02a is complete independently. Coordinate work
03a/04a addresses a separate verified computation issue while this boundary is
being designed. The focused end-to-end guide must accurately describe whichever
opening/projection features are actually implemented, not the desired future API.
