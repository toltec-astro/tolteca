# 06b — Typed product manifest and effective-config replay

Status: verified; focused commit follows.

## Evidence / decision

The product writer emitted an untyped dict, and downstream probe code manually
guessed its fields. Add one `SweepProductManifest` Pydantic model, reusing the
existing `SourceInfoModel` and `KidsConfig` rather than duplicating their fields.
It records an aware UTC creation time, format, fixed relative array path, source,
effective configuration and software versions. `read_sweep_manifest()` validates
metadata without touching numerical arrays; `open_sweep_product()` additionally
checks completion and agreement with the Zarr step configuration.

The effective config comes from the executed reduction context and the configured
output step. These are all the currently supported KidsConfig sections. No runtime
reference, raw config tiers, open resource or deferred condition is serialized.
Standalone numerical exports can honestly omit input/config; a fabricated source
or invented handler state would be worse than an explicit None.

Keep a structured source record on the live execution node, distinct from the
flat scalar projection used to evaluate conditions. Output naming obtains its
flat values from that same model. FileLoc's parent-path/context semantics and
original/effective path information survive serialization.

## Observable changes

- Development manifest format is now `tolteca.reduced-sweep.v2`. No compatibility
  reader is added for the earlier untyped v1 prototype.
- `input` is a structured shared source model, not a flat config-context dict;
  data-kind JSON uses its integer enum representation. The typed API restores
  enums and computed aliases. `config` replaces the untyped `output_config` field.
- `write_sweep_product` takes a SourceInfoModel and DataProdOutputConfig (or None).
  In-memory tests omit those arguments instead of passing an empty dict.
- Existing array format, numerical contents, exclusive destinations and
  manifest-written-last publication remain unchanged.
- Replay uses the recorded effective settings, not changed source recipes. A new
  destination must be chosen explicitly. Raw input still must be available;
  reopening/plotting the completed product does not need it.

## Upstream guidance

Used Pydantic's standard nested model serialization and validation. Current
official API documentation recommends `round_trip=True` for round-trip dumps;
this omits computed fields rather than maintaining ad-hoc exclusion lists:
https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/
https://pydantic.dev/docs/validation/latest/concepts/serialization/

## Verification / self-review

- 56 pipeline/no-op/reference-parity tests passed with the actual v2 interpreter.
  This includes all four reference cases, source/config round-trip, an actual
  CLI replay from recorded settings, rejected unsupported/path-escaping manifests,
  and rejection when recorded config disagrees with the saved step context.
- 2 Web plotting regression tests passed separately. Collecting tests from both
  repositories in one pytest process failed because both expose a `tests` package;
  separate invocations avoid that unrelated import collision.
- Scoped pre-commit and type checks passed. Two old tests constructing the flat
  input projection were updated to round-trip records; the no-op assertion now
  expects the standard integer enum JSON representation.
- Serialization explicitly calls the existing `ensure_dataset` utility: a
  DataTree node needs a Dataset, while the general view permits DataArray inputs.
  This is zero-cost for an existing Dataset and does not narrow accessor APIs.
- Self-review found no changes to reducer computation, context completion or
  source lifetimes. The completion marker validates metadata, not content hashes;
  this is not a distributed transaction or a hermetic/environment replay claim.

Next: use the validated manifest to register completed products and provenance
through the existing DB models, with a Parquet projection for Web discovery.
