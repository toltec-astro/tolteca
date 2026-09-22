# 02b4 — Header-backed catalog fields: measured gap and model boundary

This checkpoint adds a diagnostic, not a new catalog API or a claim that the
selection gap is fixed. The shared file registries and source identity remain
unchanged. The guide accurately states the currently available catalog fields.

## Actual use and fresh evidence

V2 `ToltecFileAccessor._update_from_item_meta` projects `np.isscalar` values from
each reader's metadata into the frame. Its `.open()` loads those readers first.
Taco `scripts/drivefit/dispatch_drivefit.py` uses that open catalog for `--select`
and later selects on `atten_drive`. Thus this is a real downstream feature, not
a hypothetical reason to make SourceInfo accept arbitrary extra fields.

`design/probes/catalog_metadata.py` was run separately in the installed v2 and
v3 environments on all four read-only ROACH-0 reference files. Fresh outputs:
`/tmp/tolteca-20260922-catalog-metadata-{v2,v3}.json`.

| Acquisition | v2 metadata logical bytes | v3 full metadata logical bytes | v3 catalog logical bytes |
| --- | ---: | ---: | ---: |
| VNA 145605 | 36,620 | 604 | 536 |
| Target 145606 | 24,060 | 604 | 536 |
| Tune 145609/1 | 31,140 | 14,764 | 536 |
| Timestream 145609/2 | 16,972 | 588 | 536 |

All four readers report `atten_drive=7.25` in both versions. All four current v3
catalog queries `atten_drive >= 0` raise UndefinedVariableError. Catalog identity
reads seven scalar/string headers. Full v3 metadata reads 17–21 variables; tune
adds its 3,540-element LO vector (14,160 bytes) to identify acquired blocks.
V2 also reads the 16,384-byte ADC snapshot and, for every raw sweep, the LO vector.
No detector I/Q arrays are read by either metadata path. Counts are logical
backend payloads, not physical disk/HTTP bytes or timing claims.

## Why not simply merge another dictionary?

- `SourceInfoModel` owns discovery/location evidence and effective identifiers.
  Its typed serializer, Arrow catalog, per-input configuration and product
  manifest all use that model. Extra DataFrame columns alone would disappear
  when reconstructing source records and would not reach `enable_if` reliably.
- Existing KIDs `.meta` includes actual block counts, so calling it merely to
  filter an attenuation reads LO data. Metadata is still much cheaper than I/Q,
  but that computation is unnecessary for header-only selection.
- KIDs metadata still predates the telescope decisions: raw master is converted
  to a name; `mastervar`/`repeatvar` are strings; `filename_orig` naming and several
  optional/default fields are not aligned. Extending a catalog from that model
  now would expose those unresolved semantics as new user-facing columns.
- The strict telescope model must remain strict. Filename-only or unidentified
  source rows can lack a metadata object; this is different from weakening
  required telescope fields to defaults/None on a supposedly valid model.

## Recommended next bounded design

1. Finish the KIDs **header metadata contract** against the reference files and
   v2 field inventory, before projecting it into more consumers. Preserve raw
   integer codes, derive readable labels, use `filepath_orig`, and justify each
   required/nullable field explicitly. Do not silently alter scientific arrays,
   count meanings or source-level master labels.
2. Give the reader a header/dimension-only construction path reused by the full
   metadata builder. Keep actual sweep-block derivation in the full path/view.
   Do not create a second string-key header list in the catalog layer, scan
   expressions to guess reads, or add a second cache. Common validation stays
   with the typed observation model and its schema/mapper.
3. Choose a source-record composition/projection using those types, then test
   catalog round-trip, units/Time serialization and the same conditional-config
   context. Prefer optional **typed header metadata** over arbitrary extras or
   one enormous nullable cross-instrument metadata class. Reuse Arrow struct
   projection for vectorized queries; do not query Python objects row by row.
   Whether `atten_drive` is an explicit shorthand for a nested header field
   should be one documented rule, not a special-case column assignment.

This separates a necessary field-contract decision from the otherwise working
reducer-only slice. The primitive telescope/schema/mapper/view architecture does
not need another redesign. Proposed type/member names are not fixed here; the
important boundary is file evidence versus validated acquisition metadata versus
array-derived facts. A header-only method must return a valid declared model,
not use `model_construct` to fake a partial full observation.

## Alternatives considered

- **Add `atten_drive`/`obs_goal` directly to SourceInfo:** smallest immediate
  patch, but duplicates domain fields and grows a mixed nullable schema. It
  doesn't meet the user's shared authoritative-metadata requirement.
- **Always attach existing full `.meta`:** reuses types but performs avoidable
  tune reads and exposes the unresolved KIDs contract. Appropriate for explicitly
  requested complete metadata, not an invisible discovery policy change.
- **Keep arbitrary scalar extras in DataFrame only (v2 mechanics):** loses typed
  round-trip and per-source config/manifest consistency. Retain the v2 selection
  capability, not this implementation detail.

## Self-review and verification

Both fresh diagnostic invocations finish successfully, recording the expected
missing-field query rather than swallowing it in production. Scoped Ruff
(excluding existing CPY001), formatting and ty pass. No reader, model, CLI default
or fixture changes are made by this record. A separate fresh end-to-end run after
Web 04b2 passes all four exact-v2 scientific cases and CLI/DB/Parquet/Dash checks:
`run/overnight-slice-20260922-web-reads/` under the development location.
