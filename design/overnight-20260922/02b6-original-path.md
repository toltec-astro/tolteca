# 02b6 — Use `filepath_orig` for the KIDs acquisition path

## Evidence and scope

All four ROACH-0 reference `Header.Toltec.Filename` values contain the full
`/data_lmt/toltec/tcs/toltec0/...nc` acquisition path. V2 calls the field
`filename_orig`, but the user chose `filepath_orig` for v3. Telescope metadata
and SourceInfo already use that name.

Rename the KIDs metadata field and schema mapping to `filepath_orig`, and
update the identifier to use the same mapping. Values, decoding, source/header
precedence and the current file location are unchanged. No compatibility alias,
new path parsing, header reads, cache or filename requirement is introduced.
The JSON field name changes accordingly; this is an unreleased v3 API change.

The existing missing-header behavior is deliberately preserved in this naming
commit, not endorsed as the final raw-acquisition contract. Header requiredness
must be reviewed separately against raw versus reduced data. Do not infer that
the raw-file metadata work is complete from this rename.

## Tests and self-review

- Five new tests cover the four actual paths, mapper names, independent regex
  identifier agreement, retained location of a deliberately renamed source,
  lazy I arrays, typed JSON round-trip, unrecognized text and missing header.
- 61 focused metadata/source tests pass; **961 full Tolteca tests pass in
  53.49 s**, with the actual v2 interpreter enabled and benchmarks disabled.
  Log: `/tmp/tolteca-20260922-filepath-full.log`.
- Scoped Ruff (excluding existing CPY001), formatting and typing pass. Diff
  whitespace checks pass; the large core module is not globally reformatted.
- No remaining runtime reference to `filename_orig` exists in Tolteca, DB or
  Web. Historical design records retain their original observations. This
  commit has no scientific, CLI, storage-policy or external-service change.

Next: review the shared raw/reduced metadata model before scalar catalog
projection. The four reference raw files have all established mapped fields;
the old raw reader reads these directly. Some current synthetic/reduced tests
instead rely on a partial metadata model. Those cases must not silently define
the real acquisition contract.
