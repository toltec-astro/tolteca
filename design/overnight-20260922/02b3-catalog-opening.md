# 02b3 — Catalog opening policies and shared header enrichment

## Changes and preserved contracts

`build_input_catalog` now uses the shared identifier from 02b2 and accepts the
Python keyword `header_mode`: `never`, `fallback`, or `always` (a StrEnum using
the existing lowercase/auto convention). The default remains `always` for KIDs.
Known telescope `.nc` candidates now receive strict header identification too;
previously the catalog kept their filename-only records. Unknown netCDF names
remain candidates for either reader. Unsupported known kinds retain guesses.

`never` returns filename discovery without opening any source. `fallback` opens
only candidates whose current filename did not parse; it does not opportunistically
fill arbitrary missing fields in recognized names. `always` reads supported
candidates even when the filename parses. Invalid mode strings fail before
discovery. Files close on success and exceptions; returned records own no handles.

The runner now uses the same shared identifier. Telescope no-op input context
therefore includes header-backed identifiers, notably `master='tcs'` for the
recorded fixture, instead of filename-only `master=None`. This changes bounded
metadata reads: scalar headers, two TelTime endpoints and target-coordinate pairs
are now read. Pointing streams and I/Q remain unread. The existing no-op read
guard was updated explicitly for this contract, not removed.

No CLI flag or FileAccessConfig field was added. The CLI continues its existing
all-header preparation, now also covering telescope files; `--show_inputs`,
`--select`, `--show_config`, `--dry_run`, dotted overrides and environment
boundaries are unchanged. No telescope science pipeline is enabled. Explicit
`/tel` discovery is still separate from unqualified/ROACH KIDs discovery.

## Why this size of change

The user requested creation policies for filename-only / original-name fallback /
all-open catalogs. A Python keyword is enough to expose that choice without
introducing a second CLI/config policy and its interaction with execution's
fresh read. Shared identification prevents catalog and no-op from disagreeing
about a renamed telescope source. This does not add broad metadata columns;
the separate typed scalar projection gap in 02b remains open.

Source identity is still read again when execution opens its own Dataset.
There is no shared open-file lifetime or source cache to invalidate. This costs
another bounded metadata read, and ensures per-run config uses the opened input.
Do not describe `header_mode='never'` as suppressing later execution reads.

## Evidence

- 451 focused tests pass with the actual-v2 interpreter enabled: shared file
  access, LMT readers, filename/catalog/discovery, and all KIDs tests including
  exact numerical comparisons. No tolerance changes.
- The 12-case real-file matrix covers both KIDs/telescope, known/renamed source,
  and all three modes. Backend guards verify no I/Q/pointing reads and exactly
  two telescope endpoints per identification. Open and close counts match.
- All eight actual raw KIDs/telescope payloads produce header-backed source
  identities in the default catalog. Telescope master selection works after
  enrichment. A deliberately failing identifier closes its open file.
- CLI tests select a renamed telescope source by canonical interface, master
  and observation number, with both `--show_inputs` and `--dry_run`. One/two
  opened stores respectively close; endpoint reads total two/four respectively.
- Scoped Ruff (excluding CPY001), format and ty pass on all five Python files.

## Self-review and limits

The default telescope enrichment is an observable addition, not merely a
refactor: strict incomplete metadata can now fail catalog preparation. Filename
only/fallback modes let Python callers control this cost. KIDs projections and
header-only reads remain covered by the previous tests.

ObsSpec filtering still precedes header enrichment. A master-qualified telescope
selector cannot match an unknown filename master; use `/tel`, then select the
header-enriched `master` column. No TCS value is invented from directory layout.
The source-info catalog does not yet expose `obs_goal` or `atten_drive` as typed
header metadata; the guide must not advertise them as selectable columns.

Next: consolidate the focused telescope → catalog → no-op guide, then return to
the remaining reducer/context/output and DB/Web review. Re-run full integration
after those checkpoints, with a fresh output directory.
