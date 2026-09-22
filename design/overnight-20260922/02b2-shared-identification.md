# 02b2 — Shared telescope and KIDs source identification

## Scope and behavior

Add `tolteca_datamodels.filestore.identify_source` for an already-open Dataset.
The existing telescope schema recognizes the telescope timestamp variable;
strict `LmtTelMetadata` supplies its identifiers and recorded original path.
Other datasets delegate to the existing KIDs identifier unchanged. No CLI or
catalog call site switches in this commit, and no opening-policy default changes.

The returned source record preserves its current FileLoc. Header observation
identifiers take precedence; an unrecognized current name can use the original
header path for filename evidence. A missing telescope master stays None.
Source records use the master label; the strict metadata's numeric master is not
modified. Telescope stream kind comes from the recorded prefix, with a known
current telescope kind as fallback. A timestamp alone does not invent Tel/Tel2.
Incomplete recognized telescope files propagate metadata errors. The existing
optional original-path consistency check remains off by default.

Move the unchanged `_get_filepath_fields` helper from the KIDs module to the
root file module so both identifiers share filename precedence. It does not
add a header registry, partial metadata class, new raw-name list, or extra
metadata fields to SourceInfoModel. Full scalar-metadata catalog projection is
still the separate issue described in 02b; this is identity enrichment only.

## Why this boundary

The user explicitly wants strict file-backed telescope metadata, not another
nullable partial model. Using its existing mapper preserves that definition.
KIDs full metadata is not substituted for its scalar identifier: tune metadata
would read LO for block boundaries. Root filestore assembles these existing
readers; root file remains responsible for filename records and classification.
The explicit two-reader choice is small enough not to warrant a plugin system.

## Reads and cost

Actual reference-file instrumentation permits scalar telescope headers, two
TelTime endpoint reads, and the two RA/Dec target pairs (four values), rejecting
pointing-stream reads. The two endpoint indices observed at the backend are 0
and n-1; xarray normalizes a user-level -1 index before the backend call.
All four actual KIDs payload projections are unchanged, including renamed-source
fallback, and existing scalar-only KIDs tests continue to pass.

A warm already-open tune Dataset microcheck (7 repeats of 100 calls, logging
disabled) measured median 295.45 microseconds for the existing KIDs identifier
and 346.75 microseconds for the shared entry point. The extra telescope mapping
lookup is about 51 microseconds here and reads no extra KIDs values. This is not
a file-opening, network, pipeline, peak-memory, or cold-cache measurement. No
cross-source cache or value cache was introduced to hide this lookup cost.

## Verification and self-review

- 240 focused tests pass with actual-v2 tests enabled: shared filestore, LMT,
  filename/catalog, and KIDs source tests.
- Ruff check (excluding project-wide CPY001), format check and scoped `ty` pass
  on all four changed Python files, with the development environment explicit.
- Fresh-process imports work when root file, root filestore, LMT metadata, or
  KIDs source is imported first; no circular import was observed.
- Tests cover actual telescope and KIDs records, renamed local-source fallback,
  header precedence, nullable master, corrupted/incomplete headers, and unknown
  recorded paths. No fixtures were changed.
- Self-review caught an accidental replacement of the five existing shared
  file-settings tests by new identification tests; all five were restored before
  commit. The final diff adds coverage rather than removing those checks.

Next: separately wire this into catalog enrichment, test explicit filename-only /
fallback / always opening policies, and recheck no-op resource closure. Do not
present this commit as a general scalar-metadata catalog or completed guide.
