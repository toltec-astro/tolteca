# 02b11 — Raw design dimension and ADC-reference presence

V2's raw metadata requires `loclen` and its common KIDs metadata requires
`Header.Toltec.AdcSnapData`. All four actual inputs contain them. The v3 port had
made both optional even for complete raw metadata. This step restores presence
requirements without restoring v2's eager ADC payload read.

## Focused behavior

- Raw metadata reads `n_kids_design` through the existing dimension mapping;
  an absent dimension raises the mapper's existing KeyError.
- Raw metadata obtains the native ADC DataArray handle and retains its name as
  `adc_snap`. An absent variable raises KeyError. `.name` does not load values.
  `get_name` alone would not validate presence: an unresolved NameMapping still
  retains its candidate name. No duplicate explicit existence check is added.
- Processed metadata may still omit both fields. Successful values, serialization,
  science processing and source/catalog identity are unchanged.

The lazy-reference representation is an intentional v3 behavior already in use,
not a claim of matching v2's full ADC-array metadata representation. Direct
native array access remains available through the schema/mapper. Dimension and
variable presence checks do not require ADC or detector-position data.

## Evidence and QA

Eight negative cases drop the dimension or variable from an in-memory view of
each actual input. Before the change, all eight incorrectly accept the incomplete
raw record. Two preserved cases check lazy values and processed omissions.

The old toy numerical datasets now explicitly include the recorded Xloc and ADC
values from a small session fixture. That fixture loads these small test values
once and closes its file; it neither edits the acquisition nor fabricates design
metadata. Runtime laziness is independently checked on real unopened payloads,
not inferred from the loaded toy fixture. Dataset shapes/identities in those old
numerical tests are not used as acquisition-format evidence.

**1,026 full Tolteca tests pass in 55.55 s**, actual-v2 parity enabled, benchmarks
disabled. Log: `/tmp/tolteca-20260922-design-metadata-full.log`. Scoped Ruff,
formatting and ty pass, as does `git diff --check`. Existing unrelated core and
accessor-test style debt is not swept into this patch. `just review` is regenerated
before the commit. No new product-integration run is claimed for this step;
the 02b9 run remains the latest full CLI/DB/Web integration.

Self-review confirms the only new raw failure modes are missing design dimension
or ADC variable; valid-file metadata still reads no I/Q, Xloc/Yloc or ADC arrays.
Remaining raw sweep repetition/capacity/sample-dimension handling and typed
header-only catalog composition are distinct follow-up work.
