# 00 — Registry and source-classification checkpoint

## Scope and rationale

Finish the user's approved direct-registration design before reviewing the
previous vertical slice. `FilepathInfo` represents filename evidence, while
`SourceInfoModel.data_kind` represents the current classification, which headers
can refine. A filename declaration should not bundle source classification and
instrument identity policies into a `FileSpec` object.

Root registries now collect regex→record types, capture converters, prefix
aliases and classification data. LMT/TolTEC modules populate them on import.
Root combines `LmtInfo.interface_instru` and `ToltecInfo.interface_instru`.
`guess_data_kind` is a direct normalized-key lookup callable without a filename.
The policy for which key positions are retained is declarative by instrument/
component, not tied to regex selection. Ellipsis stays an ordinary ignored-key
marker on both lookup and stored keys, not a wildcard-matching engine.

## Observable changes

- Remove `FileSpec` and rename the classification key to `DataKindKey`.
- Remove the nested filename kind fields and their catalog query aliases.
  Keep the stored top-level source kind, including explicit Unknown, enum
  serialization, and existing header precedence.
- Import-time domain registration replaces cached lazy bundle assembly. The
  TypeAdapter for stored subtypes remains cached after domain registration.

No change to accepted grammar, data acquisition values, directory master
deduction, filename time, source/original precedence, remote handling, or
scientific readers. Known ROACH numeric spelling behavior remains preserved.

## Evidence and self-review

See [C11](../reader-v2-v3-comparison.md#c11-direct-registration-and-source-owned-data-kind)
for exact commands/results and the four fixture-test corrections. Full Tolteca
pytest with v2 enabled: 894 passed. DB: 192 passed / 54 existing deselections.
Focused modern Web: 10 passed. All 25 touched/new Tolteca Python files pass
scoped Ruff, format and type checks. All 132 before/after source projections
match except the deliberately removed nested kind. No tolerance changes or
fixture writes. Existing global QA debt is recorded, not silently suppressed.

Local paired benchmark (median, µs/source): readout 53.76→55.38; telescope
57.49→54.70. This is not a speedup claim. The key guarantees are no I/O for
discovery and no repeated deduction during access, dump, or restoration.

The prior pending work is preserved as a separate baseline by staging its saved
review patch into the Git index, without changing current working files:

| Repository | Starting HEAD | Preserved baseline |
| --- | --- | --- |
| Tolteca | `94b87a0` | `269d1db` |
| DB | `292ea6c` | `317f951` |
| Web | `d58e02f` | `3bc81fe` |

Those baseline commits contain earlier daytime work, not newly authored
overnight changes. No reset, push or merge occurred. Baseline commits bypassed
commit hooks; tests and scoped QA were run explicitly as recorded above.

## Next decision

The focused follow-up is committed as Tolteca `62602a2`. Review it after the
separate `269d1db` baseline; the combined pre-commit `just review` export is
`/tmp/toltec-python-file-registry-review.diff`.

Review the previous no-op and reducer slice from the new baseline. Do not treat
passing tests as evidence that all models, field meanings and public docs agree
with the latest user decisions. Record real gaps first; make focused corrections
and re-run end-to-end comparisons before claiming the vertical slice complete.
