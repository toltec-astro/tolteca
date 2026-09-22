# 02b8 — Native repeat level and registered instrument component

The previous diagnostic found raw metadata with `repeatvar='0'` and
`instru_component=None`. The four real acquisition headers store int32 zero;
the actual v2 reader reports both `repeatvar=0` and its public `repeat=0`.
`ToltecInfo.interface_instru` already assigns each ROACH interface to
`toltec`/`roach`.

## Focused changes

- Rename the model/schema field to `repeat` and retain its integer value.
  Remove the string conversion and redundant v2 storage-oriented name. JSON
  now has an integer `repeat`, with no compatibility `repeatvar` alias.
- Read the instrument/component pair from the existing interface map when
  constructing metadata. No new registry, literal pair or header read is added.
- Preserve other fields, numerical processing, UIDs, source/catalog records and
  current absence behavior. A missing repeat remains None in this naming/value
  commit; raw-field requiredness is the next separate step, not claimed fixed.

There are no runtime `repeatvar` consumers in the v3 repositories. The diagnostic
uses the new public name; earlier saved reports retain their original evidence.
The generic model is not narrowed to a new instrument type hierarchy here.

## Verification

Five new tests cover all four real files, registered identity, integer values,
typed JSON round-trip, I-array laziness and absence without a fabricated repeat.
The actual v2 interpreter independently reports zero for both its repeat fields
on all four inputs. No payload fixture is modified.

**971 full Tolteca tests pass in 54.62 s**, with actual-v2 parity enabled and
benchmarks disabled. Log: `/tmp/tolteca-20260922-native-metadata-full.log`.
Scoped Ruff (existing CPY001 excluded), formatting and typing pass. The large
core file is not reformatted. Diff review confirms no new numerical operation,
runtime cache, dependency, source-catalog field or external action.
