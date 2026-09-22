# 02b14 — Preserve recorded acquisition timestamps on serialization

The source-metadata prototype caught a real timestamp loss before automatic
enrichment was connected. Generic `TimeField` emits the Time object's default
millisecond-precision ISO representation. The actual telescope endpoint values
contain finer precision; parsing that string does not reproduce the original
Time. Earlier telescope serialization tests used whole-second synthetic times.

## Focused correction

Use Tollan's existing `UnixTimeField` for `LmtRawObsMetadata.t0/t1`. The readers
already construct these times from numeric Unix acquisition timestamps. This
changes serialized observation timestamps from ISO strings to numeric Unix
seconds, and model inputs from ISO strings/Time to Unix numbers/Time. Callers
with an ISO string can explicitly construct `Time(text)`. No legacy-production
compatibility shim or global Tollan serializer change is introduced.

In-memory times, units, elapsed-time calculations, filename writer UT, sample
timestamps and scientific arrays are unchanged. No additional file reads are
needed. This is an explicit serialization-contract change, not an internal-only
refactor. It is appropriate to the native acquisition precision; it is not a
claim that single-double Unix serialization preserves arbitrary higher-precision
astronomical times. Such a generic codec would need the JD double pair.

Astropy documents both [numeric Unix representation](https://docs.astropy.org/en/stable/api/astropy.time.Time.html)
and its [two-double internal representation](https://docs.astropy.org/en/latest/time/).
The scoped acquisition choice avoids a broader redesign of generic Time fields.

## Evidence and self-review

The new test reads all four real telescope files, serializes/restores their
complete metadata, and requires exact Time equality for both endpoints and exact
`t_elapsed`. No tolerance is added or relaxed. The initial test also requested a
`tel_toltec2` file that is not present in this fixture tree; it was corrected to
test the actual files. Tel2 content coverage remains absent, not synthesized.

All **81 focused telescope/KIDs metadata tests pass** in 1.14 s. Log:
`/tmp/tolteca-20260922-time-focused.log`. Scoped lint/format pass. A full run with
the separate catalog prototype gives 1,055 passes and one expected catalog dtype
assertion failure (`metadata` is now Arrow null, not object). That prototype is
not included in this commit and is still being reviewed separately.

The actual-time equality test was the gate; comparing only reserialized JSON
would have missed the precision loss. Scalar metadata restoration must also
check values against the original reader, not merely serialization stability.
