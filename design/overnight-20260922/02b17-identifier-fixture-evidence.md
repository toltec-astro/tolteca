# 02b17 — Recorded filenames in identity-only tests

Final fixture review found invented ROACH-8 names/timestamps and a `timestream`
suffix in the earlier identifier-only tests. Those tests exercise useful
precedence and snapshot behavior, but parser acceptance is not acquisition-format
evidence. Replace their positive filename examples with the recorded ROACH-0
target, VNA and timestream basenames from the read-only reference files.

The target example keeps the nonempty-suffix/UTC-segment assertions. The
timestream example has its actual empty suffix. Deliberately conflicting scalar
headers remain explicit corruption tests. The copied `ics` directory and unknown
`simu` master remain intentional relocation/fallback cases, not claims of observed
acquisition locations. The processed VNA name is derived from the real raw
basename using taco's `{stem}_processed.nc` convention, confirmed in
`taco_recipes/scripts/tlaloc/bin/reduce.sh` lines 175/196/208/215. No processed
payload or its numerical correctness is claimed by that name-only test.

The small Dataset objects remain deliberately partial inputs to the low-level
identity helper (`read_metadata=False`), not fake complete raw acquisitions.
Actual catalog/runner tests use complete reference files after 02b16.

No production code, precedence, metadata value policy or numerical expectation
changes. All **35 source tests pass**; scoped Ruff (existing CPY001 excluded),
formatting, ty and Git whitespace checks pass. This is a focused correction to
the inherited test evidence, not a broad rewrite of old test suites.
