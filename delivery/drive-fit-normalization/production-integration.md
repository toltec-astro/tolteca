# Drive-fit algorithm integration and production evidence

Status: 2026-10-26. Baseline: delivery branch commit `762b05b`. Scope: drive-fit
numerics, offline regression tests, and the handoff to the detector-setup work.
No OT script, C++ command, hardware setting or primary checkout is changed here.
The combined operational specification is maintained by `work_on_tolteca_reduce`
in the primary checkout's `design/detector-setup-spec.md`. This note supplies
evidence and integration requirements; it does not replace that specification.

## Algorithm status

The fixed-total correction already exists in this delivery branch and in the
primary checkout's `drive_fit_commit/step.py`. Preserve that implementation.
For a matched reference power fraction `r_i`, normalized on the **full reference
comb**, and the attenuation `A_i` after the existing fill/cutoff policy,

\[
t_i=r_i10^{-A_i/10},\qquad b_i\propto\sqrt{t_i},\qquad
D_{\rm exact}=-10\log_{10}\sum_{i\in B}t_i.
\]

`B` contains the unique emitted DAC bins. Masks and last-row ownership of a
shared bin matter; summing all table rows is not generally equivalent. The
delivered power is `10 log10(b_i² / sum_B b²) - D`. The downward 0.25 dB
quantization leaves an excess in `[0, 0.25)` dB. The old peak-only normalization
omitted this sum. The percentile defaults, matching policy and relative
amplitude prescription are unchanged. This model does not include the
phase-dependent mean power of a peak-normalized waveform.

The analytic phase derivatives and centered solver coordinates entered in
`46a659b`. Keep the explicit solver budget and the default disabled alternate
start. No new optimizer or scientific selection rule is introduced here.

### Additional failure found during production replay

Network 5, reference observation 149325, channel 173 produces an empty nonlinear
phase window. The baseline evaluates `mean(zt[:5])` before testing whether the
four-parameter fit has enough samples. With warnings promoted to errors, this
aborts the whole network. With ordinary warning handling, it reaches a later
array-reduction error and is recorded as a failed fit.

The correction rejects windows with fewer than four samples before evaluating
their statistics. The existing drive-fit driver records the `ValueError` in the
appropriate power/channel slot. Channel 173 retains seven failed fits and no
accepted attenuation. Other channels continue normally. This is not evidence
that this particular defect caused the original January operational failures.

Changed implementation: `src/tolteca_kidsproc/_phase_fit.py`,
`ResonanceFitterSingleTone.fitphasenlin_c2`. The guard is shared by analytic and
legacy modes; it changes only an underdetermined/empty-window failure path.
The valid-window calculations and operation order are unchanged. A separate
baseline replay of all 474 nw5 channels, with ordinary warning handling,
confirmed exact equality for all 34 non-error variables; only seven error
messages changed.

## Production findings for the OT specification

The original records establish three distinct effects. They must not be combined
into one inferred power offset.

| Finding | Evidence | Integration responsibility |
| --- | --- | --- |
| Initial commit amplitudes overwritten before load | All 11 active networks: commit at 150275; 150276 records old LUT amplitudes; its reduction replaces the amplitude file before the next FILE upload. | OT must load the committed amplitudes and apply their attenuation before an installing reduction can intervene. FILE mode alone is insufficient verification. |
| Commit rows paired with a different finding result | 32 of 33 re-init network/events have different amplitude/frequency counts. Across them, 97 additional slots contain unity and 252 entries are truncated. At 150301, nw4 has 513 committed rows versus 519 loaded tones; nw5 has 489 versus 487. | Commit targets the fresh unreduced acquisition's ordered probe comb. Reject a different pending proposal. Do not change the positional Tlaloc contract or silently remap commit onto finding output. |
| Missing summed-power normalization | For the reconstructed intended input comb at 150301, the omitted terms are +1.629060 dB (nw4) and +1.843427 dB (nw5), before the unknown original quarter-dB flooring contribution. | Preserve the existing fixed-total correction and its full-reference normalization. These terms exclude the separate assignment error and crest-factor effects. |

The reciprocal frequency-match diagnostic gives approximately 2.08/2.06 dB
scatter from coefficient assignment for nw4/nw5 at 150301. It is separate from
the common normalization term. Ordinary sweep reduction's amplitude transfer is
not the origin of this commit-to-load mismatch.

Eight of eleven fresh January fits failed. Logs identify the older fit used for
each network. In particular, nw5 used 149325, not 150275. The dispatcher must
retain exact per-network reference identities and visible failure outcomes;
intentional reuse cannot be inferred from the existence of an old readable file.

Agreed operation: complete intended placement changes, acquire an unreduced
target, commit, select FILE, reload the same comb, apply attenuation, then verify
the first sweep of Tune. The existing frequency/count/order/LO installation
guard is exercised here. It does not verify all masks, phases, controller state
or actual hardware application. Those remain explicit operational checks.

## Verification and reproducibility

The new `tests/tolteca_kids/test_drive_fit_production.py` reads only the separate
capture, checks each consumed original against its retained SHA-256/size record,
and writes products to pytest temporary directories. It does not unpickle caches
or contact acquisition hosts.

1. Eight original March selection vectors (3,669 channels) are held fixed and
   evaluated under the modern fill/cutoff policy. An independent summed-power
   equation verifies amplitudes, exact attenuation and quantized power closure.
   This is a controlled normalization test, not replay of the March shell policy.
2. Complete fresh analytic fits use the logged nw4/150275 and nw5/149325 series,
   then commit to 150301 through the public workflows. Products are reopened;
   installation rejects the recorded 150302 comb without altering existing
   amplitude/attenuation files, then accepts the exact target comb.
3. The existing all-channel nw0 reference CLI replay remains part of the handoff.

| Fresh replay | Reference channels | Accepted selections | Commit channels | Global attenuation | Quantization excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| nw4, 150275 → 150301 | 510 | 402 | 513 | 8.00 dB | 0.106912 dB |
| nw5, 149325 → 150301 | 474 | 386 | 489 | 7.50 dB | 0.034039 dB |
| nw0, 145484 → 145485 | 632 | 506 | 632 | 8.25 dB | 0.098090 dB |

These are **new analytic fits**, not recovered January fit tables. The original
January ECSV products remain unavailable. Nw0 channel 596 is accepted at
5.493193500488467 dB in this replay. No hardware run was performed.

From this checkout with its dependencies installed:

```sh
TOLTECA_DRIVE_FIT_CAPTURE=/absolute/path/to/drive-fit-production-capture-20261026 \
python -m pytest -q tests/tolteca_kids/test_drive_fit_production.py

TOLTECA_DEPLOY_ROOT=/absolute/path/to/deployment \
TOLTECA_DRIVE_FIT_DATA=/absolute/path/to/tolteca_cal_data/toltec/drive_fit/ics-17068/toltec0 \
python -m pytest -q tests/tolteca_kids/test_drive_fit.py \
  tests/tolteca_kids/test_drive_fit_commit.py \
  tests/tolteca_kids/test_drive_fit_power.py \
  tests/tolteca_kidsproc/test_phase_fit_solver.py \
  tests/tolteca_kids/test_tone_prepare.py \
  tests/tolteca_kids/test_tlaloc_amplitudes.py

python delivery/drive-fit-normalization/run_reference.py \
  --data-lmt /absolute/path/to/tolteca_ref_data/data_lmt \
  --output /absolute/path/to/new-output-directory
```

The deployment path contains `run/data_lmt`; the 31-sweep baseline remains in
calibration data, outside the compact reference repository. Production tests
skip when their capture environment variable is unset. With all paths supplied,
10 production and 91 existing regression tests passed. Ruff checks passed for
the changed Python files. Runtime: Python 3.13.9, NumPy 2.5.2, SciPy 1.18.1.

Local evidence is retained separately under the workspace's `run/`:

- `drive-fit-production-capture-20261026/`: immutable original records and manifest.
- `drive-fit-production-investigation-20261026/report.html`: reconstruction,
  source provenance, limitations and original-record calculations.
- `drive-fit-integration-20261026/`: test logs, fresh product trees, nw0 configs,
  and `before-after.json` with the independent scientific-array comparison.

For main integration, apply the three-line sample-window guard and its unit test,
then add the production regression module. Do not replace the primary checkout's
`tlaloc_output.py`; it already contains subsequent shared-output work. The
normalization and analytic-solver changes predate this patch. The reconstruction
and this note are inputs to the combined detector-setup specification; OT script
admission and hardware verification remain with that workflow's owner.
