# Drive-fit power audit — 2026-10-20

## Scope and status

The user requested absolute per-tone diagnostics and questioned the commit's
amplitude/global-attenuation calculation. This audit evaluates the existing saved
685-channel reference fit and commit, observation 17068, network 0. It adds an
executable diagnostic probe; it does not change production fitting, matching,
amplitude files, global attenuation, or live instrument settings.

The earlier exact v2 parity checks establish faithful porting. They do not
establish that the original recipe's physical power model is correct.

This audit uses the recorded input probe comb as its output comb. The later C++
sequence clarification confirms that this is appropriate for commit after a
standalone, unreduced target sweep whose tone list remains current. The audit
does not evaluate a different comb created by a separate finding operation.

Sources inspected:

- V2 `tolteca_kids/roach_tone_power.py`: `RoachTonePower.P_dac_dbm_per_tone`
  distributes a configured total DAC power in proportion to squared amplitudes.
- Current Tlaloc `ToltecRoachManager::dac_comb`: rounds comb frequencies, assigns
  complex amplitudes to FFT bins, applies the inverse FFT, then divides both I
  and Q by a **shared maximum complex waveform magnitude**.
- V2 recipe `DriveFitCommit.compute_ampcor` and the v3 port: select/fill/clip
  attenuation, multiply the reference LUT by attenuation factors, normalize to
  the largest tone amplitude, and adjust global attenuation using that maximum.
- Reference native tone headers, read directly to distinguish absent phases
  from mapper-provided defaults. None of the 31 files records tone phases.

## Power contract and diagnosis

For the v2 fixed-total-power approximation, the emitted power of a unique tone is

\[
P_i = P_{\mathrm{DAC,total}} + 20\log_{10} a_i
      -10\log_{10}\sum_j a_j^2 - A - L_i.
\]

Here `A` is drive attenuation and `L_i` is downstream loss. This audit sets
`L_i = 0`, so the reference plane is **after the drive attenuator**, not at a
KID. The explicitly supplied `-12.6 dBm` total is the current v2 model's nominal
calibration, documented there as a 2025 spectrum-analyzer result. It is not a
measurement of this 2022 sequence. A calibrated detector-input estimate needs
the applicable total/transfer calibration and downstream loss.

Tlaloc instead holds the synthesized waveform peak fixed. With
`M(a, phi, f) = max(abs(ifft(spectrum(a, phi, f))))`, its normalized tone
amplitude is proportional to `a_i / M`. Changing all LUT amplitudes by the same
factor leaves the waveform unchanged. Changing relative amplitudes, frequencies,
phases or active bins can change `M`. A max-tone-amplitude correction alone cannot
in general preserve absolute powers after this normalization.

For a unique matched tone with old amplitude `a_i`, new amplitude `b_i`, selected
attenuation `A_i`, and output attenuation `A_out`, the calibration-independent
power error under the waveform model is

\[
\Delta P_i = 20\log_{10}(b_i/a_i) + A_i - A_{\mathrm{out}}
             +20\log_{10}(M_{\mathrm{old}}/M_{\mathrm{new}}).
\]

A scalar global attenuation can satisfy every tone only if their required values
agree. The current amplitude-file **relative shape** satisfies that condition
under the independent-tone assumptions. Its absolute normalization calculation
omits the waveform term. The exported amplitude file contains replacement
coefficients; it is not a multiplier to apply again to the target LUT.

The existing `tone_amps_phases_to_powers` helper does not solve this omission: it
ignores phases and frequencies. The v2 `tone_powers_to_amps` inverse also scales
I and Q separately, unlike Tlaloc's common normalization. Neither should be
promoted into a corrected commit without an independent waveform comparison.

## Reference results

Generated evidence is under `run/drive-fit-power-audit-20261020/result-final` at
the workspace root. `summary.json` records product paths, assumptions, source
hashes and phase-trial samples. `tone_power_audit.ecsv` contains per-channel
best/policy/commit powers, errors, DAC-bin ownership and overwrite flags.

| Check | Result |
| --- | --- |
| Current global attenuation | 13.75 dB |
| Original fixed-total model, treating all rows as separate tones | Commit is 0.10830 dB below the filled/clipped target powers. |
| Fixed-total model after applying actual DAC-bin ownership | Commit is 0.11346 dB below the targets; exact common attenuation would be 13.63654 dB. Rounding upward to 0.25 dB still gives 13.75 dB. |
| Intentional percentile limitation | 49 accepted channel rows are underdriven relative to their fitted best values, by up to 2.18065 dB before the normalization residual. This follows the existing tenth-percentile policy. |
| Failed/flagged selection fill | 202 rows use the existing median-fill policy. |
| Frequency collisions | 685 rows map to 654 distinct DAC bins; 30 duplicate-frequency groups contain 31 overwritten amplitude entries. Tlaloc's last assignment wins. |
| Collision effect in this commit | The matcher assigns equal requested powers within each group, so this sequence does not demonstrate incompatible powers in a shared bin. The unmodified v2 power model nevertheless double-counts duplicated bins in its total. |
| Recorded phase information | No reference or target tone-phase headers; exact waveform-normalization error cannot be reconstructed. |

The reference **does not demonstrate a large global attenuation error**. The
normalization issue is demonstrated independently by a controlled two-tone
case: equal initial amplitudes, requested attenuations 10/20 dB, no percentile
clipping. Current commit produces `[1, sqrt(0.1)]` and 10 dB global attenuation.
This overdrives both targets by 2.59637 dB under fixed-total normalization, or
3.63398 dB under Tlaloc peak normalization with known zero phases at 100/200 MHz.
This case tests the physical invariant, not v2 implementation parity.

The probe also runs 64 seeded phase scenarios for the real amplitude vectors
and DAC bins. Retaining phases gives median controlling-bin errors spanning
-0.333 to +0.320 dB; regenerating phases gives -1.436 to +1.183 dB. These are
synthetic sensitivity ranges, not measured errors or confidence intervals.

## Implementation direction

This section records the initial power-model proposal, not an accepted change
to percentile selection or attenuation rounding. The subsequent operational
clarification distinguishes commit on the current comb from a separate finding
operation that prepares a new comb.

A corrected commit should express selected powers at a named calibrated plane,
then solve its output amplitudes and global attenuation using the **full emitted
comb** and Tlaloc's normalization. Required inputs include actual target
frequencies, phases and masks, plus the reference normalization or equivalent
calibrated powers. Channel selections must not renormalize the selected subset.
Missing phase metadata must remain unknown rather than become zero phase.
Duplicate bins need an explicit shared-tone policy or a validation error when
requested powers conflict.

Commit diagnostics should retain best requested power, policy-adjusted target,
predicted achieved power, residual, fill/cutoff flags, DAC-bin ownership,
normalization, calibration provenance and reference plane. Hardware attenuation
quantization should be evaluated in that same power space. Upward rounding avoids
overdrive in the tested fixed-total case; the policy must be explicit rather
than inherited accidentally from v2's downward rounding.

Current Tlaloc writes tone phases to newer files, so this input requirement can
be met for newer data. This historical reference cannot supply it. A measured
post-commit sweep would independently test the power prediction.

## Verification and review

The final probe completed all 31 native-header consistency checks, the full
685-row calculation, original-v2 forward-model execution, scale-invariance and
unchanged-setting checks, the two-tone counterexample, and explicit
last-row-wins duplicate-bin checks. Both 64-trial phase series completed with the
recorded seed. Artifact checks cover 685 ECSV rows, 30 collision groups and all
13 figure traces. The four-panel HTML was inspected in the in-app browser.
Scoped Ruff and ty passed. Production code and public API documentation were not
changed, so the earlier workflow test counts and Sphinx results are historical,
not checks rerun for this audit.

The reference header reports `ToneAmpsMode=FILE`. Tlaloc reads replacement
amplitudes in that mode; writing the file alone does not switch a receiver from
LUT mode. This is an operational condition to retain in future commit checks,
not an observed failure of this reference.

The repository ignore list now includes the audit and probe as maintained
sources. `just review` regenerates the workspace diff. The focused diff uses the
pre-edit working copy of `.gitignore` and empty baselines for the two new files;
it does not compare unrelated pending changes with Git HEAD. No files were
staged and no Git commit or live instrument write was performed.

## Operational clarification and handoff findings

The user clarified the observing procedure after reviewing the power audit:

- Each VNA starts a new calibration group with a fixed 1000-tone comb and a
  frequency-dependent amplitude LUT. Finding produces the tones for subsequent
  target sweeps; its output count need not equal the probe-channel count.
- A commit after VNA reuses an existing drive-fit product to establish drive
  settings for the new calibration group. It does not require a new drive fit.
- For FILE-mode sweeps, finding transfers the recorded input amplitudes to the
  next tone list through its probe-channel match. Output amplitudes must align
  with the frequencies actually written for the next load.
- The percentile cutoff deliberately limits the dynamic range required by
  extreme outliers. Keep `drive_fit_commit.commit.adrv_ref_perc` and its default
  of 10 unchanged. Choosing another value is deferred. The 49 limited rows in
  the earlier table are policy outcomes, not commit defects.

These requirements are recorded with their source evidence in
[the workflow record](drive-fit-workflows-20261020.md#tone-list-handoff-clarification).
The two independent matches have different purposes: finding links output
tones to current probe channels; commit identifies the prior fitted resonators.
Finding is not a prerequisite of commit on an unreduced target sweep.

The inspected v2 ordinary sweep path implements the first relationship:
`KidsFind` assigns each detection its matched probe's `amp_tone`;
`TlalocOutput.make_roach_tone_props` uses those amplitudes by default; and
`TlalocEtcDataStore.write_tone_props(sort_fft=True)` applies one permutation to
frequencies, amplitudes, phases and masks. VNA/target configurations allow the
tone count to change; the tune configuration requests a fixed count.

The following findings were revised after the user's C++ sequence clarification:

1. **Withdrawn as a defect in the intended commit sequence:** the v2 recipe and
   v3 port write `amp_best` in input-probe order. A standalone target sweep does
   not invoke finding or change the tone-frequency files. For the user's
   unreduced-target workflow, those input probes still identify the next comb,
   so this order is appropriate. The earlier finding incorrectly assumed an
   intervening tone-list replacement. A separate explicit reduction that writes
   new tone files is a different sequence requiring its own correspondence check.
2. V3 `KidsFind.run` builds its channel table with unit amplitudes and all-true
   masks, despite available native channel metadata. It therefore cannot
   preserve the FILE-mode amplitude mapping implemented in v2.

A fresh run of `BlockCreate → SweepReduce → SweepCheck → KidsFind` on reference
target `toltec0_017068_030_0000_2022_12_11_05_47_52_targsweep.nc`, using current
v3 defaults, produced 551 detections from 685 probe channels. All 551 detection
amplitudes were 1; all differed from their matched recorded amplitudes. The
native amplitudes span 0.0319862899 to 1. This demonstrates the metadata loss
in the current port; the detection count is not asserted to reproduce the
historical production finding configuration or final hardware tone count.
Evidence: `run/drive-fit-tone-handoff-20261020/{finding.log,summary.json,detected.ecsv}`
at the workspace root. The ECSV retains both the current amplitude and the
amplitude selected from the recorded probe by the current finding match.

The previously reported 685-to-654 DAC-bin count is a separate property of the
recorded probe comb. It does not explain or invalidate the expected change in
tone count caused by finding. A power correction must evaluate the actual comb
to be loaded, retaining the reference comb's normalization when deriving its
best powers. For the clarified commit sequence this is the current probe comb.
The same-comb numerical counterexample addresses normalization independently
of whether finding runs.

This follow-up verifies workflow evidence and updates findings only. It changes
neither production code nor numerical policy. No production tests or Sphinx
build were rerun; there is no public API/documentation change. The fresh finding
probe and configuration-default check completed, and `just review` refreshed
the workspace diff. The focused diff uses the pre-edit working copies saved in
`run/drive-fit-tone-handoff-20261020/before`, not Git HEAD.

## C++ command-sequence correction

The user clarified that commit consumes an unreduced standalone target sweep.
Read-only inspection of `ToltecRoachManager.cc` confirms:

- `targ_sweep()` acquires the sweep, updates status/observation bookkeeping and
  restores LO/sample rate. It invokes neither reduction nor tone upload.
- `exec_tune()` explicitly performs sweep, reduction, upload, second sweep and
  second reduction. It does not upload after the second reduction.
- `reduce_do()` dispatches explicit reductions; `targ_load()` uploads tones.
  The command loop handles their triggers separately from standalone sweeping.

The prior statement that commit must run after finding on its target sweep is
withdrawn. Commit may use that unreduced sweep's input-probe frequencies and
ordering directly. Its prepared amplitudes and attenuation must take effect on
the next load/sweep; subsequent finding propagates recorded amplitudes through
the existing v2 match. This sequence does not require porting finding before
commit's channel mapping can be correct.

The older all-in-one OT script inspected earlier contains explicit target
reduction commands. That variant does not establish automatic reduction by
`TargSweep` or describe the unreduced-target sequence specified by the user.
Source differences must remain explicit; no historical hardware execution was
reconstructed. C++ source hashes and a focused diff against this follow-up's
pre-edit working files are under `run/drive-fit-cpp-sequence-20261020` at the
workspace root. Production code and numerical policies remain unchanged.
