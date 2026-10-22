# Phase-fit stabilization, 2026-10-22

The default drive fitter now uses analytic derivatives for both the linear
reference and nonlinear phase fits, in centered, dimensionless coordinates.
The physical phase model, circle fitting, selection thresholds and fixed-total
power normalization are unchanged. The original implementation remains available
as `phase_fit_method: legacy`. This work follows the normalization investigation
at `f8bd7c27674ba002d29bdd4ede4b496c5acf3e61`; the earlier report is historical
evidence for that implementation, not a prediction of the new fitter's output.

## Runtime and work limits

The unbounded solver is still MINPACK Levenberg-Marquardt. It now receives the
analytic weighted residual Jacobian directly. Frequency is centered on the
fitted sample window and scaled by a reference linewidth; complex signal is
scaled by the fitted circle diameter. Model/Jacobian calls reuse intermediate
calculations. Covariance is transformed back to physical parameter units,
including alpha and Hz, with the original relative-sigma convention.

Each solve has an explicit 200-function-evaluation budget by default. This is
an optimizer work limit, not a wall-clock deadline; MINPACK can finish an
iteration just beyond the evaluation limit. There is no search loop over starts
in the default path. The optional retry described below permits only one extra
solve with the same evaluation budget. Failures propagate to `fit_flag` and
`fit_error` instead of returning the legacy kernel's sentinel parameter vector.

Five warmed, interleaved runs of **632 channels × seven powers**, measured with
the checked-in benchmark on an ARM64 Mac, Python 3.13.2, NumPy 2.5.4 and SciPy
1.18.1:

| Mode | Median full-network fit time | Median channel p99 | Median slowest-channel time |
|---|---:|---:|---:|
| Legacy | 2.822 s | 12.44 ms | 39.10 ms |
| Analytic, default | **2.592 s** | **8.92 ms** | **23.56 ms** |
| Analytic, optional retry | 2.720 s | 12.99 ms | 33.04 ms |

The default is about **8% faster** in this local comparison. Commit computation
takes about 0.039 s for each mode. Timing includes fit-array assembly and the
linear reference; it excludes imports, file reading/reduction, plots, product
publication and telescope control. Telescope computers are substantially slower:
these measurements establish a local relative comparison, not telescope latency
or a guarantee for all readout networks. Retest on the actual processing host.

## Full-network repeatability

Inputs are the unchanged network-0 reference observations 145484/1–7 and target
145485, from `tolteca_ref_data` revision
`9cdbc76c4327b11e9ec45ce7895ef13b537d03a5`. Every mode was replayed at I/Q
multipliers 1 and 1 ± 1e-15, 1e-14, 1e-13, 1e-12: 27 complete network passes.

| Result across nine multipliers | Legacy | Analytic default | Optional retry |
|---|---:|---:|---:|
| Accepted channels | 505–506 | Same 506 | Same 510 |
| Worst requested tone-power range | 2.810 dB | 4.05e-10 dB | 4.05e-10 dB |
| Channel 596 attenuation range | 2.510 dB | 6.91e-13 dB | 1.08e-12 dB |
| Channel 596 unperturbed selection | 5.51823 dB | 5.49319 dB | 6.34080 dB |
| Quantized global attenuation, every run | 8.25 dB | 8.25 dB | 8.25 dB |

The default preserves the unperturbed accepted population. For those 506
channels, the median absolute selected-drive change is 0.000218 dB and the
maximum is 0.0308703 dB. The exact global attenuation changes from 8.34822501
to 8.34809026 dB. All matching indices remain identical, all committed amplitudes
are finite and positive, and modeled power residuals remain uniform and inside
the 0.25 dB quantization interval.

An independent legacy replay reproduced every existing numeric fit and commit
array from the original saved local replay exactly. The new successful-fit
diagnostics are additional fields, not changes to the selection policy.

## Optional alternate start needs scientific review

`phase_fit_retry: true` retries only failed nonlinear solves or fitted
nonlinearities outside the selector's existing interval [-0.2, 1.5]. The one
alternate start has zero nonlinearity and half the linear-reference Q. It can
replace the primary only when it converges and lowers the weighted residual
sum; a relative 1e-8 tie retains the primary. This is a limited search, not a
global-minimum guarantee. Both candidates may still be poor physical fits.

This option recovers crossings for channels **174, 458, 567 and 574** in this
sequence. For channel 596 it finds a lower-cost positive-nonlinearity solution
at the 6 dB sweep, moving the crossing from 5.49319 to 6.34080 dB. It also
recovers the lower-cost alternative at channel 586/6 dB that analytic derivatives
alone missed. These are scientific output changes, so **retry remains off by
default** despite modest aggregate runtime overhead. Its p99 channel latency is
also slightly higher than legacy in this replay.

Channel 573's 4 dB fit still reaches a higher-cost basin with either analytic
setting than in the legacy run. That channel remains flagged and filled. The
limited retry is therefore an improvement on specific difficult fits, not a
claim to find every lower-cost solution.

Raw failure counters require care: legacy reports 14 execution failures while
another 30 nonlinear optimizer failures become sentinel parameters. The new
default explicitly reports all 46 failures (14 pre-optimizer plus 32 budget
failures); the optional retry reports 24 (14 plus 10). The smaller default budget
therefore leaves two more nonlinear failures than legacy in this unperturbed
sequence, with no change to channel acceptance. It can be increased explicitly.

## Configuration and diagnostics

```yaml
drive_fit:
  fit:
    phase_fit_method: analytic  # default; legacy reproduces the prior fitter
    phase_fit_maxfev: 200      # per solve; ignored in legacy mode
    phase_fit_retry: false     # default; opt-in alternate start
```

Saved results now contain `fit_nfev`, `fit_status`, `fit_cost`, and `fit_retried`
for successful analytic fits. Evaluation counts include both solves when a
retry occurs; cost/status describe the selected solution. These fields are
unavailable for failed or legacy fits (`-1`, `NaN`, or `False`); consult
`fit_flag`/`fit_error`. Analytic failure messages include the evaluation count.
The selected solver configuration is preserved in the existing result/context
metadata. No instrument settings are written by the benchmark.

## Reproduce and validate

Install this branch in the delivery environment described in [README.md](README.md).
From the checkout, with the usual TolTECA venv:

```bash
$HOME/tolteca/bin/python delivery/drive-fit-normalization/probes/phase_fit_stability.py \
  --data-dir /path/to/tolteca_ref_data/data_lmt/toltec/tcs/toltec0 \
  --output /path/to/phase-fit-check --repeats 5 --perturbations
```

The script validates and caches inputs, warms each method, rotates benchmark
order, and saves JSON provenance/timings and NPZ scalar arrays. Omit
`--perturbations` for a shorter timing comparison. Input checksums, package
versions and source hashes are recorded. [phase-fit-stability-results.json](phase-fit-stability-results.json)
contains the compact recorded results and checks for this delivery.

Focused validation: **48 tests passed, 16 skipped** because the 31-sweep
ICS-17068 baseline and a deployed fixture are unavailable locally. Existing
golden parity tests explicitly select legacy; the new tests cover independent
model/Jacobian checks, covariance units, signal-unit invariance, bounded solves,
failure propagation, channel-596 repeatability, and the competing channel-586
solutions. Changed Python files pass Ruff 0.16.5; the changed numerical kernel
and tests pass the type checker. Full-network replay supplies additional
integration coverage using the available raw reference sequence.

The remaining acceptance gate is another calibration group and other networks,
plus timing and scientific inspection on the telescope host. Numerical
repeatability under tiny scaling does not establish robustness to measurement
noise or physical correctness of every minimum.
