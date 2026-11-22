# Compare acquired Tune models

Use this example to fit recorded target sweeps and both acquired sweep blocks
of a Tune. Keep the original NetCDF files read-only. Start with one network and
one acquisition, then expand the comparison after checking the frequencies and
flags. This example does not install files into Tlaloc or register a database.

The comparison baseline is `tolteca_kids.ModelFit` with `backend: python`:
`tolteca_kidsproc` performs the detailed isolated and joint physical-model fits.
Fast acquisition-time fitting with `kidscpp` is outside this deliverable.
Kidscpp remains the native model reader and timestream-recovery library used by
Citlali; a Kidscpp fitter executable is not needed to produce these models.

For the complete maintained workflow, start with
[Compare a Tune model through a saved timestream product](../../docs/kids-tune-comparison.md).
`replay.py` generates configurations and calls the installed `tolteca_kids`
command for fitting, group-aware solving, diagnostics and product output.
Its raw inputs, destination, native validation executable and optional response
evidence are explicit arguments. The later `network_*` sections retain the
145609 investigation recipes and evidence; they are not prerequisites for
running the maintained Python workflow on another acquisition.

## Run a fit

Install the coordinated Tolteca and Tollan source revisions supplied with the
comparison release in a Python 3.13 environment. From the Tolteca checkout:

```sh
tolteca_kids --config_paths examples/tune_model_comparison/fit.yaml \
  --show_inputs -- /absolute/path/to/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc
tolteca_kids --config_paths examples/tune_model_comparison/fit.yaml \
  --show_config -- /absolute/path/to/your_tune.nc
tolteca_kids --config_paths examples/tune_model_comparison/fit.yaml \
  -- /absolute/path/to/your_tune.nc
```

Use the actual recorded filename in all three commands. Positional inputs
after `--` are exact file locations. Output paths in this example are relative
to the working directory. Save the printed effective configuration with the
comparison. For an affine-plus-delay fit, set
`kids.model_fit.analysis.gain_family: delay` in a separate configuration.

## Inspect the result

Open the generated `index.yaml` under `comparison/dataprod_toltec`. It lists
the processed acquisition, its artifacts and exact run identity. Open its
product with `tolteca_kids.product.open_sweep_product`. Inspect each block
separately with `ModelFit.view(tree, block=0)` and `block=1` for a two-block
Tune. Do not substitute the final block for the first block.

Each fitted block retains the model analysis in `array.zarr`, a
`kids_model.ecsv` calibration table and a model-fit HTML diagnostic. The Python
backend has no native command log.

For each disputed channel, compare the acquired sweep samples, fit residuals,
fitted resonance frequency, acquisition probe frequency, fit flags and sample
solving flags. Preserve group/component identities for overlapping resonances.
A successful process exit alone does not establish a usable calibration.

## Compare frequencies and normalization

`f_tone` is signed baseband frequency. The channel probe frequency is
`f_chan = f_lo_center + f_tone`; a sweep sample uses its acquired LO frequency
plus that tone frequency. A fitted `fr` is a resonance frequency, not a new
name for the probe frequency. The affine/delay gain reference and phase
convention are part of the serialized model and must travel with its parameters.

Retain the accumulation-length metadata when comparing raw I/Q with model
values. Use the model reader/evaluator rather than copying numeric columns
into a legacy formula. The physical model and historical GainLinTrend model
have different parameter conventions.

## Pass the result to a native consumer

Select the published Python fit's exact `index.yaml`, then prepare a model stub
for the acquisition that will consume it:

```sh
python examples/tune_model_comparison/prepare_stubs.py \
  /absolute/path/to/your_tune.nc /absolute/path/to/science.nc \
  --model_index /absolute/path/to/fit/index.yaml \
  --output_path comparison/dataprod_toltec \
  --namespace tune-comparison > comparison/prepared-input.json
```

This example handles one network and one single-block timestream. The raw
headers and acquired block order identify the recorded calibration sweep. For
the recorded 145609 timestream, this is the final sweep block of its two-block
Tune. Both Tune fits remain in the fit product; preparation selects the correct
one without refitting or reading timestream I/Q samples.

| Operation | Owner | Saved result |
| --- | --- | --- |
| Fit each sweep block | `tolteca_kids.ModelFit`, Python backend | Processed Tune with `array.zarr`, per-block ECSV and diagnostics |
| Check and bind the selected fit | `prepare_model_binding` | Exact model artifact, sweep and block reference |
| Package the selected models for the target acquisition | `tolteca_reduce.stubs.materialize_model_stubs` | Acquisition-owned `dp_raw_obs_proc` with a NetCDF stub artifact |
| Retain the resolved selection | `WorkflowRun` and `DataProdStore` | Run record containing the resolved raw observation context |

Binding does not write a stub. Materialization packages the existing fit,
including model schema, normalization and joint-group metadata, into the
acquisition header representation. Reuse of one Tune by several later
acquisitions requires a matching stub for each acquisition. Raw indexes and
stub products are installed in the selected namespace; raw files remain in
their original locations. The selected fit index remains the exact dependency.

The preparation run records the selected fit as its input and produces the stub.
A separate input-check run saves the resulting context including that stub;
it does not run Citlali. Neither run rewrites an earlier run's inputs.
The script prints the installed stub index, both run UIDs, resolved context
and a `data_item` fragment ready for the existing Citlali input. The
comparison source package includes this preparation module, not the complete
`tolteca_reduce` workspace/driver CLI.

Citlali accepts a selected stub on a network data item:

```yaml
data_items:
  - meta: {interface: toltec0}
    filepath: /absolute/path/to/science.nc
    stubs:
      - filepath: /absolute/path/to/science_kids_model_stub.nc
        mode: override
```

This is a fragment within one existing Citlali input, not a complete reduction
configuration. An exact ECSV `model_params` selection is an alternative to
stubs; do not configure both on one data item. Keep the raw acquisition as the
base. The current Citlali consumer supports one timestream block per input.
Grouped recovery additionally needs explicit APT detector/component bindings.
It must not infer detector identity from channel ordinal.

The [model-report contract](../../design/spec/kids-model-reports.md) applies to
both ECSV and stubs. Each model slot retains the complete ECSV header in
`ModelParams:ecsv_header_<block>`; its numeric values and column names use the
native `ModelParams`/`ModelParamsHeader` variables. There is no separate
model-accumulation-length variable. The source normalization travels in the
ECSV metadata.

Nonlinear joint groups retain all resonators and the full affine/delay gain.
`fit_joint_required` routes a channel to joint solving; it does not make the
fit unusable. `fit_unusable` vetoes solving and preserves flagged NaNs. The
fixed-drive coefficient is retained as loss varies, so fitted nonlinearity is
not held constant while solving. A declared stationary branch is preserved;
arbitrary hysteretic histories are not inferred.

## Verify the common report locally

Build the coordinated Kidscpp revision with tests enabled. Its
`tests/kids_model_report` executable exercises the production ECSV/header readers,
independent-channel masking and routed joint solving. With reference data available
through the test environment:

```sh
KIDS_MODEL_REPORT_CLI=/absolute/path/to/kidscpp-build/tests/kids_model_report \
  pytest -q tests/tolteca_kids/test_joint_model_native.py
pytest -q tests/tolteca_reduce/test_model_table.py \
  tests/tolteca_kidsproc/test_joint_nonlinear.py
```

The native parity test fits an explicitly identified two-channel group in both
145609 Tune blocks, for affine and delay gain. It writes each ECSV and stub, then
compares nonlinear response, derivatives, source flags, recovered frequency and
loss against Python on controlled trajectories. Its per-case temporary directory
contains the reports, stubs, native output and `parity-summary.json`. This tests
numerical agreement, not acceptance of those fits for a measured timestream.
Only the final Tune block is the recorded calibration for the following
acquisition; do not test the first block by applying it to that timestream.

## Evidence to exchange

Send the source revision manifest, effective configuration, both block results,
model/stub files, fit diagnostics and relevant native logs. Identify the science
observation, network and channels exhibiting the problem. Preserve flags and
unresolved identities in the comparison. Test the selected models on the same
raw samples before interpreting a difference in a pointing map.

## Follow the sweep-anchored solution

`joint_solver(group)` now starts from the fitted sweep state and tracks the
accepted detector state through the measured samples. Its default
`state_step_max=0.25` limits each change in `dr/r_reference` and
`dfr/(2*r_reference*f_reference)`. Independent starts are compared at every
sample. Two admissible solutions remain flagged; a solution outside the allowed
motion does not silently replace the previous one.

```python
solver = joint_solver(group, state_step_max=0.25)
fr, r, flags, residual = solver(I, Q, normalization=accumulation_ratio, reset=reset)
x = solver.probes[None, :, None] / fr[:, None, :] - 1
```

`reset` is a boolean vector marking acquisition gaps or explicit sequence
boundaries. A reset returns to the sweep anchor. For xarray input, `solve`
recognizes gaps in the `sample` coordinate and carries state across Dask chunks.
Keep failed samples as flagged NaNs. Use `continuation=False` only to examine
independent solutions without the sweep/continuity assumption. Plot alternative
trajectories as Solution A and Solution B, with dimensionless `r,x`.

The local 145609 regression fits both Tune blocks with affine and delay gains,
checks controlled nonlinear trajectories, and solves the following measured
scan through Python, native ECSV and native stub readers. Only the final Tune
block is used for that measured scan. The acceptance check concerns solving and
branch tracking; it does not certify the sweep fit's scientific quality.

## Full-network coverage and previous nonlinear studies

`network_coverage.py` reads a complete saved Python fit; it does not replace
KidsFind or ModelFit with a private fitting implementation. The maintained
`native_network` CMake target links Kidscpp's production readers and array
solver. Its inputs retain the ECSV/stub model, recorded probe order and
accumulation ratio.

For the 145609 comparison, enable `nonlinear: true` and `a_nl_max: 20.0`
uniformly. This search bound admits the high-drive population into the
diagnostic; it is not a detector-selection threshold or a statement that an
optimum at the bound is a good calibration. Keep the acquired sweep direction.
Singlet/multiplet describes connected fit groups; component labels are not
persistent detector identities. Missing or ambiguous associations stay visible.

```sh
cmake -S examples/tune_model_comparison/native_network -B comparison/native \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/matching/native/install
cmake --build comparison/native
python examples/tune_model_comparison/network_coverage.py \
  /path/to/full-fit/index.yaml /path/to/tune.nc /path/to/timestream.nc \
  comparison/native/tune_network /path/to/review/coverage
```

Use `coverage-delay` for the delay-gain run, with the same finding and
scientific settings. `--prepare_only` writes the replay inputs without native
solving. Native solving retains every measured sample. Python parity checks all
independent-channel samples and the first 256 ordered samples of every measured
joint group. Controlled 16-sample trajectories check every selected group in
both blocks. Only the final recorded Tune block calibrates the measured scan.

`network_validation.py` verifies native ECSV/stub output byte for byte and saves
controlled-state errors. `network_report.py` produces the diagnostic matrix and
interactive plots. Pass the review directory to both programs. The 145609
layout uses `coverage`, `coverage-delay` and a shared
`dataprod_toltec/network-145609` namespace with `0000-models` and `0001-models`
fit realizations. The report reads `evidence.json`: prior-study snapshot paths,
original source paths, checksums and scientific qualifications.

Every component remains in the denominator, including failed fits, unselected
components and groups with no valid solved samples. ECSV report flags,
per-sample flags and fit convergence are separate diagnostics. Bins in `a_nl`
do not change the model, solver or acceptance policy.

The local solving assumption holds `b = 2*a_nl*r_reference³` fixed.
Re-expressing solved loss as a changing nonlinearity adds no independent
measurement, so that trace is omitted. Optical loading changes both resonance
frequency and loss, with local responses `delta x_opt = R_x*delta P_opt` and
`delta r = R_r*delta P_opt`. Fixed incident tone power need not give fixed
nonlinearity: the response model scales `a_nl` with `Q_r³` when its other physical
factors are stable. Earlier power and Skydip studies test those assumptions;
unchanged hardware does not prove fixed `b` under arbitrary loading.
The working-zone figure describes conditional amplitude bias for assumed errors
in `b`. It is not measured noise or a multiplet rejection mask. The equations
and noise distinction are in `design/spec/kids-model-reports.md`.

### Read the recovered signal and its quality

The report starts with the returned signal, then measured output quality,
conditional calibration-error sensitivity, science-sensitivity requirements and
numerical verification. The figures use `x′` for the effective detuning in the
nonlinear response denominator, and `x_opt` for optical detuning with the modeled
readout-power shift removed. Both refer to the same probe. The returned array
`x` estimates `x_opt = f_probe/f_resonance - 1`; it is not calibrated optical flux
or a separation of optical signals from other intrinsic frequency changes.
A median-subtracted example shows both detunings, the returned
resonance-frequency change in Hz and measured loss versus optical detuning.

`network_quality.py`, called by `network_report.py`, uses maintained
`PSDAnalysis` to measure uncleaned resonance-frequency spectra from saved native
outputs. It writes `output-quality.csv`, `output-spectra.npz` and
`output-quality-method.json` in each measured folder, plus the comparison's
`output-quality-matrix.csv`. Hann windows of 1,024 samples, 50% overlap and
constant detrending are used at the recorded sample rate. The 5–20 Hz summary
band is an overview, not a white-noise claim or selection cut. Only complete
valid records enter the spectra; incomplete and failed components remain counted.
Several valid singlet probes contribute their median PSD, not a combined
timestream. A component is never counted once per probe.

`network_response.py` adds the measured loss–detuning comparison. It reads the
pinned `references/skydip-network.json` from the independent Skydip 142440 study
and its 112-trace response distribution. Every component has a row in
`response-components.csv`; `response-trajectories.npz` retains ordered one-second
means and `response-matrix.csv` groups the results. A short-record slope is a
mixed-fluctuation statistic, not a measured optical response. No detector identity
is transferred between observations. Singlets use the complete probe nearest
the fitted resonance; joint components use their full-group native solution.

The conditional two-coordinate projection learns weights from the first half
and evaluates the second half, preserving unit response to the independent
Skydip median direction. Its signal-transfer range uses the Skydip detector
spread as a sensitivity scenario, not an uncertainty interval for this Tune.
`response-method.json` records the bands, selection and assumptions. The new
figures are analysis artifacts; they do not replace any solved timestream.

Frequency ASD in Hz/√Hz is not optical sensitivity. The tables leave optical
S/N and mapping speed unmeasured until calibrated optical response, signal
transfer and noise after science processing are available. The conditional
5%/10% errors in `b` are scenarios, not measured calibration uncertainties.
For correctly calibrated signals their modeled amplitude bias is zero. The
report retains unresolved parity discrepancies alongside the measured spectra.


After both native routes finish, `network_parity.py REVIEW/coverage/measured`
can resume unfinished Python checks with four independent worker processes.
Each worker preserves sample order and continuity within its group. Reuse saved
Python arrays only with the unchanged request, fit and solver revision; use a
new review folder for a different calibration. Normal `network_coverage.py`
execution recomputes Python arrays and does not implicitly reuse cached results.

### Qualify response populations before interpreting noise

The noise report separates response evidence from fit and solving quality.
Generate its component ledger before rendering:

```sh
python examples/tune_model_comparison/network_populations.py REVIEW \
  /path/to/apt-evolution-comparison-20261109
python examples/tune_model_comparison/network_report.py REVIEW
```

The identity-study argument is the saved dark-candidate study containing
`rows.json`, `model-products.json` and `publications.json`. The recipe verifies
identical acquired sweep arrays and original finding measurements, then follows
original peak links into the current model. Tentative associations remain
tentative. It keeps every component of both blocks and both model families in
`response-qualification/components.csv`. The evidence JSON records exact sources
and hashes. It does not match nearby frequencies or reuse model-local integers
as persistent identities.

The report uses response class and acquisition-specific readout suitability as
the primary scientific assessments. Its `science_qualified` field remains false
until optical response and readout suitability are established; numerical native
acceptance does not fill that gap. Finder dark flags remain candidate evidence,
and unknown components are not called active. The existing nonlinearity matrices
remain numerical diagnostics, not an optical-power performance comparison.
See `design/manuscripts/detector-noise-quality.md` for ideal signal/noise
expectations, the measured comparison and the separate role of dark references.

### Compare detectors with individual Skydip response laws

After producing the fits, native arrays and dark-candidate ledger above, run:

```sh
PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/tune_model_comparison/network_active.py REVIEW \
  /path/to/dev_toltec/python/run
```

The second argument contains the maintained Skydip 142440 response study,
baseline replay and APT roster. `active-response/sources.json` lists the exact
files and their checksums. The recipe uses full shared tone-matcher edges,
requires repeatable support in both Tune blocks, and retains ambiguous links.
It adds reviewable observation associations; it does not alter the source APT or
its physical response classes. Current model terms are linked through their
stored finding plans, with seed and group equality checks.

Open `active-response/index.html` through the local report server. It contains
the response/readout populations, signal-referred noise comparison, paired
affine/delay results and selection sensitivity. `active-response/inspect.html`
lets you select each qualified detector and model family to see the measured
sweep, IQ cloud, Skydip response, trajectory, time series and spectra together.

The 145609 replay supports 80 active-response candidates, of which 74 pass the
declared acquired-readout checks in both families. Five quality exclusions,
one unselected solver route, fourteen
dark candidates and fifteen dark-indicator/loading-response conflicts remain
separate. Another 145 components have loading-response evidence without a
qualified individual loss law; 371 have unresolved identity or response.
The 74 qualified components are singlets. Their noise results do not establish
multiplet sensitivity or a benefit from heavy driving.

`components.csv` keeps every final-block component and its exclusions.
`signal-noise.csv` records individual response directions, transformed to the
actual probe frequency, and the measured and conditional noise quantities.
`match-edges.csv` preserves all considered associations. The first Tune block
supports identity consistency; only the final block calibrates scan 2.
Study thresholds and alternate counts are explicit. No fitting, native solving,
raw-data change, roster promotion or science-mask installation occurs here.

### Inspect full network coverage and every exclusion

The 74-detector result is a response-law subset, not network yield. To compare
that subset with all established identities and every fitted component, run:

```sh
PYTHONPATH="$PWD/src" python examples/tune_model_comparison/network_census.py \
  REVIEW /path/to/dev_toltec/python/run
```

Open `network-census/index.html`. It separates missing identity transfer and
response-law evidence from unselected solver routes, invalid samples and
fit-quality exclusions. Both fit families and all multiplicities remain in the
tables. `components.csv` accounts for all 625 final-block model components per
family; `roster-coverage.csv` accounts for all 556 established network-0 UIDs.
Neither is a physical-completeness estimate.

The latest consolidated APT/Skydip receipts preserve the 240 supported transfers.
Of 334 affine components passing the readout screen, 64 are multiplet components
and 260 lack the evidence required for the individual-law comparison. The source
response study omitted 338 traces in connected groups before testing individual
loss/detuning laws. That omission is not a failure of their 145609 fits.

The recipe checks updated receipts against the existing frequency-correspondence
edges and refuses to silently change a supported identity in the saved numerical
comparison. It does not refit, rerun solving, or promote any roster observation.

### Reconcile complete groups and include supported multiplets

The current comparison is the group-aware replay. The isolated-response and
census reports above remain earlier evidence; their 74-detector subset is not
network yield.

```sh
PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/tune_model_comparison/network_groups.py \
  --root run/network-145609 \
  --evidence /path/to/dev_toltec/python/run \
  --output run/network-145609/group-response
```

Open `group-response/index.html` on the report server. The recipe consumes the
saved full tone-match edges, current consolidated Skydip/APT receipts, native
Skydip fit covariances and both existing 145609 model families. It writes no
roster records and does not repeat fitting or native solving.

The group reconciliation preserves whole target fitting groups and source
Skydip groups. All order-preserving assignments that cover the smaller feature
list contribute alternatives. A pair must be forced in every assignment;
unknown source identities still veto an association. Unmatchable complete groups
remain unresolved. This is conditional on the matcher's frequency-order
assumption, not acceptance of a new detector identity. The 200 kHz gate,
five-acquisition support and both-Tune-block requirements remain unchanged.

The local replay supports 292 provisional identities in both blocks. Independent
Skydip response fitting now includes connected-group constituents and produces
208 source component laws, with explicit gaps and quality reasons for every
other source trace. It retains seven acquisitions, exact accepted finding
receipts and labelled frequency/loss covariance; it does not substitute a
population-average response for a missing measurement.

Before fixed-probe routing, the signal/noise comparison included **103 components
per family: 99 singlets and four multiplet components**. The multiplet UIDs are 31, 228, 475 and 546,
across three joint groups. Fourteen multiplet components have transferred
individual response evidence; the other ten were excluded by a mixture of
fit adequacy, unassessed routes, numerical failures and review-policy limits.
All 625 final-block components per family and all 556 roster identities remain
in the accounting. Missing source evidence is not labelled a target fit failure.

The component explorer shows the independent Skydip response, measured joint
trajectory, solved time series and PSD. `joint-covariance.csv` retains measured
cross-component coherence. The conditional per-component two-coordinate
projection does not mix different detector outputs or establish an absolute
NEP/mapping speed. Source fit reduced chi-square and leave-one-acquisition signal
transfer make the response assumptions visible. Public columns use
`loss_detuning_slope` for R_r/R_x and reserve beta for R_x/R_r.

`group-edges.csv` and `group-topology.json` retain every geometric decision.
`source-response-ledger.csv` separates missing receipts/measurements from source
fit/readout quality. `signal-noise.csv`, `matrix.csv` and `evidence-readout.csv`
retain the full target population. `sources.json` pins the inputs, and
`verification.json` checks unchanged scientific artifacts and independent
held-out time-domain projections, including the multiplets.


### Replay fixed-probe routing and inspect separate diagnostics

Use a fresh output directory to regenerate reports and stubs from the saved fits,
then run the native ECSV/stub readers and Python parity checks. No optimizer is
rerun. The original products and previous solved arrays remain read-only.

```sh
PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/tune_model_comparison/network_routing.py \
  run/network-145609 run/network-145609-routed \
  --native run/network-145609/native-build/tune_network

PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/tune_model_comparison/network_groups.py \
  --root run/network-145609-routed \
  --evidence /path/to/dev_toltec/python/run \
  --output run/network-145609-routed/group-response
```

UID 492's doublet is group 313, components 348/349 in the final Tune block.
Channel 622 contributes only edge sweep samples to this fit; its actual probe
belongs to group 312. The group-313 readout uses probes 623/624 and retains both
resonators. A genuinely overlapping range at the actual probe remains unselected.
The same rule applies throughout both blocks and model families. Selecting a
route does not certify its rank, fit adequacy or scientific calibration.

`routing-changes.json` records each changed channel route. The comparison retains
the 5% residual threshold, all fit-warning cuts and the original branch/parity
limits. `signal-noise.csv` now separates:

| Column | Interpretation |
| --- | --- |
| `route_status` | Selected or unassessed route |
| `fit_adequacy`, `residual_limit` | Measured sweep residual against the unchanged study limit |
| `numerical_validity`, `numerical_flags` | Complete estimates versus recorded numerical/model failures |
| `recorded_sample_flags` | All native sample flags, including unused singlet probe estimates |
| `selection_policy_limits` | Bound/seed warnings, domain/branch limits and parity tolerances |
| `parity_status` | Matching masks/values, mask disagreement, tolerance miss or unassessed |
| `comparison_exclusions` | All reasons retained by the comparison policy |

The report provides a table of all response-supported multiplets, including
excluded components, and independent diagnostic counts for the complete network.
A domain flag can mean a physical limit or a sweep/continuity restriction;
it does not establish numerical nonconvergence. A complete native estimate
likewise does not establish that its sweep model meets the residual criterion.

The local replay retains all 625 final-Tune components per gain family:

| Measured result | Affine, before → after | Delay, before → after |
| --- | --- | --- |
| Complete native component estimates | 490 → 552 | 492 → 554 |
| Passes unchanged comparison checks | 334 → 369 | 332 → 367 |
| Has independent Skydip response and passes comparison | 103 → 113 | 103 → 113 |
| Multiplet components in that response comparison | 4 → 5 | 4 → 5 |

UID 492 has valid native solutions for all 7,802 samples, with sweep residuals
of 1.5648% (affine) and 1.4374% (delay). Native outputs are byte-identical through
the ECSV and stub readers for both controlled Tune blocks and the measured
timestream. Python/native comparison covers every independent-channel sample
and the first 256 measured samples of each joint group; group 313 passes.
An extended check of group 313 also compares all 7,802 samples: validity flags
match for both models, maximum frequency differences are below 0.00025 Hz,
and maximum loss differences are below 2.5e-13.
The remaining nine response-supported multiplet components all exceed 5%.
Their additional numerical and selection-policy limits remain recorded.

The comparison checks all 4,512 original scientific-file hashes, unchanged fit
values and identities, and the same selection predicates for all 1,250 model
component records. The refreshed report is
`run/network-145609-routed/group-response/index.html`; the earlier report is
retained under `run/network-145609/group-response/index.html`.


### Post-solving coordinates, cross-spectra and network performance

Use the existing routed fits, ECSV/stub solutions and Skydip/APT evidence:

```sh
PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/tune_model_comparison/network_postsolve.py \
  --root run/network-145609-routed \
  --output run/network-145609-routed/post-solving
```

Open `post-solving/index.html` through the existing local report server.
The explorer contains every final-Tune component, including those without a
complete estimate. It displays `x_prime` versus `x_opt`, loss trajectories,
coordinate PSDs, complex prime/loss cross-spectra, coherence and signed local
noise contributions. Full samples and spectra remain downloadable. The same
fitted group response generates the prime coordinate and reconstructs measured
I/Q; no new fitting or solving path is introduced.

```text
post-solving/
├── index.html
├── components.csv             # all 1,250 model/component records
├── performance.csv            # evidence cohorts and measured performance
├── matrix.csv                 # singlet/multiplet × fitted nonlinearity
├── iq-residuals.csv            # predicted versus measured group I/Q
├── cross-components.csv       # correlations between solved components
├── catalog.json
├── cases/<model>-<group>-<component>.json
├── arrays/<model>-<group>-<component>.npz
├── arrays/<model>-group-<group>-cross.npz
├── measurement.json           # raw I/Q equality and acquisition continuity
├── method.json
├── sources.json               # exact input and code hashes
└── verification.json
```

All 625 components per family remain visible: 552 affine and 554 delay have
complete measured trajectories. Independent response-qualified comparisons
still contain 108 singlets and five multiplet components per family; all
selection and evidence fields are checked against the previous report.
The 5–20 Hz affine median coordinate ASD ratio is 1.377 for those singlets and
1.226 for those multiplets. Accounting for the independently assumed Skydip
signal transfer gives median ratios 1.125 and 1.201. The report separates this
conditional comparison from coordinate-noise changes. It does not infer an
instantaneous measured nonlinearity or claim absolute mapping speed.

The shared numerical implementation and assumptions are specified in
[the model-report contract](../../design/spec/kids-model-reports.md#post-solving-response-and-noise-diagnostics).
