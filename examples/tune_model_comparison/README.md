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
