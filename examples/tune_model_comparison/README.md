# Compare acquired Tune models

Use this example to fit recorded target sweeps and both acquired sweep blocks
of a Tune. Keep the original NetCDF files read-only. Start with one network and
one acquisition, then expand the comparison after checking the frequencies and
flags. This example does not install files into Tlaloc or register a database.

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
backend has no native command log. The `kidscpp` backend additionally records
native command and block/channel diagnostics; it is a separate fitting
implementation and must not be treated as an alias for the Python physical fit.

For each disputed channel, compare the acquired sweep samples, fit residuals,
fitted resonance frequency, acquisition probe frequency, fit flags and inverse
recovery flags. Preserve group/component identities for overlapping resonances.
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

Select the exact recorded calibration sweep and its fitted block. Prepare a
model stub for the acquisition that will consume that calibration using
`tolteca_kids.prepare_model_binding`. The original fit product belongs to the
sweep; the stub describes the consuming acquisition. Reuse of one Tune by
several later acquisitions requires a matching stub for each acquisition.

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

## Evidence to exchange

Send the source revision manifest, effective configuration, both block results,
model/stub files, fit diagnostics and relevant native logs. Identify the science
observation, network and channels exhibiting the problem. Preserve flags and
unresolved identities in the comparison. Test the selected models on the same
raw samples before interpreting a difference in a pointing map.
