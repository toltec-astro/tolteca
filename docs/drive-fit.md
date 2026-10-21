# Fit and commit drive attenuation

Use `tolteca_kids.drive_fit` to fit an attenuation sequence and publish its
scientific results. Use `tolteca_kids.drive_fit_commit` with that saved product
and a target sweep to compute tone amplitudes and global attenuation.
Configuration belongs to the root `drive_fit` and `drive_fit_commit` sections;
settings under `kids` do not affect either workflow.

| Workflow | Inputs | Steps | Product |
| --- | --- | --- | --- |
| `drive_fit` | One attenuation sequence, across selected readout interfaces | `DriveFitStep`, optional `DriveFitPlotStep`, product output | `dp_drivefit` |
| `drive_fit_commit` | Existing drive-fit product and one target acquisition | `DriveFitCommitStep`, optional `DriveFitCommitPlotStep`, product output, optional `TlalocOutput` | `dp_drive_fit_commit` |

Each fit sequence needs at least five distinct attenuation settings by default,
one master, consistent channels and tone amplitudes within each interface, and
one sweep block per file. Commit needs exactly one target sweep per interface,
all from the same acquisition. Select one sequence or target acquisition at a time.

## Fit and publish from Python

The `tolteca_ref_data` collection contains a compact initialization sequence,
observations 145480–145486 on networks 0, 7 and 11 (one per array). The fit inputs
are the seven target sweeps at 145484/1–7, from 4 to 16 dB attenuation.
The following example uses network 0; repeat with network 7 or 11, or pass
all three networks to the same workflow. Set `sequence_dir` to your local
read-only network directory.

```python
from pathlib import Path
from tollan.config import RuntimeContext
from tolteca_kids.drive_fit import (
    DriveFit, DriveFitOutputStep, DriveFitStep, open_drive_fit_product,
)

sequence_dir = Path("/path/to/tolteca_ref_data/data_lmt/toltec/tcs/toltec0")
sources = sorted(sequence_dir.glob("toltec0_145484_*_targsweep.nc"))
runtime = RuntimeContext({
    "drive_fit": {
        "fit": {"channels": [0, 100, 300]},
        "plot": {"enabled": True},
        "output": {"enabled": True, "path": "run/dataprod_toltec"},
    },
})
working = DriveFit(runtime)(sources)
result = DriveFitStep.view(working["interfaces/toltec0"])
print(result.table)
fit_path = DriveFitOutputStep.get_context(working).data

with open_drive_fit_product(fit_path, interface="toltec0") as saved:
    print(DriveFitStep.view(saved).n_success)
```

Omit `fit.channels`, or set it to `None`, to fit every recorded channel.
`DriveFit` closes input files before fitting. Its working collection retains
loaded scientific arrays and typed step contexts. Published products include
Zarr arrays, an ECSV channel summary, optional figures, and `index.yaml`.
Reopened arrays may be lazy: keep the returned tree open until values load.
Plotting saved results does not repeat the fit.

## Inspect fit diagnostics

Enabling `plot` produces three diagnostic views:

- `drive_fit_summary.html`: selected-attenuation and nonlinearity histograms,
  plus attenuation versus tone frequency and loaded Qr. Colors distinguish
  accepted from flagged finite selections. The title also counts channels
  without a selection.
- `drive_fit_grid.html`: one channel per panel, with 12 channels per page by
  default. Use the channel-page slider to inspect the full network. Each panel
  shows the nonlinearity curve, selected point, reference line, and failed or
  excluded fit counts. Hover shows Qr and phase-quality flags.
- `drive_fit_sweeps.html`: a channel browser listing flagged channels first.
  Each channel opens a collated four-panel view of all attenuation settings:
  magnitude, I/Q circles and fitted models, phase, and phase residuals (data
  minus model). Colors identify attenuation; the same color spans every panel.
  Use the attenuation selector, or double-click its legend entry, to isolate
  one power. Hover shows `a`, `Qr`, `fit_flag`, and `phase_flag`. Failed fits
  retain measured magnitude and I/Q; saved errors appear under **Fit failures**.

The grid initially shows the nonlinearity selection interval. Autoscale reveals
outliers; their recorded values are retained. Failed fits leave gaps in curves.
The displayed phase-quality flags do not change drive-selection acceptance.
In the sweep plots, dots show measurements, solid lines show models over the
fitted interval, dotted lines show fitted circles, and faint dots show full-sweep
phase. I/Q axes have equal scale. Magnitude and I/Q use the fitter's normalization
to the lowest-power sweep's final sample and its cable-delay correction. Frequency
offsets are relative to the channel's tone frequency in kHz.
These figures use saved diagnostics without reading raw files or refitting. If an
older product lacks circle diagnostics, rerun drive fitting to record them.
Keep `drive_fit_sweeps.html` with its sibling `drive_fit_sweeps` directory when
copying the plots; all pages work offline.

Configure `plot.artifacts.grid.page_size` and `columns` to change the layout,
or `channels` to select recorded channel IDs. For example, inspect saved results
without repeating the fit:

```python
from tolteca_kids.drive_fit import DriveFitGridViz

with open_drive_fit_product(fit_path, interface="toltec0") as saved:
    grid = DriveFitGridViz(page_size=12, columns=4)(DriveFitStep.view(saved))
    grid.write_html("run/drive-fit-grid.html")
```

To export detailed sweep plots for particular channels, configure
`drive_fit.plot.artifacts.sweeps.channels`, or use the saved
product directly:

```python
from tolteca_kids.drive_fit import DriveFitSweepArtifacts

with open_drive_fit_product(fit_path, interface="toltec0") as saved:
    sweeps = DriveFitSweepArtifacts(channels=[100, 300])
    figures = sweeps(DriveFitStep.view(saved))
sweeps.write(figures, path=Path("run/drive-fit-details"))
```

Channel selections default to all fitted channels. The browser, channel pages,
and local plotting library are included in the published product index.

## Commit to a target sweep

Continue with `fit_path` from the preceding example:

```python
from tolteca_kids.drive_fit_commit import (
    DriveFitCommit, DriveFitCommitStep, open_drive_fit_commit_product,
)

commit_runtime = RuntimeContext({
    "drive_fit_commit": {
        "plot": {"enabled": True},
        "output": {"enabled": True, "path": "run/dataprod_toltec"},
    },
})
target = next(sequence_dir.glob("toltec0_145485_*_targsweep.nc"))
committed = DriveFitCommit(commit_runtime)([target], reference=fit_path)
commit_path = DriveFitOutputStep.get_context(committed).data
with open_drive_fit_commit_product(commit_path) as saved:
    settings = DriveFitCommitStep.view(saved)
    print(settings.table)
    print(settings.data.attrs["atten_drive_global"])
```

The commit uses the shared KIDs `Match1D` matcher with `dtw_python`. As in v2,
its shift-search spacing is half the target sweep's frequency step. Failed
or excessive drive selections receive the configured fill percentile; global
attenuation is quantized downward to 0.25 dB by default. Tone amplitudes remain
in the target's recorded channel order.

Commit predicts tone power using a **fixed total DAC power** approximation,
distributed over the emitted DAC bins by squared tone amplitude. It uses the
full reference comb's power fractions, including when the fit selected fewer
channels, and normalizes the complete output comb after matching. This corrects
the earlier max-amplitude attenuation calculation. The fill and cutoff
percentiles retain their existing meaning. Downward quantization can produce a
model power excess of less than one attenuation step.

The commit table records `power_policy_db`, `power_commit_db` and `power_error_db`;
the first two are relative to total DAC power at the drive-attenuator output.
`atten_drive_exact` in the metadata is the unquantized model solution.
The commit plot includes the residual after quantization. These are model
predictions: Tlaloc's phase-dependent waveform peak can change the actual total
DAC power. Absolute detector-input dBm additionally requires calibration and
the downstream loss.

Rerun drive fitting for older products that lack `power_fraction`. New products
retain the full-comb normalization without requiring the reference raw files
during commit. Output tones sharing a DAC bin must request the same power;
conflicting requests stop commit with an error.

The product includes matching diagnostics, the commit table, and Tlaloc-format
`toltecN/default_targ_amps.dat` and `toltecN/atten_drive.dat` files. Publishing
these artifacts does not apply them to an instrument directory.

## Write Tlaloc settings

Enable `drive_fit_commit.tlaloc_output` and supply its `path` to install settings
after configured product publication and registration succeed. It is disabled
by default. The destination is the Tlaloc `etc` root containing `toltecN`
directories. The writer replaces amplitude and attenuation files and preserves
other files, including phases. Each file replacement is atomic; the pair is
not a filesystem transaction.

The existing `toltecN/targ_freqs.dat` must describe the commit target's recorded
LO and ordered probe frequencies, compared at the Tlaloc DAC frequency resolution.
A missing file or a changed LO, count or tone order stops installation before
replacement. If another reduction has prepared a different tone list, load that
list and acquire a new standalone target sweep before making its commit.

You can also install a saved commit without reading the target sweep or matching
again. This example assumes the demonstration directory already contains the
matching v2 tone list:

```python
from tolteca_kids.tlaloc_output import TlalocOutput

with open_drive_fit_commit_product(commit_path) as saved:
    TlalocOutput(enabled=True, path=Path("run/tlaloc-example/etc"))(saved)
```

## Run configured discovery

Save this as `drive_fit.yaml`, replacing the input path:

```yaml
input_discovery:
  inputs:
    - sources:
        - glob: /path/to/tolteca_ref_data/data_lmt/toltec/tcs/toltec0/toltec0_145484_*_targsweep.nc

drive_fit:
  fit:
    n_adrvs_min: 5
    weight_window_Qr: 12000.0
    a_ref: 0.2
    channels: [0, 100, 300]
  plot:
    enabled: true
  output:
    enabled: true
    path: run/dataprod_toltec
```

```bash
python -m tolteca_kids.drive_fit --config_paths drive_fit.yaml
python -m tolteca_kids.drive_fit_commit \
    --drive_fit_commit.reference /path/to/published/fit/index.yaml \
    --drive_fit_commit.plot.enabled true \
    --drive_fit_commit.output.enabled true \
    --drive_fit_commit.output.path run/dataprod_toltec \
    -- /path/to/target_targsweep.nc
```

Installed console commands are `tolteca_drive_fit` and `tolteca_drive_fit_commit`.
Both accept dotted overrides such as `--drive_fit.fit.a_ref 0.15` and
`--show_config` to inspect settings without reading inputs. Positional arguments
after `--` replace configured input discovery with exact file locations.
Fit, plot, and output settings are separate; plot and output default to disabled.

Configured discovery uses the same
[input entries and observation selectors](kids-inputs.md) as other TolTECA
workflows. Metadata queries run after headers have been read. Conditional
configuration can use `master`, `roach` and `interface`; product output settings
must agree across interfaces. Python callers can iterate over the same workflows
with each module's `run(runtime)`.

## Register and reopen products

Set the workflow's `output.registration: true` with a configured `data_prod.index`
DB binding to register its published indexes. Register input raw acquisitions
first. Commit also requires its referenced drive-fit realization in that DB.
Fit products retain all sequence-member associations; commit products retain the
target acquisition and the exact namespaced fit realization.

Products use the configured output root, namespace (default `scratch`), run and
tag (default `main`). The returned path pins a completed realization. A version
group path resolves its current `main` realization once when opening it.
Both `open_drive_fit_product` and `open_drive_fit_commit_product` also accept
`DataProdIndex` descriptors returned by DPDB, with `interface` required when
several networks are present. Reopening and plotting use saved arrays.

If registration fails, published files remain available. Retry ingestion of the
realization's `index.yaml`, its parent version-group `index.yaml`, and the
namespace's `namespace.yaml` through `DataProdHandler(runtime).index.ingest(...)`.
This does not rerun fitting or matching. Tlaloc application has not run when
registration fails.

## Interpret the result

| Value | Meaning |
| --- | --- |
| `adrv_best` | Selected drive attenuation in dB. |
| `p_best` | Negative selected attenuation; this preserves the recipe's power convention and is not an absolute calibrated power. |
| `a_best`, `Qr_best` | Nonlinearity and loaded quality factor at the selected power. |
| `p_flag` | 0 for an accepted first crossing of `a_ref`; 1 for an endpoint, a single valid fit, or no valid fit. |
| `fit_flag` | Per-power execution status: 0 for a completed phase fit, 1 for a failed fit. |
| `phase_flag` | The fitter's phase-quality flag: 0 good, 1 fit failure, 2 residual threshold exceeded. |
| `fit_error` | Exception text for an unsuccessful fit. |

Drive selection uses completed fits whose nonlinearity lies within
`[-0.2, 1.5]`, preserving the production recipe's crossing tolerance and endpoint
rules. Phase-quality flags remain available separately; they do not filter the
drive selection. Channels with no usable fits have NaN selected values and
`p_flag = 1`.

`result.data` exposes the per-power `a`, `Qr`, `Qc`, `fr`, reduced I/Q,
frequency grids, and fitted phase samples. Phase arrays are padded with NaN
after the fitted interval. Input identities, source locations, effective
configuration and cable delay accompany the arrays.

