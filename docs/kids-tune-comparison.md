# Compare a Tune model through a saved timestream product

For the focused recipient release, see the
[handoff README](https://github.com/toltec-astro/tolteca/blob/codex/grant-tune-comparison-20261120/examples/tune_model_comparison/handoff/README.md).
It includes the compact 152434 comparison PDF, pinned dependency revisions and
a portable fitting-to-map recipe. This guide retains the detailed evidence,
controls and earlier investigations, including unresolved solver parity.

Start with one raw Tune and the timestream that records it as its calibration.
The comparison fits every acquired Tune block, prepares a model stub for the
timestream, then solves and diagnoses the timestream through the KIDs workflow.
The raw files stay unchanged. A database and an acquisition controller are not
required.

For 145609, use the `toltec0` Tune scan 1 and timestream scan 2. The Tune contains
two sweep blocks. Both are fitted, but only the final block calibrates scan 2.

## Run the maintained recipe

Install the coordinated Tolteca and Tollan versions from the comparison's
release manifest. From that Tolteca source tree, with its installed
`tolteca_kids` command on `PATH`:

```sh
python examples/tune_model_comparison/replay.py \
  /data/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc \
  /data/toltec0_145609_000_0002_2025_10_31_09_27_57.nc \
  comparison-145609 --workers 4
```

Supply your actual file paths and a new output directory. The recipe generates
separate affine and delay configurations and invokes the packaged command for
each fit and solve. Both use the same continuous nonlinear model, with the
reviewed fitting range `a_nl_max: 20`. A small fitted nonlinearity does not select
a different solving algorithm or discard a detector.

For the native ECSV/stub comparison, build the supplied
`examples/tune_model_comparison/native_network` CMake target against the pinned
Kidscpp installation, and add `--native /path/to/tune_network`. This executable
tests the production Kidscpp readers and array solver used by Citlali. Python
remains the fitting backend. The verification covers controlled data for both
Tune blocks and recorded timestream data for the final block.
For the measured acquisition, ECSV and stub transport must agree on every
native sample. The replay reuses its saved Python workflow states for the
numerical comparison, checking the model, normalization and raw I/Q first.
Failure flags and finite-value masks must also agree. When verification runs
without a solved product, it computes an explicitly reported Python prefix.
The recipe saves its Python product receipts before starting native verification.
A failed native check exits unsuccessfully, retains `solving.json`, and marks
`native_verification: failed` in `replay.json`; it does not discard the Python
products or label the comparison as passed.

### Installed-wheel check on 145609

The 2026-11-22 installation check fitted both acquired Tune blocks and processed
the first 256 measured samples of scan 2 with all 632 acquired probes. This short
sample tests the installed workflow and artifact round trip; it is not the
full-record signal/noise measurement. The same fitting configuration routes 480
groups in block 0 and 473 groups in block 1.

| Check | Affine baseline | Affine plus delay |
| --- | ---: | ---: |
| Fitted components retained for scan 2 | 625 | 625 |
| Components with complete solutions and spectra in the 256-sample check | 549 | 552 |
| Declared artifacts in the solved product | 452 | 453 |
| Controlled Python/native samples per routed group, both Tune blocks | 16, passed | 16, passed |
| Native ECSV/stub equality over all measured samples | 7,802, passed | 7,802, passed |
| Strict Python/native comparison on the measured prefix | Failed | Failed |

The measured comparison exposes differences that the controlled trajectories do
not exercise. Some rejected samples have different reason flags; some
overdetermined groups disagree on numerical convergence or the returned state
within the comparison tolerance. These remain explicit failures of the native
comparison. Transport equality does not imply Python/native solving parity, and
neither establishes scientific acceptance of a fitted detector. The comparison
bundle includes the group-by-group report and does not present its native path
as an interchangeable, validated replacement for the Python baseline.

Both installed-wheel products were subsequently reopened with
`diagnose_product.py` and the exact Skydip/APT evidence table. The new
`0001-diagnostics/index.yaml` products retain all `fr`, `r`, `x_opt` and flag
arrays exactly, together with the original calibration binding and solve record.
They add the evidence artifact and identify 113 response-qualified components;
the remaining evidence categories stay separate in the 625-row ledger. This
classification comes from the supplied baseline evidence, not from the short
installation-test timestream.

### Full-record replay from a fresh directory

An isolated Python 3.13 environment installed the pinned application wheels and
public-index dependencies. All 54 focused numerical and workflow tests passed
with the normal warning-as-error settings. Package imports and dependency
consistency were checked independently of the development environment.

The raw-data replay used those wheels and the same pinned dependency versions
in the local test runtime. It processes all 7,802 measured samples and all 632
acquired probes. It fits both Tune blocks in each gain
family, prepares the target-acquisition stub, and publishes the solved arrays,
diagnostics and figures through `DataProdOutput`. All four newly generated
block reports match the bundled ECSV reports byte-for-byte. Both raw-file
hashes remain unchanged.

| Full-record check | Affine baseline | Affine plus delay |
| --- | ---: | ---: |
| Fitted components retained | 625 | 625 |
| Routed groups | 473 | 473 |
| Components with complete trajectories and spectra | 544 | 547 |
| Components passing the separate 5% fit-residual policy | 400 | 401 |
| Response-qualified signal/noise comparisons | 113 | 113 |
| Declared artifacts, all verified on disk | 453 | 454 |
| Groups disagreeing with native solving over the full record | 14 | 16 |

These rows describe independent checks. A complete numerical trajectory does
not establish optical response or acceptable sweep fit. The full ledger keeps
incomplete solutions, missing probes, fit-policy decisions and response-evidence
categories separate. Both families retain all 625 component records, including
those without a usable spectrum.

For this fresh replay, each namespace contains `0000-models` for the Tune fit
and `0001-solved` for the timestream product:

```text
dataprod_toltec/
├── affine/
│   ├── 145609-0-1/dp_raw_obs_proc/0000-models/index.yaml
│   └── 145609-0-2/dp_raw_obs_proc/
│       ├── <stub-run>/index.yaml
│       └── 0001-solved/
│           ├── index.yaml
│           ├── array.zarr
│           └── artifacts/toltec0/blocks/0/
│               ├── tolteca_kids.timestream_diagnostics:TimestreamDiagnostics/
│               │   ├── detectors.ecsv
│               │   ├── performance.ecsv
│               │   └── response_evidence.ecsv
│               └── tolteca_kids.timestream_plot:TimestreamPlot/
│                   ├── timestream.html
│                   ├── summary.html
│                   ├── group_*.html
│                   └── plotly.min.js
└── delay/                  # same layout, its own exact model and stub
```

The bundle's `evidence/full-replay.json` identifies every actual realization,
including the UUID-named stub runs. `evidence/full-product-snapshots/` contains
inspection copies of the generated configurations, indexes, completed run
records and diagnostic tables. The index captures the execution during output
creation; `run.yaml` records its completion. These copies omit the large Zarr
payloads and are not relocated inventories. Running the maintained recipe
recreates the complete product tree.

The full-record native check remains failed. Exact native ECSV/stub equality
passes on all samples, but Python/native validity flags, rejection reasons and
some numerical solutions still differ. The bundle retains the strict comparison
and its group-by-group results; no tolerance was relaxed to obtain a passing
handoff.

## Follow the selected model

| Entity or operation | What it contributes |
| --- | --- |
| Raw observation index | Acquisition identity, file locations and recorded calibration history |
| Python `ModelFit` product | The exact fitted groups and channel selections, plus one ECSV per sweep block |
| Resolved raw observation context | The timestream's recorded sweep and exact selected model artifact |
| Model stub product | That model packaged for the consuming acquisition's header slots |
| `TimestreamSolve` | Accepted detector states and flags from the same fitted groups |
| `TimestreamDiagnostics` | Coordinates, cross-spectra, sample checks and a complete component ledger |
| `TimestreamPlot` | Offline Plotly figures made from the saved diagnostics |
| `DataProdOutput` | The scientific Zarr tree, declared artifacts and portable `index.yaml` |

The stub and scientific model arrays describe the same fit. They serve different
readers: the Python workflow reads the full model product, while the native
reader consumes the common ECSV or its NetCDF stub representation. Before
solving, the Python workflow checks that the full product exports the bound
ECSV exactly.

## Inspect and change the configuration

The generated `fit-affine.yaml` contains:

```yaml
kids:
  sweep_check: {enabled: true}
  kids_find: {enabled: true}
  model_fit:
    backend: python
    analysis:
      gain_family: affine
      nonlinear: true
      a_nl_max: 20
  model_fit_plot: {}
  tone_prep: {enabled: false}
  tlaloc_output: {enabled: false}
```

`solve-affine.yaml` points to the installed raw indexes and exact fitted product:

```yaml
kids:
  input:
    select: obsnum == 145609 and subobsnum == 0 and scannum == 2
    raise_on_errors: true
  context:
    raw_indexes:
      - /comparison/dataprod_toltec/affine/145609-0-1/dp_raw_obs.yaml
      - /comparison/dataprod_toltec/affine/145609-0-2/dp_raw_obs.yaml
    models:
      - index: /comparison/dataprod_toltec/affine/145609-0-1/dp_raw_obs_proc/0000-models/index.yaml
        producer: tolteca_kids.model_fit:ModelFit
  timestream_solve: {enabled: true, backend: python, chunk_samples: 4096}
  timestream_reduce: {psd: null}
  timestream_diagnostics: {nperseg: 1024, fit_residual_max: 0.05}
  timestream_plot: {}
  output:
    enabled: true
    registration: false
    path: /comparison/dataprod_toltec
    namespace: affine
    run_name: {generator: counter, key: solved}
```

Paths above illustrate the generated configuration; use its actual absolute
paths. `replay.json` records the exact output indexes, so inspection does not
depend on whichever realization a later `main` tag selects.

The 5% diagnostic cut compares the fit residual with measured sweep variation,
centered separately within each probe. The total-IQ residual ratio remains a
separate column. Neither cut removes solved states. Numerical flags and external
response evidence are also separate from that policy decision.

## Inspect the saved output

The recipe separates configurations and logs from the shared product tree:

```text
comparison-145609/
├── fit-affine.yaml, fit-delay.yaml
├── solve-affine.yaml, solve-delay.yaml
├── prepared-affine.json, prepared-delay.json
├── replay.json
├── logs/
├── native-affine/, native-delay/          # when native verification is enabled
└── dataprod_toltec/
    ├── affine/
    │   ├── namespace.yaml
    │   ├── .runs/by_uid/<execution>/run.yaml
    │   ├── 145609/dp_obs.yaml
    │   ├── 145609-0-1/
    │   │   ├── dp_raw_obs.yaml
    │   │   └── dp_raw_obs_proc/0000-models/
    │   │       ├── index.yaml
    │   │       ├── array.zarr
    │   │       └── artifacts/toltec0/blocks/{0,1}/…/kids_model.ecsv
    │   └── 145609-0-2/
    │       ├── dp_raw_obs.yaml
    │       └── dp_raw_obs_proc/
    │           ├── <stub-run>/index.yaml
    │           ├── <stub-run>/*_kids_model_stub.nc
    │           └── <solve-run>/
    │               ├── index.yaml
    │               ├── array.zarr
    │               └── artifacts/toltec0/blocks/0/
    │                   ├── tolteca_kids.timestream_diagnostics:TimestreamDiagnostics/
    │                   │   ├── detectors.ecsv
    │                   │   ├── performance.ecsv
    │                   │   └── response_evidence.ecsv   # when supplied
    │                   └── tolteca_kids.timestream_plot:TimestreamPlot/
    │                       ├── timestream.html
    │                       ├── summary.html
    │                       ├── group_*.html
    │                       └── plotly.min.js
    └── delay/                              # same organization, separate fits
```

Run directory names are generated by the store. This is the layout contract;
`replay.json` and each `index.yaml` identify the actual realizations.

Open `timestream.html` to inspect the network summary and individual groups.
The HTML works offline and shares one Plotly library. The exported index lists
the figures and tables as artifacts; these files are not an extra report tree
outside the data product.

Within `array.zarr`, `toltec0/blocks/0/solved_groups` retains all fitted
components and independent group nodes. Each group has detector `fr,r` and
probe-by-detector `x_opt`, sample flags and the complete fixed-drive model.
`timestream_diagnostics` retains evaluated `x_prime` and `x_opt`, measured and
predicted I/Q, signed cross-spectral matrices and cross-component spectra.
Missing or rejected samples remain at their original positions.

## Include independent response evidence

Without Skydip/APT evidence, the recipe reports coordinate noise and keeps every
component's evidence status as `missing_evidence`. It cannot infer optical
sensitivity from the timestream alone.

Supply `--affine_evidence evidence-affine.ecsv` and
`--delay_evidence evidence-delay.ecsv` to reproduce an independently prepared
response comparison. Each table identifies its exact model report with a
SHA-256 hash and records the response slope's reference frequency. The workflow
retains those input bytes in the exported product. It rejects another model's
evidence instead of silently matching row numbers.

The performance table keeps singlets, multiplets and nonlinearity bins separate.
Its signal-transfer comparison is conditional on the supplied response and
loading assumption. It does not label missing evidence as detector failure or
sum correlated multiplet components into a mapping speed.

See the [model-report contract](kids-model-reports.md) for the coordinate and
spectral definitions, flag meanings, calibration transport and verification
requirements.

```{toctree}
:hidden:

kids-model-reports
```

## Rerun diagnostics without fitting or solving

To change the spectral window, plotting, or independent evidence assessment,
start from the pinned solved product. The maintained recipe reopens its accepted
states, checks the exact fitted-group identity, runs the diagnostic and plot
steps, and writes a new realization:

```sh
python examples/tune_model_comparison/diagnose_product.py \
  /comparison/dataprod_toltec/affine/145609-0-2/dp_raw_obs_proc/0001-solved/index.yaml \
  /comparison/dataprod_toltec/affine/145609-0-1/dp_raw_obs_proc/0000-models/index.yaml \
  /comparison/dataprod_toltec --namespace affine \
  --evidence /comparison/response-evidence-affine.ecsv
```

Use the actual pinned paths from `replay.json`. Omit `--evidence` when no
independent response table is available. The earlier product and its review
history remain unchanged; the new run records the source product and retains
its resolved calibration inputs and step records. A changed fit is rejected.

Group figures include the complex cross-spectrum and coherence of x' and r,
in addition to their trajectories, coordinate spectra and signed noise terms.
The full numerical arrays remain in Zarr; figure downsampling only changes the
displayed time samples.

## Compare reduced maps with a legacy reduction

Use two controls to distinguish changes in the reduction engine from changes
in the fitted detector model. First hold the raw files, APT, telescope data,
pointing correction, model coefficients and requested science settings fixed.
Then hold the v3 engine fixed and change only its selected model. A comparison
with an archived map alone does not establish which change caused a difference.

`prepare_map_control.py` accepts a fully resolved, single-observation legacy
Citlali configuration. Its file paths and fit-report directory must already be
local. It validates every legacy coefficient against the APT, adds explicit
channel labels for v3, and records the input hashes. It refuses ambiguous report
selection. Both engines use serial execution; `--single-pass` disables iterative
source feedback in both configurations.

```sh
python examples/tune_model_comparison/prepare_map_control.py \
  local-citlali.yaml map-control --single-pass

# Run the actual legacy and coordinated v3 executables separately.
/path/to/legacy/citlali /path/to/legacy/default-config.yaml \
  map-control/v2/config.yaml
/path/to/v3/citlali map-control/v3-legacy-model/config.yaml

python examples/tune_model_comparison/compare_maps.py \
  map-control/v2/redu00/152434/raw \
  map-control/v3-legacy-model/redu00/152434/raw \
  map-control/figures
```

Use the output directories actually reported by the selected engines. The
comparison uses the FITS WCS to identify identical sky pixels, including opposite
pixel storage directions. It rejects unequal pixel scales or fractional pixel
alignment. It does not register the images, interpolate their values, or rescale
one map to resemble the other. Signal panels share a color scale; the difference
panel has its own symmetric scale. The JSON metrics use a common finite,
positive-weight mask, a 10% coverage threshold and a 40 arcsec aperture.

### A model-change control with an existing APT

Fit the actual Tune with the maintained Python configuration and use
`prepare_stubs.py` to bind its final sweep block to the target acquisition.
The full model product retains both acquired blocks and all fitted groups.

```sh
python examples/tune_model_comparison/prepare_singlet_map_control.py \
  map-control/v3-legacy-model/config.yaml \
  /path/to/exact-model/index.yaml prepared.json model-control
/path/to/v3/citlali model-control/legacy-singlets/config.yaml
/path/to/v3/citlali model-control/python-singlets/config.yaml
python examples/tune_model_comparison/compare_maps.py \
  model-control/legacy-singlets/redu00/152434/raw \
  model-control/python-singlets/redu00/152434/raw \
  model-control/figures --arrays a1100 \
  --left-label 'Legacy Tune model' --right-label 'Python affine nonlinear model' \
  --difference-label 'Python - legacy'
```

This first control selects groups containing exactly one acquired probe and one
fitted resonator, with an existing usable APT row and zero model flags. It applies
the same selected channels, constant weights and disabled adaptive
inverse-variance rejection to both maps. PCA cleaning, filtering, pixelization,
geometry and the original flux conversion remain fixed. The native consumer
reads the complete Python model stub, including nonlinearity; no coefficients
are refitted or linearized for this comparison.

For the local 152434 test, network 0 has 630 acquired channels, 553 usable legacy
APT rows and 373 one-channel/one-resonator groups. Of those groups, 358 have zero
model flags and 346 also have a usable APT row. The other channels remain in the
fit product. Their omission from this map control does not classify them as
failed detectors. Multiplets require verified component-to-APT associations
before their recovered signals can be assigned beam positions or flux scales.

The original flux conversion is intentionally fixed. These maps test signal and
shape changes, rather than establishing a new absolute photometric calibration.
The unresolved full-record Python/native solving differences documented above
also remain a separate validation requirement.

### Historical single-network 152434 control

The 2026-11-22 replay produced three array maps with the same legacy Tune
coefficients, then an a1100 map using the new Python affine, nonlinear fit to
network 0. Both Tune blocks were fitted; the raw observation context selected
the final sweep block for the following timestream. The comparison artifacts
are in `run/map-comparison-152434/`, with `index.html` as the review entry point.

| Control | Selection | Map correlation | Difference RMS (mJy/beam) | Fitted offset change (Az, El; arcsec) |
| --- | --- | ---: | ---: | --- |
| Legacy/v3 engines, same legacy models | Original APT, a1100 | 0.99858 | 7.929 | +0.017, −0.001 |
| Legacy/v3 engines, same legacy models | Original APT, a1400 | 0.98784 | 12.274 | −0.050, −0.143 |
| Legacy/v3 engines, same legacy models | Original APT, a2000 | 0.99999 | 0.342 | +0.011, −0.003 |
| Legacy/v3 engines, same legacy models | Same 346 singlets, constant weights | 1.00000 | 6.7 × 10⁻¹² | 0.000, 0.000 |
| Legacy/Python models, same v3 engine | Same 346 singlets, constant weights | 0.92896 | 22.211 | +0.004, −0.269 |

For the model-change control, the fitted peak changes from 541.56 to 586.93
mJy/beam (+8.4%). The fitted widths change from 5.81 × 6.68 to 5.70 × 6.85
arcsec. The difference image includes structure away from the source. These
are measured changes with a fixed legacy flux conversion, not a demonstrated
improvement in sensitivity or absolute flux accuracy. The shared-selection
engine control agrees to numerical precision, so its engine change does not
explain this model-change residual.

The legacy executable is the available local v4.0.0 development build, not a
reconstruction of the exact production executable that made the archived v2
maps. It required one documented packet-counter bounds correction before it
could read the observation. The correction was applied to an isolated build;
the original checkout and executable remain unchanged. The report retains
the patch, build commands, configuration and source hashes, logs, FITS, PPTs,
model index and stub index. Archived maps are a separate historical reference.

### Full acquired-network reduction, by array

The current performance comparison uses the corrected gap/PCA implementation
and explicit numerical/physical sample acceptance, described below after input
preparation. The original maps, coordinate experiment and strict-subset PCA
trace are retained as the earlier investigation, not current performance.

The full 152434 recipe uses all locally available acquisitions: networks
0–5, 7–9, 11 and 12. Networks 6 and 10 are absent. Its manifest has one row per
network, with `interface`, `tune` and `timestream` paths. Fit all Tune files in
one ordinary KIDs invocation using the affine/nonlinear configuration above:

```python
import json
import subprocess

manifest = json.load(open("inputs.json"))
subprocess.run(
    ["tolteca_kids", "--config_paths", "fit-affine.yaml", "--"]
    + [row["tune"] for row in manifest],
    check=True,
)
```

`DataProdOutput` writes one acquisition product containing all selected
interfaces and both sweep blocks. A group with insufficient independent samples
is retained in the fit plan with an explicit unattempted status. It does not
abort the network or receive fabricated coefficients. The channel report keeps
the usual no-model/unusable flags where no model is available.

Use the actual resulting model index to prepare the comparison:

```sh
python examples/tune_model_comparison/prepare_array_map_control.py \
  map-control/v3-legacy-model/config.yaml inputs.json \
  /comparison/dataprod_toltec/python-affine/152434-0-1/dp_raw_obs_proc/0001-models/index.yaml \
  full-array/prepared --dataprod-path /comparison/dataprod_toltec

/path/to/v3/citlali full-array/prepared/legacy-common/config.yaml
/path/to/v3/citlali full-array/prepared/python-full/config.yaml

python examples/tune_model_comparison/array_map_report.py \
  full-array/prepared map-control/v2/redu00/152434/raw \
  full-array/prepared/legacy-common/redu00/152434/raw \
  full-array/prepared/python-full/redu00/152434/raw full-array/report \
  --legacy-current map-control/v3-legacy-model/redu00/152434/raw \
  --legacy-manifest map-control/control-manifest.json
```

Preparation uses the same raw-index, calibration-history, model-binding and
stub-artifact APIs as reduction input preparation. It installs the raw indexes
and target-acquisition model stubs beside the fit product in the shared
`python-affine` namespace. Each target network selects the final recorded Tune
block. Both Tune blocks remain available in the model product.

| Output | What it allows the reviewer to check |
| --- | --- |
| `prepared.json` | Exact model and stub indexes, resolved raw context and network counts |
| `channel-coverage.ecsv` | Every acquired channel, APT flag, model flags and mapping eligibility |
| `plan-coverage.ecsv` | Every planned group in both blocks, including insufficient-data groups |
| `group-coverage.ecsv`, `component-coverage.ecsv` | All exported groups and fitted components, including multiplicity and nonlinearity |
| `legacy-common/config.yaml` | Legacy coefficients with the same model-eligible channel set |
| `python-full/config.yaml` | Full Python model stubs with the original APT and science settings |
| `report/index.html`, `gap-maps.png`, `map-quality-summary.json` | Current policy comparison, all-array maps and sample-cost accounting |
| `a1100.png`, `a1400.png`, `a2000.png`, `metrics.json` | Historical original maps, fitted offsets/widths and contributor breakdown |
| `report/metrics.json`, `native-coverage.json` | Numerical comparisons and actual channel/chunk contributions from saved native weights and flags |
| `report/populations.json`, `population-ledger.ecsv` | Executed APT hashes and exact channel populations, including beammap flags, selected model groups, per-channel component counts, nonlinearity, fit quality and contributing chunk IDs |

The original comparison preserved filtering, cleaning, inverse-variance
weighting and flux conversion. Warning-only fits remain eligible. This differs
from the earlier 346-channel control, which deliberately used zero model flags
and constant weights. The common-APT control separates changed model eligibility
from changed model coefficients; subsequent science cuts may still retain
different channels, and the report shows those counts.

The historical APT does not contain accepted component-to-beam associations for
the newly fitted joint readout groups. Full group outputs therefore cannot enter
the map as newly identified detectors. Independent channel rows remain available
where that probe couples to one component, including probes within larger fit
groups and several probes observing the same resonator. Rows requiring joint
solving are excluded. The report distinguishes group multiplicity from the
number of components coupled to each probe. It does not assign component beam
positions by frequency proximity or count channel rows as independent optical
detectors.

The per-array report includes a common-covered 20–40 arcsec background annulus.
Its RMS includes correlated residual structure. Neither this RMS, a peak ratio,
nor the contributing-channel count establishes mapping speed without an
independent signal-transfer and noise assessment.

For the full local replay, final-block channel accounting is:

| Array | Acquired channels | Original APT excluded | Unusable flag | Joint output not mapped | Scalar-map eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1100 | 3,173 | 444 | 213 | 581 | 1,935 |
| a1400 | 1,368 | 318 | 126 | 261 | 663 |
| a2000 | 977 | 110 | 71 | 298 | 498 |

The four status columns partition the acquired channels: original APT exclusions
take precedence, followed by the unusable model flag, missing joint bindings,
then eligible channels. The unusable flag includes probe selection and solving
constraints as well as fitting failures; the sweep-quality audit below separates
these causes. Individual flag-bit counts can overlap and are retained separately.
The final Tune block contains 5,177 fitted components across all networks.
One group in network 4's first block has insufficient data and remains in the
plan; the other groups complete. Both network-0 ECSV reports are byte-identical
to the earlier fit. All 11 destination stubs decode to the exact selected source
reports, including their complete group metadata.

The actual full comparison tree is:

```text
run/map-comparison-152434/full-array/
├── inputs.json, input-hashes.json, sample-counts.json
├── fit-affine.yaml, fit-retry-command.json, fit-affine-retry.log
├── dataprod_toltec/python-affine/
│   ├── namespace.yaml
│   ├── 152434/dp_obs.yaml
│   ├── 152434-0-1/
│   │   ├── dp_raw_obs.yaml
│   │   └── dp_raw_obs_proc/0001-models/
│   │       ├── index.yaml
│   │       ├── array.zarr                 # all 11 networks, both blocks
│   │       └── artifacts/<interface>/blocks/{0,1}/
│   │           └── tolteca_kids.model_fit:ModelFit/kids_model.ecsv
│   └── 152434-0-2/
│       ├── dp_raw_obs.yaml
│       └── dp_raw_obs_proc/e154bf98-feeb-4d8f-88f9-552baacae078/
│           ├── index.yaml
│           └── <acquisition-stem>_kids_model_stub.nc  # one per network
├── prepared/
│   ├── prepared.json, common-apt.ecsv, *-coverage.ecsv
│   ├── legacy-common/config.yaml, redu01/152434/raw/
│   └── python-full/config.yaml, redu01/152434/raw/
└── report/
    ├── index.html, metrics.json, native-coverage.json
    ├── populations.json, population-ledger.ecsv
    ├── population-flow.{png,pdf}, model-coverage.{png,pdf}
    ├── model-exclusions.{png,pdf}, sweep-model-quality.{png,pdf,json}
    ├── sweep-model-{channels,responses}.ecsv, sweep-example-selection.json
    ├── a{1100,1400,2000}-{populations,sweep-example}.{png,pdf}
    └── a1100.{png,pdf}, a1400.{png,pdf}, a2000.{png,pdf}
```

The completed working `redu01` directories retain native FITS, PPT, diagnostic NetCDF and
Citlali indexes. They are comparison trials; this recipe does not promote them
to an accepted `dp_pointing` realization or change inventory tags. Model and
stub products use the ordinary portable data-product store.


#### Map the broader population with joint components

The broader 152434 comparison includes a corrected legacy-model reference on the same expanded APT eligibility. It starts from the same final Tune-block Python fits,
model stubs, beammap-good APT gate and scientific configuration. It adds supported
joint components without refitting or assigning new beam positions:

```text
ModelFit plan component
  → saved KidsFind finding
  → unique saved finding/channel match
  → observation APT row and detector UID
  → native group response at the recorded probe
  → filtering, PCA and map accumulation
```

`prepare_joint_map_control.py` verifies the plan's initial frequency, loss and
finding-group identity against the saved finding. It requires the stored match
to be one-to-one on both sides. The selected group must have sufficient acquired
probes and usable model flags. Multiple candidate owners of an APT identity are
excluded explicitly. No new nearest-frequency matching or residual cut is used.

| Array | Original APT good | Scalar eligible | Added joint components | Broader eligible |
| --- | ---: | ---: | ---: | ---: |
| a1100 | 2,729 | 1,935 | 545 | 2,480 |
| a1400 | 1,050 | 663 | 226 | 889 |
| a2000 | 867 | 498 | 274 | 772 |
| Total | 4,646 | 3,096 | 1,045 | 4,141 |

Of the original 1,140 rows excluded only for missing joint bindings, 1,045 now
have supported bindings and 95 remain unmatched. The original 410 fitting or
measurement-limit exclusions are unchanged. All 872 beammap-bad rows remain
excluded. These are **APT-row counts**. The component ledger includes findings
without a channel match and therefore has a different denominator.
Here “scalar control” denotes the earlier independent-channel solving route;
it does not imply that all those fitted groups are singlets. Both broader V3
policies return optical detuning from the same nonlinear model, with different
sample acceptance rather than different physical models.

The APT's `kids_readout` metadata declares joint component rows and explicit
`channel_rows` for unchanged scalar rows. A scalar entry records its position
within that network's APT slice, its existing UID and acquired channel. This
preserves repeated placeholder UIDs on excluded APT rows without treating them
as component identities. The native adapter verifies the recorded row/UID pair.
It requests only groups actually referenced by component rows, preserving the
complete model calibration. Joint rows consume the fitted group's response;
scalar rows keep the existing channel solution and acquired-frequency nearby-tone veto. Joint components use group-recovery validity instead of that scalar proximity cut.

Beam coordinates, flux factors and sensitivity columns are unchanged. Newly
admitted components use the acquired probe as their detuning reference, while
scalar-control probe frequencies remain unchanged. The standard median-weight outlier reset is population dependent: three scalar a1400 weights change by about 2%, although the APT sensitivity values and weighting settings are unchanged. The audit records the exact differences. The native checks compare actual
scalar responses and masks, selected group identities and component projection.
Underlying joint-solving flags agree across the policies. When the bounded
joint solver returns no finite state, numerical projection adds a nonfinite
flag; physical acceptance already rejects the domain flag. The audit verifies
that neither policy maps those samples. Numerical acceptance does not expand
the joint solver's search domain.

The completed full-integration runs are `numerical/redu02`, `physical/redu02`
and `legacy/redu00`. Of the 1,045 added components, **1,007 contribute valid
samples** under either V3 policy; 38 remain entirely rejected. Numerical V3
has 4,061 contributors in total, and physical V3 has 4,057. Failed or interrupted
earlier attempts retain receipts and do not enter the comparison.

| Array | Added contributors | Broader V3 contributors, numerical / physical | Legacy RMS | V3 numerical RMS | V3 physical RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1100 | 519 | 2,421 / 2,419 | 57.57 | 38.44 | 35.02 |
| a1400 | 215 | 869 / 867 | 43.12 | 47.40 | 44.85 |
| a2000 | 273 | 771 / 771 | 8.40 | 9.01 | 7.94 |

RMS is in mJy/beam on the same 952 covered pixels in the 20–40 arcsec annulus.
Adding joint components reduces physical-policy RMS by 15.9%, 16.6% and 17.8%
relative to the corresponding scalar-only controls. Relative to the broader
legacy reference, physical V3 improves a1100 and a2000 while a1400 remains
4.0% higher. These changes are map-fluctuation measurements, not validated
mapping-speed gains. The broader legacy and both V3 policies have bitwise
identical per-row weights over the complete integration; the common valid
scalar precleaning signals are also unchanged from the scalar controls.

```bash
base=run/map-comparison-152434/full-array
python examples/tune_model_comparison/build_gap_control.py \
  run/gap-handling-native run/joint-map-native --patch-prefix joint-map
python examples/tune_model_comparison/prepare_joint_map_control.py \
  "$base" "$base/joint-control"
for policy in legacy numerical physical; do
  engine=run/joint-map-native/build/citlali/citlali
  if [ "$policy" = legacy ]; then
    engine=run/gap-handling-native/build/citlali/citlali
  fi
  python examples/tune_model_comparison/run_xprime_control.py \
    "$engine" \
    "$base/joint-control/$policy.yaml" "$base/joint-control/$policy" \
    --no-variance-cuts
done
python examples/tune_model_comparison/joint_map_report.py "$base"
```

The preparation command refuses an existing output directory. The report
requires exactly one successful realization per policy and ignores failed
attempts only when their execution receipts explicitly record failure. It never
silently chooses the newest successful `reduNN`.

```text
joint-control/
├── joint-apt.ecsv
├── selection.json
├── component-bindings.ecsv       # every fitted multiplet component
├── component-bindings.json
├── channel-disposition.ecsv      # every original acquisition/APT row
├── joint-contributors.ecsv       # added component identity, a_nl, accepted samples
├── numerical.yaml
├── physical.yaml
├── legacy.yaml
├── legacy-apt.ecsv               # same eligibility; legacy channel responses
├── legacy/reduNN/
├── numerical/reduNN/             # exact config, execution receipt, FITS, RTC/PTC
└── physical/reduNN/              # same fitted calibration; different acceptance
report/
├── joint-control.json            # counts, policy checks, weights and hashes
├── joint-map-summary.{png,pdf}    # scalar vs broader at each policy
├── joint-legacy-maps.{png,pdf}    # legacy vs V3 on the broader population
└── joint-maps.{png,pdf}           # all arrays, shared intensity scale per array
```

The broad maps remain comparison trials. An inherited acquisition-to-APT
correspondence is not a new component beam/flux calibration. Joint covariance
is retained, but the map accumulator uses per-row weights; correlated
components are not independent sensitivity gains. The report distinguishes
eligibility, actual contributors and accepted samples. Annulus RMS is not
validated sensitivity or mapping speed, and the earlier measured
Python/native parity discrepancy remains a release limitation.

#### Scalar controls: numerical versus physical acceptance

The maintained native correction is in `native/gap-citlali.patch` and
`native/gap-kidscpp.patch` beside the comparison recipes. These patches apply
to the pinned comparison sources; they do not overwrite the earlier executable
or its evidence. `build_gap_control.py` builds an isolated installation.

There are two independent contracts:

| Operation | Numerical data | Map eligibility |
| --- | --- | --- |
| Solve | Retain diagnostics; reject hard failures | Apply the recorded acceptance policy |
| Fill gaps | Endpoint baseline + observed network fluctuations + reproducible noise | Preserve the rejection mask |
| Filter | Convolve the filled series | Reject every output whose FIR support touches a gap |
| PCA | Estimate covariance and mode amplitudes from valid samples | Reject times with insufficient mode support |
| Map | Accumulate valid cleaned samples with the APT weights | Filled samples never receive weight |

The shared gap implementation follows the despiker construction and handles
endpoint gaps and fully missing channels explicitly. It runs even when spike
detection is disabled. Filling with zero is no longer the gap preparation
policy. The PCA projection uses valid-sample least squares; it accounts for
missing basis rows rather than just zeroing their contributions.

The native configuration now exposes:

```yaml
kids:
  solver:
    sample_acceptance: numerical  # default: physical
```

`physical` rejects finite domain/stability violations. `numerical` retains those
finite algebraic results with their diagnostic bits, making the model comparison
less dependent on newly introduced physical selection cuts. Invalid models,
singular or nonfinite responses and failed joint recovery remain rejected.
Neither policy refits the sweep or changes the nonlinear correction. Saved
`kids_sample_flag` metadata records the acceptance policy and exclusion bitmask;
the saved `flags` array is the actual map mask after filtering and other cuts.

Both policies use the same nonlinear V3 model and optical-detuning output;
this is an acceptance comparison, not numerical versus physical modeling.

All three corrected runs use the original `prepared/common-apt.ecsv`, including all
1,935 intended a1100 rows. “Full intended” means all 3,096 initially eligible
channel/APT rows, not all 4,646 beammap-good channels. The missing component/APT
bindings and unusable models remain excluded in these scalar controls; the broader runs above restore supported component bindings. There is no selection based on a channel's sample
history. The legacy numerical control, V3 physical policy and V3 numerical
policy use identical per-channel weights. Effective coverage can still differ
with sample acceptance; the report measures this rather than assuming it equal.

The completed **numerical-policy pair** has identical sample masks, map weights
and coverage, as well as per-channel weights. It contributes 1,902 / 654 / 498
channel/APT rows in a1100 / a1400 / a2000. The same 33 a1100 nearby-tone exclusions
remain; the 45 intermittently rejected V3 channels are no longer removed by
physical acceptance. Their finite solved values are used, not filled values.

| Array | Legacy numerical | V3 numerical | V3 physical |
| --- | ---: | ---: | ---: |
| a1100 | 63.37 | 45.69 | 41.62 |
| a1400 | 47.37 | 54.31 | 53.75 |
| a2000 | 10.22 | 10.96 | 9.66 |

These are annulus RMS values in mJy/beam. The numerical pair is the direct
comparison with matched acceptance. The physical column retains additional
rejections, so its effective coverage differs. V3 improves this a1100 metric;
a1400 and a2000 do not improve under numerical acceptance. Compared with
numerical acceptance, physical acceptance lowers RMS by 8.91% / 1.02% / 11.90%
while retaining 99.491% / 98.879% / 99.595% of accepted channel-samples.
This includes a change of the cleaning basis as well as final map acceptance. None of these values
alone establish astronomical signal transfer or sensitivity.

```bash
base=run/map-comparison-152434/full-array
python examples/tune_model_comparison/build_gap_control.py \
  run/sweep-anchored-solving run/gap-handling-native
native=run/gap-handling-native/build/citlali/citlali
python examples/tune_model_comparison/gap_control.py prepare "$base"
for model in legacy v3-physical v3-numerical; do
  python examples/tune_model_comparison/run_xprime_control.py "$native" \
    "$base/gap-control/$model/source.yaml" \
    "$base/gap-control/$model" --no-variance-cuts
done
python examples/tune_model_comparison/gap_control.py audit "$base"
python examples/tune_model_comparison/gap_control_report.py "$base"
```

`gap-control.json` records the accepted population, exact weights check,
unchanged diagnostic flags across V3 policies, map coverage comparisons and
masked native PCA replay. `gap-maps` shows all three arrays; `gap-trajectories`
shows the six a1100 network means without including filled samples. Map RMS
remains a fluctuation metric, not a validated sensitivity or mapping speed.


`map-quality-summary` for the scalar controls reads only the
corrected `gap-control.json` audit. It plots RMS and the additional rejected
sample fraction, rather than the superseded original map excess. The summary
records the audit hash; the PDF build rejects a stale summary. Earlier maps
and contributor counts are grouped in the investigation appendix.

#### Sweep model quality versus map eligibility

The sweep-model audit retains `population-flow.png` and `model-coverage.png` for the original scalar selection, followed by a paired sweep-quality plot. The report now leads with the broader joint maps. Per-array stacked bars show actual
contributors by singlet/multiplet group membership, fitted nonlinearity and
quality flags. Exact counts remain in expandable HTML tables and ECSV ledgers.
Every figure has a vector PDF counterpart.

**Scalar-map eligibility is not a fit-success count.** Among 4,646 original
beammap-good channel rows, 4,495 (96.7%) have a fitted response and 4,409 (94.9%)
have at least one optimizer-converged response. Only 3,096 (66.6%) pass the
scalar-map gate. Its 1,140 joint-required exclusions retain fitted models. The broader comparison above now admits 1,045 using saved component/channel/APT correspondences, leaving 95 without a unique supported binding.

`model-exclusions.png` separates the other 410 unusable flags into 151 without
a fitted response, 104 selected optimizer failures, 28 probes outside fitted
windows and 127 converged fits with insufficient simultaneous measurements
for independent component solving. Missing fits or probe-window gaps are not
evidence of wrong APT associations. These are first-exclusion categories, not
independent flag counts or a claim that every converged fit is adequate.

`examples/tune_model_comparison/sweep_model_comparison.py` evaluates the saved
legacy `tune.txt` equation and the full saved v3 M-channel/N-resonator prediction
at identical finite data-valid measured frequencies in each response window.
The v3 prediction includes nonlinearity and the affine baseline. The audit checks
IQ accumulation and verifies the fitting samples against `SweepReduce` data;
it applies no refit, gain adjustment or frequency shift. The legacy files are
the original calibration reports, not newly refitted legacy models.

The paired error is RMS complex-IQ residual divided by RMS measured complex-IQ
excursion about the window mean. This avoids diluting the error by the large IQ
baseline. `sweep-model-quality.png` shows paired residuals and their cumulative
distributions for each array, including map-ineligible channels. Shape marks
one versus several resonators; color marks map status. The selected group is
fixed by the saved probe binding, not selected for its residual.

For a1100, a1400 and a2000 respectively, v3 has lower residuals for
98.7%, 97.7% and 97.9% of the 2,691/966/810 paired channels. Median within-pair
v3/legacy ratios are 0.187, 0.654 and 0.858. This is an **in-sample description
test**, with more parameters in v3; it does not establish held-out performance,
unique timestream solutions or improved optical sensitivity. The poorer maps
were part of the gap/PCA investigation, not a current performance result despite the generally
better sweep descriptions.

Each array also has a measured-sweep overlay for an M=2, N=2 group, including
both channels and fitted `a_nl`. The example is nearest the median paired
residual ratio among this class of joint-output exclusions, rather than a best
fit. `sweep-example-selection.json` records its identity and selection rule.
The report retains all alternative group/channel predictions in
`sweep-model-responses.ecsv`, and all missing or unselected channels in
`sweep-model-channels.ecsv`. `sweep-model-quality.json` records the method,
original legacy-model hashes, exact counts and summary statistics.

#### Follow a singlet and a joint group into the measured timestream

The report now includes network-0 group 20 (channel 297, one resonator) and
group 404 (channels 99 and 100, two resonators). Selection uses **sweep evidence
before inspecting timestream noise**: all member channels must be APT-good,
converged, have no group warning and an excursion-normalized residual below
0.05. Choose the group closest to the median maximum channel residual within
each M=N=1 or M=N=2 pool. This leaves 329 singlet and 25 joint candidates.
The selection and complete model coefficients are saved in
`timestream-examples/selection.json`.

`array_timestream_examples.py` reads the saved final Tune block and calls the
maintained `joint_models` / `joint_solver` machinery, including the same nonlinear
affine model for both cases. It reads all simultaneous I/Q channels and checks
the recorded probe frequencies, packet continuity and accumulation ratio.
Both cases recover all 7,864 measured samples without flagged failures. This
establishes numerical validity for these examples, not independent proof of
physical uniqueness or joint-component/APT identity.

Each component is displayed at the recorded probe nearest its fitted resonance.
The legacy `tune.txt` uses `f_in` 6,457.47 Hz below that recorded probe. The
comparison preserves this as-used frequency convention and records the offset;
it does not silently correct the legacy model. Legacy x and r are effective
channel coordinates, not separate solutions for the resonators in a blend.
The new joint signals are diagnostic products, **not contributors to the saved
maps**, whose component/APT bindings are still missing.

The figures show the same measured sweep and model predictions, followed by
median-subtracted x′, x_opt and r timestreams, one-sided PSDs and detuning–loss
coherence. Welch estimation uses the longest common contiguous valid interval
without closing sample gaps, 1,024-sample Hann windows, 50% overlap and constant
detrending. The shaded 5–20 Hz band summarizes measured fluctuations before
decorrelation or optical calibration. Joint figures additionally show component
frequency coherence and the signed terms D² PSD(x′), μ² PSD(r) and
2Dμ Re C(x′,r), evaluated at the median recovered state. The fixed-b assumption
remains explicit; x_opt is not an independent instantaneous-nonlinearity estimate.
Smaller raw-coordinate PSD does not establish better optical sensitivity.

The main `array_map_report.py` command regenerates these examples and plots.
They can also be reproduced separately without rerunning fitting or map making:

```bash
PYTHONPATH=src python examples/tune_model_comparison/array_timestream_examples.py \
  run/map-comparison-152434/full-array/prepared \
  run/map-comparison-152434/control-manifest.json \
  run/map-comparison-152434/full-array/report
PYTHONPATH=src python examples/tune_model_comparison/array_example_plots.py \
  run/map-comparison-152434/full-array/prepared \
  run/map-comparison-152434/control-manifest.json \
  run/map-comparison-152434/full-array/report
```

`summary.json` records inputs and hashes, frequency conventions, validity flags,
rank, conditioning and band ASD. `*-solved.npz` retains every sample and flag;
`*-comparison.npz` retains the common time series and spectral matrices;
`*-spectra.nc` retains signed coordinate spectra and noise propagation.
Each figure is available as PNG and vector PDF. The handoff PDF freezes these
records in its evidence directory.

#### Interpreting performance

These pointing trials have not established matched astronomical signal transfer
or noise depth versus integration time and area. Annulus RMS includes correlated
structure and source wings. The old formal inverse-weight uncertainty was not a
second RMS measurement and did not capture the gap/PCA contamination. The current
policy comparison reports the actual accepted sample cost instead. Do not use
raw detector PSD, channel count or 1/RMS² as validated mapping speed.

#### Earlier investigation: superseded map controls

The following maps and contributor counts use the earlier processing. They
explain why the gap correction was needed and are not the current numerical
versus physical policy results. The current RMS values and accepted samples are
reported above. The separate model-quality and solved-example evidence remains
applicable because the acceptance-policy change does not refit the models.

##### Original full-array results

The complete integration finishes in 12 chunks and writes maps and a native PPT
for all three arrays. These are comparison results, not accepted pointing
calibration. In particular, the a1100 and a1400 maps have substantially larger
background residuals than the matched legacy control. The a1400 fitted major
width reaches the fitter's 12.6 arcsec default upper bound; its fitted pointing
and width therefore need review rather than automatic adoption.

| Array | Legacy contributing channels | Python contributing channels | Peak ratio to common legacy | Common-pixel correlation | Difference RMS (mJy/beam) |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1100 | 2,638 | 1,882 | 0.981 | 0.177 | 198.84 |
| a1400 | 1,034 | 648 | 0.812 | 0.387 | 136.75 |
| a2000 | 865 | 498 | 1.120 | 0.961 | 20.48 |

The peak ratios and pixel comparisons use legacy models restricted to the same
model-eligible channel set, with the same v3 engine. The contributing counts
refer to the full legacy and Python reductions and include channels contributing
in any chunk. The report also gives counts contributing in every chunk and the
mean count per chunk. On the common-covered 20–40 arcsec annulus, background RMS
changes from 70.78 to 139.88, 51.36 to 130.43, and 10.22 to 15.46 mJy/beam,
respectively. These include residual structure and must not be interpreted as
noise-only sensitivity or mapping speed. The full array result does not support
a blanket improvement claim based on the earlier network-0 test.

##### Coordinate control: map the same V3 fits as x′

The updated comparison includes full-integration x′ maps for all three arrays.
The isolated diagnostic build changes only the returned scalar coordinate:
it computes x_opt and its validity checks, then returns the pre-correction x′
for accepted physical-model samples. Loss values and solver flags are retained.
The fitted nonlinearity and model stubs are unchanged. This does **not** set
a_nl or b to zero, refit the sweeps, or add blended-component mapping.

The 20–40 arcsec background RMS in mJy/beam is:

| Array | Legacy common eligibility | V3 x′ | V3 x_opt |
| --- | ---: | ---: | ---: |
| a1100 | 70.78 | 138.36 | 139.88 |
| a1400 | 51.36 | 75.98 | 130.43 |
| a2000 | 10.22 | 14.96 | 15.46 |

**In this pre-correction experiment, the a1100 excess remained in x′.** The nonlinear coordinate correction cannot
by itself explain the large regression. The a1400 residual structure depends
strongly on that coordinate choice; a2000 changes little. The HTML report and
handoff PDF put the RMS plot and three-array maps beside the existing summary.

Changing the coordinate also changes downstream variance-cut decisions under
the recorded settings. A second pair disables only the lower and upper
inverse-variance cuts in both raw and processed time chunks. Saved contribution
decisions, flagged fractions and per-channel weights then match exactly.
Its x′ / x_opt RMS values are 119.02 / 127.68 for a1100, 69.63 / 124.66 for
a1400, and 14.95 / 15.45 for a2000. The shared contributing channel/APT rows are
1,900 / 652 / 498. PCA modes are still recomputed per coordinate; equal saved
flagged fractions alone do not prove bitwise identity of final sample masks.

All runs retain the same APT flux conversion and sensitivity values. Thus
the lower x′ RMS is not a validated sensitivity gain or a decision to use x′
as the science coordinate. Signal transfer, the fixed-b assumption and the
remaining a1100 residuals motivated the subsequent PCA investigation. Each comparison uses shared,
integer-aligned sky pixels without interpolation or signal rescaling.

The following commands use the completed local comparison and its pinned
native build. The builder copies the native source and archive, changes one
scalar output projection, and links a private executable. It verifies that
the original source, archive and executable hashes remain unchanged. Controlled
affine, delay and zero-b trajectories cover 18 samples, including rejected
states; all seven valid x′ values and all loss values/flags pass the checks.

```bash
base=run/map-comparison-152434/full-array
native=run/sweep-anchored-solving
python examples/tune_model_comparison/build_xprime_control.py \
  "$native" "$base/xprime-control/native"

python examples/tune_model_comparison/run_xprime_control.py \
  "$base/xprime-control/native/citlali-xprime" \
  "$base/prepared/python-full/config.yaml" "$base/xprime-control/standard"
python examples/tune_model_comparison/run_xprime_control.py \
  "$native/build/citlali/citlali" \
  "$base/prepared/python-full/config.yaml" \
  "$base/xprime-control/no-variance-cuts-opt" --no-variance-cuts
python examples/tune_model_comparison/run_xprime_control.py \
  "$base/xprime-control/native/citlali-xprime" \
  "$base/prepared/python-full/config.yaml" \
  "$base/xprime-control/no-variance-cuts-prime" --no-variance-cuts

python examples/tune_model_comparison/array_coordinate_control.py \
  "$base" "$base/report" --update-report
python examples/tune_model_comparison/handoff/build_pdf.py
```

The recorded controls are explicitly `redu00` in the three new trial folders;
the existing legacy-common and x_opt references are `redu01`. Repeating a
native invocation creates another `reduNN`; the report never silently switches
to the latest directory. Review any changed trial paths before comparing a new
realization. Each new `redu00/execution.json` records binary, config and input
hashes. `xprime-control/native/build.json` records compiler/link commands,
fixture results and the original-file checks; `xprime-output.patch` is the
isolated change. `report/xprime-control.json` holds the map hashes, exact RMS
values and contribution audit. The HTML renderer also includes this saved
control when rebuilding the full report. This is a diagnostic experiment,
not a production coordinate-selection option.

##### Trace the a1100 excess through PCA

The next control compares the v2 `tune.txt` response and V3 `x_opt` through the
**same pinned Citlali engine**. It freezes 1,855 a1100 channel/APT rows that are
finite and unflagged at every saved sample in both model paths. The six network
counts are 357, 276, 292, 329, 297 and 304. This is a validity intersection,
not a new residual, nonlinearity or noise cut: 33 of the original 1,935 eligible
channels fail this condition in legacy, and 80 in V3, including those same 33.

Both matched native reductions have exactly the same 3,628 samples, sample
masks, timestamps, pointing coordinates and per-channel weights. Map coverage
and weights also agree exactly. Their common 20–40 arcsec annulus RMS is
**55.13 mJy/beam for legacy and 40.76 for V3**. The earlier V3 excess is absent
in this control. These maps retain the existing APT flux calibration; this
result does not by itself validate sensitivity or astronomical signal transfer.

The trace identifies a native **PCA flag-handling inconsistency**. Covariance
estimation masks flagged samples, but `Cleaner::remove_eig_values` computes
mode amplitudes using the entire stored signal matrix. A partially flagged
probe can therefore contribute its stored rejected values to a mode, which is
then subtracted from valid probes. Legacy has no partially flagged probes in
these captures; V3 does. This is not a failure of the new sweep model to fit
the good detectors.

Using the saved basis, the replay reproduces native post-PCA values within
1.22e-9 mJy/beam. The same always-valid target channels have bitwise identical
pre-PCA values in the broader and strict captures. The diagnostic then compares
the native projection with (a) zeroed flagged contributions and (b) mode
amplitudes estimated by least squares on valid probes. It retains the original
captured basis and the same target channels and weights. The second replay
reduces the V3 weighted per-channel residual RMS from 1,651 to 672 mJy/beam in
network 1 and from 2,538 to 642 in network 4. These are **timestream fluctuation
RMS values**, not map RMS or noise-only sensitivity. The report shows the
weighted network trajectories and all six networks.

The broader result includes both this leakage and model-dependent cleaning
bases. On the strict population, the report also compares pre/post-PCA spectra,
pre-PCA regression gains, and exchanged bases. The input-difference and
basis-change contributions have a substantial signed cross term; their powers
must not be added as independent causes. No empirical regression gain is
applied to the maps. The native algorithm is unchanged by this investigation.

Reproduce the captures and matched runs from the existing local comparison:

```bash
base=run/map-comparison-152434/full-array
native=run/sweep-anchored-solving/build/citlali/citlali
python examples/tune_model_comparison/pca_trace.py prepare-capture "$base"
for model in legacy v3; do
  python examples/tune_model_comparison/run_xprime_control.py "$native" \
    "$base/pca-control/capture/$model/source.yaml" \
    "$base/pca-control/capture/$model" --no-variance-cuts
done
python examples/tune_model_comparison/pca_trace.py prepare-matched "$base"
for model in legacy v3; do
  python examples/tune_model_comparison/run_xprime_control.py "$native" \
    "$base/pca-control/matched/$model/source.yaml" \
    "$base/pca-control/matched/$model" --no-variance-cuts
done
python examples/tune_model_comparison/pca_trace_analysis.py "$base" "$base/report"
python examples/tune_model_comparison/pca_flag_projection.py "$base" "$base/report"
python examples/tune_model_comparison/pca_trace_report.py "$base/report"
python examples/tune_model_comparison/handoff/build_pdf.py
```

The recipe explicitly reads `redu00` in each new folder. Reruns create later
realizations; the report does not silently adopt the newest one. Each execution
retains the native config, input hashes and log. `channel-selection.ecsv` keeps
all excluded rows, while `matched-apt.ecsv` changes only flags. Other arrays
retain their common eligibility; the new analysis is restricted to a1100.

Native RTC output is after filtering, downsampling and APT flux calibration,
immediately before PCA. PTC follows mean subtraction, five-mode network PCA and
weighting. Both variance cuts are disabled. Spectra use a Hann window, 256
samples and 128 overlap within each chunk, and include source and sky signal.
Saved `scan_indices` have incorrect starts when the chunk lengths differ. The
reader reconstructs append-order bounds from `raw_scan_indices` and the
downsampling factor, and verifies the complete native PCA replay. It does not
change the acquisition, sample values or native map computation.

The report retains `pca-trace.json`, `pca-flag-projection.json`, per-channel/chunk
ECSV records and NPZ spectra/trajectories, with PNG and vector PDF figures.
The strict 1,855-channel subset is a diagnostic control, not a proposed
permanent detector exclusion.

##### Original contributor audit

The full legacy and Python-model runs use the **same beammap APT**,
`apt_152434_matched.ecsv`, with SHA-256
`5f9a99b50829f2d44746b40d16503e4528f2375fc5d71b345bece16216d22831`.
Of its 5,518 rows, 4,646 have `flag == 0`. Neither reduction admits an APT-bad
row. The common-eligibility legacy control changes only the `flag` column,
leaving 3,096 eligible rows; all other values, units and table metadata are
unchanged. This is a restriction of the existing APT, not a new beammap
calibration or a reassessment of optical detector quality using v3 models.

“Contributing” means a finite positive saved map weight and a finite
`flagged_frac < 1` in at least one of the 12 chunks. It counts acquired
channel/APT rows, not unique physical detectors, not all-sample validity, and
not every fitted resonator component.


The final-block Python fit provides consistent diagnostic labels for **both**
the legacy and Python contributor populations in the report plots. These labels describe the
same raw channels, not the fitting or solving method used by legacy Citlali.
Group multiplicity and probe blending are different: a probe in a multi-resonator
fit group can couple to only one fitted component.


The 90 contributing Python channels in multi-resonator groups each have exactly
one nonzero component coupling in the exported model. Their independent ECSV
row contains that component's frequency and nonlinearity, verified against the
full group payload. This is not full blended-component map recovery. The 83
contributing Python channels in multi-probe singlet groups also must not be
counted as 83 independent detector identities.

For a single-component channel, the nonlinearity bins use its Python fitted
`a_nl`. Blended channels have multiple values and remain a separate category.
Legacy gainlintrend asymmetry `A` is not the same parameter. All Python fits
include nonlinearity; these diagnostic intervals do not define different
solvers or scientific acceptance thresholds.


Fit quality is a separate axis. No residual-fraction or conditioning threshold
was enabled. The plotted states partition each population, taking unusable
before joint-required before warning-only. These are **Python fit diagnostics**
even for legacy contributors; they are not retrospective legacy failure flags.


Among Python contributors, 368/278/209 channels respectively carry parameter
bound warnings and 14/7/0 carry initialization-fallback warnings; bits can
overlap. Zero flags do not establish scientific adequacy. The HTML report also
shows saved residual fractions and reduced chi-square per unique selected
group. Residual fraction is normalized by total IQ amplitude, including its
baseline; reduced chi-square assumes independent repeat means and quadratures.
Neither is a calibrated measure of detector optical performance.

The common-eligibility legacy and Python maps share 1,871/643/498 contributors.
An additional 12/4/0 occur only in that legacy control and 11/5/0 only in Python.
Thus even the common-eligibility comparison does not hold the final channel
population, sample validity or weights fixed. It measures the combined effect
of the model and downstream cuts. The channel ledger records the exact
network/channel identity and contributing chunk IDs for all three runs.

##### Original native implementation checks

Two native changes were required for a practical complete replay. Their patches
are retained in `examples/tune_model_comparison/native/`:

* `citlali.patch`, based on `e426b15a527d55b1e8f9880dc057ea76dd29fd18`,
  rejects channels with fewer than two valid samples before PCA and guards pairs
  without enough shared samples to estimate covariance. The first attempt failed
  in network 12, chunk 5: channel 250 retained one valid sample, producing 449
  nonfinite covariance entries. The captured data are a regression fixture.
* `kidscpp.patch`, based on `81f531996225f28873cef1f38cf5544048ed931d`,
  lets a consumer omit unused joint outputs. Citlali requests them whenever its
  APT has the necessary component bindings. The standalone solver still defaults
  to complete output. Calibration metadata and independent-channel results and
  flags remain identical; joint-required rows are never substituted by singlets.

Apply the patches to those native source revisions and rebuild kidscpp before
Citlali. `native-build.json` records the patches, source and executable hashes,
and checks. Both the original and optimized diagnostic runs produced identical
pre-cleaning samples and flags. All seven previously completed control FITS
files remain numerically identical after the fixes, including the 346-channel
Python-model map. The final full reduction finishes in about 25 seconds locally;
the abandoned run spent roughly seven minutes per chunk computing joint outputs
that its APT could not use.

The original successful full-array trial is `prepared/python-full/redu01`; `redu00`
retains the failed attempt. Each completed trial has its executed config, log
and `execution.json`. The historical appendix links to that trial; current policy results are under `gap-control/`.
The native tests cover the covariance failure, preservation of channel outputs
across all eleven model stubs, and physical solving. An additional older
joint-map fixture still fails before solving because its 20261110 stub predates
the common ECSV/stub contract; its failure is retained in `native-build.json`.
This does not resolve the separate Python/native joint-parity limits described
above, nor provide missing component-to-beam associations for this observation.
