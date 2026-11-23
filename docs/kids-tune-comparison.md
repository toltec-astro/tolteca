# Compare a Tune model through a saved timestream product

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
