# Tune-model comparison release

Start with [the six-page comparison summary](tune-model-comparison-summary.pdf).
It presents the completed 152434 reduction first; controls and unresolved
questions are in its appendices. [AGENT_HANDOFF.md](AGENT_HANDOFF.md) is the
entry point for an agent helping with a different observation.

This release uses **Python ModelFit** for the detailed nonlinear group fits.
Kidscpp is the native model reader/solver used by Citlali; its fast fitting
backend is not required. The native changes target the coordinated Spack-based
Citlali branch, not a separate newer refactor. Keep these repositories together.

## 1. Fetch and install

```sh
git clone --branch codex/grant-tune-comparison-20261120 \
  https://github.com/toltec-astro/tolteca.git grant-tune
cd grant-tune
git rev-parse HEAD
export TUNE_SOURCE="$PWD"
export TUNE_STACK="$PWD/../grant-tune-stack"
python3 examples/tune_model_comparison/handoff/bootstrap.py \
  "$TUNE_STACK" --spack-env
```

Requires Git and Python 3.13 (or `uv`, which can obtain it). The bootstrap
creates a new isolated directory, fetches the exact source revisions in
[revisions.json](revisions.json), installs Tolteca and Tollan with
[Python constraints](python-constraints.txt), and checks imports, the KIDs CLI
and dependency consistency. `uv --no-sources` deliberately bypasses the
development workspace's source mapping. No existing environment is modified.

`--spack-env` **prepares but does not build the native stack**. With a current
Spack 1.x installation supporting the supplied repository API v2 namespaces:

```sh
spack -e "$TUNE_STACK/spack" concretize
spack -e "$TUNE_STACK/spack" install --test=root
export TUNE_CITLALI="$TUNE_STACK/spack-view/bin/citlali"
"$TUNE_CITLALI" --version
```

Select your site's C++ compiler with bootstrap's `--spack-compiler`, for example
`gcc@14.2.0`, if needed. The generated environment uses develop paths for all
four pinned native projects, so package recipes cannot silently replace them
with their default release commits. Save the resulting `spack.lock` and
compiler details with the run. This Linux/Spack installation route is supplied
for the recipient; it has not been executed as a fresh Linux build here.

If suitable native dependencies are already installed, bootstrap can instead
build all four projects with CMake (CMake >=3.25 and a modern C++23 compiler):

```sh
python3 examples/tune_model_comparison/handoff/bootstrap.py \
  /absolute/new-stack --deps-prefix /absolute/dependencies-prefix \
  --cxx /absolute/compiler/bin/clang++ --jobs 2
```

This builds the pinned sources, not an existing binary. External CMake
dependencies include Eigen, Ceres, FFTW, NetCDF C/C++, Boost, Spectra,
CFITSIO/CCfits, yaml-cpp, fmt/spdlog, GoogleTest and the Tula wrapper packages
declared in the bundled Spack recipes. `--deps-prefix` may be repeated. The
compiled CLI is `<new-stack>/install/bin/citlali`. `--python-only`,
`--native-only` and `--fetch-only` allow installation in stages, each into a
fresh destination. All commands and failures are retained in `bootstrap.json`
and `logs/`.

## 2. Supply one observation and its recorded calibration

Prepare a working native Citlali YAML for the observation with:

- one `inputs` acquisition entry and explicit `filepath` values;
- raw KIDs files for the selected networks, plus telescope/HWPR files as needed;
- the original **observation-matched APT**, including acquisition channel IDs,
  detector UIDs, beammap flags, beam/flux/sensitivity columns;
- the required pointing and photometry calibration, and your map settings.

Relative file paths are resolved against that YAML's directory. Keep its
legacy model selection if you want a legacy reference run. Use the recorded
Tune/target sweep from the acquisition headers, which can belong to another
obsnum. Supply exactly one calibration file per selected network. Do not use an
unmatched global beammap APT or a previous comparison's restricted APT.
`--apt` can explicitly select the original observation-matched table.

```sh
"$TUNE_STACK/venv/bin/python" \
  "$TUNE_SOURCE/examples/tune_model_comparison/handoff/run_observation.py" \
  --config /data/my-observation/citlali.yaml \
  --obsnum 152434 \
  --tune /data/my-calibration/toltec*_tune.nc \
  --apt /data/my-observation/apt_matched.ecsv \
  --output /results/152434-v3 \
  --workers 4 --citlali "$TUNE_CITLALI" --also-legacy
```

Replace **all** observation-specific paths and the optional obsnum assertion.
The shell expands the Tune list; header identities determine the correspondence.
The recipe never chooses a calibration by nearest filename or by an assumed
scan number. The raw-context machinery resolves the recorded sweep/block.

By default it fits both Tune blocks using the maintained `tolteca_kids` CLI,
with the Python nonlinear affine group model, then prepares stubs and maps the
target. Use `--gain-family delay` for the affine-plus-delay comparison. A whole
network fit and full-array mapping can take substantial time; follow
`logs/fit.log` and `maps/citlali.log`. Start with one network if checking the
installation, using an APT and input list restricted to that same network.

`--prepare-only` performs fitting/binding/materialization without running
Citlali. `--model-index /path/index.yaml` reuses an exact existing fit (including
its accessible artifact payloads). `--also-legacy` runs the original legacy
coefficients through the same corrected engine and broader APT eligibility;
it rejects an input configuration already selecting KIDs stubs. Telescope
correction stubs are preserved. The recipe supports solved `xs`/`rs` map
quantities; a component APT is not a raw I/Q map layout.

Physical sample acceptance is the default. `--acceptance numerical` is a
separate control; it does not expand the joint search domain. Both exclude
hard failures. Existing science settings and variance cuts are retained unless
you request `--no-variance-cuts`, the setting used in the matched-weight 152434
comparison. Use fresh output directories for different controls.

## 3. Inspect the generated products

```text
152434-v3/
  input-config.yaml             # resolved source config, before model replacement
  acquisitions.json             # exact network/calibration/target file list
  fit.yaml                     # generated when fitting is requested
  handoff-run.json              # context, product identities, hashes, outcome
  apt.ecsv                     # existing optical calibration + explicit joint binding
  legacy-apt.ecsv               # same eligibility, without V3 component metadata
  channels.ecsv                # every acquired APT row and its disposition
  components.json / .ecsv       # saved group/component/channel/UID correspondence
  citlali.yaml / legacy.yaml    # exact prepared native configurations
  logs/fit.log
  dataprod_toltec/comparison/
    <calibration-raw-uid>/dp_raw_obs_proc/<store-run>/index.yaml
    <target-raw-uid>/dp_raw_obs_proc/<stub-run>/
      index.yaml
      <acquisition-stem>_kids_model_stub.nc
  maps/
    config.yaml
    citlali.log
    redu00/
      execution.json
      <observation>/raw/        # FITS, statistics, optional saved timestreams
  legacy-maps/                 # only with --also-legacy
```

The model product retains both blocks, per-block ECSV and Zarr artifacts. Stub
files belong to the target acquisition, with exact model and calibration
links in their indexes. Existing model indexes are reused by reference, so
their complete payloads must remain available. This focused recipe does not
automatically export maps to DPDB, change `main` tags, or install Tlaloc files.
It is not the full operational `tolteca_reduce` service distribution.

To compare intensity maps on unchanged linear azimuth/elevation grids:

```sh
"$TUNE_STACK/venv/bin/python" \
  "$TUNE_SOURCE/examples/tune_model_comparison/compare_maps.py" \
  /results/152434-v3/legacy-maps/redu00/152434/raw \
  /results/152434-v3/maps/redu00/152434/raw \
  /results/152434-v3/comparison \
  --left-label 'Legacy reference' --right-label 'V3 full model'
```

Check `execution.json` for the actual reduNN and observation folder, rather
than assuming `redu00`. The utility rejects incompatible sky grids instead
of silently registering/rescaling maps. Inspect contributor counts, APT flags
and sample flags alongside the figures; a successful native exit does not
establish scientific quality.

## 4. Evidence and remaining limits

- The portable preparation reproduced every APT column, the complete
  `kids_readout` model payload and native scientific configuration of the
  completed broader 152434 run (4,141 eligible rows).
- The pinned native Git trees match the source trees used for those maps.
  Existing checks include the full reductions, 17 selected kidscpp tests and
  10 selected Citlali tests; three older external-fixture cases were skipped.
- A fresh Python 3.13 installation passed imports, CLI and dependency checks.
  It then refitted both network-0 Tune blocks with identical calibration-table
  columns and full joint metadata. Full-array preparation reproduced all 11
  stubs' numeric variables and the 4,141-row eligibility selection. The local
  focused test suite passed all 74 tests, including the real-data fixtures;
  Ruff and the handoff entry-point type checks passed.
  The handoff tests cover explicit paths, acquisition identity, reused Tune
  obsnums and rejection of misleading legacy labels/unsupported map types.
- Exact model transport is not full solver parity. On the separate 145609
  full record, 14 affine / 16 delay groups out of 473 still differ between
  Python and native solving. Keep those limitations visible.
- The V2-derived APT has not been recalibrated for resolved components.
  Joint covariance is diagnostic; map weighting remains per row. The a1400
  residual RMS remains 4% above the corrected legacy reference. No validated
  mapping-speed claim is made.

[evidence/summary.json](evidence/summary.json) contains aggregate results and
source hashes; [validation.json](validation.json) records the release checks.
Raw data and large reduction products are not in Git.
Rebuild the compact PDF with Python plus `reportlab` and Pillow:

```sh
python build_summary.py tune-model-comparison-summary.pdf
```

Run that command in this handoff directory. `collect_summary.py` regenerates
its compact evidence from the completed local comparison tree; it does not
refit or rerun a reduction. The [extended guide](../../../docs/kids-tune-comparison.md)
and [investigation recipes](../README.md) retain the detailed controls/history.
