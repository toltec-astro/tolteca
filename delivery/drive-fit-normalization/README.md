# Drive-fit normalization investigation snapshot

This temporary branch captures the full pending TolTECA source changes as of
2026-10-21, based on `542c86a0a955edaa4a293521c55a18221952cbb1`. It is an
investigation snapshot, not a reviewed release or a merge proposal. The original
checkout remains on its review branch; packaging does not stage, commit, or
modify its working files. This is a point-in-time capture; subsequent local
edits are not included. The capture time is in `source-provenance.json`.
The delivery branch may be deleted after the investigation.

## Read the report

Open [report.html](report.html) in a browser, or serve this directory locally:

```bash
python -m http.server 8000 --directory delivery/drive-fit-normalization
```

Then open `http://127.0.0.1:8000/report.html`. The equations, figures, and
[six saved fit/commit plots](index.html#plots) work without a CDN. Numerical
comparisons are in [comparison/summary.json](comparison/summary.json) and the
accompanying per-channel ECSV files. The six-network cross-check with the
independent investigation is in [independent-comparison.json](independent-comparison.json).

The manuscript source is included as [report.md](report.md). `legacy/` contains the exact v2
power model and drive-fit recipe used as references, with original paths and
hashes in [source-provenance.json](source-provenance.json). These historical
files are for inspection; they are not an independently installed v2 environment.

## Install the source snapshot

Use Python 3.13 or newer (the local validation used Python 3.14), with `uv` or
an equivalent installer. Tollan must be the matching v3 source revision; a
published older Tollan package is insufficient.

```bash
git clone --branch delivery/drive-fit-normalization-20261021 \
  https://github.com/toltec-astro/tolteca.git tolteca-investigation
git clone https://github.com/toltec-astro/tollan.git tollan-investigation
git -C tollan-investigation checkout 71d4373522d677de740c6c637d7751e52e714207
uv venv --python 3.14 .venv-investigation
uv pip install --no-sources --python .venv-investigation/bin/python \
  -e ./tollan-investigation -e ./tolteca-investigation
```

The `--no-sources` option is required for this standalone installation: it
ignores TolTECA's development workspace mapping while the explicit editable
Tollan checkout supplies the pinned dependency. No workspace generation is
needed. Package versions from the original report run are in `environment.json`;
the clean standalone replay and its versions are in `snapshot-validation.json`.

## Reference inputs

The compact drive-fit reference is observation **145484**, subobservations
**1–7**, plus standalone commit target **145485/0/1**, on network **0**. The
seven fit powers are 4–16 dB in steps of 2 dB; all 632 channels are used.
The `145484/0` tune is excluded from fitting.

Raw data are published in the separate `tolteca_ref_data` repository at commit
[`9cdbc76c4327b11e9ec45ce7895ef13b537d03a5`](https://github.com/toltec-astro/tolteca_ref_data/commit/9cdbc76c4327b11e9ec45ce7895ef13b537d03a5).
Install Git LFS, then fetch the eight files needed for these workflows:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/toltec-astro/tolteca_ref_data.git tolteca_ref_data
git -C tolteca_ref_data lfs install --local
GIT_LFS_SKIP_SMUDGE=1 git -C tolteca_ref_data checkout 9cdbc76c4327b11e9ec45ce7895ef13b537d03a5
git -C tolteca_ref_data lfs pull \
  --include="data_lmt/toltec/tcs/toltec0/toltec0_145484_*_targsweep.nc,data_lmt/toltec/tcs/toltec0/toltec0_145485_*_targsweep.nc" \
  --exclude=""
```

To download every reference observation, use `git -C tolteca_ref_data lfs pull`
without the include/exclude options. Paths, sizes and checksums are in
[reference-network0.json](reference-network0.json), relative to that
repository's `data_lmt/` directory. Use all seven fit powers and the separate
commit target; do not substitute a shorter power grid.

The report retains previously computed results for networks 0, 7 and 11.
Only network 0 is retained in the compact reference repository. Reproducing
networks 7 and 11 requires acquiring their original files separately; the
full acquisition manifest is [reference-acquisition.json](reference-acquisition.json).
The older 31-sweep ICS 17068 case belongs to `tolteca_cal_data`, not this
reference-data repository.

## Run both workflows

From the snapshot checkout, using the installed environment:

```bash
../.venv-investigation/bin/python delivery/drive-fit-normalization/run_reference.py \
  --data-lmt /absolute/path/to/tolteca_ref_data/data_lmt \
  --output /absolute/path/to/new-results \
  --plots
```

The runner invokes the public `drive_fit` and `drive_fit_commit` CLIs, writes
both resolved configurations and logs, and pins commit to the newly created
fit product. Omit `--plots` for a faster numerical run. The output directory
must be new or empty. Product registration and live Tlaloc installation are
disabled. Instrument settings are retained as product artifacts for inspection.

Expected network-0 corrected attenuation: continuous **8.3526392661 dB**,
written **8.25 dB**; fixed-total-model residual **+0.1026392661 dB**. This is
not a calibrated detector-input power or a guarantee of waveform-peak closure.
The tenth-percentile cutoff, median filling and 0.25 dB floor rule are retained.

The original three-network YAML files in this directory retain their original
local paths as provenance; use the runner to generate paths for your machine.

## Focused verification

Install `pytest`, `pytest-doctestplus`, `pytest-remotedata`, and
`pytest-benchmark` in the same environment, then run:

```bash
python -m pytest -q tests/tolteca_kids/test_drive_fit_power.py
```

The analytic power tests run without raw data. The calibration-dependent test
requires `TOLTECA_DRIVE_FIT_DATA` pointing to the 31-sweep ICS 17068 directory;
it skips if that data is unavailable. The original 45-test fit/commit/power
validation is recorded in `provenance.json`. Snapshot-specific checks and
workflow replay results are recorded in `snapshot-validation.json`.

The standalone instructions were also tested from fresh GitHub clones in a new
Python 3.14 environment on macOS arm64, with local workspace access blocked and
network access disabled during computation. Both workflows completed for all
632 channels. All committed amplitudes and channel flags matched the published
comparison table exactly; power diagnostics differed by at most 2.14e-14 dB.
The run produced 636 HTML pages with no broken local links. See the
`clean_environment` entry in `snapshot-validation.json` for revisions, package
versions, input checksum verification and numerical comparisons.

## Scope of the snapshot

Tracked working-tree contents, deletions and non-ignored untracked source
files are captured, with one design-directory exception: only `design/spec.md`
is added or updated. Other design files already tracked in the base commit
remain at their base versions; new local design notes, probes and slides are
excluded. The ignore rule permits only `design/spec.md` for future additions.
Finder metadata, virtual environments, caches and generated build products are
excluded. The `delivery/` directory contains the portable report, its renderer,
and investigation material; runtime source files retain their captured contents.

The runtime requires the code in this branch, not only the normalization module:
it includes the pending v3 data-model and workflow changes. No scientific
algorithm was altered while packaging this branch. Deleting the remote delivery
branch later does not change the original local review branch or working files.
