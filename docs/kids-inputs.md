# Inspect KIDs inputs and check a run

Use `tolteca_kids --dry_run` to check input selection and configuration before
running a reduction. It reads source-identifying metadata, validates configuration for
each selected input, and executes a no-op pipeline. It does not read signal
arrays or write products, even when output is enabled in the configuration.

## Choose inputs

Put options before `--` and one or more local NetCDF files, quoted globs, or
[observation selectors](obs_spec.md) after it:

```console
tolteca_kids --dry_run -- sweep.nc
tolteca_kids --show_inputs -- '/data/lmt/toltec/tcs/toltec0/*.nc'
tolteca_kids --data_lmt_path /data/lmt --dry_run -- '123456/0'
```

Selectors require `data_lmt_path`; explicit files and globs do not. A file matched
more than once, including through a symbolic link, is processed once.

ObsSpec first narrows file discovery with glob patterns, then filters filename
metadata with the complete selector. Selected candidates are opened to read
scalar header metadata. Raw acquisitions must contain their required headers;
no-op execution does not bypass metadata validation. Use `--select` to filter
that enriched catalog:

```console
tolteca_kids --select "roach < 7 & file_suffix in ['vnasweep', 'targsweep', 'tune']" \
  --dry_run -- '/data/lmt/toltec/tcs/toltec*/*.nc'
```

For an unrecognizable filename such as `downloaded.nc`, identification falls
back to the acquisition filename from `Header.Toltec.Filename`. The usual
`file_suffix` selection therefore also works for these renamed files. Header
identifiers correct filename guesses before `--select` is evaluated. A
recognizable current filename keeps its own suffix; use
`filepath_info_orig.suffix` to explicitly select by the acquisition suffix.
See the [observation-file guide](input-discovery.md) for the nested
fields, telescope identification, and Parquet examples.

`--show_inputs` prints the selected catalog and exits without executing a
pipeline. An empty catalog is allowed for inspection; a run with no selected
inputs fails.

## Configure a run

Paths and selection can also be supplied in a YAML configuration:

```yaml
files:
  data_lmt_path: /data/lmt
  dataprod_toltec_path: /data/products
  select: roach < 7
```

`dataprod_toltec_path` supplies the default product-output directory. It does not
create directories or enable output. An explicit `kids.output.path` takes
precedence over this shared default.

`--config_paths` accepts files or directories of numbered configuration files.
Later paths override earlier paths. The explicit path and selection options
override their `files` YAML fields. Dotted `--kids.*` options override processing
settings; shared discovery settings use the explicit options above.

```console
tolteca_kids --config_paths recipes/ override.yaml --show_config -- sweep.nc
tolteca_kids --config_paths recipes/ --dry_run --kids.sweep_reduce.block 0 -- sweep.nc
tolteca_kids --config_paths recipes/ --config_paths override.yaml --dry_run -- sweep.nc
```

`--show_config` prints the effective, validated `files` and `kids` configuration
sections for each selected input and exits. Conditional sources use that input's
metadata for `enable_if`; these conditions control stage configuration, not
input selection. Set discovery paths and catalog selection in unconditional
configuration, since they are needed before per-input metadata is available.
Scalar header names such as `atten_drive`, `obs_type` and `n_chans` are available
in both `--select` and `enable_if`. Actual tune block counts are not read during
catalog preparation. See the guide for metadata types and field availability.

Kids consumes explicit path settings. It does not discover a deployment root
or derive paths from process environment variables. Relative configuration
paths use the runtime validation context's `rootpath`, or the working directory
when no root is supplied; they are not relative to each YAML file.

## Launch through Deploy

Use Deploy's `run_kids` to supply deployment paths and forward application
configuration, as shown in the [observation-file guide](input-discovery.md).
You can also pass an explicitly configured command through its generic runner:

```console
tolteca_deploy run -- tolteca_kids \
  --config_paths recipes/ --data_lmt_path /data/lmt --dry_run -- '123456/0'
```

Options before Deploy's `--` configure the launcher; options after it belong
to Kids. A config passed only to Deploy is not automatically passed to Kids.

## Diagnose a run

```console
tolteca_kids --log_level DEBUG --config_paths recipes/ --dry_run -- sweep.nc
```

A successful check prints `no-op completed` for each selected input and exits
with status zero. Selection, configuration, and execution failures produce a
nonzero exit status. Open input files close when execution finishes or fails.

The check supports raw VNA, target, and tune sweeps, and raw timestreams.
Success does not mean a scientific processing pipeline exists for every kind.
Debug logging shows discovery, selection, per-input configuration, and step
execution. It includes paths and configuration values, so handle saved logs
accordingly.

`--show_inputs` and `--show_config` remain inspection-only when combined with
`--dry_run`; neither executes the no-op pipeline.
