# From telescope files to a checked KIDs run

Read an LMT telescope file, select observation files by metadata, and check a
KIDs run before reducing anything. This walkthrough uses observation 145609
from the reference data. Set `/data_lmt` to your raw-data directory; the examples
read inputs in place. Save catalogs and products outside that directory.

## Read the telescope metadata

Open a telescope file with `open_lmt_tel()` and use its `lmt_tel` accessor:

```python
from pathlib import Path

import astropy.units as u
from tolteca_datamodels.lmt import open_lmt_tel

data_root = Path("/data_lmt")
tel_path = data_root / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"

with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    meta = tel.meta
    print(meta.uid_raw_obs_file, meta.obs_pgm, meta.obs_goal)
    print(meta.t_exp.to_value(u.s), meta.t_elapsed.to_value(u.s))
    print(meta.target.ra.to_value(u.deg), meta.target_ref.ra.to_value(u.deg))
    az = tel.az.isel(time=slice(0, 100)).load()
```

`ds` is an xarray Dataset; `meta` is frozen `LmtTelMetadata`. Opening checks for
the telescope timestamp variable without reading its values. The first `.meta`
access reads the required headers, two timestamp endpoints, and main/reference
target coordinates. It does not read pointing streams. Missing required metadata
raises an error; an unavailable master may be `None`.

| Metadata | Meaning |
| --- | --- |
| `obsnum`, `subobsnum`, `scannum` | Recorded observation identifiers |
| `master`, `master_str` | Acquisition-master code and readable name, or `None` |
| `quartet`, `uid_raw_obs` | The same `master-obsnum-subobsnum-scannum` identifier |
| `uid_raw_obs_file` | Quartet followed by interface, such as `tcs-145609-0-2-tel` |
| `instru`, `instru_component`, `interface` | Instrument, component and readout actor: `lmt`, `tcs`, `tel` |
| `obs_pgm`, `obs_goal` | Observation program and goal |
| `t0`, `t1` | First and last telescope sample times, as Astropy Time objects |
| `t_exp` | Integration time from the header; it may be zero |
| `t_elapsed` | Actual timestamp span `t1 - t0`; it need not equal `t_exp` |
| `target`, `target_ref` | Main and reference SkyCoord positions, not an offset vector |
| `filepath_orig` | Original acquisition path recorded in the header |

The UIDs identify the acquisition, not its current storage location. They have
no zero padding and are `None` when the master is unavailable. Metadata is
cached on the accessor and remains usable after closing the file. Open a fresh
dataset/accessor if the underlying headers change.

### Slice arrays before loading

| Accessor property | Values | Reading behavior |
| --- | --- | --- |
| `az`, `alt`, `ra`, `dec` | Actual pointing angles in radians | Lazy; slice before loading |
| `az_des`, `alt_des` | Desired horizontal pointing | Lazy; slice before loading |
| `az_map`, `alt_map` | Map coordinates | Lazy; slice before loading |
| `tel_time` | Native Unix timestamps | Lazy; slice before loading |
| `t` | Elapsed seconds relative to the first sample | Computed and cached; deferred with Dask |
| `hold_flag` | Integer flags without a physical unit | Converted and cached; deferred with Dask |

Keep the Dataset open while using arrays backed by its file. Pass `chunks={}`
to `open_lmt_tel()` for Dask-backed arrays. For a native flag slice, use the
schema and mapper directly:

```python
with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    flags = tel.mapper.get_arr(tel.data_source, tel.mapper.schema.hold_flag)
    first_flags = flags.isel(time=slice(0, 100)).load()
    timestamps = tel.tel_time.isel(time=slice(0, 100)).load()
    print(tel.az.u.unit)
```

The accessor works on a Dataset or DataTree. Its schema names describe the
fields; its mapper connects them to the stored variables. Use these APIs when
you need a native array rather than spelling the netCDF header names yourself.

## Distinguish acquisition metadata from file locations

A file's current location and its recorded acquisition path can differ after
copying or renaming it. `filepath_info_orig` parses the recorded path once:

```python
print(meta.filepath_orig)
print(meta.filepath_info_orig.prefix, meta.filepath_info_orig.ut)
```

`ut` is the writer's file-creation time encoded in the filename, not filesystem
mtime or telescope time. Telescope names encode only a date, represented as
midnight UTC; readout names also encode the time of day. Do not use filename UT
as the exposure start time.

An unrecognized recorded name raises `ValueError` when
`meta.filepath_info_orig` is requested, without preventing ordinary metadata
reading. To check header identifiers against that recorded name explicitly:

```python
with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    checked = tel.mapper.get_metadata(tel.data_source, check_filepath_orig=True)
```

The check is off by default. Conflicting observation numbers or available
master labels raise `ValueError`. It does not compare the current filename or
require the different clocks to agree.

## Build a file catalog

A catalog is a pandas DataFrame with one row per source. Start with filenames
when you want fast discovery without opening files:

```python
from tolteca_datamodels.file import guess_info_from_sources

kids_paths = sorted((data_root / "toltec/tcs/toltec0").glob("*145609*.nc"))
paths = [tel_path, *kids_paths]
guessed = guess_info_from_sources(paths)
print(guessed[["source", "interface", "obsnum", "data_kind"]])
```

`guess_info_from_source(path)` returns one frozen `SourceInfoModel`.
An unrecognized name raises `ValueError`; `raise_on_error=False` returns `None`
instead. The plural function keeps unrecognized sources as rows with unknown
identifiers. A malformed date in an otherwise recognized name still raises.

Read authoritative header identifiers with `build_input_catalog()`:

```python
from tolteca_datamodels.toltec.filestore import build_input_catalog

catalog = build_input_catalog(paths)
telescope = catalog.toltec_file.query("interface == 'tel' & master == 'tcs'")
records = telescope.toltec_file.to_info_list()
print(records[0].filepath_orig)
```

Files close before this function returns. Telescope identification uses its
strict metadata reader; KIDs identification reads scalar headers without
loading I/Q or scanning tune LO values. The catalog holds source identity and
filename fields, not every field of the observation metadata. Read `.meta`
for fields such as telescope `obs_goal` or KIDs `atten_drive`.

Choose when sources are opened with the Python `header_mode` keyword:

| Mode | Files opened for identification |
| --- | --- |
| `"never"` | None; keep filename guesses |
| `"fallback"` | Only unrecognized current filenames |
| `"always"` (default) | All supported telescope/KIDs and unknown netCDF candidates |

Fallback does not fill missing fields in an otherwise recognized name. Errors
from recognized but incomplete files propagate; unsupported formats retain
filename guesses. This policy controls catalog construction, not subsequent
execution reads.

For an already-open dataset, use the shared identifier directly:

```python
from tolteca_datamodels.file import SourceInfoModel, ToltecFileAccessor
from tolteca_datamodels.filestore import identify_source

with open_lmt_tel(tel_path) as ds:
    record = identify_source(ds, SourceInfoModel(source=tel_path))
one_file = ToltecFileAccessor.from_info_list([record])
```

The same call recovers a renamed file such as `downloaded.nc` from its headers.
It returns a fresh record without changing the current source. Header identifiers
correct guesses; an unrecognized current filename falls back to the recorded
original filename. A recognized current name retains its own suffix and UT.
Use `check_filepath_orig=True` for optional header/original-name consistency.

### Select rows using path fields

`filepath_info` holds effective filename information; `filepath_info_orig`
holds only the recorded acquisition-name information. Either may be absent.
Their fields include `prefix`, `interface`, `instru`, `instru_component`,
`master_str`, `obsnum`, `subobsnum`, `scannum`, `ut`, `suffix`, and `ext`.
TolTEC readout path records also have `roach`.

The native prefix `tel_toltec` or `tel_toltec2` maps to interface `tel` and
instrument/component `lmt`/`tcs`. Select a specific stream by prefix, not UID.
ROACH interfaces such as `toltec0` belong to `toltec`/`roach`. A filename does
not supply a telescope master. The source-level `master` column is a label;
the strict telescope metadata separately retains its numeric code.

Use either dotted fields or short aliases in queries:

```python
selected = catalog.toltec_file.query(
    "roach == 0 & file_suffix in ['vnasweep', 'targsweep', 'tune']"
)
by_recorded_name = catalog.toltec_file.query(
    "filepath_info_orig.suffix == 'tune'"
)
```

| Structured field | Query alias |
| --- | --- |
| `filepath_info.prefix` | `file_prefix` |
| `filepath_info.ut` | `file_ut` |
| `filepath_info.suffix` | `file_suffix` |
| `filepath_info.ext` | `file_ext` |
| `filepath_info_orig.suffix` | `file_suffix_orig` |

This rule applies to every path field and its `_orig` variant. Aliases are not
extra stored columns. Obtain a column with
`catalog["filepath_info"].struct.field("suffix")`. Path fields are stored as
Arrow columns, so querying them does not reparse paths or traverse Python
models row by row. `data_kind` belongs to the source record, not the path struct.

Queries use pandas syntax, including `&`, `|`, `in [...]`, and null checks such
as `file_suffix_orig.isna()`. Use timezone-qualified strings for UT comparisons,
for example `file_ut >= '2025-10-31T00:00:00Z'`. Evaluate only trusted expressions;
pandas queries are not a sandbox for untrusted input.

### Save catalogs or metadata

Parquet preserves the nested fields and UTC timestamps. Load with Arrow dtypes:

```python
import pandas as pd

catalog.to_parquet("catalog.parquet", index=False)
restored = pd.read_parquet("catalog.parquet", dtype_backend="pyarrow")
selected_again = restored.toltec_file.query("file_suffix_orig == 'tune'")
```

Import `tolteca_datamodels.file` in a fresh process to register the accessor.
Use Pydantic to save full telescope metadata independently of an open source:

```python
from pydantic import TypeAdapter
from tolteca_datamodels.lmt import LmtTelMetadata

adapter = TypeAdapter(LmtTelMetadata)
payload = adapter.dump_json(meta, round_trip=True)
metadata = adapter.validate_json(payload)
```

Use `round_trip=True` when saving model inputs for later validation. Ordinary
metadata serialization also includes computed fields, including the parsed
original path. To save the full source location, including relative-path context:

```python
payload_source = record.model_dump_json(round_trip=True)
record_restored = SourceInfoModel.model_validate_json(payload_source)
```

`source` is a Tollan `FileLoc`, supporting local paths, host paths and URLs.
Describing a remote source does not download it. Catalogs store resolved local
paths or remote URLs; raw readers require local files. Stage remote data locally
before opening it. `record.filepath` raises for a remote source.

## Check a KIDs run from the CLI

Use the same raw-data root to inspect the KIDs files for observation 145609:

```console
tolteca_kids --data_lmt_path /data_lmt --show_inputs -- '145609/0'
tolteca_kids --data_lmt_path /data_lmt --dry_run -- '145609/0'
```

The [ObsSpec](obsspec.md) `145609/0` selects ROACH 0. Selectors first narrow
filesystem discovery, then filter filename identifiers. Headers are read before
the CLI's `--select` expression. Explicit files and quoted globs need no data
root. Overlapping inputs, including symlinks, are deduplicated.

An explicit `/tel` selector searches the telescope directory; unqualified and
ROACH selectors search TolTEC files. To select telescope master, use `/tel`
and then `--select "master == 'tcs'"`. A master-qualified filename selector
cannot match a telescope master that is only available in the header.

Configure paths and input selection in a recipe, for example `check.yaml`:

```yaml
files:
  data_lmt_path: /data_lmt
  dataprod_toltec_path: /data/products
  select: roach == 0 & file_suffix == 'tune'
kids:
  sweep_reduce:
    block: -1
```

```console
tolteca_kids --config_paths check.yaml --show_config -- '145609/0'
tolteca_kids --config_paths check.yaml --dry_run --kids.sweep_reduce.block 0 -- '145609/0'
```

Put positional inputs after `--` to delimit multi-value options. Later config
paths override earlier ones; explicit path/selection options override `files`
settings. Dotted `--kids.*` options override processing settings. Conditional
source-list recipes use each input's metadata for `enable_if`; `--select`
instead chooses which inputs to run. See [KIDs inputs](kids-inputs.md) for
these options and conditional-configuration rules.

`--show_inputs` prints the selected catalog; `--show_config` prints each input's
validated configuration. Both exit without executing steps. `--dry_run` executes
a real no-op step with a live context, but does not reduce signals or write
products. The product root alone does not enable output. Source files close on
success and failure. Add `--log_level DEBUG` for discovery, condition decisions,
configuration and step timings; logs include paths and config values.

A successful no-op prints `no-op completed` for each input. It checks the
input/configuration/execution path, not scientific reduction. The available
science pipeline reduces raw sweeps; it does not yet reduce timestreams or
telescope pointing data. Continue with [KIDs reduction](kids-reduction.md) when
you are ready to compute and export a result.

## Invoke the same check through Deploy

For a deployed location, let Deploy supply raw-data/product paths:

```console
tolteca_deploy run_kids --location /path/to/location -- --dry_run -- '145609/0'
tolteca_deploy run_kids --show_command --location /path/to/location -- --dry_run -- '145609/0'
```

`run_kids` turns deployment settings into explicit KIDs arguments. It consumes
environment settings and forwards config paths for per-input evaluation by
Kids. Kids itself uses explicit file settings, not deployment environment
variables. Later KIDs path options can override the supplied roots.

`--show_command` prints `argv` and `cwd` without opening raw files or starting
the subprocess. An orchestrator can execute that argument vector without a
shell, using the recorded working directory. Retain the referenced recipes and
Python environment to replay it. The generic `tolteca_deploy run -- COMMAND`
can also execute an explicitly configured command, but does not prepare KIDs
arguments or automatically forward its own config options.
