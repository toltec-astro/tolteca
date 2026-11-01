# From telescope files to a checked KIDs run

Read an LMT telescope file, select observation files by metadata, and check a
KIDs run before reducing anything. This walkthrough uses observation 145609
from the reference data. Set `/data_lmt` to your raw-data directory; the examples
read inputs in place. Save tables and products outside that directory.

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
    print(meta.t_int.to_value(u.s), meta.t_elapsed.to_value(u.s))
    print(meta.target.ra.to_value(u.deg), meta.target_ref.ra.to_value(u.deg))
    az = tel.az.isel(time=slice(0, 100)).load()
```

`ds` is an xarray Dataset; `meta` is frozen `LmtTelMetadata`. Opening checks for
the telescope timestamp variable without reading its values. The first `.meta`
access reads the required headers, two timestamp endpoints, and main/reference
target coordinates. It does not read pointing streams. Missing required metadata
raises an error.

| Metadata | Meaning |
| --- | --- |
| `obsnum`, `subobsnum`, `scannum` | Recorded observation identifiers |
| `master_code`, `master` | Telescope controller code `0` and label `"tcs"` |
| `quartet` | Observation label in `master-obsnum-subobsnum-scannum` form |
| `uid_obs` | Identifier shared by acquisitions of one observation |
| `uid_raw_obs` | Identifier of the raw observation |
| `uid_raw_obs_file` | Identifier of the observation's interface data |
| `instru` | Producing instrument: `"lmt"` for telescope files |
| `data_models` | Immutable set of applicable data models, e.g. `frozenset({"lmt", "toltec"})` |
| `interface_kind`, `interface` | Interface class and instance: `"tel"`, `"tel"` for telescope files; `"roach"`, `"toltec7"` for detector readout |
| `obs_pgm` | Observation program as `LmtObsPgm`, e.g. `LmtObsPgm.lissajous` |
| `obs_goal` | Observation goal as `LmtObsGoal`, e.g. `LmtObsGoal.pointing` |
| `t0`, `t1` | First and last telescope sample times, as Astropy Time objects |
| `t_int` | Integration time from the header; it may be zero |
| `t_elapsed` | Actual timestamp span `t1 - t0`; it need not equal `t_int` |
| `target`, `target_ref` | Main and reference SkyCoord positions, not an offset vector |
| `filepath_orig` | Original acquisition path recorded in the header |

The UIDs identify observations, acquisitions or interface files, not their
current storage locations. They have no zero padding. For this telescope file,
`uid_obs` is `tcs-145609` and `uid_raw_obs` is `tcs-145609-0-2`.
Metadata is cached on the accessor and remains usable after closing the file.

Empty or unrecognized observation programs and goals raise `ValueError` when
reading metadata. When constructing `LmtTelMetadata` directly, pass `master="tcs"`
and enum members for `master_code`, `obs_pgm`, and `obs_goal`. The telescope master
code is `LmtRawObsMasterCode.TCS`, from `tolteca_datamodels.lmt.types`;
the master and code appear as `"tcs"` and `0` in table columns and JSON. Programs
and goals use
lowercase values there, such as `"lissajous"` and `"pointing"`.
Direct construction also requires `instru="lmt"`, `data_models`,
`interface="tel"`, and `interface_kind="tel"`; none has a default.

Metadata values are read-only, and both supplied values and declared defaults
are validated strictly. For additional data models, use
`tolteca_datamodels.core.types.FrozenDataModel`; override validation or mutability only
when the model needs a different input or editing protocol.

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

### Plot the telescope pointing

Load the desired pointing samples while the file is open, then plot them:

```python
import matplotlib.pyplot as plt

with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    az_deg = (tel.az.isel(time=slice(0, 1000)).load().to_numpy() * u.rad).to_value(u.deg)
    alt_deg = (tel.alt.isel(time=slice(0, 1000)).load().to_numpy() * u.rad).to_value(u.deg)

fig, ax = plt.subplots()
ax.plot(az_deg, alt_deg)
ax.set(xlabel="Azimuth [deg]", ylabel="Altitude [deg]", title=meta.uid_raw_obs_file)
fig.savefig("telescope-pointing.png")
plt.close(fig)
```

## Distinguish acquisition metadata from file locations

Raw observation metadata includes the required acquisition path `filepath_orig`.
A file's current location can differ from that path after copying or renaming it.
`filepath_info_orig` parses the recorded path once:

```python
info = meta.filepath_info_orig
assert info is not None
print(meta.filepath_orig)
print(info.prefix, info.ut)
```

The recorded path is parsed using all registered filename layouts, independently
of the metadata's instrument. This property describes the filename, not the
file's header; it does not check whether they agree.

`ut` is the writer's file-creation time encoded in the filename, not filesystem
mtime or telescope time. Telescope names encode only a date, represented as
midnight UTC; readout names also encode the time of day. Do not use filename UT
as the exposure start time.

An unfamiliar recorded filename provides basic information: `ext` and an empty
`data_models` set. A location without a filename gives
`meta.filepath_info_orig = None`; the raw `filepath_orig` remains available.

Use `FilepathInfo.from_source()` to parse a local path or remote file location
without opening it:

```python
from tolteca_datamodels.filepath import FilepathInfo

filepath_info = FilepathInfo.from_source(tel_path)
print(filepath_info.interface, filepath_info.ut)
print(filepath_info.quartet)
```

The factory returns the information available in the recognized filename layout;
for example, TolTEC readout names also provide `roach`. It accepts a `FileLoc`.
Unfamiliar filenames use the basic parser, which supplies `ext` and an empty
`data_models` set. A location without a filename raises `ValueError`, or returns
`None` with `raise_on_error=False`. Invalid values in a recognized layout, such
as an invalid date, still raise.

## Select telescope observations

Use `LmtFileAccessor` for a table of telescope observations, without detector
data. Read headers to select by observation goal or program:

```python
from tolteca_datamodels.lmt.file import LmtFileAccessor

tel_paths = (data_root / "tel").glob("tel*.nc")
tel_table = LmtFileAccessor.from_sources(tel_paths, mode="read").table
pointing = tel_table.lmt_file.query(
    'obs_goal == "pointing" and obs_pgm == "lissajous"'
)
print(pointing[["uid_raw_obs_file", "obs_goal", "obs_pgm", "t_int_s"]])

for row in pointing.itertuples(index=False):
    with open_lmt_tel(row.source) as ds:
        az = ds.lmt_tel.az.isel(time=slice(0, 100)).load()
```

Reference observations 145609, 145610 and 145611 include Lissajous pointing
scans. Their headers also supply `t0_mjd`, `t1_mjd`,
`t_int_s`, `t_elapsed_s` and target-coordinate columns such as `target_ra_deg`,
`target_dec_deg` and `target_frame`. The reference target uses the same naming
with `target_ref_`. Both targets in these telescope files use `"icrs"`.
Reading the table closes each file without loading pointing streams.

Filter by position and duration using the same query API:

```python
selected = pointing.lmt_file.query(
    'target_frame == "icrs" and 70 <= target_ra_deg < 80 '
    'and 60 <= target_dec_deg < 70 and t_int_s > 60 and t_elapsed_s > 65'
)
print(selected[["uid_raw_obs_file", "target_frame", "t_int_s", "t_elapsed_s"]])
```

This selects scan 2 of observation 145609 in the reference subset. Coordinate
columns are in degrees and duration columns in seconds. `t_int_s` is the
header integration time; `t_elapsed_s` is the span between telescope timestamp
endpoints. They are different measurements, and an integration time of zero
does not imply an empty file.

The default `mode="guess"` uses filenames only. To read headers later, call
`tel_table.lmt_file.read_metadata()`; it updates that table in place.
With `mode="guess_or_read"`, filenames with no data-model membership
trigger header reads. Other instruments' recognized filenames retain their
information without triggering telescope header reads.
Use `tel_table.lmt_file.query_obs_spec("tcs-145609/tel")` to select by observation
identifiers instead of a pandas expression.

Open the selected telescope files to work with their pointing arrays:

```python
with selected.lmt_file.open() as opened_tel:
    pointing_arrays = [
        row.io_obj.lmt_tel.az.isel(time=slice(0, 100)).load()
        for row in opened_tel.itertuples(index=False)
        if row.io_obj is not None
    ]
```

The scope identifies readers and closes files it opened on exit. It does not
populate metadata. Supplied Dataset and DataTree
objects are borrowed and remain open. An opening failure leaves `io_obj=None`;
an identification failure retains the resource but leaves `accessor_obj=None`.
Use `on_error="raise"` to propagate source failures. A metadata failure retains
the live reader for inspection and leaves previously collected values unchanged.

For an already opened telescope dataset, use its named accessor directly:

```python
import xarray as xr
from tolteca_datamodels.lmt import LmtTelAccessor

with xr.open_dataset(tel_path) as data:
    reader = data.lmt_tel
    assert isinstance(reader, LmtTelAccessor)
    print(reader.meta.uid_raw_obs_file)
```

Accessing `data.lmt_tel` validates telescope structure and reuses the accessor
on subsequent access. It does not read sample arrays or validate all observation
headers; invalid headers raise when accessing `reader.meta`. Keep the dataset
open while using the reader. For objects whose reader is not known, use a file
table to identify them through the selected data models.

`LmtFileAccessor` accepts telescope data for any instrument. `ToltecFileAccessor`
accepts telescope data when `Header.Dcs.Receiver` identifies TolTEC,
independently of the filename. Its metadata has `instru="lmt"`,
`data_models=frozenset({"lmt", "toltec"})`, and `interface_kind="tel"`.

## Build and query a file table

A file table collects filenames, metadata and processing issues without running
an analysis pipeline. Factories return an accessor over a new DataFrame; `.table`
returns that DataFrame. Methods on an existing accessor update its table in place.

```python
from tolteca_datamodels.toltec.file import ToltecFileAccessor

file_accessor = ToltecFileAccessor.from_sources([
    data_root / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
    data_root / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
], mode="read")
table = file_accessor.table
selected = file_accessor.query("obsnum == 145609")
```

The concrete class selects the capabilities: `LmtFileAccessor` handles telescope
files; `ToltecFileAccessor` handles TolTEC detector, HWPR and companion telescope
files. Every input retains a row, including unsupported or invalid entries.
Filename parsing always uses all currently registered parsers, independently of
which accessor will handle the row.

Choose how much information to collect before querying:

| Mode | Matching filename | Recognized foreign filename | Unidentified filename |
| --- | --- | --- | --- |
| `guess` (default) | Guess | Keep filename information | Keep format information |
| `guess_or_read` | Guess | Keep filename information | Open, identify, read metadata |
| `read` | Open, identify, read metadata | Keep filename information | Open, identify, read metadata |

A recognized filename has nonempty `data_models`. Empty or missing memberships
mean unidentified. Matching uses current table values, not historical trace
entries or the `file_data_models` snapshot. A generic filename such as
`renamed.nc` still supplies `file_ext="nc"`; its contents can identify the reader.

`file_*` columns preserve filename evidence. Marked filename fields initialize
unprefixed columns such as `obsnum`, `interface` and `data_models`. Successful
metadata reads replace the columns declared by the concrete accessor's metadata
mapping; other columns remain unchanged. A failed read leaves the row's values
unchanged. Metadata supplies its own identifiers; no
filename derivation runs after a metadata update.

`file_accessor.read_metadata()` reads metadata into the table and returns the
same accessor. Each call processes matching and unidentified rows, even if their
metadata was populated earlier. Metadata is obtained through each reader's
`.meta` property.

### Open resources and read metadata

Plain `open()` opens and identifies eligible sources without reading metadata.
The context yields the same table with temporary `io_obj` and `accessor_obj`
columns. Use `accessor_obj` to work with the selected data reader:

```python
with file_accessor.open() as opened:
    file_accessor.read_metadata()
    selected = file_accessor.query("interface == 'tel'")
    for row in selected.itertuples(index=False):
        if row.accessor_obj is not None:
            az = row.accessor_obj.az.isel(time=slice(0, 100)).load()
```

For metadata alone, no surrounding open context is needed:

```python
file_accessor.read_metadata()
print(file_accessor.table[["obsnum", "interface"]])
```

The update reuses live readers. Otherwise it opens, reads and closes the required
sources itself. Context exit closes resources opened by the table and removes
the temporary columns; metadata remains. A repeated path is opened once per
opener within the context. Supplied open objects are borrowed and never closed
by the table:

```python
with open_lmt_tel(tel_path) as dataset:
    borrowed = ToltecFileAccessor.from_sources([
        {"source": dataset, "source_loc": tel_path},
    ])
    with borrowed.open():
        borrowed.read_metadata()
        print(borrowed.table.obs_goal.tolist())
```

Keep row positions stable while a context is open, and load any arrays needed
after the context exits.

Opening rules use the row's current values, with `file_*` as a fallback only when
there is no corresponding current column. Reader rules select a class from those
values, or inspect the opened contents when the values cannot decide. Once a
reader is selected, a binding or metadata failure is recorded; it does not trigger
a search for a different reader.

Processed KIDs data can remain openable when acquisition metadata is unavailable.
A metadata read records `metadata_unavailable` and retains the row's existing
metadata values.
Acquisition lineage for these products is a remaining reader-level gap.

### Inspect processing issues

The default `on_error="record"` retains failed rows. Use `on_error="raise"` to
propagate source validation, opening, identification or reading failures. Invalid
configuration and projection/dtype errors always raise.

```python
print(file_accessor.source_trace_summary())
for row in file_accessor.table.itertuples(index=False):
    for issue in row.source_trace.iter_error_entries():
        print(row.source_loc, issue.kind.value, issue.message, issue.context)
```

Traces retain the latest issue of each kind, not success events. Repeated issues
replace earlier details of that kind. `trace.has_error` reports recorded failures;
it is not a readiness flag. A later successful retry does not erase those failures.
Inspect current metadata columns and, inside `open()`, `accessor_obj`
for the information available to your analysis. `filepath_rule_unmatched`, `file_accessor_rule_unmatched`,
and `opener_rule_unmatched` are informational, even with `on_error="raise"`.
Failure to obtain a live reader records `reader_creation_failed` and follows
`on_error`. This includes no reader being found after inspecting the opened
contents, an inspection error, or an error constructing the reader. Successful
content-based identification records no issue. Invalid entries remain in the
table with their validation failure.

### Query the flat columns

Use `.query()` for metadata expressions and `.query_obs_spec()` for observation
selectors. Both return a DataFrame and preserve row order and index labels.

```python
pointing = file_accessor.query("obs_goal == 'pointing'")
toltec_related = file_accessor.query('contains(data_models, "toltec")')
sweeps = file_accessor.query("bits_any_set(data_kind_toltec, raw_sweep)")
observation = file_accessor.query_obs_spec("tcs-145609/{tel,0}")
```

TolTEC and other instruments keep independent classification columns such as
`data_kind_toltec` and `data_kind_dummy_instru`; their flags are not combined.
In a composite query, qualify flag names, for example
`ToltecDataKind.raw_sweep`. The concrete TolTEC accessor also accepts `raw_sweep`.

Observation selectors are context-free parsed constraints. The accessor supplies
interface numbering. A composite combines its selected interface vocabularies
and applies positional selection once to the combined table. `tcs-[-1]` selects
the largest available TCS observation number; it does not mean latest by time
across controllers. See [observation selectors](obs_spec.md).

### Discover files in a data tree

Stores discover candidates without opening them. Compose discovery and file
access explicitly when working in Python:

```python
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

store = LmtFileStore(data_root)
spec = ObsSpec.parse("145609/{tel,0}")
paths = store.partial(type(store).tel, type(store).toltec).glob_obs_spec(spec)
file_accessor = ToltecFileAccessor.from_sources(paths, mode="read")
selected = file_accessor.query_obs_spec(spec)
```

Or use the integrated operation:

```python
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import query_file_store

models = DataModelSet.from_select(["toltec"])
selected = query_file_store(store, spec, mode="read", data_models=models)
```

The selected data models provide store requirements and file-accessor types.
Discovery first finds candidates, then populates the table, then applies the full
observation selector. Metadata can therefore correct a guessed observation
number before final selection. Store candidate globs assume organized filenames;
use explicit sources or glob inputs for renamed files.

### Use registered or explicit capabilities

The assembled `FileAccessor` uses available registered data-model specs.
Importing it enables LMT and TolTEC. Extensions explicitly register their specs.
The concrete classes work directly without constructing a `DataModelSet`.

```python
from tolteca_datamodels.file import FileAccessor

mixed = FileAccessor.from_sources(paths)
toltec_view = mixed.partial(accessor_types=(ToltecFileAccessor,))
toltec_view.read_metadata()
selected = toltec_view.query_obs_spec(spec)
```

`partial()` restricts capabilities, not rows. It shares the original table and
cannot enable a capability excluded by an earlier view. Each matched row is
assigned to one concrete accessor. TolTEC supersedes generic LMT handling on
its telescope companions, so aliases, UIDs and metadata are not populated twice.

For a new table with an explicit capability selection:

```python
from tolteca_datamodels.core.file import FileAccessorView

models = DataModelSet.from_select(["toltec"])
file_accessor = FileAccessorView.from_sources(
    paths, accessor_types=models.file_accessor_types, mode="read",
)
```

`DataModelSpec` associates a data-model identity with accessors and stores.
`DataModelSet` selects these declarations; file accessors do not depend on a spec
or registry. The root entry point assembles registrations. Core imports remain
neutral but use filename registrations already available in the process.

### Configure input discovery

Each input entry discovers and populates its own table, applies its observation
selector, then contributes to the combined table. `data_model_select` narrows
capabilities globally or per entry. It does not change filename parsing.

```yaml
input_discovery:
  data_model_select: [toltec]
  inputs:
    - data_lmt_path: /data/data_lmt
      obs_spec: tcs-145609/{tel,0}
      mode: guess
    - sources:
        - {glob: /cache/*.nc}
      mode: guess_or_read
```

Exact source locations are literal; only `{glob: ...}` expands patterns. `*`
searches one directory, `**` permits recursion, and no match yields an empty
list. Paths follow the runtime configuration's `root_path`. Combined discovery
retains the first selected occurrence of each location or local symlink target.
Pipeline-specific selection remains a later pipeline step.

```python
from tollan.config import RuntimeContext
from tolteca_datamodels.input_discovery import InputDiscovery

runtime = RuntimeContext({"input_discovery": {"inputs": [{
    "data_lmt_path": data_root,
    "obs_spec": "145609/{tel,0}",
    "mode": "read",
}]}})
table = InputDiscovery(runtime).discover(data_models=models)
```

### Define an extension

Import building blocks from `tolteca_datamodels.core`. A concrete file accessor
provides row match rules, guess derivation, opener and reader policies, metadata
column mappings, and observation-selector interpretation. A composite supplies
component accessor types and delegates to their declarations. Neither requires a
data-model tag on the accessor class.

`tolteca_dummy_instru` is a minimal separate instrument demonstrating this split,
including separate counter and telescope policies. Its `register()` enables its
data-model spec, store and discovery entry in assembled workflows. Importing only
its filepath module enables its filename parsers without enabling its workflow.
See [the extension API](datamodels-api.rst) for the hook contracts.


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
ROACH selectors search TolTEC files. Telescope filenames identify master `tcs`,
so `tcs-145609/tel` selects telescope files without opening their headers.

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
