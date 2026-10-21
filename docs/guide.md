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
| `data_models` | Applicable data models, e.g. `["lmt", "toltec"]` |
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

An unrecognized recorded name gives `meta.filepath_info_orig = None`; the raw
`filepath_orig` remains available.

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
Unrecognized names raise `ValueError`, or return `None` with
`raise_on_error=False`. Invalid dates still raise.

## Select telescope observations

Use `LmtFileAccessor` for a table of telescope observations, without detector
data. Read headers to select by observation goal or program:

```python
from tolteca_datamodels.lmt.file import LmtFileAccessor

tel_paths = (data_root / "tel").glob("tel*.nc")
tel_table = LmtFileAccessor.from_sources(tel_paths, mode="read")
pointing = tel_table.lmt_file.query(
    'obs_goal == "pointing" and obs_pgm == "lissajous"'
)
print(pointing[["uid_raw_obs_file", "obs_goal", "obs_pgm", "t_int_s"]])

for row in pointing.itertuples(index=False):
    with open_lmt_tel(row.source) as ds:
        az = ds.lmt_tel.az.isel(time=slice(0, 100)).load()
```

The seven reference telescope files include three matching Lissajous pointing
scans: 145609, 145610 and 145611. Their headers also supply `t0_mjd`, `t1_mjd`,
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
`tel_table.lmt_file.update_from_source_metadata()`; use `inplace=True` to update that table.
With `mode="guess_or_read"`, filenames not recognized by any registered parser
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

The scope fills the table with header metadata and closes files it opened on
exit. Supplied Dataset and DataTree objects are borrowed and remain open.
Both opened and supplied objects must be recognized telescope data. An
unrecognized or unavailable payload retains its row with an opening diagnostic
and `io_obj=None`. Use `on_error="raise"` to fail immediately instead. A metadata
failure retains the recognized handle for inspection; check metadata diagnostics
before relying on header-derived columns. Filename guessing and metadata-only
reads retain unidentified rows.

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
`data_models=["lmt", "toltec"]`, and `interface_kind="tel"`.

## Build and enrich a file table

Choose an import for the task:

| Task | Import from |
| --- | --- |
| Parse filenames using built-in layouts | `tolteca_datamodels.filepath` |
| Build a mixed file table | `tolteca_datamodels.file` |
| Select registered data-model capabilities | `tolteca_datamodels.data_model` |
| Discover inputs from configuration | `tolteca_datamodels.input_discovery` |
| Use a concrete instrument accessor or store | `tolteca_datamodels.lmt` or `tolteca_datamodels.toltec` submodules |
| Implement an extension without enabling built-ins | `tolteca_datamodels.core` submodules |

The four root entry points enable built-in LMT and TolTEC support. Optional
extensions still require their own registration. Use `core.file`,
`core.filepath`, `core.file_store`, `core.source`, `core.obs_spec`,
`core.data_model`, and `core.input_discovery` when defining an extension;
these imports do not load instrument registrations. They do not isolate or
restrict registrations already available in the process.
The core and root imports of `FileAccessor` refer to the same class, as do those
of `FilepathInfo`. Both use available registrations, including support loaded
after the core import. Filename parsing considers all registered parsers,
independently of the selected data-model scope.

Use concrete filename model types for annotations or instance checks. Import
them from the instrument's `filepath` module, and parse through the universal
factory:

```python
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.toltec.filepath import ToltecLmtFilepathInfo

tel_filepath_info = FilepathInfo.from_source(
    "tel_toltec_2025-10-31_145609_00_0002.nc"
)
assert isinstance(tel_filepath_info, LmtFilepathInfo)
assert isinstance(tel_filepath_info, ToltecLmtFilepathInfo)
```

Look up a registered data model by name or use the built-in named properties:

```python
from tolteca_datamodels.data_model import data_models

toltec = data_models.toltec
assert toltec is data_models["toltec"]
assert data_models.lmt.name == "lmt"
```

The lookup is read-only and includes new registrations. Use
`DataModelSet.from_select(...)` when selecting capabilities for a workflow.

Use membership in the `data_models` column to collect TolTEC readout files and their
telescope companions. Use `instru == "lmt"` for telescope-produced records, or
`interface_kind == "roach"` for detector readouts without enumerating boards.

| Filename prefix | `instru` | `data_models` | `interface_kind` | `interface` |
| --- | --- | --- | --- | --- |
| `toltec7` | `toltec` | `["toltec"]` | `roach` | `toltec7` |
| `hwpr` | `toltec` | `["toltec"]` | `hwpr` | `hwpr` |
| `tel_toltec` | `lmt` | `["lmt", "toltec"]` | `tel` | `tel` |
| `tel_dummy_instru` | `lmt` | `["lmt", "dummy_instru"]` | `tel` | `tel` |

For filename information alone, use the shared factory:

```python
from tolteca_datamodels.filepath import FilepathInfo

info = FilepathInfo.from_source(
    "tel_toltec_2025-10-31_145609_00_0002.nc",
)
assert (info.instru, info.interface_kind) == ("lmt", "tel")
assert info.data_models == ["lmt", "toltec"]
```

`FilepathInfo.from_source()` accepts all registered layouts. Telescope names
without registered instrument-specific support provide `instru="lmt"` and
`data_models=["lmt"]`. Use the returned attributes without importing the private
result classes. Memberships are unique names, not an ordered reader preference.
Choosing a handling context does not change them.

A file table is a pandas DataFrame with one row per resource. Build it from paths
without opening files:

```python
from tolteca_datamodels.toltec.file import guess_info_from_sources

kids_paths = sorted((data_root / "toltec/tcs/toltec0").glob("*145609*.nc"))
paths = [tel_path, *kids_paths]
guessed = guess_info_from_sources(paths)
assert guessed.filepath_info.notna().all()
assert guessed.filepath_info.iloc[0].prefix == "tel_toltec"
print(guessed[["source", "interface", "obsnum", "data_kind_toltec"]])
toltec_related = guessed.toltec_file.query('contains(data_models, "toltec")')
assert len(toltec_related) == len(guessed)
```

Importing `tolteca_datamodels.toltec.file` registers the `toltec_file` pandas
accessor used below.

Filename parsing supplies both canonical identifiers, such as `obsnum`, and
filename columns, such as `file_obsnum`, `file_suffix`, and `file_ut`.
It also supplies guessed data kinds and UIDs. Missing values stay null, including
UIDs when a required identifier is unknown. Unrecognized filenames remain in the
table for subsequent header discovery. Invalid dates in recognized names raise
an error.

The object column `filepath_info` retains each parse result from `source_loc`;
None means parsing did not succeed. Use `table.filepath_info.notna()` to select
successfully parsed filenames, or `row.filepath_info` to inspect a result while iterating.
These objects stay attached to their rows through filtering and sorting.

All file accessors retain information from every registered filename parser.
TolTEC classifies telescope companions with prefixes `tel_toltec` and
`tel_toltec2`; the dummy accessor classifies `tel_dummy_instru`. Other recognized
filenames retain their information without a filepath diagnostic. Use `read`
to check headers for recognized files in the accessor's scope as well as
unrecognized names; supported headers replace guessed identifiers.

To populate a table with authoritative metadata during construction:

```python
from tolteca_datamodels.toltec.file import read_info_from_sources

table = read_info_from_sources(paths)
telescope = table.toltec_file.query("interface == 'tel' & master == 'tcs'")
kids_selected = table.toltec_file.query("atten_drive == 7.25 & roach == 0")
tel_selected = table.toltec_file.query("obs_goal == 'pointing'")
print(telescope["filepath_orig"])
```

Header values replace canonical identifiers and add scalar metadata columns.
The header's computed UIDs replace the filename guesses too. Source locations
and filename provenance (`filepath_info` and `file_*` columns) remain unchanged.
`data_models` and `file_data_models` are list-valued columns: the former holds
current memberships, while the latter preserves the filename guesses. For
telescope metadata, `Header.Dcs.Receiver` supplies the additional membership
alongside `"lmt"`.

`instru` identifies the producing instrument. A TolTEC telescope companion has
`instru="lmt"` and belongs to both `"lmt"` and `"toltec"` data models; a ROACH
file has `instru="toltec"` and belongs to `"toltec"`.

Use `contains(data_models, "toltec")` in query strings for exact membership;
empty or missing lists do not match. Combine it with other conditions using
`and`, `or`, and `not` (or `&`, `|`, and `~`).

Choose how to populate a table with
`ToltecFileAccessor.from_sources(paths, mode=...)`, or use a batch shortcut:

| Mode | Shortcut | Information collected |
| --- | --- | --- |
| `"guess"` (default) | `guess_info_from_sources()` | Filename guesses from `source_loc`, without opening sources |
| `"guess_or_read"` | `guess_or_read_info_from_sources()` | Guesses, then metadata for filenames not recognized by any registered parser |
| `"read"` | `read_info_from_sources()` | Guesses, then metadata for filenames in the accessor's scope and unrecognized names |

Missing information in a recognized filename does not trigger `"guess_or_read"`.
Recognized files outside the accessor's scope are not opened in any construction
mode. They retain their filename-derived information and have an unknown TolTEC
data kind in a TolTEC file table.
Built-in openers require local files. Unsupported sources and failed reads
retain guesses and leave diagnostics by default; `on_error="raise"` stops on
these failures. Owned files close before construction returns.

For one source, `get_info_from_source()` returns a row with attribute access:

```python
from tolteca_datamodels.toltec.file import get_info_from_source

info = get_info_from_source(tel_path, mode="read")
print(info.uid_raw_obs_file, info.obs_goal)
```

It accepts the same modes and an optional `source_loc`. Unidentified information
remains null; the row is still returned.

For an existing table, `table.toltec_file.update_from_source_metadata()` uses current
`data_models` memberships and identity columns to select supported readers.
`data_models` is the authority for current scope: None means unknown and permits
content identification; an empty collection matches no handler. A nonempty
collection permits only handlers whose data models it contains. Successful
filename parsing initially supplies a collection, possibly empty; later metadata
or explicit table updates may replace it.
Original `filepath_info` and `file_*` snapshots remain
unchanged when metadata enrichment replaces current values and do not override
the current scope.
It refreshes metadata in a new table, preserving row order, index and additional
columns. Pass
`inplace=True` to update the existing table instead. Unsupported sources follow
`on_error`, retaining previously collected metadata in the default recording
mode. Neither querying nor editing columns triggers automatic reading.

Metadata reads use scalar headers and dimension lengths. Telescope timestamps
read only their first and last samples; KIDs identification does not load I/Q
or scan tune LO values. Required acquisition headers remain required. Acquisition
settings such as `atten_drive`, `obs_type_code`, and `n_chans` are available for
selection without reducing the detector data.

### Inspect source processing

`from_sources()`, `update_from_source_metadata()` and `open()` default to `on_error="record"`,
retaining failed rows for inspection. Use `on_error="raise"` to stop on a
filename-validation, opening, identification, metadata-reading or column-projection
failure.
Invalid arguments and ambiguous filename or reader registrations raise in either
mode.

```python
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.core.file import source_trace_issues

inspected = ToltecFileAccessor.from_sources(
    [tel_path, Path("missing.nc")], mode="read", on_error="record"
)
for row in inspected.itertuples(index=False):
    for event in source_trace_issues(row.source_trace):
        print(row.source_loc, event.stage, event.message)
print(inspected.toltec_file.source_trace_summary())
```

Each `source_trace` cell holds a history of immutable `SourceTraceEvent` models.
Stages are `filepath`, `scope`, `open`, `identify`, and `metadata`. Events describe
successes, skips, failures, and whether current information puts a row inside or
outside the selected data models. `exception_type` names an exception when one
occurred. An unrecognized filename
alone is recorded without raising in either mode, with `exception_type=None`.
If no permitted reader recognizes an opened object, all three methods follow
`on_error`: record a `ValueError` event and continue, or raise immediately.
Invalid API arguments still raise.
Use `event.model_dump()` for a dictionary. The CLI's `--show_inputs` output
includes these named fields. `source_trace_summary()` counts rows by their latest
outcome at each stage, with the table size in `total`. Debug logging reports the
outcomes of each operation in the same tabular form.

Metadata and data kinds update together for each row. If identification,
reading or projection into columns fails, that row retains its collected
values, including guessed data kinds. In record mode, other rows still update;
an error event does not supply a replacement classification.
With `update_from_source_metadata(inplace=True, on_error="raise")`, a failure leaves the
original table unchanged.

Retries append events. `source_trace_issues()` returns unresolved errors and
identification misses; a successful retry resolves the previous issue, while
skipping a stage does not. A failed open leaves `io_obj=None`
and does not attempt a metadata read. Identification failures retain `io_obj`
for inspection with `accessor_obj=None`. A metadata failure retains the live accessor
for inspection, but previously guessed columns are not authoritative.

A diagnostic alone does not make a row unusable. For example, a renamed file
can retain a filepath diagnostic after its header is read successfully. The
KIDs input step applies its query first, then skips unusable selected rows by
default. Set `kids.input.raise_on_errors: true` to raise instead. See
[input selection](kids-inputs.md).

### Query the flat columns

Queries use the current table values without opening resources. Canonical labels
are lowercase: `master == "tcs"` or `obs_pgm == "lissajous"`. Numeric header
codes are separate columns, such as `master_code` and `obs_type_code`.

Use `uid_obs` to group an observation's acquisitions and `uid_raw_obs` to
distinguish their sub-observation and scan numbers. When a filename does not
identify its master, these columns omit it: for example, `145609` and
`145609-0-1`. These identifiers are not unique across masters. Reading the
header replaces them with master-qualified identifiers. If only `obsnum` is
available, `uid_obs` can be populated while `uid_raw_obs` remains null.

```python
selected = table.toltec_file.query("bits_any_set(data_kind_toltec, raw_sweep)")
by_filename = table.toltec_file.query("file_suffix == 'tune'")
for row in selected.itertuples(index=False):
    print(row.uid_raw_obs_file, row.obsnum, row.source)
```

`data_kind_toltec` is a nullable integer flag column. Use
`bits_any_set(data_kind_toltec, raw_sweep)` or `bits_all_set(data_kind_toltec, mask)`;
flag constants accept `|`, `&` and `~`. Combine these with pandas comparisons
and membership expressions. Use `.isna()` or `.notna()` to select missing values.
Expressions are trusted Python-facing configuration, not a sandbox.

Use the file accessor's population and metadata-update methods to change collected
information. They update metadata and its derived identifiers together; direct
row or column edits are not a supported update workflow. Use
`update_from_source_metadata(force=True)` to explicitly reread source metadata.

### Discover files in a data tree

`LmtFileStore` represents a `data_lmt` tree; its `toltec` child is a
`ToltecFileStore` rooted at `data_lmt/toltec`. Stores discover paths. File tables
operate on paths regardless of where they are located.
LMT searches its managed `tel` directory and registered instrument stores, not
files directly under `data_lmt`. Selecting a location does not recursively scan
arbitrary subdirectories.

For direct Python use, reuse one selector for candidate discovery and the final
table query:

```python
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore

store_spec = ObsSpec.parse("145609/{tel,0}")
store = LmtFileStore(data_root)
store_paths = store.partial("tel", "toltec").glob_obs_spec(store_spec)
store_table = ToltecFileAccessor.from_sources(store_paths, mode="read")
store_selected = store_table.toltec_file.query_obs_spec(store_spec)
```

The store interprets interface numbers for each search location; the accessor
interprets them for the table query. Positional selectors apply once to the
populated table, not separately to each substore's candidates. `partial()`
selects declared locations without requiring their directory names.
Use `store.tel.glob_obs_spec(spec)` for telescope files alone, or
`store.toltec.glob_obs_spec(spec)` for the TolTEC subtree.

Use class descriptors to select paths and substores, and ObsSpec to constrain
the master and interface:

```python
S = type(store)
selected_store = store.partial(S.tel, S.toltec)
candidate_patterns = list(selected_store.iter_glob_patterns("tcs-156755/{tel,7}"))
selected_paths = selected_store.glob_obs_spec("tcs-156755/{tel,7}")
without_tel = selected_store.partial(exclude=(S.tel,))
readout_store = store.substores[ToltecFileStore]
```

This searches telescope files plus ROACH 7's TCS files. The `toltec` descriptor
selects the entire TolTEC subtree, including HWPR and other interfaces;
ObsSpec constrains which files to discover there. The view leaves the original
store unchanged and can be reused.
Further `partial()` calls narrow the view. Use positional selections to include
their union, `exclude=` to remove selections, or `store_types=` to select by
the classes managing each location. Supply exactly one selection form.
Calling `partial()` without a selection raises `ValueError`;
`partial(exclude=())` removes nothing and `partial(store_types=())` selects nothing.
String selections such as `"tel"` are descriptor names, not path expressions.

`iter_glob_patterns()` also works on whole stores and does not inspect the
filesystem. Patterns are relative to the store's `path`, or a view's `root_path`.
A view's `path` is available when it selects a complete concrete directory:
`store.tel.path.glob("*.nc")` is the low-level telescope-directory API.
A selection spanning only parts of several directories has no single `path`.

To choose capabilities by data model rather than by store layout, use the same
set for searching and table access:

```python
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.file import FileAccessor

store_models = DataModelSet.from_select(["toltec"])
model_search = store.partial(store_types=store_models.file_store_types)
store_files = FileAccessor.partial(accessor_types=store_models.file_accessor_types)
model_table = store_files.from_sources(
    model_search.glob_obs_spec(store_spec), mode="read"
)
model_selected = store_files(model_table).query_obs_spec(store_spec)
```

The accessor view retains its handler selection for building tables and accessing
them afterward. Further `store_files.partial(accessor_types=...)` calls can only narrow
that selection. Filename parsing continues to use all available registrations;
foreign rows keep their filename information. Neither view changes its original
store or the global registrations.

For the combined operation, call `query_file_store()` without creating a runtime
configuration:

```python
from tolteca_datamodels.input_discovery import query_file_store

queried_store = query_file_store(
    store, store_spec, mode="read", data_models=store_models
)
assert queried_store.source_loc.tolist() == model_selected.source_loc.tolist()
```

This discovers candidates, populates their information, and applies the complete
ObsSpec query. It also accepts an already restricted store view, such as
`store.tel`. Omitting `data_models` uses currently registered capabilities;
an explicitly empty set searches no locations. Configured store discovery uses
this same operation.

```python
from tollan.config import RuntimeContext
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_kids.core import Kids

runtime = RuntimeContext({
    "input_discovery": {
        "inputs": [{"data_lmt_path": data_root, "obs_spec": "145609/tel"}]
    },
    "kids": {"input": {"select": 'obs_goal == "pointing"'}},
})
models = DataModelSet.from_select(["toltec"])
discovered = InputDiscovery(runtime).discover(data_models=models)
with discovered.toltec_file.open() as opened:
    opened.toltec_file.update_from_source_metadata(inplace=True)
    identified = Kids(runtime).input.select(opened)
    print(identified[["uid_raw_obs_file", "obs_goal"]])
```

Each discovery entry selects its own candidates before the tables are combined.
A `data_lmt_path` entry searches the LMT tree; `data_toltec_path` searches only
the supplied TolTEC master/interface subtree. An LMT entry includes telescope
files when its interface is unconstrained or includes `tel`. A constraint such
as `/0` excludes telescope files. Explicit file locations can also supply them.
`data_model_select: [toltec]` searches the telescope and TolTEC stores;
`data_model_select: [lmt]` searches only telescope files. A directly supplied
TolTEC subtree stays within its root and does not search for sibling telescope
files.
See
[input discovery settings](kids-inputs.md).
Both tree entries use filename guesses by default; an explicit `sources` entry
accepts exact FileLoc-compatible locations and `{glob: pattern}` resolvers,
reading unrecognized filenames by default. Modes are `guess`, `guess_or_read`,
and `read`. Matches retain entry and glob enumeration order; duplicate locations
and symlink targets keep the first matching row. Unmatched discovery entries
contribute no rows, and discovery may return an empty table.

By default, `InputDiscovery.discover()` uses all registered data-model support and
populates a separate classification column for each instrument. Set
`input_discovery.data_model_select` to choose store requirements and the handlers
used for enrichment;
each input entry can narrow this selection with its own `data_model_select`.
Use a list for exact names, `{"excludes": [...]}` to exclude names, or a regex
string for a fullmatch. An empty selection searches no stores. Explicit sources
still produce a filename-only table without opening sources or adding derived
columns. These settings do not remove other
registered filename information. ObsSpec can still select exact interface
labels; bare numeric interface shortcuts require the corresponding handler.

Pass a `DataModelSet` as above to restrict the capabilities available to a
workflow. Configuration cannot enable handlers outside that set.
Each selected data model supplies the store classes it needs, including companion
data. Explicit `sources` are not filtered by these store requirements.
The KIDs CLI supplies a TolTEC-only set
automatically; the discovery configuration is unchanged.

Stores provide `glob_obs_spec(spec)` for coarse directory discovery. Whole-store
searches include all declared locations and registered substores. Configured
discovery uses `store.partial(store_types=models.file_store_types)` to select the
data models' required locations, including companions. It can reach a required
descendant without searching the parent's own files.
`InputDiscovery` applies the complete selector after populating each entry.
`discovered.toltec_file.open()` opens and identifies the combined candidates.
`opened.toltec_file.update_from_source_metadata(inplace=True)` reads authoritative
header information from those live accessors. Each pipeline selects from that table using
its own input step; KIDs uses `kids.input.select`. Selecting rows does not change
the shared table. Handles remain live within the open scope; signal arrays are
not loaded for selection.

### Build a table across registered instruments

Use `FileAccessor` when the inputs may belong to different instruments.
Importing `tolteca_datamodels.file` enables LMT and TolTEC support, as does input
discovery. Optional instruments opt in:

```python
from tolteca_datamodels.file import FileAccessor
from tolteca_dummy_instru import register

register()  # Optional synthetic instrument support.
mixed = FileAccessor.from_sources([
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "chan01-tcs-1-0-2-20251031T120000.nc",
    "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
])
counts = FileAccessor(mixed).query(
    "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
)
```

To limit discovery's enrichment without losing other registered filename
information, configure the data-model selection:

```python
scoped_runtime = RuntimeContext({"input_discovery": {
    "data_model_select": ["toltec"],
    "inputs": [{"sources": mixed.source.tolist(), "mode": "guess"}],
}})
scoped_mixed = InputDiscovery(scoped_runtime).discover()
scoped_rows = scoped_mixed.toltec_file.query(
    "bits_any_set(data_kind_toltec, lmt_tel)"
)
```

This table still includes the dummy filename information, but only has a
TolTEC classification column. `data_model_select` accepts a
{py:class}`tollan.config.types.StringSelector`: an exact-name list, an exclusion
mapping, or a fullmatch regex. Registered names include `lmt`, `toltec`, and,
after opt-in, `dummy_instru`. The default selects all registered support.
Discovery returns an ordinary table, not a stored workflow context. Use the
appropriate concrete accessor for subsequent instrument-specific queries,
metadata reads, or opening. Direct `FileAccessor.from_sources()` uses all
registered handlers; concrete factories use only their declared support.

The same `guess`, `guess_or_read`, and `read` modes apply. Unidentified rows
remain in the table. Instrument-specific telescope companions receive that
instrument's classification; other recognized telescope files use LMT support.
Two equally specific accessors recognizing a file produce an error, not an
import-order-dependent choice.

The table has separate integer flag columns: `data_kind_toltec` and
`data_kind_dummy_instru`. Each classifier assigns its own `unknown` flag to
unmatched rows, including foreign files and unidentified inputs. Concrete
factories use the same column names, but populate only their own classifications.

Use the matching enum class in each flag query. Combine query masks to select
across instruments:

```python
selected_kinds = FileAccessor(mixed).query(
    "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep) | "
    "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
)
```

There is no combined `data_kind` column. Successful header reads replace
contradictory filename classifications with the corresponding instrument's
`unknown` flag.

Use `FileAccessor(mixed).update_from_source_metadata()` to enrich the table or
`with FileAccessor(mixed).open() as opened:` to access its files. Header
recognition can recover renamed files independently of their filename guesses.
Opening selects a reader by file format: netCDF produces an xarray Dataset,
and ECSV produces an Astropy QTable. For cached files, `source_loc` supplies the
format extension while `source` supplies the file to open. Supplied objects are
borrowed unchanged. Current table values select an instrument reader when a
reader rule matches; otherwise, content identification selects the reader.
Opening stops after identification. Request metadata updates explicitly when
header values are needed for querying.

Filename models populate the generic table even without an instrument accessor.
Derived columns such as data kinds require registered instrument support.
Register filename recognition with `register_filepath_parser` from
`tolteca_datamodels.core.filepath`: provide a named-capture regex, an ordered tuple of
field converters, and the result model class. Each converter receives the
current capture dict and path, then returns values to merge into the dict.
Defining a model alone does not register a filename parser.

For LMT acquisition models, inherit `FilepathInfoMixin` from
`tolteca_datamodels.lmt_base.filepath`. Parsers must supply `master` and `suffix`, using
`None` when unavailable; neither field has a default. Optional named regex
groups supply `None` when unmatched. If the pattern has no group for a required
field, a converter must supply it.

Supported scalar fields, including fields added by
subclasses, automatically receive `file_*` columns and unprefixed aliases.
Filename-only fields (`prefix`, `suffix`, `ext`, `ut`, and `proc_suffix`) receive
only `file_*` columns. Other filename models without declared columns receive
scalar `file_*` columns at class definition. Each filename model uses its own
column mapping. Different attributes and converters may populate the same column,
but must supply values compatible with its destination dtype. Incompatible
writes raise during table population; they do not widen the column to object.

Use `SourceOpener` from `tolteca_datamodels.core.source` to define format-opening
rules. All constraints must match; the first matching rule wins. Extensions omit
the leading dot. Put an override before the built-in `source_opener.rules` to
customize opening without changing another data model's policy:

```python
from astropy.table import QTable
from tolteca_datamodels.core.source import SourceOpener, source_opener

def open_csv(path, **kwargs):
    return QTable.read(path, format="ascii.csv", **kwargs)

csv_opener = SourceOpener(
    rules=(({"ext": "csv"}, open_csv), *source_opener.rules)
)
```

Use `csv_opener.open({"ext": "csv"}, source=path)` to open a file without
instrument identification. Supply rule values as a mapping or an object with
matching attributes; the source is passed unchanged to the chosen callable.
For multiple rows, `csv_opener.iter_open_table(table, source_col="source",
column_names={"ext": "file_ext"})` yields opened objects in row order. The caller
owns each yielded object, including when a later row fails. Matching and source
collection happen when the iterator is created; opening advances one row at a
time. For example, with in-memory text sources:

```python
from io import StringIO
import pandas as pd
from tolteca_datamodels.core.source import SourceOpener

opener = SourceOpener(rules=(({"format": "text"}, StringIO),))
with opener.open({"format": "text"}, source="measurement") as stream:
    assert stream.read() == "measurement"

inputs = pd.DataFrame({
    "format": ["text", "text"],
    "payload": ["first", "second"],
})
contents = []
for stream in opener.iter_open_table(inputs, source_col="payload"):
    with stream:
        contents.append(stream.read())
assert contents == ["first", "second"]
```

`opener(record)` and `opener.map_table(table)` return callables without opening
sources; unmatched rows receive None. A selected callable's failure propagates
without trying another rule or closing previously yielded objects.

Typed openers such as `open_lmt_tel()` open a known format and bind its known
accessor without selecting among readers. When implementing a typed opener for
an extension, close newly opened resources if accessor initialization fails.
Return successful resources to the caller, who owns their cleanup. Full metadata
and samples remain available on demand through the accessor.

Define your data model's identity with `DataModelInfo`, then bind the file
accessor to that class. The concrete accessor can be used directly without
registration. To enable it in generic workflows, declare its capabilities in a
`DataModelSpec` and register it before selecting a context or building tables:

```python
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.types import DataModelInfo

class MyDataModel(DataModelInfo):
    name = "my_data_model"

class MyFileAccessor(FileAccessor[MyDataModel]):
    _source_opener = csv_opener

assert MyFileAccessor.data_model_cls is MyDataModel
data_model = DataModelSpec(file_accessor_cls=MyFileAccessor)
assert data_model.name == MyDataModel.name
data_model.register()
```

When selected data models share an opener for a row, the file is opened once.
Different opener choices for the same row raise an ambiguity before that file
is opened. Already-open objects bypass opener selection and go to the readers.
File-table workflows pass the original `source` to the selected callable, not
`source_loc`. Custom openers may accept remote locations; the built-in NetCDF
and ECSV openers require local files.

The built-in definitions are available as
`from tolteca_datamodels.lmt.data_model import data_model` and
`from tolteca_datamodels.toltec.data_model import data_model`.
These imports do not register the definitions. Import
`tolteca_datamodels.data_model`, `tolteca_datamodels.file`, or
`tolteca_datamodels.input_discovery` to enable the built-in workflow support.
Importing a component such as `tolteca_datamodels.toltec.file_store` does not
assemble workflow capabilities. The `tolteca_datamodels.filepath` entry point
enables filename parsers without registering workflow support.
For store-backed discovery, include `file_store_types` in the definition.
For example, TolTEC declares `(LmtFileStore, ToltecFileStore)` so a TolTEC
workflow searches telescope companions as well as instrument data. Omit this
field for a data model used only with explicit sources. Shared stores are
searched once when several selected data models require them.
In Python, use `LmtDataModel.name` or `ToltecDataModel.name` from their respective
`types` modules for the built-in handler names. For example, `ToltecFileAccessor`
inherits `FileAccessor[ToltecDataModel]`. Its `data_model_cls` identifies the
handling it supplies; a record's `data_models` lists the handling that applies
to that record. A `tel_toltec` record supports both `"lmt"` and `"toltec"`,
without giving either accessor more than one data-model identity.
Configuration and membership queries still use the string names
`"lmt"` and `"toltec"`.
Registering an existing name raises `ValueError`. Existing contexts retain their
selected definitions when additional names are registered. Direct concrete-accessor
calls use only their declared support. For example, `LmtFileAccessor` accepts telescope data for
any instrument; `ToltecFileAccessor` accepts TolTEC KIDs, HWPR and telescope
companions identified by their recorded instrument header, even under unfamiliar
filenames.

Declare `_reader_rules` with `SourceReaderRules` from
`tolteca_datamodels.core.source`. Rules return accessor classes directly.
For example, both LMT and TolTEC can select `LmtTelAccessor` for telescope data:

```python
from tollan.accessor.pandas.model_table import Contains
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.types import ToltecDataModel

readers = SourceReaderRules(rules=((
    {"interface": "tel", "data_models": Contains(value=ToltecDataModel.name)},
    LmtTelAccessor,
),))
assert readers({"interface": "tel", "data_models": ["lmt", "toltec"]}) is LmtTelAccessor
assert readers({"interface": "tel", "data_models": ["lmt"]}) is None
```

Supply `prepare_record(opened_object)` when unfamiliar filenames need content
identification. Return a mapping or object containing routing values, or None
for an unsupported payload. Read only the needed identity headers, not full
metadata or signal arrays. `readers.identify(opened_object)` applies the same
rules to those values and returns a class without constructing it.

Use `rules` to select readers from current query values. Each entry pairs a
constraint dict with an accessor class; the first matching rule wins within
that data model. Reader selection uses `data_models` for
current scope and identity columns such as `interface`, not their `file_*`
snapshots. File-only attributes such as `ext` use `file_ext` when there is no
current column with that name. A current null does not restore an older value.

A unique rule result constructs the selected accessor on the opened object.
Otherwise, content identification resolves the remaining choices. Multiple
remaining accessor classes raise. Failed construction or metadata reading
does not try another reader. A shared accessor reads `meta` once; each accepting
data model applies its own `_metadata_columns` and optional
`_classify_metadata(reader, metadata)` hook for its classification flag.

For already-open objects, pass them as table sources with `source_loc` and use
the file accessor's `open()` method. It identifies the borrowed objects without
closing them. Use the instrument's `identify_*` function when you already know
the reader you need, or its `open_*` function to open a new resource.

Declare `_metadata_columns` on the accessor registering the reader. These mappings
apply to metadata returned by that accessor's readers; they need not use the same
attribute paths or converters as another accessor's mappings. Successful reads
replace metadata-covered query values, clearing stale guesses absent from the
recovered metadata. Filename-derived `file_*` values remain unchanged.

### Enable an additional discovery input

An extension can supply its own settings model and discovery operation. Enable
the extension before loading configuration. For the synthetic test instrument:

```python
from tolteca_dummy_instru import register

register()
runtime = RuntimeContext({
    "input_discovery": {"inputs": [{
        "data_dummy_instru_path": "/data/data_lmt/dummy_instru",
        "obs_spec": "1/(1,3)",
        "mode": "guess",
    }]},
})
discovered = InputDiscovery(runtime).discover()
```

Use `FileAccessor(discovered)` to query or open this table. For example,
`FileAccessor(discovered).query("bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)")`
selects counter observations. A configuration can combine this entry with
TolTEC and telescope inputs; each row retains its instrument's classification.
For a dummy-only task, pass
`DataModelSet.from_select(["dummy_instru"])`
as the `data_models` argument to `discover()`.

To provide a new input type, subclass `InputDiscoveryEntry` from
`tolteca_datamodels.core.input_discovery`, declare its settings, and implement
`discover(*, data_models=None, on_error="record")` to return a file table. Select
the data models with `models = self._select_data_models(data_models)`, then create
`files = FileAccessor.partial(accessor_types=models.file_accessor_types)` from
`tolteca_datamodels.core.file`. Use `files.from_sources(...)` with the resolved
candidates and `files(table).query_obs_spec(...)` when an observation selector is
provided. This applies the entry's data-model selection within the workflow's scope.
Store-backed entries can inherit `FileStoreInputDiscovery[ObsSpecField]` from
`tolteca_datamodels.core.input_discovery` and provide a private `_store`
property. Bind the generic parameter to the annotated selector field type for
the store; it controls validation and serialization, including the `"*"` default.
The entry searches the selected data models' required stores, then applies
ObsSpec to the populated table using the same data-model support.
For an LMT-layout store, inherit `LmtBaseFileStoreInputDiscovery` from
`tolteca_datamodels.lmt_base.input_discovery` to use the LMT observation selector.
Register the settings model with
`InputDiscoveryConfig.register_input(key, model_cls)`. The key must be a required,
unaliased field of the model. Each configured entry must contain exactly one
registered identifying key; conflicting keys are rejected. Registering an existing
key or model raises an error, even when repeating the same pair.
Register extensions before defining enclosing configuration models or creating
Pydantic type adapters for discovery settings.

Python callers can instantiate the concrete settings model and call its
`discover()` method directly, without registering the entry type. Pass a
`DataModelSet` explicitly, or use the registered support enabled by the
entry points above. Import `LmtInputDiscovery`
from `tolteca_datamodels.lmt.input_discovery`, or `ToltecInputDiscovery` from
`tolteca_datamodels.toltec.input_discovery`.

### Use a different identification path

`source` is the resource to open. `source_loc` optionally identifies a cached
resource by a different path, such as a location returned by a remote database.
The entire identification path is used; fields are not mixed between names.

```python
cached = guess_info_from_sources(
    [{
        "source": "cached.nc",
        "source_loc": "archive:/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
    }],
)
print(next(cached.itertuples(index=False)).obsnum)
```

Both columns use object dtype. `source` preserves the supplied resource;
`source_loc` contains a validated `FileLoc`. An omitted or None `source_loc`
uses `source` for validation, so open handles require an explicit location. Access
the columns through `table.toltec_file.source` and `.source_loc`, or on each
row from `itertuples()`. Describing a remote source does not download it.
Stage it locally or supply an already-open xarray object before
reading its data. Header `filepath_orig` is acquisition metadata, not a fallback
identification path; a renamed file keeps null filename columns unless an
explicit `source_loc` is supplied.

Filter identification locations with `source_loc_is_local`, `source_loc_host`
and `source_loc_scheme`:

```python
remote = cached.toltec_file.query(
    'not source_loc_is_local and source_loc_host == "archive"'
)
```

These columns describe `source_loc`, even when `source` is a local cached file
or an open object. Local file locations have an empty host; HTTP/HTTPS locations
are remote, including those on localhost. No availability checks are performed.

### Work with open resources

```python
with table.toltec_file.open() as opened:
    opened.toltec_file.update_from_source_metadata(inplace=True)
    selected_tel = opened.toltec_file.query("interface == 'tel'")
    for accessor in selected_tel.accessor_obj:
        az = accessor.az.isel(time=slice(0, 100)).load()
```

Each distinct local file opens once per scope. `io_obj` holds the opened payload;
`accessor_obj` holds its identified data accessor. Opening does not read or
project full metadata. The explicit update above reuses those live accessors;
omit it when only direct data access is needed. Use `accessor_obj` to reuse the
table's metadata cache; a separately obtained xarray accessor has its own cache.
An unrecognized payload has
`accessor_obj=None` and remains available in `io_obj` for inspection.

For both operations at once, use `open(update_from_metadata=True)`. The
`metadata_read` column is False for filename guesses and becomes True only after
a successful source read and table update. Updates skip those completed rows by
default, including rows populated during construction. Use
`update_from_source_metadata(force=True)` or
`open(update_from_metadata=True, force=True)` to reread them. A failed forced
read retains the previous values and flag, and adds an error to `source_trace`.

Owned files close on success or failure. Both temporary object columns are
removed on exit; projected metadata remains. Load any file-backed arrays inside
the scope. Tables returned by `mode="read"` or `mode="guess_or_read"` retain
successful reader choices for reopening without repeating content identification.
Reopening binds a new accessor to the newly opened payload. Changes to the source
or applicable reader choices require identification again.

An entry may supply an already-open Dataset or DataTree as `source` and its
identification path as `source_loc`. Alternatively, pass one Dataset, DataTree,
or None per row as `io_objs` (`None` uses that row's source). Borrowed objects
are never closed. Selection keeps those objects aligned with their rows.

### Project other objects into a table

Use `ModelColumnMapper` to select columns from LMT metadata. See
[Tollan's model-table guide](https://toltec-astro.github.io/tollan/model-tables.html)
for attribute projection, name and type rules, and positional updates.

For metadata quantities, configure explicit units in the column mapper:

```python
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tollan.accessor.pandas.model_table import (
    ModelColumnMapper, ModelTableAccessor, QuantityColumnRule, ScalarColumnRule,
    SkyCoordColumnRule, TimeColumnRule,
)

mapper = ModelColumnMapper(
    name_rules=[(r".+", r"\g<0>")],
    type_rules=[
        QuantityColumnRule(physical_type="time", unit="s"),
        TimeColumnRule(format="mjd"),
        SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
        ScalarColumnRule(),
    ],
    include_computed=True,
    default=None,
)
metadata_columns = mapper(LmtTelMetadata)
metadata_table = ModelTableAccessor.from_models([meta], columns=metadata_columns)
print(metadata_table[["t0_mjd", "t_int_s", "target_ra_deg"]])
```

This example represents integration times in seconds, timestamps in MJD in their
own scale, and target coordinates in degrees. To project the LMT target under
two column prefixes, supply two matching name rules:

```python
coordinates = ModelColumnMapper(
    name_rules=[(r"target", r"myobj_\g<0>"), (r"target", r"\g<0>")],
    type_rules=[SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg")],
)
coordinate_columns = coordinates(LmtTelMetadata)
assert list(coordinate_columns) == [
    "myobj_target_ra_deg", "myobj_target_dec_deg", "myobj_target_frame",
    "target_ra_deg", "target_dec_deg", "target_frame",
]
```

Choose `attrs=["az", "alt"]` for horizontal coordinates or `attrs=["l", "b"]`
for Galactic coordinates. The values must already provide those attributes;
projection does not transform frames. Include `target_frame` in positional
queries when observations may use different frames.

### Consume selected data

Use pandas `itertuples()` for row-wise query information. Open selected local
files for their typed metadata and arrays. For the `by_filename` selection above:

```python
from tolteca_datamodels.toltec import open_toltec_kids

for source in by_filename.toltec_file.source:
    with open_toltec_kids(source) as data:
        print(data.toltec_kids.meta.uid_raw_obs_file)
```

`open_toltec_kids()` requires a recognized KIDs classification; it closes the dataset
and raises if identification fails. Use `open_lmt_tel()` for telescope files
and `open_toltec_hwpr()` for half-wave plate readout files.
Full observation metadata and signal arrays remain deferred until accessed.
These openers accept a path, a local file URL or a `FileLoc`. Remote locations
must be staged locally before opening.

Processed files recognized by the native KIDs reader remain openable and
classifiable, but their `meta` is currently `None`. Acquisition identity and
links back to acquisition metadata are not yet exposed through this reader.
File tables retain collected filename information without adding acquisition
metadata for these files; do not treat those guesses as header-verified values.
Classification alone does not set `metadata_read=True`; the trace records that
metadata population was skipped.

When passing an already-open DataTree to a native reader or file table, supply
the node that contains the acquisition dataset. Native readers inspect that
node only; they do not search its children. For a KIDs processing tree, use
`tree.toltec_kids.data_source` to obtain its acquisition dataset.

When constructing KIDs or HWPR metadata directly in Python, use `"tcs"`, `"ics"`,
or `"clip"` for `master` and `ToltecDBRawObsMasterCode` members from
`tolteca_datamodels.toltec.toltec_db` for `master_code`. JSON and projected table
columns use string labels and integer codes, respectively. `ToltecInfo.MasterT`
and `ToltecInfo.ObsTypeT` constrain the accepted labels; `ToltecInfo.ArrayT` and
`ToltecInfo.ArrayNameT` constrain the numeric `array` and string `array_name`.
For display descriptions, use `ToltecInfo.master_description[master]` and
`ToltecInfo.array_name_description[array_name]`.
Both models require `instru="toltec"` and `data_models=["toltec"]`. HWPR metadata
also requires `interface="hwpr"` and `interface_kind="hwpr"`; KIDs metadata
requires the detector interface and its kind. These identity fields have no defaults.

### Work with HWPR and detector files together

Select HWPR by its interface label, not a ROACH number. For example, observation
156755 in the reference data contains an HWPR nominal scan and tune/nominal
scans for ROACHes 7 and 11:

```python
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore

readout_paths = ToltecFileStore(data_root / "toltec").glob_obs_spec(
    "tcs-156755/{hwpr,7,11}"
)
readouts = ToltecFileAccessor.from_sources(readout_paths, mode="read")
print(readouts[["interface", "readout_code", "roach", "scannum", "obs_type"]])
hwpr = readouts.toltec_file.query("bits_any_set(data_kind_toltec, hwpr)")
detectors = readouts.toltec_file.query("bits_any_set(data_kind_toltec, raw_kids_data)")

with hwpr.toltec_file.open(on_error="raise") as opened:
    row = next(opened.itertuples(index=False))
    encoder = row.io_obj.toltec_hwpr.enc_val.isel(time=slice(0, 10)).load()
```

`readout_code` is available in KIDs file and block metadata, HWPR metadata, and
header-populated table rows. It retains the recorded code: 13 for this HWPR file,
7 and 11 for its detector companions. HWPR has no ROACH identity; its `roach`,
`nw`, and array columns remain null. A numeric ObsSpec interface selector such as `/13`
does not select HWPR. Use `/hwpr` or a mixed set such as `/{hwpr,7,11}`.

For a single file, use the typed opener:

```python
from tolteca_datamodels.toltec import open_toltec_hwpr

with open_toltec_hwpr(hwpr.source.iloc[0]) as data:
    hwpr_meta = data.toltec_hwpr.meta
    print(hwpr_meta.readout_code, hwpr_meta.uid_raw_obs_file)
    encoder = data.toltec_hwpr.enc_val.isel(time=slice(0, 10)).load()
```

The accessor exposes native encoder, zero-point, PPS and sensor arrays with
their recorded shapes and values. It does not convert encoder counts to angle
or align HWPR samples to detector timestamps. Metadata reads load headers only.
`t_int` is the readout record count divided by the recorded record rate;
`t_elapsed` is the difference between the acquisition end/start headers.
The reference HWPR encoder is stationary, so it does not exercise rotation.

### Use table rows in conditional configuration

For conditional KIDs configuration, pass a row's values as a mapping:

```python
row = next(table.itertuples(index=False))
context = row._asdict()
print(context["data_kind_toltec"])
```

The values retain their table types, including integer data-kind flags,
timestamps and nullable cells.

## Check a KIDs run from the CLI

Use the same raw-data root to inspect the KIDs files for observation 145609:

```console
tolteca_kids --data_lmt_path /data_lmt --show_inputs -- '145609/0'
tolteca_kids --data_lmt_path /data_lmt --dry_run -- '145609/0'
```

The [ObsSpec](obs_spec.md) `145609/0` selects numbered interface 0 (`toltec0`
in the TolTEC store). `145609/toltec(0,5)` restricts selection to ROACHes 0
through 4; `145609/(0,5)` uses each store's interface-to-number mapping, which
can include interfaces from other instruments too. Selectors first narrow
filesystem discovery, then filter filename identifiers. Headers are read before
the CLI's `--select` expression. Explicit file locations need no data
root. Overlapping inputs, including symlinks, are deduplicated.

An unqualified selector searches both telescope and instrument trees. `/tel`
searches only the telescope directory; ROACH selectors skip it. Telescope
filenames identify master `tcs`,
so `tcs-145609/tel` selects telescope files without opening their headers.

Configure paths and input selection in a recipe, for example `check.yaml`:

```yaml
input_discovery:
  inputs:
    - data_lmt_path: /data_lmt
      obs_spec: 145609/0
kids:
  input:
    select: contains(data_models, 'toltec') & roach == 0 & file_suffix == 'tune'
  sweep_reduce:
    block: -1
```

```console
tolteca_kids --config_paths check.yaml --show_config
tolteca_kids --config_paths check.yaml --dry_run --kids.sweep_reduce.block 0
```

Put positional inputs after `--` to delimit multi-value options. Later config
paths override earlier ones. Positional inputs replace configured discovery entries;
`--select` overrides `kids.input.select`. Dotted `--kids.*` options override
processing settings. Conditional
source-list recipes use each input's metadata for `enable_if`; `--select`
instead chooses which inputs to run. See [KIDs inputs](kids-inputs.md) for
these options and conditional-configuration rules.

`--show_inputs` prints the selected table; `--show_config` prints each input's
validated configuration. Both exit after input selection, before reduction or
output. `--dry_run` executes
a real no-op step with a live context, but does not reduce signals or write
products. The product root alone does not enable output. Source files close on
success and failure. Add `--log_level DEBUG` for discovery, condition decisions,
configuration and step timings; logs include paths and config values.

A successful no-op prints `no-op completed` for each input. It checks the
input/configuration/execution path, not scientific reduction. Continue with
[KIDs reduction](kids-reduction.md) when you are ready to compute and export a result.

## Invoke the same check through Deploy

Use Deploy's `run` command to invoke the same KIDs command:

```console
tolteca_deploy run -- tolteca_kids --config_paths check.yaml --dry_run
tolteca_deploy run -- tolteca_kids --data_lmt_path /data_lmt --dry_run -- '145609/0'
```

Options after Deploy's `--` belong to the launched command. Supply the KIDs
recipe or data path explicitly: `run` does not derive KIDs arguments from
deployment settings or automatically forward its own config options. Kids does
not consume deployment environment variables.

Configured `input_discovery.inputs` remain authoritative when the command also
supplies `--data_lmt_path`. That root is used for positional ObsSpec, or as the
discovery default when no entries are configured. An orchestrator can invoke
the same argument vector from the same working directory.
