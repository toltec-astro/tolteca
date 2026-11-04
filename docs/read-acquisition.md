# Read one acquisition

Use a typed opener when you know which file you want and which format it holds.
This is the lowest level of the [observation-data stack](datamodels.md#choose-a-level):
one resource, one reader, no table. To compare or select among many files, build a
[file table](file-tables.md) instead.

The examples use observation 145609 from the reference data. `data_root` is the
directory that holds your raw data, with the `tel/` and `toltec/` trees below it.
Examples read inputs in place; save tables and products outside that directory.

```{testsetup} read_acquisition
data_root = workspace / "toltec_astro_dev/run/data_lmt"
```

## Choose an opener

| Data | Python entry point | What it exposes |
| --- | --- | --- |
| Telescope telemetry | `tolteca_datamodels.lmt.open_lmt_tel` | Observation metadata and pointing arrays through `.lmt_tel` |
| TolTEC detector readout | `tolteca_datamodels.toltec.open_toltec_kids` | Acquisition metadata and detector data through `.toltec_kids` |
| TolTEC half-wave plate | `tolteca_datamodels.toltec.open_toltec_hwpr` | HWPR metadata and readout through `.toltec_hwpr` |

Metadata describes the recorded acquisition; the filename is preliminary evidence.
Keep the resource open while reading arrays backed by that resource.

## Read the telescope metadata

Open a telescope file with `open_lmt_tel()` and use its `lmt_tel` accessor:

```{testcode} read_acquisition
import astropy.units as u
from tolteca_datamodels.lmt import open_lmt_tel

tel_path = data_root / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"

with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    meta = tel.meta
    print(meta.uid_raw_obs_file, meta.obs_pgm, meta.obs_goal)
    print(f"{meta.t_int.to_value(u.s):.2f} {meta.t_elapsed.to_value(u.s):.2f}")
    print(f"{meta.target.ra.to_value(u.deg):.4f} {meta.target_ref.ra.to_value(u.deg):.4f}")
    az = tel.az.isel(time=slice(0, 100)).load()

assert az.sizes["time"] == 100
```

```{testoutput} read_acquisition
tcs-145609-0-2-tel lissajous pointing
60.32 68.51
72.3471 72.3471
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

```{testcode} read_acquisition
with open_lmt_tel(tel_path) as ds:
    tel = ds.lmt_tel
    flags = tel.mapper.get_arr(tel.data_source, tel.mapper.schema.hold_flag)
    first_flags = flags.isel(time=slice(0, 100)).load()
    timestamps = tel.tel_time.isel(time=slice(0, 100)).load()
    print(tel.az.u.unit)

assert first_flags.sizes["time"] == timestamps.sizes["time"] == 100
```

```{testoutput} read_acquisition
rad
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

## Use a reader on an open dataset

For an already opened telescope dataset, use its named accessor directly:

```{testcode} read_acquisition
import xarray as xr
from tolteca_datamodels.lmt import LmtTelAccessor

with xr.open_dataset(tel_path) as data:
    reader = data.lmt_tel
    assert isinstance(reader, LmtTelAccessor)
    print(reader.meta.uid_raw_obs_file)
```

```{testoutput} read_acquisition
tcs-145609-0-2-tel
```

Accessing `data.lmt_tel` validates telescope structure and reuses the accessor
on subsequent access. It does not read sample arrays or validate all observation
headers; invalid headers raise when accessing `reader.meta`. Keep the dataset
open while using the reader. For objects whose reader is not known, use a
[file table](file-tables.md) to identify them through the selected data models.

## Distinguish acquisition metadata from file locations

Raw observation metadata includes the required acquisition path `filepath_orig`.
A file's current location can differ from that path after copying or renaming it.
`filepath_info_orig` parses the recorded path once:

```{testcode} read_acquisition
info = meta.filepath_info_orig
assert info is not None
print(meta.filepath_orig)
print(info.prefix, info.ut)
```

```{testoutput} read_acquisition
/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc
tel_toltec 2025-10-31 00:00:00+00:00
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

(observation-source-access)=

### Sources, locations and metadata

Use `SourceSourceLoc` from `tolteca_datamodels.source` to prepare one resource
with the filename used to recognize it:

| Value | Purpose |
| --- | --- |
| `source` | The resource to open or borrow: a path, FileLoc, or already-open object. |
| `source_loc` | The identification location used to parse its filename. It defaults to `source`; supply it for an open object or when the useful filename differs from the resource location. |
| `filepath_info` | Information inferred from that filename, without opening the resource. |
| Reader metadata | Information obtained from the opened data, used to replace filename guesses. |

Here, “source” means a data resource; an astronomical source is the observed
target. Validation parses the filename and opens nothing. Importing
`SourceSourceLoc` enables the built-in filename support:

```{testcode} read_acquisition
from tolteca_datamodels.source import SourceSourceLoc

item = SourceSourceLoc.model_validate("archive:/renamed.nc")
assert item.filepath_info.ext == "nc"
assert item.filepath_info.data_models == frozenset()
assert item.source_loc_host == "archive"
```

Use `with item.open() as resource:` for local NetCDF or ECSV input. An explicit
`source_loc` supplies the original filename for a renamed local source or an
already-open object, but opening uses `source`, not `source_loc`; the default
opener requires remote resources to be staged locally first. A resource opened by
this context is closed on exit when closable, while a borrowed object stays with
its caller. Pass a `SourceOpener` policy with `item.open(opener=my_opener)` to
customize opening; see `SourceOpener` for composing policies.

Standalone validation and opening raise on failure and do not read metadata. To
prepare many sources and record per-source failures instead, build a
[file table](file-tables.md#build-a-table).

### Parse a filename

Use `FilepathInfo.from_source()` to parse a local path or remote file location
without opening it:

```{testcode} read_acquisition
from tolteca_datamodels.filepath import FilepathInfo

filepath_info = FilepathInfo.from_source(tel_path)
print(filepath_info.interface, filepath_info.ut)
print(filepath_info.quartet)

assert filepath_info.interface == "tel"
```

```{testoutput} read_acquisition
tel 2025-10-31 00:00:00+00:00
tcs-145609-0-2
```

The factory returns the information available in the recognized filename layout;
for example, TolTEC readout names also provide `roach`. It accepts a `FileLoc`.
Unfamiliar filenames use the basic parser, which supplies `ext` and an empty
`data_models` set; every parsed object provides `data_models` as a frozenset, and
an empty set means the filename identifies no data model, not that the contents are
unreadable. A location without a filename raises `ValueError`, or returns `None`
with `raise_on_error=False`. Invalid values in a recognized layout, such as an
invalid date, still raise.

`FilepathInfo.from_source()` considers every built-in layout. A concrete layout's own
`from_source()` accepts only that class and its registered descendants. It raises
`ValueError` when no layout is eligible, unless `raise_on_error=False` is supplied:

```{testcode} read_acquisition
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo

tel_info = LmtFilepathInfo.from_source(tel_path)
assert type(tel_info).__name__ == "ToltecLmtFilepathInfo"

readout_name = "toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
assert LmtFilepathInfo.from_source(readout_name, raise_on_error=False) is None
```

## Next

- Compare, select and open many files: [file tables](file-tables.md).
- Search a data tree for the files of an observation: [file stores](file-stores.md).
- Select by observation identifiers: [observation selectors](obs_spec.md).
