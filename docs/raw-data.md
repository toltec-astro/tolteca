# Reading observation data

Open a selected KIDs or telescope netCDF file as an xarray Dataset. Its accessor
provides typed metadata and arrays that can be sliced before loading:

```python
from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.lmt import open_lmt_tel

with open_toltec("toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc") as ds:
    kids = ds.toltec_kids
    print(kids.meta.obsnum, kids.meta.n_chans, kids.meta.atten_drive)
    print(kids.f_tone, kids.tone_mask)

with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    tel = ds.lmt_tel
    print(tel.meta.obs_goal, tel.meta.target)
    az = tel.az.isel(time=slice(0, 100)).load()
```

Keep the dataset open until the desired arrays are loaded. Both accessors also
work on an xarray DataTree, using the current node's dataset. For discovering
and selecting input files, see [Discovering and selecting files](file-catalogs.md).

`open_lmt_tel()` accepts local paths, file URLs, and local `FileLoc` objects.
Stage remote telescope files locally before opening them.

Opening resolves the telescope schema and checks for its timestamp variable
without reading sample values or observation headers. A dataset without that
variable raises `ValueError`; the helper closes it before raising. Accessing
`.lmt_tel` on an already-open Dataset or DataTree applies the same check.
Required metadata fields and timestamp values are checked when you request
`.meta`.

## Inspect observation metadata

`kids.meta` returns `ToltecSweepMetadata` or `ToltecTimeStreamMetadata`;
`tel.meta` returns `LmtTelMetadata`. Time, quantity, and sky-coordinate fields
use Astropy objects, so units and coordinate conversions are available directly:

```python
import astropy.units as u

with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    meta = ds.lmt_tel.meta
    print(meta.t0.isot, meta.t1.isot)
    print(meta.t_exp.to_value(u.s), meta.t_elapsed.to_value(u.s))
    print(meta.target.ra.to_value(u.deg), meta.target.dec.to_value(u.deg))
    print(meta.obs_pgm, meta.obs_goal)
    print(meta.master, meta.master_str)
    print(meta.uid_raw_obs, meta.uid_raw_obs_file)
```

| Telescope metadata | Meaning |
| --- | --- |
| `t0`, `t1` | First and last recorded sample times |
| `t_exp` | Recorded integration time, which may be zero and need not equal `t1 - t0` |
| `t_elapsed` | Elapsed time `t1 - t0`, derived from the first and last telescope timestamps |
| `target`, `target_ref` | Main and reference sky positions; the latter is a position, not an offset vector |
| `obs_pgm`, `obs_goal` | Observation program and goal, such as `Lissajous` and `pointing` |
| `master`, `master_str` | Recorded acquisition-master code and name, such as `0` and `tcs`; `None` if unavailable |
| `instru`, `instru_component`, `interface` | Instrument, component, and readout actor: `lmt`, `tcs`, and `tel` |
| `filepath_orig` | Original acquisition file path recorded in the header |

Use `quartet` or its alias `uid_raw_obs` to identify an observation across
interfaces: for example, `tcs-145606-0-1`. `uid_raw_obs_file` appends the
interface, giving `tcs-145606-0-1-tel`. These identifiers have no zero padding
and are `None` when the master name is unavailable. They identify acquisitions,
not storage locations: copies of the same acquisition retain the same ID.

KIDs integration time is the sample count divided by the sample rate. Its
acquisition timestamps describe the containing file. Unavailable optional
fields, such as LO-center frequency, remain `None`; missing required headers
raise an error.

Metadata is frozen and cached after the first read. Create a new dataset/accessor
if you change the underlying headers. Reading metadata does not load detector
I/Q or telescope pointing streams. Telescope metadata reads the endpoints of
the time array; tune metadata also scans LO frequency to identify sweep blocks.

### Inspect the original acquisition name

The telescope metadata's `filepath_info_orig` property parses the recorded
acquisition path on first access and caches the result:

```python
with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    meta = ds.lmt_tel.meta
    print(meta.filepath_orig)
    print(meta.filepath_info_orig.ut, meta.filepath_info_orig.ext)
```

Its fields have the same meanings as the
[catalog's parsed path information](file-catalogs.md).
An unrecognized original filename raises `ValueError` when this property is
accessed; it does not prevent ordinary metadata reading.

To check that header observation numbers and master agree with the recorded
path, request the optional consistency check:

```python
with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    tel = ds.lmt_tel
    checked = tel.mapper.get_metadata(tel.data_source, check_filepath_orig=True)
```

Conflicts raise `ValueError`. This check is off by default.

## Read telescope pointing and flags

Select a slice of native pointing or timestamp arrays before loading it:

```python
with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    tel = ds.lmt_tel
    az = tel.az.isel(time=slice(0, 100)).load()
    timestamps = tel.tel_time.isel(time=slice(0, 100)).load()
    print(az.u.unit)
```

| Accessor property | Values | Loading behavior |
| --- | --- | --- |
| `az`, `alt`, `ra`, `dec` | Native pointing angles in radians | Lazy; slice before loading |
| `az_des`, `alt_des` | Desired horizontal pointing in radians | Lazy; slice before loading |
| `az_map`, `alt_map` | Map coordinates in radians | Lazy; slice before loading |
| `tel_time` | Native Unix timestamps | Lazy; slice before loading |
| `t` | Elapsed seconds from the first sample | Computed and cached on ordinary datasets; deferred with Dask |
| `hold_flag` | Integer hold flags, without a physical unit | Converted and cached on ordinary datasets; deferred with Dask |

`meta.t_elapsed` uses the endpoints of `tel_time` and does not load the full
time array. `meta.t_exp` instead comes from the header's integration timer;
zero integration time can accompany a nonzero elapsed span.

Pass `chunks={}` to `open_lmt_tel()` to use Dask-backed arrays. Raw flags can
be selected through the mapper:

```python
with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    tel = ds.lmt_tel
    schema = tel.mapper.schema
    flags = tel.mapper.get_arr(tel.data_source, schema.hold_flag)
    first_flags = flags.isel(time=slice(0, 100)).load()
```

## Inspect a sweep response

For a KIDs sweep, reduce the repeated samples at each LO frequency before
inspecting `S21 = I + 1j Q`. `SweepReducer` can be used directly from Python:

```python
import numpy as np
from tolteca_datamodels.toltec.kids import SweepReducer

with open_toltec("toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc") as ds:
    tree = SweepReducer()(ds)
    sweep = tree.toltec_kids.sweep
    S21 = sweep.I + 1j * sweep.Q
    S21_db = 20 * np.log10(np.abs(S21))
    frequency = sweep.frequency
```

The reduced view provides I/Q and the per-channel frequency array used for
S21-versus-frequency and I/Q plots. Reduction reads detector data. See
[KIDs reduction](kids-reduction.md) to run configured pipeline steps and export
their results.

### Select sweep-block metadata

A flat KIDs acquisition has file-level metadata. For a tune, the reader scans
LO frequency to count recorded sweep blocks; `n_blocks_max` describes the
allocated capacity. An explicitly block-indexed dataset returns a list of
metadata objects.

To inspect one block in a flat acquisition:

```python
with open_toltec("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc") as ds:
    meta = ds.toltec_kids.mapper.get_metadata(ds, block=1)
    print(meta.n_times, meta.t_exp)
```

The result describes that block's sample count and integration time. Acquisition
start/end timestamps still describe the containing acquisition.

## Save metadata without an open file

Use a Pydantic TypeAdapter to serialize and validate the metadata:

```python
from pydantic import TypeAdapter
from tolteca_datamodels.lmt import LmtTelMetadata

adapter = TypeAdapter(LmtTelMetadata)
with open_lmt_tel("tel_toltec_2025-10-31_145606_00_0001.nc") as ds:
    payload = adapter.dump_json(ds.lmt_tel.meta, round_trip=True)
metadata = adapter.validate_json(payload)
```

Use `round_trip=True` when you intend to validate the saved metadata again.
Ordinary serialization also includes computed fields such as `master_str` and
`filepath_info_orig`; serializing the latter parses the recorded filename if it
has not already been accessed.
