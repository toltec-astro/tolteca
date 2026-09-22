# Discovering and selecting files

A file catalog is a pandas DataFrame with one row per source. Build one from
paths, then select files using their observation numbers, interfaces, or suffixes:

```python
from tolteca_datamodels.file import guess_info_from_sources

catalog = guess_info_from_sources([
    "/data/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
    "/data/toltec9_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
])
selected = catalog.toltec_file.query("roach == 8 & obsnum == 145605")
records = selected.toltec_file.to_info_list()
```

These operations do not open files. Filename information is a guess: use the
file's headers when you need to verify or complete it. Unrecognized filenames
remain in the catalog with unknown identifiers and `data_kind="Unknown"`.

## Describe one source

Use `guess_info_from_source()` for a single `SourceInfoModel`:

```python
from tolteca_datamodels.file import guess_info_from_source

record = guess_info_from_source(
    "/data/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
)
print(record.obsnum, record.roach, record.data_kind)
print(record.file_ut, record.file_suffix, record.file_ext)
```

An unrecognized filename raises `ValueError`; pass `raise_on_error=False` to
receive `None` instead. A recognized filename with an invalid date or identifier
still raises a validation error. A recognized filename with an unknown suffix
can provide observation identifiers even when its data kind is unknown.

Records are frozen. To change input fields, construct and validate a fresh record:

```python
from tolteca_datamodels.file import SourceInfoModel

updated = SourceInfoModel.model_validate({
    **record.model_dump(round_trip=True),
    "obsnum": 145610,
})
```

## Local and remote locations

`source` holds a Tollan `FileLoc`. It accepts local paths, `host:/path`
locations, and file/HTTP/HTTPS URLs. Describing a remote source does not download it:

```python
from tollan.utils.fileloc import fileloc

source = fileloc("archive:/data/tel_toltec_2025-10-31_145609_00_0002.nc")
remote = guess_info_from_source(source)
print(remote.source.url)
print(remote.source.as_rsync_arg())
```

For local sources, `record.filepath` gives the resolved filesystem path. It
raises `ValueError` for remote sources; stage those files locally before using
the raw readers. Use `fileloc("relative.nc", local_parent_path=...)` to resolve
a relative path against a known directory.

Catalogs store resolved local path strings or remote URLs. If you need to retain
the relative spelling and parent-path context, save the model itself:

```python
payload = record.model_dump_json(round_trip=True)
restored_record = SourceInfoModel.model_validate_json(payload)
```

## Renamed files and acquisition paths

The current location is `source`; `filepath_orig` is the acquisition path
recorded in a file header. If you already know that path, use it to identify a
renamed file without opening either location:

```python
record = guess_info_from_source(
    "/downloads/sweep.nc",
    filepath_orig="/data_lmt/toltec/tcs/toltec8/"
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
)
print(record.source.path)       # /downloads/sweep.nc
print(record.file_suffix)       # vnasweep
print(record.master)            # tcs
```

A recognized current filename supplies its own suffix, date, and observation
numbers. An unrecognized current filename falls back to `filepath_orig`.
The recorded acquisition directory takes precedence when guessing the master.
The source location itself is unchanged.

Two properties expose parsed path records: `record.filepath_info`
describes the effective path information, while `record.filepath_info_orig`
describes only the recorded acquisition path. The latter is parsed on first
access and cached. Either can be `None` when its information is unavailable.

Once a `FilepathInfo` has been created, its segments and inferred identity are
stored fields. Reading attributes, serializing the record and selecting catalog
rows do not repeat filename identification. To describe another path, call
`FilepathInfo.from_path(new_path)` to create a new record. Round-trip dumps
retain all of these stored fields.

| Field | Meaning |
| --- | --- |
| `prefix` | Literal filename prefix, such as `toltec8` or `tel_toltec` |
| `interface` | Acquisition interface, such as `toltec8` or `tel` |
| `instru`, `instru_component` | Paired instrument/component labels; `None` when unknown |
| `master_str` | Acquisition master name from the directory, including names such as `simu` without a database record |
| `obsnum`, `subobsnum`, `scannum` | Observation identifiers |
| `ut` | Writer's file-creation time encoded in the filename, in UTC at the filename's precision |
| `suffix` | Filename suffix, such as `vnasweep`; `None` when absent |
| `ext` | Extension without the leading dot |
| `roach` | ROACH index; present on the TolTEC readout record |
| `data_kind` | Filename-derived `ToltecDataKind`, or `Unknown` |

The model derives `interface` from `prefix`, then `instru` and
`instru_component` from the interface during parsing. Telescope prefixes such as
`tel_toltec` and `tel_toltec2` both identify interface `tel`, instrument `lmt`, and component
`tcs`. Other prefixes retain their spelling as the interface token; parsing
does not require it to be a registered readout. Unknown instrument or component
values remain `None`. The shared parser returns a TolTEC record for a
`toltec<N>` prefix, with the additional integer `roach` field. Other filename
records do not have this field. Mixed catalogs expose it as a nullable column,
so `file_roach.isna()` selects rows without a ROACH index.

Instrument/component labels are paired strings: `tel` maps to `("lmt", "tcs")`,
while `toltec_hk` maps to `("toltec", "hk")`. Housekeeping belongs to its
instrument; a bare `hk` prefix does not identify one. ROACH, HWPR and Wyatt
interfaces map to `toltec` with their respective components.

`ut` records when the writer created the file, independently of header and
telescope observation times. It is not filesystem modification time. A date-only
telescope filename supplies only the creation date, represented as midnight UTC;
it does not supply the time of day. Readout filenames include the creation time
to the second. Both filename formats require their date/time segment.
Top-level record identifiers may be corrected from headers; the parsed path
fields remain available for comparison. Likewise, `record.data_kind` may be
refined by header identification, while `record.filepath_info.data_kind`
preserves the filename-derived kind.

## Complete KIDs records from headers

To recover identifiers from a renamed KIDs file, open it and enrich a record:

```python
from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.file import SourceInfoModel, ToltecFileAccessor
from tolteca_datamodels.toltec.kids.source import identify_source

record = SourceInfoModel(source="sweep.nc")
with open_toltec(record.filepath) as ds:
    record = identify_source(ds, record)
catalog = ToltecFileAccessor.from_info_list([record])
```

`identify_source()` reads scalar headers, including the recorded acquisition
path, and returns a fresh record. It does not read detector arrays. Header
identifiers replace filename guesses; missing headers leave existing values
unchanged. A more specific filename kind is retained when compatible with the
header kind, for example a particular processed-sweep kind.

Pass `check_filepath_orig=True` to check header identifiers against the recorded
acquisition path. A mismatch raises `ValueError`. This optional check does not
compare the current filename or require the writer's file-creation time to equal
header or telescope times.

`ToltecFileAccessor.from_info_list(records)` accepts any iterable of records,
including an empty one. It does not open their sources. To recover records,
use `catalog.toltec_file.to_info_list()`.

## Select catalog rows

Use `catalog.toltec_file.query()` for ordinary columns, dotted path fields,
and short aliases in the same expression:

```python
selected = catalog.toltec_file.query(
    "roach == 8 & file_suffix in ['vnasweep', 'tune']"
)
records = catalog.toltec_file.to_info_list("obsnum == 145609")
```

| Structured field | Query alias |
| --- | --- |
| `filepath_info.prefix` | `file_prefix` |
| `filepath_info.ut` | `file_ut` |
| `filepath_info.suffix` | `file_suffix` |
| `filepath_info.ext` | `file_ext` |
| `filepath_info.data_kind` | `file_data_kind` |
| `filepath_info_orig.ut` | `file_ut_orig` |
| `filepath_info_orig.data_kind` | `file_data_kind_orig` |
| `filepath_info_orig.suffix` | `file_suffix_orig` |
| `filepath_info_orig.ext` | `file_ext_orig` |

Select all telescope files by interface, or one telescope stream by prefix:

```python
telescope = catalog.toltec_file.query('interface == "tel"')
supplementary = catalog.toltec_file.query('file_prefix == "tel_toltec2"')
```

Derived identity fields are also stored in the Arrow struct. For example,
`filepath_info.instru_component == "roach"` selects ROACH files without
reconstructing models for each row.

The same rule applies to every parsed path field, such as `file_obsnum` or
`file_master_str_orig`. Unqualified aliases refer to effective path information;
`_orig` aliases refer to the recorded acquisition path. They are query aliases,
not additional stored columns. To obtain a column, use
`catalog["filepath_info"].struct.field("suffix")`.

Expressions follow pandas syntax, including `&`, `|`, `in [...]`, and null
handling. For example, `file_suffix_orig.isna()` selects rows without an
original suffix. Use timezone-qualified strings for timestamp comparisons:

```python
selected = catalog.toltec_file.query(
    "file_ut_orig >= '2025-10-31T00:00:00Z' "
    "& file_suffix_orig == 'vnasweep'"
)
```

Only evaluate trusted expressions: pandas queries are not a sandbox for
untrusted remote input.

## Use a record in configuration conditions

The model's computed `file_prefix`, `file_ut`, `file_suffix`, and `file_ext`
properties are
included in ordinary Pydantic dumps. Request a JSON-compatible scalar
projection when supplying a configuration validation context:

```python
context_input = record.model_dump(
    mode="json",
    context={"flat": True},
    exclude={"filepath_info", "filepath_info_orig"},
)
```

`flat` serializes the current source as a resolved path or URL and the data kind
as its enum name. Excluding the nested path fields avoids parsing the original
filename for this operation. UTC datetimes use ISO 8601 strings ending in `Z`.
Without `flat`, serialization retains FileLoc's full structure and the enum's
ordinary representation. Use `round_trip=True` when saving model inputs for
subsequent validation; it omits computed fields.

## Save and reload a catalog

Parquet preserves nested path fields and UTC timestamps. Request Arrow dtypes
when loading the saved catalog:

```python
import pandas as pd

catalog.to_parquet("catalog.parquet", index=False)
restored = pd.read_parquet("catalog.parquet", dtype_backend="pyarrow")
selected = restored.toltec_file.query("file_suffix_orig == 'vnasweep'")
```

Import `tolteca_datamodels.file` to register the accessor in a fresh
process. Continue with [Reading observation data](raw-data.md) to open selected
sources and work with their metadata and arrays.
