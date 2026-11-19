# File tables

A **model table** is a pandas DataFrame whose columns hold information collected
from model objects. A **file accessor** prepares and operates on a model table of
file records. A **data accessor**, such as `.lmt_tel`, exposes one opened object's
metadata and arrays. Use a file table to compare many files and choose inputs; this is
separate from an astronomical source catalog.

This is the second level of the [observation-data stack](datamodels.md#choose-a-level).
To work with a single known file, use a [typed opener](read-acquisition.md). To let
configuration or a directory layout find the files, continue with
[input discovery](input-discovery.md).

The examples that use real files read observation 145609 and its neighbours from the
reference data. `data_root` is the directory that holds your raw data.

```{testsetup} file_tables
data_root = workspace / "toltec_astro_dev/run/data_lmt"
```

## Build a table

Choose `LmtFileAccessor` for telescope data, `ToltecFileAccessor` for detector, HWPR
and TolTEC telescope companions, or the assembled `FileAccessor` for all enabled
data models. Each supplied entry retains a row, including unsupported or invalid
entries. Filename parsing always uses all currently registered parsers,
independently of which accessor will handle the row.

Factories return an accessor over a new DataFrame; `.table` returns that DataFrame.
Methods on an existing accessor update its table in place. This example needs no
files on disk:

```{testcode} file_tables
from tolteca_datamodels.file import FileAccessor

file_accessor = FileAccessor.from_sources([
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
])
table = file_accessor.table
assert table.interface.tolist() == ["tel", "toltec0"]
assert all("toltec" in memberships for memberships in table.data_models)

telescope = file_accessor.query_obs_spec("tcs-145609/tel")
assert telescope.interface.tolist() == ["tel"]
```

`file_table_from_sources()` returns the DataFrame directly when you do not need the
accessor. Both add observation columns such as `obsnum` and `interface` to parsed
rows, alongside the universal `file_*` columns. Rows that carry the observation
identifiers also receive `quartet`, `uid_obs`, `uid_raw_obs` and `uid_raw_obs_file`;
an identifier is null when one of its components is missing. A filename that no layout
recognizes keeps `file_ext` and a base `filepath_info` whose `data_models` is empty.

By default, an invalid recognized filename or an invalid source/location pair also
keeps its row, with the failure recorded in `source_trace`
([below](#inspect-processing-issues)); pass `on_error="raise"` to stop at the first one.
To prepare a single resource and have failures raise, see
[sources, locations and metadata](read-acquisition.md#sources-locations-and-metadata).

```{testcode} file_tables
from tolteca_datamodels.file import file_table_from_sources

unfamiliar = FileAccessor.from_sources(["example.nc"])
assert unfamiliar.table.file_ext.tolist() == ["nc"]
assert unfamiliar.source_trace_summary().kind.tolist() == [
    "file_accessor_rule_unmatched"
]
assert file_table_from_sources(["example.nc"]).file_ext.tolist() == ["nc"]
```

Pass a finite iterable of entries, such as a list, tuple or glob iterator; the table
follows its iteration order. Wrap a single resource or a source/location dictionary in
a list. An invalid outer collection always raises; invalid entries within a valid
collection follow `on_error`.

The accessor's `source`, `source_loc`, `filepath_info`, `data_models` and
`source_trace` properties return the corresponding columns as pandas Series with the
table's index. They create no columns, process no sources and raise `KeyError` when
the column is absent. Retain the `source_trace` column when you select rows if you
want to keep their recorded issues.

`FileAccessor(table)` binds an existing table without requiring columns or modifying
it, whereas `from_sources()` validates the entries and installs them in a new table:

```{testcode} file_tables
bound = unfamiliar.table
bound.index = [7]
assert FileAccessor(bound).table is bound
assert bound.index.tolist() == [7]
assert bound.file_ext.tolist() == ["nc"]
```

### Choose how much to read

The `mode` argument chooses how much information to collect before querying:

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

## Select telescope observations

Use `LmtFileAccessor` for a table of telescope observations, without detector
data. Read headers to select by observation goal or program:

```{testcode} file_tables
from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.file import LmtFileAccessor

tel_paths = sorted((data_root / "tel").glob("tel*.nc"))
tel_table = LmtFileAccessor.from_sources(tel_paths, mode="read").table
pointing = tel_table.lmt_file.query(
    'obs_goal == "pointing" and obs_pgm == "lissajous"'
)
print(pointing[["uid_raw_obs_file", "obs_goal", "obs_pgm", "t_int_s"]].to_string(index=False))

for row in pointing.itertuples(index=False):
    with open_lmt_tel(row.source) as ds:
        az = ds.lmt_tel.az.isel(time=slice(0, 100)).load()

assert sorted(pointing.uid_raw_obs_file) == [
    "tcs-145609-0-2-tel",
    "tcs-145610-0-2-tel",
    "tcs-145611-0-2-tel",
]
```

```{testoutput} file_tables
:options: +NORMALIZE_WHITESPACE

  uid_raw_obs_file obs_goal   obs_pgm  t_int_s
tcs-145609-0-2-tel pointing lissajous    60.32
tcs-145610-0-2-tel pointing lissajous    60.48
tcs-145611-0-2-tel pointing lissajous    60.68
```

Reference observations 145609, 145610 and 145611 include Lissajous pointing
scans. Their headers also supply `t0_mjd`, `t1_mjd`,
`t_int_s`, `t_elapsed_s` and target-coordinate columns such as `target_ra_deg`,
`target_dec_deg` and `target_frame`. The reference target uses the same naming
with `target_ref_`. Both targets in these telescope files use `"icrs"`.
Reading the table closes each file without loading pointing streams.

Filter by position and duration using the same query API:

```{testcode} file_tables
selected = pointing.lmt_file.query(
    'target_frame == "icrs" and 70 <= target_ra_deg < 80 '
    'and 60 <= target_dec_deg < 70 and t_int_s > 60 and t_elapsed_s > 65'
)
print(selected[["uid_raw_obs_file", "target_frame", "t_int_s", "t_elapsed_s"]].to_string(index=False))

assert selected.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
```

```{testoutput} file_tables
:options: +NORMALIZE_WHITESPACE

  uid_raw_obs_file target_frame  t_int_s  t_elapsed_s
tcs-145609-0-2-tel         icrs    60.32    68.508325
```

This selects scan 2 of observation 145609 in the reference subset. Coordinate
columns are in degrees and duration columns in seconds. `t_int_s` is the
header integration time; `t_elapsed_s` is the span between telescope timestamp
endpoints. They are different measurements, and an integration time of zero
does not imply an empty file.

With the default `mode="guess"`, call `tel_table.lmt_file.read_metadata()` later to
read headers; it updates that table in place. With `mode="guess_or_read"`,
filenames with no data-model membership trigger header reads. Other instruments'
recognized filenames retain their information without triggering telescope header
reads. Use `tel_table.lmt_file.query_obs_spec("tcs-145609/tel")` to select by
observation identifiers instead of a pandas expression.

Open the selected telescope files to work with their pointing arrays:

```{testcode} file_tables
with selected.lmt_file.open() as opened_tel:
    pointing_arrays = [
        row.io_obj.lmt_tel.az.isel(time=slice(0, 100)).load()
        for row in opened_tel.itertuples(index=False)
        if row.io_obj is not None
    ]

assert len(pointing_arrays) == 1
```

`LmtFileAccessor` accepts telescope data for any instrument. `ToltecFileAccessor`
accepts telescope data when `Header.Dcs.Receiver` identifies TolTEC,
independently of the filename. Its metadata has `instru="lmt"`,
`data_models=frozenset({"lmt", "toltec"})`, and `interface_kind="tel"`.

## Combine telescope and detector files

Files from the same observation can have different sampling rates and time
coordinates. Select the companion records together, then align their samples
explicitly when combining telescope and detector measurements.

A concrete class selects the capabilities: `LmtFileAccessor` handles telescope
files; `ToltecFileAccessor` handles TolTEC detector, HWPR and companion telescope
files.

```{testcode} file_tables
from tolteca_datamodels.toltec.file import ToltecFileAccessor

tel_path = data_root / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
toltec_path = (
    data_root
    / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
)
file_accessor = ToltecFileAccessor.from_sources([tel_path, toltec_path], mode="read")
table = file_accessor.table
selected = file_accessor.query("obsnum == 145609")

assert selected.interface.tolist() == ["tel", "toltec0"]
```

## Query the flat columns

Use `.query()` for metadata expressions and `.query_obs_spec()` for observation
selectors. Both return a DataFrame and preserve row order and index labels.

```{testcode} file_tables
pointing = file_accessor.query("obs_goal == 'pointing'")
toltec_related = file_accessor.query('contains(data_models, "toltec")')
observation = file_accessor.query_obs_spec("tcs-145609/{tel,0}")

assert pointing.interface.tolist() == ["tel"]
assert toltec_related.interface.tolist() == ["tel", "toltec0"]
assert observation.interface.tolist() == ["tel", "toltec0"]
```

TolTEC and other instruments keep independent classification columns such as
`data_kind_toltec` and `data_kind_dummy_instru`; their flags are not combined.
In a composite query, qualify flag names, for example
`ToltecDataKind.raw_sweep`. The concrete TolTEC accessor also accepts `raw_sweep`.
Filenames alone provide the classification:

```{testcode} file_tables
sweeps = ToltecFileAccessor.from_sources([
    data_root / "toltec/tcs/toltec0/toltec0_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
    toltec_path,
]).query("bits_any_set(data_kind_toltec, raw_sweep)")

assert sweeps.interface.tolist() == ["toltec0"]
assert sweeps.obsnum.tolist() == [145605]
```

Observation selectors are context-free parsed constraints. The accessor supplies
interface numbering. A composite combines its selected interface vocabularies
and applies positional selection once to the combined table. `tcs-[-1]` selects
the largest available TCS observation number; it does not mean latest by time
across controllers. See [observation selectors](obs_spec.md).

## Open resources and read metadata

Plain `open()` opens and identifies eligible sources without reading metadata.
The context yields the same table with temporary `io_obj` and `accessor_obj`
columns. Use `accessor_obj` to work with the selected data reader:

```{testcode} file_tables
with file_accessor.open() as opened:
    file_accessor.read_metadata()
    selected = file_accessor.query("interface == 'tel'")
    for row in selected.itertuples(index=False):
        if row.accessor_obj is not None:
            az = row.accessor_obj.az.isel(time=slice(0, 100)).load()

assert az.sizes["time"] == 100
```

For metadata alone, no surrounding open context is needed:

```{testcode} file_tables
file_accessor.read_metadata()
print(file_accessor.table[["obsnum", "interface"]])
```

```{testoutput} file_tables
:options: +NORMALIZE_WHITESPACE

   obsnum interface
0  145609       tel
1  145609   toltec0
```

The update reuses live readers. Otherwise it opens, reads and closes the required
sources itself. Context exit closes resources opened by the table and removes
the temporary columns; metadata remains. A repeated path is opened once per
opener within the context. Supplied open objects are borrowed and never closed
by the table:

```{testcode} file_tables
with open_lmt_tel(tel_path) as dataset:
    borrowed = ToltecFileAccessor.from_sources([
        {"source": dataset, "source_loc": tel_path},
    ])
    with borrowed.open():
        borrowed.read_metadata()
        print(borrowed.table.obs_goal.tolist())

assert borrowed.table.obs_goal.tolist() == ["pointing"]
```

```{testoutput} file_tables
['pointing']
```

Keep row positions stable while a context is open, and load any arrays needed
after the context exits.

Opening rules use the row's current values, with `file_*` as a fallback only when
there is no corresponding current column. Reader rules select a class from those
values, or inspect the opened contents when the values cannot decide. Once a
reader is selected, a binding or metadata failure is recorded; it does not trigger
a search for a different reader.

For a native processed KIDs file, `filepath_orig` holds the acquisition path
recorded by Kidscpp. Reading this field works even when the processed file has
been renamed or the acquisition is offline. Open the recorded acquisition to
read its full headers. When the writer supplied no source reference, the table
retains its classification and records `metadata_unavailable`.

## Inspect processing issues

The default `on_error="record"` retains failed rows. Use `on_error="raise"` to
propagate source validation, opening, identification or reading failures. Invalid
configuration and projection/dtype errors always raise.

```{testcode} file_tables
print(file_accessor.source_trace_summary())
for row in file_accessor.table.itertuples(index=False):
    for issue in row.source_trace.iter_error_entries():
        print(row.source_loc, issue.kind.value, issue.message, issue.context)

assert not any(row.source_trace.has_error for row in file_accessor.table.itertuples())
```

An empty summary means no issue was recorded:

```{testoutput} file_tables
Empty DataFrame
Columns: [kind, count, total]
Index: []
```

Each row's `source_trace` is a `SourceTrace` that retains the latest issue of each
kind, not success events. A repeated issue replaces the earlier details of its kind,
and a later successful retry does not erase recorded failures. Use
`trace.iter_entries()` for every recorded issue, `trace.iter_error_entries()` for
failures only, and `trace.has_error` to check for failures; it is not a readiness
flag. `file_accessor.source_trace_summary()` counts the rows that contain each kind.
Selecting or reordering rows keeps their trace objects, and `trace.record(entry)`
also updates `has_error` in every table that shares the object.

A location without a matching filename pattern, such as `/`, records
`filepath_rule_unmatched`. The accessor then matches each valid row against its
accessor rules; a row that matches none records `file_accessor_rule_unmatched`,
including when `data_models` is empty or missing. A row whose source/location
validation failed keeps that failure without a rule miss. These diagnostics are
recorded before observation columns are derived, so they remain available if
derivation fails:

```{testcode} file_tables
diagnosed = FileAccessor.from_sources(["example.nc", "other.nc", "/"])
counts = dict(zip(*diagnosed.source_trace_summary()[["kind", "count"]].to_numpy().T))

assert counts["file_accessor_rule_unmatched"] == 3
assert counts["filepath_rule_unmatched"] == 1
assert diagnosed.table.filepath_info.notna().sum() == 2
assert not any(trace.has_error for trace in diagnosed.source_trace)
```

Both rule misses, and `opener_rule_unmatched`, are informational, even with
`on_error="raise"`. No recorded issue does not mean that every later processing step
has run; use `table.filepath_info.notna().sum()` to count parsed filenames, and inspect
the current metadata columns and, inside `open()`, `accessor_obj` for the information
available to your analysis. Failure to obtain a live reader records
`reader_creation_failed` and follows `on_error`. This includes no reader being found
after inspecting the opened contents, an inspection error, or an error constructing the
reader. Successful content-based identification records no issue. Invalid entries remain
in the table with their validation failure.

## Choose registered or explicit capabilities

The assembled `FileAccessor` uses available registered data-model specs.
Importing it enables LMT and TolTEC. Extensions explicitly register their specs.
The concrete classes work directly without constructing a `DataModelSet`.

```{testcode} file_tables
paths = [tel_path, toltec_path]
mixed = FileAccessor.from_sources(paths)
toltec_view = mixed.partial(accessor_types=(ToltecFileAccessor,))
assert toltec_view.table is mixed.table
assert toltec_view.accessor_types == (ToltecFileAccessor,)

toltec_view.read_metadata()
selected = toltec_view.query_obs_spec("145609/{tel,0}")

assert selected.interface.tolist() == ["tel", "toltec0"]
```

`partial()` restricts capabilities, not rows: it does not filter rows, run checks or
open resources, and it shares the original table. The view keeps its selected accessor
classes, even if another data model is registered later. Further `partial()` calls
can only narrow that selection, so a view cannot enable a capability that an earlier
view excluded. `mixed.view` creates an unrestricted view of the accessor's available
capabilities, and the view's `file_accessor` property returns the originating accessor.
Concrete accessors provide neither `partial()` nor `view`; they already select one
policy.

Each matched row is assigned to one concrete accessor. With LMT and TolTEC both
enabled, TolTEC handles the telescope files of its own instrument and LMT handles the
other telescope files, so aliases, UIDs and metadata are not populated twice. With
only LMT selected, LMT handles either instrument's telescope files. Matching is
repeated against the current values for each operation; the table does not retain the
selected accessor class.

For a new table with an explicit capability selection:

```{testcode} file_tables
from tolteca_datamodels.core.file import FileAccessorView
from tolteca_datamodels.data_model import DataModelSet

models = DataModelSet.from_select(["toltec"])
file_accessor = FileAccessorView.from_sources(
    paths, accessor_types=models.file_accessor_types, mode="read",
)

assert file_accessor.table.interface.tolist() == ["tel", "toltec0"]
```

A row's `data_models` is the set of models that can interpret its data. A TolTEC
telescope companion belongs to both `lmt` and `toltec`, while its producing
instrument remains `instru="lmt"`. `DataModelSet` selects the capabilities
available to a task; it does not filter the rows of an existing table.

`from_sources()` on a view produces the same columns and traces as the concrete
accessor's factory with the same inputs. Both retain foreign and unidentified rows; the
selected capabilities decide which rows receive UIDs and classifications.

### Classify files

Each enabled data model adds its own nullable-integer flag column: `data_kind_toltec`
for TolTEC and `data_kind_dummy_instru` for the dummy instrument. They are not
combined into one `data_kind`. A classifier gives unsupported filenames, foreign files
and invalid sources its `unknown` flag, and its column is present even when the input
is empty. If derivation fails, existing derived columns keep their values and no new
ones are added.

Qualify the flag names when you query a mixed table. A concrete accessor also accepts
its shorthand member names. A composite or view accepts a shorthand when all of its
selected concrete classes define the same object under that name. Querying a view
evaluates the expression over all rows; it does not discard rows outside the selected
capability:

```{testcode} file_tables
classified = FileAccessor.from_sources([
    "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
    "example.nc",
])
sweeps = classified.query("bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)")
assert sweeps.interface.tolist() == ["toltec8"]

toltec_classified = classified.partial(accessor_types=(ToltecFileAccessor,))
unknown = toltec_classified.query("bits_any_set(data_kind_toltec, unknown)")
assert unknown.source.tolist() == ["example.nc"]
```

## Process groups of rows

`iter_accessors()` on a composite accessor or its partial view yields disjoint groups
of matching rows. Each pair holds row positions and a concrete accessor over the
**original full table**, not a sliced copy; apply updates at those positions to leave
other rows unchanged, even when index labels repeat:

```{testcode} file_tables
grouped = FileAccessor.from_sources([
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "example.nc",
])
grouped.table["inspected"] = False
for positions, group in grouped.partial(
    accessor_types=(ToltecFileAccessor,)
).iter_accessors():
    assert isinstance(group, ToltecFileAccessor)
    assert group.table is grouped.table
    column = group.table.columns.get_loc("inspected")
    group.table.iloc[positions, column] = True

assert grouped.table.inspected.tolist() == [True, False]
```

Matching runs once, when iteration starts: all groups, and the traces for unmatched or
unidentified rows, are determined before the first group is yielded. Those rows are not
yielded but remain in the table. Groups follow their first matching row, and an active
iterator keeps its original groups. You may add or update columns, but do not change
the row order or count during iteration. Iterating alone derives no columns and opens
no resources.

## Next

- Search a data tree for the files of an observation: [file stores](file-stores.md).
- Find files from configuration or the command line: [input discovery](input-discovery.md).
- Selector syntax: [observation selectors](obs_spec.md).
- Signatures and defaults: [observation data API](datamodels-api.rst).
