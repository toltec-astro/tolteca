# Working with observation data

Use TolTECA to select observation files, inspect their metadata, and read the
arrays you need. You can start with individual files without setting up a
database or a deployment.

For example, select the ROACH 8 sweep from a directory and read its metadata:

```python
from pathlib import Path

from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.file import guess_info_from_sources

catalog = guess_info_from_sources(Path("sweeps").glob("*.nc"))
records = catalog.toltec_file.to_info_list(
    "roach == 8 & file_suffix == 'vnasweep'"
)
for record in records:
    with open_toltec(record.filepath) as ds:
        print(ds.toltec_kids.meta)
```

Building and selecting this catalog only parses filenames. Opening a selected
file lets its accessor read metadata from the headers. Reading detector arrays
or running a reducer is a separate operation.

Follow the chapters in order, or go directly to the task you need:

```{toctree}
:maxdepth: 2

file-catalogs
raw-data
```

For command-line processing, see [KIDs inputs](kids-inputs.md). For observation
number selectors, see [ObsSpec](obsspec.md). To configure and run reductions,
see [KIDs reduction](kids-reduction.md).
