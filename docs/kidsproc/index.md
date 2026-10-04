# Work with KIDs signals

Use `tolteca_kidsproc` to construct I/Q datasets, inspect transmission, compute
sweep derivatives, and estimate timestream spectra. Import the package to register
the `.kids` accessor on xarray Datasets and DataTrees.

| Task | Entry point | Result |
| --- | --- | --- |
| Create data from arrays | `make_kids_dataset(...)` | Dataset with physical coordinates |
| Open saved data | `open_datatree(...)` | DataTree; close it after use |
| Inspect transmission | `source.kids.s21` | I, Q, complex S21, amplitude, amplitude in dB |
| Differentiate a sweep | [D21Analysis](d21.md) | Matched derivative and unified magnitude |
| Estimate noise spectra | [PSDAnalysis](psd.md) | Welch spectra and robust statistics |
| Adapt existing field names | `KidsMapper` with `KidsSchema` | Views over existing data |
| Find signatures and settings | [API reference](api.rst) | Generated classes, fields, constraints, and methods |

```{graphviz}
digraph workflow {
    rankdir=TB;
    node [shape=box];
    arrays [label="Arrays or saved data"];
    source [label="Dataset / DataTree"];
    view [label=".kids.sweep / .kids.timestream"];
    analysis [label="D21Analysis / PSDAnalysis"];
    result [label="Result DataTree"];
    read [label=".kids.d21 / .kids.psd"];
    arrays -> source -> view -> analysis -> result -> read;
}
```

## Create a sweep or timestream

1. Supply equal-shaped I and Q arrays: `(sample,)` or `(chan, sample)`.
2. Supply physical coordinates as Astropy quantities.
3. Select the view needed for the next operation.

| Coordinate input | Required shape | Sample dimension | Sampling rate |
| --- | --- | --- | --- |
| `frequency` | Same as I/Q | `sweep` | Not required |
| `time` | `(sample,)`, shared by channels | `time` | Inferred from uniform spacing |
| `f_smp` | Positive finite scalar frequency quantity | `time` | Stored in Hz; elapsed seconds generated from zero |
| `time` and `f_smp` | As above | `time` | Checked against spacing with relative tolerance `1e-5` |
| Neither | — | `sample` | Not required for `.s21` |

If both frequency and time are supplied, the sample dimension is `time` and
frequency still spans the full I/Q shape. Coordinate units are retained.

```{testcode}
import numpy as np
from astropy import units as u
from tolteca_kidsproc.accessors import make_kids_dataset

frequency = np.linspace(1e9, 1.1e9, 101)
signal = 1 / (1 + 1j * (frequency - 1.05e9) / 1e6)
sweep = make_kids_dataset(
    signal.real, signal.imag, frequency=frequency * u.Hz
)
assert sweep.kids.sweep.S21.shape == (101,)
timestream = make_kids_dataset(
    np.ones((3, 128)), np.zeros((3, 128)), f_smp=64 * u.Hz
)
assert timestream.kids.multi_timestream.n_chans == 3
assert timestream.kids.timestream.time.dims == ("time",)
assert timestream.kids.timestream.f_smp == 64 * u.Hz
```

## Select data and views

| View | Required fields | Additional access |
| --- | --- | --- |
| `.s21` | I and Q | `S21 = I + 1j*Q`, `aS21`, `aS21_db = 20*log10(aS21)` |
| `.sweep` | I/Q and frequency with frequency units | `frequency` |
| `.multi_sweep` | Two-dimensional sweep with a mapped channel axis | `n_chans` |
| `.timestream` | I/Q, shared one-dimensional time with time units, recorded rate | `time`, `f_smp` |
| `.multi_timestream` | Two-dimensional timestream with a mapped channel axis | `n_chans` |

The sweep and timestream views support single-channel and multi-channel data.
Transmission properties preserve lazy array operations.

For an existing Dataset, select its payload directly. For a DataTree, select the
observation block first; `.kids` searches only that node and these relative paths:

| View | Child candidates in order, before the current node |
| --- | --- |
| `.s21` | `sweep`, `timestream`, `_input` |
| `.sweep` / `.multi_sweep` | `sweep`, `_input` |
| `.timestream` / `.multi_timestream` | `timestream`, `_input` |

The first existing candidate must satisfy the view requirements. An invalid
candidate raises an error; selection does not search sibling observation blocks.
The accessor's `data_source` is the supplied container; a data view's `data_source`
is its selected Dataset payload.

```{testcode}
import xarray as xr

observations = xr.DataTree.from_dict({"scan/sweep": sweep})
view = observations["scan"].kids.sweep
assert view.data_source.identical(sweep)
```

Views check field presence and the physical coordinate types described above.
For existing datasets, callers must also ensure I/Q dimensions and coordinates
match. D21 requires an ordered sweep grid; PSD requires complete, uniformly sampled
signals. `make_kids_dataset` checks array shapes and time spacing, but does not
establish all D21 sampling requirements.

## Use existing names

Use a schema to describe field names, a mapper to resolve them against a dataset,
and a view to read the signals without renaming or copying the source arrays.

```{testcode}
from dataclasses import replace
from tolteca_kidsproc import KidsMapper
from tolteca_kidsproc.accessors.views import SweepView

renamed = sweep.rename({"I": "in_phase", "Q": "quadrature"})
names = {"I": "in_phase", "Q": "quadrature"}
original = KidsMapper.from_data_source(sweep)
mapper = KidsMapper({
    field: replace(resolved, name=names.get(resolved.name, resolved.name))
    for field, resolved in original.mapped_fields.items()
})
mapped = SweepView(renamed, mapper)
np.testing.assert_allclose(mapped.S21, signal)
```

Data views are cached on an accessor. After changing field names or input paths,
construct a new `KidsAccessor(source)` or an explicitly mapped view. Result views
(`.kids.d21`, `.kids.psd`) are fresh selections: reacquire one after recomputing an
analysis. A previously acquired result view continues to refer to its old result.

## Save and reopen an analysis

Call an analysis without `output` to retain the input at `_input` in a standalone
result tree. Zarr supports the complex arrays produced by D21.

```{testcode}
from pathlib import Path
from tempfile import TemporaryDirectory
from tolteca_kidsproc.accessors import open_datatree
from tolteca_kidsproc.analysis import D21Analysis

analysis = D21Analysis(method="gradient", smooth=5)
result = analysis(sweep.kids.sweep)
with TemporaryDirectory() as directory:
    target = Path(directory) / "sweep.zarr"
    result.to_zarr(target, zarr_format=2)
    with open_datatree(target, engine="zarr") as reopened:
        assert reopened.kids.d21.d21_analysis == analysis
        xr.testing.assert_allclose(reopened.kids.sweep.S21, sweep.kids.sweep.S21)
```

Use `engine` and `chunks` as for `xarray.open_datatree`. The saved hierarchy is
preserved. Complete computations that need file-backed arrays before closing the
tree, or explicitly load those arrays while it is open.
