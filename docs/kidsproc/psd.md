# Estimate timestream spectra

Use `PSDAnalysis` for Welch power spectra, frequency-band statistics, and an
optional common spectrum across channels.

## Prepare and analyze I/Q data

| Requirement | What to supply |
| --- | --- |
| Signals | Complete, uniformly sampled I/Q; at least two samples |
| Time | Shared one-dimensional coordinate with time units, labeling the sample axis |
| Rate | Positive finite sampling frequency recorded in Hz |
| Common spectrum | A mapped channel dimension and `compute_common=True` |
| Missing samples | Select or repair a continuous interval before analysis |

The analysis uses the recorded sampling rate without reading timestamp values.
It does not detect gaps, remove invalid samples, or interpolate missing data.

```{testcode}
import numpy as np
from astropy import units as u
from tolteca_kidsproc.accessors import make_kids_dataset
from tolteca_kidsproc.analysis import PSDAnalysis

time = np.arange(256) / 128
signal = np.sin(2 * np.pi * 16 * time)
source = make_kids_dataset(
    np.stack([signal, 2 * signal]),
    np.stack([signal, 3 * signal]),
    f_smp=128 * u.Hz,
)
analysis = PSDAnalysis(
    nperseg=128, stat_f_lims=(8 * u.Hz, 32 * u.Hz), compute_common=True
)
result = analysis(source.kids.timestream)
view = result.kids.psd
spectra = view.psd.compute()
assert set(spectra.data_vars) == {"I", "Q"}
assert spectra.I.dims == ("chan", "f_psd")
assert float(spectra.I.isel(chan=0).idxmax()) == 16
assert view.psd_median.I.dims == ("chan",)
assert view.psd_common.I.dims == ("f_psd",)
```

## Choose settings

| Setting | User-visible behavior |
| --- | --- |
| `nperseg` | Samples per segment; defaults to 1024 and is capped at the available input length |
| `noverlap=None` | Half the effective segment length; explicit overlap must be smaller than that length |
| `window` | SciPy window name; use `hann` for the default |
| `detrend` | `"constant"`, `"linear"`, or `False` |
| `scaling="density"` | Squared signal units per Hz; `"spectrum"` gives squared signal units |
| `return_onesided=True` | Nonnegative frequencies for real data; any complex series makes every series two-sided |
| `stat_f_lims` | Inclusive frequency band for statistics; omitted uses all bins |
| `compute_common=True` | Median and scaled MAD across channels at each frequency |
| `batch_size` | Chunk length along each non-time dimension |

See the {ref}`configuration reference <kidsproc-analysis-models>` for defaults and
constraints, and [SciPy's Welch reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
for spectral conventions. Two-sided frequencies use NumPy FFT order, including
negative frequencies. Explicit statistic bounds must be finite, nonnegative,
increasing, and contain at least one output bin.

Computations remain lazy until `.compute()`, `.load()`, or export. Each task reads
the entire selected time span: select a shorter interval first to limit memory
use. Unsupported SciPy window or detrending choices can fail when evaluating the
result, so compute it before relying on a saved analysis as complete.

## Read spectra and statistics

Each result Dataset exposes semantic variables `I` and `Q`, including when the
input uses custom field names. Input signal `units` attributes determine output
units; absent attributes mean dimensionless signals.

| Property on `result.kids.psd` | Reduction | Remaining axes |
| --- | --- | --- |
| `psd` | Welch spectrum | Non-time axes + `f_psd` |
| `psd_median` | Median over selected frequency bins | Non-time axes |
| `psd_mad_std` | `1.4826 * median(abs(x - median(x)))` over those bins | Non-time axes |
| `psd_common` | Median across channel spectra | Other non-time axes + `f_psd` |
| `psd_common_mad_std` | Scaled MAD across channel spectra | Other non-time axes + `f_psd` |
| `f_psd` | Frequency coordinate in Hz | `f_psd` |
| `f_smp`, `n_samples`, `nperseg` | Recorded rate in Hz, input count, effective segment length | Scalars |
| `psd_analysis` | Producing configuration | — |

Statistics propagate NaNs and describe dispersion across frequency bins or
channels. A common median combines spectra after their computation.
Common-spectrum properties raise `KeyError` unless
common results were requested and are present.

## Analyze arbitrary series

Use `make_psd` for a DataArray or Dataset of named numeric series. It returns only
spectra, preserves series names, and accepts an explicit sample dimension and
sampling frequency. Dataset variables without that dimension are omitted.

```{testcode}
import xarray as xr
from tollan.utils.xarray import reduce_median_mad

series = xr.DataArray(
    signal, dims="sample", name="signal", attrs={"units": "V"}
)
spectra_only = analysis.make_psd(series, dim="sample", f_smp=128 * u.Hz)
assert set(spectra_only.data_vars) == {"signal"}
assert spectra_only.signal.attrs["units"] == "V2 / Hz"
median, mad_std = reduce_median_mad(
    spectra_only.sel(f_psd=slice(8, 32)), dim="f_psd"
)
assert median.signal.ndim == mad_std.signal.ndim == 0
```

## Compute directly from numerical samples

Use `PSDAnalysis.compute_psd` without constructing a model when the input is a
NumPy array and all spectral settings are supplied explicitly. The sampling rate
is a plain value in Hz. The returned frequency and power arrays contain no unit
metadata. Use `analysis.make_psd` above for configured, lazy xarray computations
with units.

```{testcode}
frequency, power = PSDAnalysis.compute_psd(
    signal, f_smp=128.0, window="hann", nperseg=128, noverlap=64,
    detrend="constant", scaling="density", return_onesided=True, axis=-1,
)
np.testing.assert_array_equal(frequency, spectra_only.f_psd.values)
np.testing.assert_array_equal(power, spectra_only.signal.values)
```

## Store and reuse results

Calling the analysis without `output` returns a standalone DataTree containing
`_input` and the `tolteca_kidsproc.analysis.psd` namespace. That namespace contains
`psd` and, when requested, `psd_common`. Use `output` and `parent_namespace` with
the same [destination rules as D21](d21.md#store-results-in-an-observation-tree).

Reuse requires matching configuration, recorded sampling rate, sample count, and
required outputs. Input values are **not compared**. Pass `force=True` after
editing or replacing signals, and reacquire `.kids.psd` to read the new result.
The [save/reopen example](index.md#save-and-reopen-an-analysis) applies to PSD
trees too; complete file-backed computations before closing the source.
