"""Welch spectra and statistical summaries for KIDs timestreams.

API
---
- High-level: ``analysis(timestream_view)`` computes I/Q spectra and frequency-band
  statistics from a ``TimestreamView`` or ``MultiTimestreamView``. Set
  ``compute_common=True`` to include median and scaled-MAD spectra across channels.
- Mid-level: ``analysis.make_psd(data, f_smp=..., dim=...)`` computes spectra for
  arbitrary numerical DataArrays or Dataset variables using configured spectral
  settings and an explicit sampling rate.
- Low-level: ``PSDAnalysis.compute_psd`` accepts a numerical array, a sampling
  rate in Hz, and explicit spectral settings; it returns frequency and power arrays.
- Tollan utility: ``reduce_median_mad(data, dim=...)`` summarizes selected values
  along any dimension. Select or normalize spectra before reducing them when needed.
- Result access: ``result.kids.psd``, ``PSDView(result)``, or
  ``PSDView.from_namespace(node)`` exposes spectra, statistics, and configuration.

Examples
--------
1. High-level API: compute spectra from a single-channel timestream view.

>>> import numpy as np
>>> import xarray as xr
>>> from astropy import units as u
>>> from tolteca_kidsproc.accessors import make_kids_dataset
>>> from tolteca_kidsproc.analysis import PSDAnalysis, PSDView
>>> from tollan.utils.xarray import reduce_median_mad
>>> time = np.arange(1024) / 128.0
>>> signal = np.exp(2j * np.pi * 8 * time)
>>> source = make_kids_dataset(
...     signal.real, signal.imag, time=time << u.s, f_smp=128 << u.Hz)
>>> source["I"] = source.I.u.set(u.V)
>>> source["Q"] = source.Q.u.set(u.V)
>>> timestream = source.kids.timestream
>>> timestream.f_smp
<Quantity 128. Hz>
>>> analysis = PSDAnalysis(nperseg=256, stat_f_lims=(2 << u.Hz, 20 << u.Hz))
>>> result = analysis(timestream)
>>> view = result.kids.psd
>>> view.signal_names
('I', 'Q')
>>> peak = view.psd.I.argmax(dim="f_psd").compute().item()
>>> view.f_psd[peak].item()
8.0
>>> view.psd.I.u.unit
Unit("V2 / Hz")
>>> view.psd_analysis == analysis
True
>>> view.psd_median.I.dims
()
>>> result.kids.timestream.data_source.identical(source)
True

2. Mid-level API: analyse any time series with an explicit sample rate.

>>> temperature = xr.DataArray(
...     np.sin(2 * np.pi * 8 * time), dims="sample", name="temperature",
... ).u.set(u.K)
>>> spectra = analysis.make_psd(temperature, f_smp=128 << u.Hz, dim="sample")
>>> spectra.temperature.dims
('f_psd',)
>>> spectra.temperature.u.unit
Unit("K2 / Hz")
>>> band = spectra.sel(f_psd=slice(2, 20))  # f_psd is in Hz
>>> band_median, band_mad = reduce_median_mad(band, dim="f_psd")
>>> band_median.temperature.dims
()

A Dataset can contain signals with different units and non-time dimensions.
Each variable carrying the supplied sample dimension is analysed independently:

>>> pressure = (temperature.u.unset() * 2).u.set(u.Pa)
>>> signals = xr.Dataset({"temperature": temperature, "pressure": pressure})
>>> spectra = analysis.make_psd(signals, f_smp=128 << u.Hz, dim="sample")
>>> spectra.pressure.u.unit
Unit("Pa2 / Hz")

3. Multiple channels: compute per-channel summaries and a common spectrum.

>>> multi = make_kids_dataset(
...     np.stack([signal.real, signal.real * 2, signal.real * 3]),
...     np.stack([signal.imag, signal.imag * 2, signal.imag * 3]),
...     time=time << u.s, f_smp=128 << u.Hz,
... )
>>> combined_analysis = PSDAnalysis(nperseg=256, compute_common=True)
>>> combined = combined_analysis(multi.kids.multi_timestream).kids.psd
>>> combined.psd.I.dims
('chan', 'f_psd')
>>> combined.psd_median.I.dims
('chan',)
>>> combined.psd_common.I.dims
('f_psd',)
>>> chan_dim = multi.kids.multi_timestream.mapper.get_name(
...     multi.kids.multi_timestream.mapper.schema.chan_dim)
>>> common, common_mad = reduce_median_mad(combined.psd, dim=chan_dim)
>>> xr.testing.assert_identical(combined.psd_common, common)
>>> xr.testing.assert_identical(combined.psd_common_mad_std, common_mad)

4. Input and output scopes: retain input or install results beside an existing block.

Without ``output``, the returned tree retains the view's source at ``_input``.
To analyse an existing block in place, select its timestream view and pass the
block as the destination:

>>> tree = xr.DataTree.from_dict({"blocks/0/timestream": source})
>>> block = tree["blocks/0"]
>>> analysis(block.kids.timestream, output=block) is block
True
>>> block.kids.psd.nperseg
256

Select the same destination through its enclosing tree with ``parent_namespace``:

>>> analysis(block.kids.timestream, output=tree, parent_namespace="blocks/0") is tree
True

5. Reuse results or force recomputation after changing input values.

>>> saved = block.kids.psd
>>> analysis(block.kids.timestream, output=block) is block
True
>>> block.kids.psd.data_source is saved.data_source
True
>>> analysis(block.kids.timestream, output=block, force=True) is block
True
>>> block.kids.psd.data_source is saved.data_source
False

The view's ``data_source`` is the complete analysis namespace. Its
``psd_analysis`` describes the selected results. Replacing the namespace leaves
an existing view attached to its old results; a new view selects the replacement.

6. Export and reopen: a detached analysis namespace contains spectra and
configuration. Export the enclosing scope to retain the input as well.

>>> detached = block.kids.psd.data_source.copy(deep=False)
>>> PSDView.from_namespace(detached).signal_names
('I', 'Q')
>>> from tempfile import TemporaryDirectory
>>> from pathlib import Path
>>> with TemporaryDirectory() as directory:
...     target = Path(directory) / "spectra.zarr"
...     _ = detached.to_zarr(target, zarr_format=2)
...     with xr.open_datatree(target, engine="zarr") as reopened:
...         restored = PSDView.from_namespace(reopened)
...         assert restored.psd_analysis == analysis
...         xr.testing.assert_equal(restored.psd, block.kids.psd.psd)

7. Low-level API: estimate numerical spectra without an analysis instance.

>>> frequency, power = PSDAnalysis.compute_psd(
...     signal.real, f_smp=128.0, window="hann", nperseg=256, noverlap=128,
...     detrend="constant", scaling="density", return_onesided=True, axis=-1,
... )
>>> frequency[np.argmax(power)]
np.float64(8.0)

Notes
-----
The high-level call uses the view's I/Q, shared time coordinate, and recorded
sampling frequency. Input must already contain complete, uniformly sampled data
at that rate. Validate sample continuity and handle missing samples in the
surrounding workflow before analysis. PSD analysis does not inspect timestamps
or infer a sampling rate.

``make_psd`` requires a frequency quantity and assumes uniformly sampled input;
it does not infer the rate from attributes or timestamps. Signal units come from
``.u``. Signals without a declared unit are treated as dimensionless, producing
``Hz^-1`` with density scaling. Real input uses a one-sided grid by default;
complex input uses a shared two-sided grid in NumPy FFT order. Unnamed DataArrays
use the variable name ``data``.

``__call__`` and ``make_psd`` construct lazy results without evaluating timestamps;
``compute_psd`` evaluates numerical arrays eagerly. Each spectral
task uses a complete time series. Both scatter
estimates use the median absolute deviation multiplied by 1.4826, with NaNs
propagated. Band statistics describe variation across frequency bins; common
statistics describe variation among spectra at each frequency. Neither is an
uncertainty estimate for Welch. A common median spectrum is not the spectrum of
a combined time-domain signal. Aggregation combines the selected dimension in
each task; select or normalize spectra before calling ``reduce_median_mad`` to
control the members and relative scales.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from astropy import units as u
from pydantic import ConfigDict, Field, model_validator
from scipy.signal import welch
from tollan.accessor import NameMapping, SubpathMapping
from tollan.accessor.xarray import (
    Analysis,
    AnalysisMapper,
    AnalysisSchemaMixin,
    AnalysisView,
    ensure_analysis_scope,
    ensure_dataset,
)
from tollan.config.types import (
    FrequencyQuantityField,  # noqa: TC002 -- Pydantic fields.
)
from tollan.utils import xarray as xarray_utils

if TYPE_CHECKING:
    import numpy.typing as npt
    from tollan.accessor import Mapping

    from ..accessors.views import TimestreamView as KidsTimestreamView

__all__ = ["PSDAnalysis", "PSDMapper", "PSDSchema", "PSDView"]

_MIN_PSD_SAMPLES = 2
_ANALYSIS_ATTR = f"{__name__}.analysis"


def _combine_statistics(**statistics: xr.Dataset) -> xr.Dataset:
    """Combine statistics using their series names and statistic suffixes."""
    return xr.merge(
        [
            data.rename({name: f"{name}_{statistic}" for name in data.data_vars})
            for statistic, data in statistics.items()
        ]
    )


@dataclass
class PSDSchema(AnalysisSchemaMixin):
    """Namespace, output, and sampling mappings for spectral analysis.

    Attributes
    ----------
    _namespace : SubpathMapping
        Analysis namespace relative to the working scope. Defaults to
        ``tolteca_kidsproc.analysis.psd``.
    analysis : NameMapping
        JSON-encoded producing configuration attribute on the analysis namespace.
    psd_subpath : SubpathMapping
        Spectral Dataset path within the analysis namespace; defaults to ``psd``.
    f_psd : Mapping
        Spectral frequency coordinate in Hz, on the output Dataset.
    f_smp : Mapping
        Sampling frequency in Hz, stored as an output Dataset attribute.
    n_samples : Mapping
        Sample count per series, stored as an output Dataset attribute.
    nperseg : Mapping
        Effective Welch segment length, stored as an output Dataset attribute.
    psd_common_subpath : SubpathMapping
        Common-spectrum Dataset path; defaults to ``psd_common``.
    """

    _namespace: SubpathMapping = SubpathMapping.path(__name__)
    analysis: NameMapping = NameMapping.attr(_ANALYSIS_ATTR)

    psd_subpath: SubpathMapping = SubpathMapping.path("psd")
    f_psd: Mapping = psd_subpath / NameMapping.var("f_psd")
    f_smp: Mapping = psd_subpath / NameMapping.attr("f_smp")
    n_samples: Mapping = psd_subpath / NameMapping.attr("n_samples")
    nperseg: Mapping = psd_subpath / NameMapping.attr("nperseg")

    psd_common_subpath: SubpathMapping = SubpathMapping.path("psd_common")


class PSDMapper(AnalysisMapper[PSDSchema]):
    """Field mapping and named-series projections for spectral results."""

    def get_result(
        self,
        data: xr.Dataset,
        statistic: Literal[
            "psd", "psd_median", "psd_mad_std", "psd_common", "psd_common_mad_std"
        ],
    ) -> xr.Dataset:
        """Group one statistic under the original series names.

        Parameters
        ----------
        data : xarray.Dataset
            Spectral results with ``<series>_<statistic>`` variable names.
        statistic : str
            ``psd``, ``psd_median``, or ``psd_mad_std`` for individual spectra
            and band statistics; ``psd_common`` or ``psd_common_mad_std`` for
            the common spectrum and its scaled MAD.

        Returns
        -------
        xarray.Dataset
            Lazy columns named after their input series, retaining coordinates
            and each variable's units.
        """
        suffix = f"_{statistic}"
        names = {
            name: name.removesuffix(suffix)
            for name in data.data_vars
            if isinstance(name, str) and name.endswith(suffix)
        }
        return data[list(names)].rename(names)


class PSDAnalysis(Analysis[PSDMapper]):
    """Configuration for Welch spectra and frequency-band statistics.

    Parameters
    ----------
    nperseg : int, default 1024
        Samples per Welch segment, at least two. Shorter inputs use their
        available sample count.
    noverlap : int or None, default None
        Overlapping samples between segments. ``None`` uses half the effective
        segment length. A supplied count must be nonnegative and smaller than
        both ``nperseg`` and the effective segment length.
    window : str, default "hann"
        Window name accepted by ``scipy.signal.welch``.
    detrend : {"constant", "linear", False}, default "constant"
        Remove a constant or linear trend within each segment, or disable
        detrending with ``False``.
    scaling : {"density", "spectrum"}, default "density"
        Spectral units: squared signal units per Hz for ``density``, or squared
        signal units for ``spectrum``.
    return_onesided : bool, default True
        Use a one-sided frequency grid for real input. If any series is complex,
        all series use a two-sided grid in NumPy FFT order.
    stat_f_lims : tuple of frequency quantities or None, default None
        Inclusive lower and upper frequencies for median and scaled MAD.
        Bounds must be finite, nonnegative, and increasing. ``None`` uses every
        frequency bin; a specified band must contain at least one bin.
    compute_common : bool, default False
        Also compute the median spectrum and scaled MAD across channels at each
        frequency. The input mapper identifies the channel dimension through
        ``chan_dim``; a supplied view retains its own mapping.
    batch_size : int, default 32
        Chunk length along each non-time dimension, at least one. Each task
        requires the complete selected time span.

    Notes
    -----
    - High-level: calling the analysis returns a DataTree with individual spectra,
      band statistics, and an optional common spectrum.
    - Mid-level: ``make_psd`` returns standalone spectra. Use
      ``tollan.utils.xarray.reduce_median_mad`` to summarize selected values
      along a supplied dimension.
    - Low-level: ``compute_psd`` returns numerical frequency and power arrays
      using explicit settings, without an analysis instance or xarray container.
    - Configured computations are lazy; ``compute_psd`` evaluates arrays eagerly.
      All levels assume complete, uniformly sampled input. Select a time interval
      before calling to limit each task's memory use.
    - SciPy validates window and detrending choices when the lazy result is
      evaluated. Compute the result before treating an analysis as complete.
    """

    model_config = ConfigDict(extra="forbid")

    nperseg: int = Field(
        default=1024,
        ge=2,
        description="Number of samples per Welch segment, limited by the input length.",
    )
    noverlap: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of overlapping samples between segments. "
            "Defaults to half the effective segment length."
        ),
    )
    window: str = Field(
        default="hann",
        description="Window used for each Welch segment.",
    )
    detrend: str | bool = Field(
        default="constant",
        description="Trend removed from each segment: constant, linear, or False.",
    )
    scaling: Literal["density", "spectrum"] = Field(
        default="density",
        description="Spectral density per Hz or power spectrum scaling.",
    )
    return_onesided: bool = Field(
        default=True,
        description="Use a one-sided frequency grid for real-valued input.",
    )
    stat_f_lims: tuple[FrequencyQuantityField, FrequencyQuantityField] | None = Field(
        default=None,
        strict=False,
        description=(
            "Inclusive frequency limits for median and scaled-MAD statistics. "
            "Defaults to all frequency bins."
        ),
    )
    compute_common: bool = Field(
        default=False,
        description="Compute the median and scaled MAD across channel spectra.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Chunk length along each non-time dimension.",
    )

    @model_validator(mode="after")
    def _validate_spectral_settings(self):
        if self.noverlap is not None and self.noverlap >= self.nperseg:
            msg = "noverlap must be smaller than nperseg"
            raise ValueError(msg)
        if self.stat_f_lims is not None:
            low, high = (bound.to_value(u.Hz) for bound in self.stat_f_lims)
            if not np.isfinite([low, high]).all() or low < 0 or high <= low:
                msg = "PSD statistic bounds must be finite and increasing"
                raise ValueError(msg)
        return self

    def __call__(
        self,
        timestream_view: KidsTimestreamView,
        *,
        output: xr.DataTree | None = None,
        parent_namespace: str | None = None,
        force: bool = False,
    ) -> xr.DataTree:
        """Compute I/Q spectra and statistics from a timestream view.

        Parameters
        ----------
        timestream_view : TimestreamView or MultiTimestreamView
            Complete, uniformly sampled I/Q timestream with a physical time
            coordinate and recorded sampling frequency. The caller is responsible
            for sample continuity and missing-data handling. The view's mappings
            identify the signals, time, rate, and channel dimension. Only I/Q
            signals participate.
        output : xarray.DataTree, optional
            Tree in which to install the analysis namespace. Omit to prepare a
            standalone scope with the view's source retained at ``_input``.
        parent_namespace : str, optional
            Working scope in ``output``. Relative paths start at the supplied
            node; absolute paths start at its root. Omit to use that node.
        force : bool, default False
            Recompute existing results. Otherwise allow reuse when configuration,
            sampling rate, sample count, and required outputs match. Input values
            are not compared; use ``True`` after changing them.

        Returns
        -------
        xarray.DataTree
            Supplied output tree, or a new standalone scope. The ``psd`` Dataset
            contains I/Q spectra and frequency-band median and scaled MAD.
            With ``compute_common=True``, ``psd_common`` contains the median
            spectrum and scaled MAD across the mapped channel dimension.
            The enclosing analysis namespace retains the producing configuration.

        Raises
        ------
        TypeError
            Input is not a timestream view, output is not a DataTree, or a parent
            namespace is supplied without an output tree.
        ValueError
            Sampling or spectral settings are invalid, the statistic band contains
            no bins, the requested channel dimension is missing, or the output
            scope or namespace is invalid.
        KeyError
            The selected parent namespace does not exist.
        """
        from ..accessors.views import TimestreamView  # noqa: PLC0415

        if not isinstance(timestream_view, TimestreamView):
            msg = "PSDAnalysis requires a timestream view"
            raise TypeError(msg)
        if output is not None and not isinstance(output, xr.DataTree):
            msg = "PSD output must be a DataTree"
            raise TypeError(msg)
        if output is None and parent_namespace is not None:
            msg = "Selecting a parent namespace requires an output tree"
            raise TypeError(msg)
        scope = ensure_analysis_scope(
            timestream_view.data_source if output is None else output,
            parent_namespace=parent_namespace,
        )
        mapper = self._mapper_cls.from_data_source(scope)
        if self.compute_common:
            timestream_view.mapper.validate_has_field(
                timestream_view.mapper.schema.chan_dim
            )
        self._call(
            scope,
            mapper=mapper,
            compute=lambda: self._compute_result(timestream_view, mapper=mapper),
            can_reuse=lambda saved: self._can_reuse(
                timestream_view, saved, mapper=mapper
            ),
            force=force,
        )
        return scope if output is None else output

    def _get_output_subpaths(self, *, mapper: PSDMapper) -> tuple[SubpathMapping, ...]:
        """Declare individual and optional common-spectrum Dataset locations."""
        if not self.compute_common:
            return (mapper.schema.psd_subpath,)
        return (mapper.schema.psd_subpath, mapper.schema.psd_common_subpath)

    def _can_reuse(self, timestream_view, result, *, mapper):
        names = self._get_signals(timestream_view).data_vars
        try:
            spectra = mapper.get_node(result, mapper.schema.psd_subpath).to_dataset()
        except KeyError:
            return False
        if not (
            spectra.attrs.get("f_smp") == timestream_view.f_smp.to_value(u.Hz)
            and spectra.attrs.get("n_samples") == timestream_view.time.size
            and "f_psd" in spectra.coords
            and all(
                f"{name}_{statistic}" in spectra
                for name in names
                for statistic in ("psd", "psd_median", "psd_mad_std")
            )
        ):
            return False
        if not self.compute_common:
            return True
        try:
            common = mapper.get_node(result, mapper.schema.psd_common_subpath)
        except KeyError:
            return False
        return all(
            f"{name}_{statistic}" in common.dataset
            for name in names
            for statistic in ("psd_common", "psd_common_mad_std")
        )

    def _get_signals(self, timestream_view):
        """Select the signals to analyse under their semantic names."""
        return xr.Dataset({"I": timestream_view.I, "Q": timestream_view.Q})

    def _compute_result(self, timestream_view, *, mapper):
        signals = self._get_signals(timestream_view)
        spectra = self.make_psd(
            signals, f_smp=timestream_view.f_smp, dim=timestream_view.time.dims[0]
        )
        selected = self._statistic_band(spectra["f_psd"].u.to(u.Hz).values)
        median, mad = xarray_utils.reduce_median_mad(
            spectra.isel(f_psd=selected), dim="f_psd"
        )
        outputs = {
            mapper.schema.psd_subpath: _combine_statistics(
                psd=spectra, psd_median=median, psd_mad_std=mad
            )
        }
        if self.compute_common:
            input_mapper = timestream_view.mapper
            chan_dim = input_mapper.get_name(input_mapper.schema.chan_dim)
            common, common_mad = xarray_utils.reduce_median_mad(spectra, dim=chan_dim)
            outputs[mapper.schema.psd_common_subpath] = _combine_statistics(
                psd_common=common, psd_common_mad_std=common_mad
            )
        return outputs

    def _prepare(self, data, *, dim):
        data = ensure_dataset(data)
        names = [name for name, arr in data.data_vars.items() if dim in arr.dims]
        if not names:
            msg = f"No time series on dimension {dim!r}"
            raise ValueError(msg)
        if any(data[name].dtype.kind not in "biufc" for name in names):
            msg = "PSDAnalysis requires numerical time series"
            raise ValueError(msg)
        return data[names]

    @classmethod
    def compute_psd(  # noqa: PLR0913 -- Numerical settings are explicit, without a config model.
        cls,
        values: npt.NDArray,
        *,
        f_smp: float,
        window: str,
        nperseg: int,
        noverlap: int,
        detrend: str | bool,
        scaling: Literal["density", "spectrum"],
        return_onesided: bool,
        axis: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Estimate Welch spectra from numerical samples and explicit settings.

        Parameters
        ----------
        values : numpy.ndarray
            Complete, uniformly sampled numerical series. Other axes identify
            independent series; this call does not inspect timestamps.
        f_smp : float
            Positive sampling frequency in Hz.
        window : str
            SciPy window name.
        nperseg : int
            Segment length in samples, no larger than the available time span.
        noverlap : int
            Overlap between segments, nonnegative and smaller than ``nperseg``.
        detrend : {"constant", "linear", False}
            Trend to remove from each segment.
        scaling : {"density", "spectrum"}
            Spectral density per Hz or power spectrum.
        return_onesided : bool
            True requests nonnegative frequencies for real input. Complex input
            always produces a two-sided spectrum.
        axis : int
            Sample axis; independent series occupy the remaining axes.

        Returns
        -------
        frequency : numpy.ndarray
            One-dimensional frequencies in Hz; two-sided output uses FFT order.
        power : numpy.ndarray
            Power estimates with the sample axis replaced by frequency bins.
            Values have squared input units per Hz for density, or squared input
            units for spectrum. Unit metadata is not attached. Inputs are unchanged.

        Raises
        ------
        ValueError
            SciPy rejects the numerical inputs or spectral settings.
        """
        return welch(
            values,
            fs=f_smp,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling,
            return_onesided=return_onesided,
            axis=axis,
        )

    @classmethod
    def _compute_psd_values(cls, values, **kwargs):
        """Select power values for an xarray task with a shared frequency coordinate."""
        return cls.compute_psd(values, **kwargs)[1]

    def make_psd(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        f_smp: u.Quantity,
        dim: str = "time",
    ) -> xr.Dataset:
        """Estimate a Welch spectrum for each numerical time series.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Numerical series on ``dim``. Dataset variables without that
            dimension are ignored. Non-time dimensions identify independent
            series. An unnamed DataArray uses the variable name ``data``.
            Use ``.u.set(unit)`` on each input DataArray to specify signal units.
            Each Dataset variable supplies its own unit through the ``.u``
            accessor; signals without a declared unit are treated as dimensionless.
        f_smp : frequency quantity
            Positive finite scalar sampling frequency, such as ``128 << u.Hz``.
            The caller supplies the rate for uniformly sampled values; timestamps
            and sampling-rate attributes are not interpreted by this calculation.
        dim : str, default "time"
            Sample dimension. Other dimensions identify independent series.

        Returns
        -------
        xarray.Dataset
            Lazy spectra with the original series names and non-time dimensions,
            and frequency dimension ``f_psd`` in Hz. Variable units are squared
            input units per Hz for density, or squared units for spectrum.
            Signals without a unit produce ``Hz^-1`` for density scaling and
            dimensionless results for spectrum scaling.
            Attributes record ``f_smp`` in Hz, ``n_samples``, and effective ``nperseg``.
            No band statistics, configuration attribute, or DataTree is created.

        Raises
        ------
        ValueError
            Numerical series are absent, the supplied rate is not finite and
            positive, fewer than two samples are available, or overlap exceeds
            the effective segment length.
        astropy.units.UnitConversionError
            The sampling frequency has a unit incompatible with Hz.
        """
        ds = self._prepare(data, dim=dim)
        fs = float(f_smp.to_value(u.Hz))
        if not np.isfinite(fs) or fs <= 0:
            msg = "Sampling frequency must be finite and positive"
            raise ValueError(msg)
        time_dim = dim
        nperseg = min(self.nperseg, ds.sizes[time_dim])
        if nperseg < _MIN_PSD_SAMPLES:
            msg = "Welch spectra require at least two samples"
            raise ValueError(msg)
        if self.noverlap is not None and self.noverlap >= nperseg:
            msg = "noverlap must be smaller than available segment length"
            raise ValueError(msg)
        onesided = self.return_onesided and all(
            arr.dtype.kind != "c" for arr in ds.data_vars.values()
        )
        frequencies = (
            np.fft.rfftfreq(nperseg, 1 / fs)
            if onesided
            else np.fft.fftfreq(nperseg, 1 / fs)
        )
        result = xr.Dataset(
            coords={"f_psd": xr.DataArray(frequencies, dims="f_psd").u.set(u.Hz)}
        )

        for name, signal in ds.data_vars.items():
            chunks = {dim: self.batch_size for dim in signal.dims if dim != time_dim}
            chunks[time_dim] = -1
            psd = xr.apply_ufunc(
                self._compute_psd_values,
                signal.chunk(chunks),
                kwargs={
                    "f_smp": fs,
                    "window": self.window,
                    "nperseg": nperseg,
                    "noverlap": (
                        nperseg // 2 if self.noverlap is None else self.noverlap
                    ),
                    "detrend": self.detrend,
                    "scaling": self.scaling,
                    "return_onesided": onesided,
                    "axis": -1,
                },
                input_core_dims=[[time_dim]],
                output_core_dims=[["f_psd"]],
                keep_attrs=False,
                dask="parallelized",
                dask_gufunc_kwargs={
                    "output_sizes": {"f_psd": len(frequencies)},
                    "meta": np.empty((0,) * signal.ndim, dtype=float),
                },
            )
            unit = (signal.u.unit or u.one) ** 2
            if self.scaling == "density":
                unit /= u.Hz
            psd = psd.u.set(unit)
            psd.attrs["long_name"] = f"{name} Welch {self.scaling}"
            result[name] = psd
        result.attrs.update(
            {
                "f_smp": fs,
                "n_samples": ds.sizes[time_dim],
                "nperseg": nperseg,
            }
        )
        return result

    def _statistic_band(self, frequencies):
        if self.stat_f_lims is None:
            return np.ones(len(frequencies), dtype=bool)
        low, high = (bound.to_value(u.Hz) for bound in self.stat_f_lims)
        selected = (frequencies >= low) & (frequencies <= high)
        if not selected.any():
            msg = "PSD statistic range contains no frequency bins"
            raise ValueError(msg)
        return selected


class PSDView(AnalysisView[PSDMapper, PSDAnalysis]):
    """Spectra and frequency-band statistics with their producing configuration.

    Parameters
    ----------
    data_source : xarray.DataTree
        Working scope containing the PSD analysis namespace. Use
        ``from_namespace`` for an already selected namespace. The namespace
        contains the spectral Dataset at ``psd`` by default and becomes the
        view's ``data_source``.
    mapper : PSDMapper, optional
        Namespace and output mappings. If omitted, construct the mapper from the
        supplied tree. Output fields use their schema-declared parent paths within
        the analysis namespace.

    Raises
    ------
    ValueError
        The producing configuration is absent or invalid, or the
        spectral Dataset lacks its frequency coordinate or spectra.
    KeyError
        The namespace or spectral Dataset path is absent. Access to a common
        result also raises if that output was not computed.

    Notes
    -----
    ``psd_analysis`` is the typed producing configuration, reconstructed once per
    view. ``psd``, ``psd_median``, and ``psd_mad_std`` are Datasets whose variables
    use the semantic signal names ``I`` and ``Q``. The latter two contain
    statistics over the configured frequency band. With ``compute_common=True``,
    ``psd_common`` and ``psd_common_mad_std`` expose the median spectrum and
    scaled MAD across channels, retaining ``f_psd`` and other dimensions.
    """

    @functools.cached_property
    def _psd_dataset(self) -> xr.Dataset:
        """Dataset at the declared psd output subpath."""
        return self.mapper.get_node(
            self.data_source, self.mapper.schema.psd_subpath
        ).to_dataset()

    def _validate(self) -> None:
        super()._validate()
        if not self.mapper.has_var(self.data_source, self.mapper.schema.f_psd):
            msg = "Missing required spectral frequency coordinate"
            raise ValueError(msg)
        if not self.psd.data_vars:
            msg = "No PSD spectra found in dataset"
            raise ValueError(msg)

    @property
    def psd_analysis(self) -> PSDAnalysis:
        """Saved spectral analysis configuration."""
        return self._analysis_snapshot

    @functools.cached_property
    def psd(self) -> xr.Dataset:
        """Spectra grouped by signal name."""
        return self.mapper.get_result(self._psd_dataset, "psd")

    @functools.cached_property
    def psd_median(self) -> xr.Dataset:
        """Frequency-band medians grouped by signal name."""
        return self.mapper.get_result(self._psd_dataset, "psd_median")

    @functools.cached_property
    def psd_mad_std(self) -> xr.Dataset:
        """Frequency-band scaled MADs grouped by signal name."""
        return self.mapper.get_result(self._psd_dataset, "psd_mad_std")

    @functools.cached_property
    def _psd_common_dataset(self) -> xr.Dataset:
        """Dataset containing the common spectrum and its scaled MAD."""
        return self.mapper.get_node(
            self.data_source, self.mapper.schema.psd_common_subpath
        ).to_dataset()

    @functools.cached_property
    def psd_common(self) -> xr.Dataset:
        """Median spectra across the mapped channel dimension."""
        return self.mapper.get_result(self._psd_common_dataset, "psd_common")

    @functools.cached_property
    def psd_common_mad_std(self) -> xr.Dataset:
        """Scaled MAD among spectra at each frequency."""
        return self.mapper.get_result(self._psd_common_dataset, "psd_common_mad_std")

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Signal names in result order."""
        return tuple(str(name) for name in self.psd.data_vars)

    @property
    def f_psd(self) -> xr.DataArray:
        """Spectral frequency coordinate in Hz."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_psd)

    @property
    def f_smp(self) -> float:
        """Sampling frequency in Hz."""
        return float(self.mapper.get_value(self.data_source, self.mapper.schema.f_smp))

    @property
    def n_samples(self) -> int:
        """Number of analysed samples per series."""
        return int(
            self.mapper.get_value(self.data_source, self.mapper.schema.n_samples)
        )

    @property
    def nperseg(self) -> int:
        """Effective Welch segment length in samples."""
        return int(self.mapper.get_value(self.data_source, self.mapper.schema.nperseg))
