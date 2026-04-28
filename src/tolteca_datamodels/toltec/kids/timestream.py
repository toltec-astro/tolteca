"""TolTEC timestream data schema, reducer, and views.

This module provides the schema for timestream processing with proper
namespacing to avoid conflicts. Includes PSD (Power Spectral Density)
analysis as part of the reduction process.

Schema fields are namespaced as: tolteca_datamodels.toltec.kids.timestream.*
"""

from __future__ import annotations

import functools
from typing import ClassVar

import numpy as np
import xarray as xr
from pydantic import Field
from pydantic.dataclasses import dataclass
from tollan.accessor import Mapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, ensure_dataset
from tollan.config import FrozenBaseModel

__all__ = [
    "ReducedTimestreamView",
    "TimestreamReducer",
    "ToltecTimestreamMapper",
    "ToltecTimestreamSchema",
]


@dataclass
class ToltecTimestreamSchema(Schema):
    """Schema for TolTEC timestream data.

    This schema maps to timestream data variables with proper namespacing.

    **Timestream Data (Namespaced):**
    - r, x: Solved timestream components

    **PSD Data (Namespaced):**
    - f_psd: Frequency axis for PSD [Hz]
    - I_psd: PSD of in-phase component [ADU²/Hz]
    - Q_psd: PSD of quadrature component [ADU²/Hz]
    - r_psd: PSD of solved r component [ADU²/Hz]
    - x_psd: PSD of solved x component [ADU²/Hz]

    **PSD Summary Statistics (Namespaced):**
    - I_psd_median: Median PSD in specified frequency range [ADU²/Hz]
    - Q_psd_median: Median PSD in specified frequency range [ADU²/Hz]
    - r_psd_median: Median PSD in specified frequency range [ADU²/Hz]
    - x_psd_median: Median PSD in specified frequency range [ADU²/Hz]
    - I_psd_mad_std: MAD-based std of PSD in range [ADU²/Hz]
    - Q_psd_mad_std: MAD-based std of PSD in range [ADU²/Hz]
    - r_psd_mad_std: MAD-based std of PSD in range [ADU²/Hz]
    - x_psd_mad_std: MAD-based std of PSD in range [ADU²/Hz]

    **Coordinates:**
    - time: Time axis
    - chan: Channel index
    - f_psd: PSD frequency axis

    **Metadata:**
    - reducer_config: TimestreamReducer configuration
    """

    # Solved timestream (namespaced)
    r: Mapping = Mapping(f"{__name__}.r")
    x: Mapping = Mapping(f"{__name__}.x")

    # PSD data (namespaced)
    f_psd: Mapping = Mapping(f"{__name__}.f_psd")
    I_psd: Mapping = Mapping(f"{__name__}.I_psd")
    Q_psd: Mapping = Mapping(f"{__name__}.Q_psd")
    r_psd: Mapping = Mapping(f"{__name__}.r_psd")
    x_psd: Mapping = Mapping(f"{__name__}.x_psd")

    # PSD summary statistics (namespaced)
    I_psd_median: Mapping = Mapping(f"{__name__}.I_psd_median")
    Q_psd_median: Mapping = Mapping(f"{__name__}.Q_psd_median")
    r_psd_median: Mapping = Mapping(f"{__name__}.r_psd_median")
    x_psd_median: Mapping = Mapping(f"{__name__}.x_psd_median")
    I_psd_mad_std: Mapping = Mapping(f"{__name__}.I_psd_mad_std")
    Q_psd_mad_std: Mapping = Mapping(f"{__name__}.Q_psd_mad_std")
    r_psd_mad_std: Mapping = Mapping(f"{__name__}.r_psd_mad_std")
    x_psd_mad_std: Mapping = Mapping(f"{__name__}.x_psd_mad_std")

    # Coordinates (not namespaced)
    time: Mapping = Mapping(("time", "ntimes"))
    chan: Mapping = Mapping(("chan", "channel", "n_chans"))

    # Metadata (namespaced)
    reducer_config: Mapping = Mapping(f"{__name__}.reducer_config")


class ToltecTimestreamMapper(XarrayMapper[ToltecTimestreamSchema]):
    """Mapper for TolTEC timestream data."""



# Module-level namespace constant (matches schema field prefix)
_TIMESTREAM_NAMESPACE: str = __name__  # "tolteca_datamodels.toltec.kids.timestream"


# View for Reduced Timestream Data
# ==================================


class ReducedTimestreamView(XarrayAccessorBase[ToltecTimestreamMapper]):
    """View for reduced TolTEC timestream/PSD data.

    Provides property-based access to PSD data and summary statistics.
    Transparently resolves DataTree child nodes — pass either a flat
    Dataset or a DataTree returned by TimestreamReducer.

    Examples
    --------
    >>> dt = TimestreamReducer(psd_nperseg=1024)(ds_timestream)
    >>> view = ReducedTimestreamView(dt)
    >>> f = view.f_psd       # frequency axis
    >>> psd_I = view.I_psd   # I component PSD
    """

    @functools.cached_property
    def data_source(self) -> xr.Dataset:
        """Resolve to reduced timestream Dataset.

        If given a DataTree (from TimestreamReducer), resolves to the
        timestream child node at ``_TIMESTREAM_NAMESPACE``. Falls back
        to root dataset or wraps a plain Dataset.
        """
        if isinstance(self._data_source, xr.DataTree):
            if _TIMESTREAM_NAMESPACE in self._data_source.children:
                return self._data_source.children[_TIMESTREAM_NAMESPACE].dataset
            return self._data_source.dataset
        return ensure_dataset(self._data_source)

    @property
    def f_psd(self) -> xr.DataArray | None:
        """PSD frequency axis [Hz]."""
        if self.mapper.schema.f_psd not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_psd)

    @property
    def I_psd(self) -> xr.DataArray | None:
        """PSD of in-phase component [ADU²/Hz]."""
        if self.mapper.schema.I_psd not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.I_psd)

    @property
    def Q_psd(self) -> xr.DataArray | None:
        """PSD of quadrature component [ADU²/Hz]."""
        if self.mapper.schema.Q_psd not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.Q_psd)

    @property
    def I_psd_median(self) -> xr.DataArray | None:
        """Median I PSD in the configured frequency range [ADU²/Hz]."""
        if self.mapper.schema.I_psd_median not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.I_psd_median)

    @property
    def Q_psd_median(self) -> xr.DataArray | None:
        """Median Q PSD in the configured frequency range [ADU²/Hz]."""
        if self.mapper.schema.Q_psd_median not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.Q_psd_median)


class TimestreamReducer(FrozenBaseModel):
    """Reduce raw TolTEC timestream data and compute PSDs.

    Similar to SweepReducer - configurable with pydantic, cacheable.
    Computes Power Spectral Density (PSD) using Welch's method.

    Parameters
    ----------
    time_axis : str, default "time"
        Name of time axis dimension
    compute_psd : bool, default True
        Compute power spectral density
    psd_nperseg : int, default 1024
        Length of each segment for Welch's method
    psd_noverlap : int or None, default None
        Number of points to overlap (None = nperseg // 2)
    psd_window : str, default "hann"
        Window function for Welch's method
    psd_detrend : str or bool, default "constant"
        Detrend type: "constant", "linear", or False
    psd_scaling : str, default "density"
        "density" (PSD) or "spectrum" (power)
    psd_stat_freq_range : tuple[float, float] or None, default None
        Frequency range [Hz] for computing summary statistics
    compute_r_x : bool, default True
        Compute PSDs for r and x components (if available)

    Examples
    --------
    >>> reducer = TimestreamReducer(psd_nperseg=1024, psd_stat_freq_range=(10.0, 100.0))
    >>> dt = reducer(ds_timestream)
    >>> ds_psd = dt.children["tolteca_datamodels.toltec.kids.timestream"].dataset
    >>> view = ReducedTimestreamView(dt)
    >>> view.I_psd  # I component PSD
    """

    time_axis: str = Field(
        default="time",
        description="Name of time axis dimension",
    )
    compute_psd: bool = Field(
        default=True,
        description="Compute power spectral density",
    )
    psd_nperseg: int = Field(
        default=1024,
        ge=2,
        description="Length of each segment for Welch's method",
    )
    psd_noverlap: int | None = Field(
        default=None,
        description="Number of overlapping points",
    )
    psd_window: str = Field(
        default="hann",
        description="Window function name",
    )
    psd_detrend: str | bool = Field(
        default="constant",
        description="Detrend type",
    )
    psd_scaling: str = Field(
        default="density",
        description="PSD scaling type",
    )
    psd_stat_freq_range: tuple[float, float] | None = Field(
        default=None,
        description="Frequency range for statistics [Hz]",
    )
    compute_r_x: bool = Field(
        default=True,
        description="Compute r, x PSDs if available",
    )

    _mapper: ClassVar[ToltecTimestreamMapper] = ToltecTimestreamMapper.from_defaults()

    def __call__(
        self,
        ds: xr.Dataset | xr.DataTree,
        *,
        force: bool = False,
    ) -> xr.DataTree:
        """Reduce raw timestream data and compute PSDs, returning a DataTree.

        The returned DataTree has the raw dataset at the root and the
        reduced/PSD dataset as a child at ``_TIMESTREAM_NAMESPACE``.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataTree
            Raw timestream dataset (or DataTree from a previous call)
        force : bool, default False
            If True, reprocess even if already reduced

        Returns
        -------
        xr.DataTree
            DataTree with root = raw data, child at
            ``tolteca_datamodels.toltec.kids.timestream`` = PSD data

        Raises
        ------
        ValueError
            If required data is missing
        """
        raw_ds = ds.dataset if isinstance(ds, xr.DataTree) else ds
        existing_dt = (
            ds if isinstance(ds, xr.DataTree) else xr.DataTree(dataset=raw_ds)
        )

        # Return early if already reduced with same config
        if not force and _TIMESTREAM_NAMESPACE in existing_dt.children:
            child_ds = existing_dt.children[_TIMESTREAM_NAMESPACE].dataset
            config_key = self._mapper.get_name(self._mapper.schema.reducer_config)
            if config_key in child_ds.attrs:
                cached_config = self.model_validate_json(child_ds.attrs[config_key])
                if cached_config == self:
                    return existing_dt

        ds_reduced = self._compute(raw_ds)

        return xr.DataTree(
            dataset=raw_ds,
            children={_TIMESTREAM_NAMESPACE: xr.DataTree(dataset=ds_reduced)},
        )

    def _compute(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute PSD reduction on a raw timestream dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Raw timestream dataset with I, Q data

        Returns
        -------
        xr.Dataset
            Reduced dataset with PSD variables (namespaced)

        Raises
        ------
        ValueError
            If required data is missing
        """
        from scipy.signal import welch

        # Validate required data
        if "I" not in ds or "Q" not in ds:
            msg = "Dataset must contain 'I' and 'Q' data variables"
            raise ValueError(msg)

        result = ds.copy()

        # Compute PSDs if requested
        if self.compute_psd:
            # Get time dimension and sampling frequency
            time_dim = self._get_time_dim(ds)
            fsmp = self._get_sampling_frequency(ds, time_dim)
            time_axis = self._get_time_axis(ds, time_dim)

            I_data = ds["I"].values
            Q_data = ds["Q"].values
            ndim = I_data.ndim

            # Compute I and Q PSDs
            f_psd, I_psd = welch(
                I_data,
                fs=fsmp,
                window=self.psd_window,
                nperseg=self.psd_nperseg,
                noverlap=self.psd_noverlap,
                detrend=self.psd_detrend,
                scaling=self.psd_scaling,
                axis=time_axis,
            )

            _, Q_psd = welch(
                Q_data,
                fs=fsmp,
                window=self.psd_window,
                nperseg=self.psd_nperseg,
                noverlap=self.psd_noverlap,
                detrend=self.psd_detrend,
                scaling=self.psd_scaling,
                axis=time_axis,
            )

            # Store PSD data
            if ndim == 1:
                f_psd_key = self._mapper.get_name(self._mapper.schema.f_psd)
                I_psd_key = self._mapper.get_name(self._mapper.schema.I_psd)
                Q_psd_key = self._mapper.get_name(self._mapper.schema.Q_psd)

                result[f_psd_key] = xr.DataArray(
                    f_psd,
                    dims=["f_psd"],
                    attrs={"units": "Hz", "long_name": "PSD frequency"},
                )
                result[I_psd_key] = xr.DataArray(
                    I_psd,
                    dims=["f_psd"],
                    attrs={"units": "ADU²/Hz", "long_name": "I component PSD"},
                )
                result[Q_psd_key] = xr.DataArray(
                    Q_psd,
                    dims=["f_psd"],
                    attrs={"units": "ADU²/Hz", "long_name": "Q component PSD"},
                )
            else:
                # Multi-channel
                chan_dim = self._get_chan_dim(ds)
                f_psd_key = self._mapper.get_name(self._mapper.schema.f_psd)
                I_psd_key = self._mapper.get_name(self._mapper.schema.I_psd)
                Q_psd_key = self._mapper.get_name(self._mapper.schema.Q_psd)

                result[f_psd_key] = xr.DataArray(
                    f_psd,
                    dims=["f_psd"],
                    attrs={"units": "Hz", "long_name": "PSD frequency"},
                )
                result[I_psd_key] = xr.DataArray(
                    I_psd,
                    dims=[chan_dim, "f_psd"],
                    attrs={"units": "ADU²/Hz", "long_name": "I component PSD"},
                )
                result[Q_psd_key] = xr.DataArray(
                    Q_psd,
                    dims=[chan_dim, "f_psd"],
                    attrs={"units": "ADU²/Hz", "long_name": "Q component PSD"},
                )

            # Compute summary statistics
            if self.psd_stat_freq_range is not None:
                f_min, f_max = self.psd_stat_freq_range
                freq_mask = (f_psd >= f_min) & (f_psd <= f_max)

                I_psd_median_key = self._mapper.get_name(
                    self._mapper.schema.I_psd_median,
                )
                Q_psd_median_key = self._mapper.get_name(
                    self._mapper.schema.Q_psd_median,
                )
                I_psd_mad_std_key = self._mapper.get_name(
                    self._mapper.schema.I_psd_mad_std,
                )
                Q_psd_mad_std_key = self._mapper.get_name(
                    self._mapper.schema.Q_psd_mad_std,
                )

                if ndim == 1:
                    result[I_psd_median_key] = xr.DataArray(
                        np.median(I_psd[freq_mask]),
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[Q_psd_median_key] = xr.DataArray(
                        np.median(Q_psd[freq_mask]),
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[I_psd_mad_std_key] = xr.DataArray(
                        self._mad_std(I_psd[freq_mask]),
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[Q_psd_mad_std_key] = xr.DataArray(
                        self._mad_std(Q_psd[freq_mask]),
                        attrs={"units": "ADU²/Hz"},
                    )
                else:
                    result[I_psd_median_key] = xr.DataArray(
                        np.median(I_psd[:, freq_mask], axis=-1),
                        dims=[chan_dim],
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[Q_psd_median_key] = xr.DataArray(
                        np.median(Q_psd[:, freq_mask], axis=-1),
                        dims=[chan_dim],
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[I_psd_mad_std_key] = xr.DataArray(
                        self._mad_std(I_psd[:, freq_mask], axis=-1),
                        dims=[chan_dim],
                        attrs={"units": "ADU²/Hz"},
                    )
                    result[Q_psd_mad_std_key] = xr.DataArray(
                        self._mad_std(Q_psd[:, freq_mask], axis=-1),
                        dims=[chan_dim],
                        attrs={"units": "ADU²/Hz"},
                    )

            # Compute r, x PSDs if available and requested
            if self.compute_r_x and "r" in ds and "x" in ds:
                _, r_psd = welch(
                    ds["r"].values,
                    fs=fsmp,
                    window=self.psd_window,
                    nperseg=self.psd_nperseg,
                    noverlap=self.psd_noverlap,
                    detrend=self.psd_detrend,
                    scaling=self.psd_scaling,
                    axis=time_axis,
                )
                _, x_psd = welch(
                    ds["x"].values,
                    fs=fsmp,
                    window=self.psd_window,
                    nperseg=self.psd_nperseg,
                    noverlap=self.psd_noverlap,
                    detrend=self.psd_detrend,
                    scaling=self.psd_scaling,
                    axis=time_axis,
                )

                r_psd_key = self._mapper.get_name(self._mapper.schema.r_psd)
                x_psd_key = self._mapper.get_name(self._mapper.schema.x_psd)

                if ndim == 1:
                    result[r_psd_key] = xr.DataArray(
                        r_psd,
                        dims=["f_psd"],
                        attrs={"units": "ADU²/Hz", "long_name": "r component PSD"},
                    )
                    result[x_psd_key] = xr.DataArray(
                        x_psd,
                        dims=["f_psd"],
                        attrs={"units": "ADU²/Hz", "long_name": "x component PSD"},
                    )
                else:
                    result[r_psd_key] = xr.DataArray(
                        r_psd,
                        dims=[chan_dim, "f_psd"],
                        attrs={"units": "ADU²/Hz", "long_name": "r component PSD"},
                    )
                    result[x_psd_key] = xr.DataArray(
                        x_psd,
                        dims=[chan_dim, "f_psd"],
                        attrs={"units": "ADU²/Hz", "long_name": "x component PSD"},
                    )

                # Summary statistics for r, x
                if self.psd_stat_freq_range is not None:
                    r_psd_median_key = self._mapper.get_name(
                        self._mapper.schema.r_psd_median,
                    )
                    x_psd_median_key = self._mapper.get_name(
                        self._mapper.schema.x_psd_median,
                    )
                    r_psd_mad_std_key = self._mapper.get_name(
                        self._mapper.schema.r_psd_mad_std,
                    )
                    x_psd_mad_std_key = self._mapper.get_name(
                        self._mapper.schema.x_psd_mad_std,
                    )

                    if ndim == 1:
                        result[r_psd_median_key] = xr.DataArray(
                            np.median(r_psd[freq_mask]),
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[x_psd_median_key] = xr.DataArray(
                            np.median(x_psd[freq_mask]),
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[r_psd_mad_std_key] = xr.DataArray(
                            self._mad_std(r_psd[freq_mask]),
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[x_psd_mad_std_key] = xr.DataArray(
                            self._mad_std(x_psd[freq_mask]),
                            attrs={"units": "ADU²/Hz"},
                        )
                    else:
                        result[r_psd_median_key] = xr.DataArray(
                            np.median(r_psd[:, freq_mask], axis=-1),
                            dims=[chan_dim],
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[x_psd_median_key] = xr.DataArray(
                            np.median(x_psd[:, freq_mask], axis=-1),
                            dims=[chan_dim],
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[r_psd_mad_std_key] = xr.DataArray(
                            self._mad_std(r_psd[:, freq_mask], axis=-1),
                            dims=[chan_dim],
                            attrs={"units": "ADU²/Hz"},
                        )
                        result[x_psd_mad_std_key] = xr.DataArray(
                            self._mad_std(x_psd[:, freq_mask], axis=-1),
                            dims=[chan_dim],
                            attrs={"units": "ADU²/Hz"},
                        )

        # Cache configuration in the result dataset attrs
        config_key = self._mapper.get_name(self._mapper.schema.reducer_config)
        result.attrs[config_key] = self.model_dump_json()

        return result

    def _get_time_dim(self, ds: xr.Dataset) -> str:
        """Get time dimension name."""
        time_candidates = ["time", "ntimes", "t"]
        for dim in time_candidates:
            if dim in ds.dims:
                return dim
        msg = f"Could not find time dimension in {list(ds.sizes.keys())}"
        raise ValueError(msg)

    def _get_chan_dim(self, ds: xr.Dataset) -> str:
        """Get channel dimension name."""
        chan_candidates = ["chan", "channel", "n_chans"]
        for dim in chan_candidates:
            if dim in ds.dims:
                return dim
        msg = f"Could not find channel dimension in {list(ds.sizes.keys())}"
        raise ValueError(msg)

    def _get_time_axis(self, ds: xr.Dataset, time_dim: str) -> int:
        """Get time axis index."""
        I_dims = list(ds["I"].dims)
        return I_dims.index(time_dim)

    def _get_sampling_frequency(self, ds: xr.Dataset, time_dim: str) -> float:
        """Infer sampling frequency from time coordinate.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with time coordinate
        time_dim : str
            Name of time dimension

        Returns
        -------
        float
            Sampling frequency in Hz
        """
        if time_dim in ds.coords:
            time = ds.coords[time_dim].values
            dt = np.median(np.diff(time))
            return 1.0 / dt

        # Try f_smp attribute
        if "f_smp" in ds.attrs:
            return float(ds.attrs["f_smp"])

        msg = (
            f"Could not infer sampling frequency. Please provide '{time_dim}' "
            "coordinate or 'f_smp' attribute."
        )
        raise ValueError(msg)

    @staticmethod
    def _mad_std(data: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute MAD-based standard deviation estimate.

        MAD (Median Absolute Deviation) provides robust estimate of
        standard deviation: σ ≈ 1.4826 * MAD

        Parameters
        ----------
        data : np.ndarray
            Input data
        axis : int or None
            Axis for computation

        Returns
        -------
        np.ndarray
            MAD-based standard deviation
        """
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis)
        return 1.4826 * mad
