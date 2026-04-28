"""D21 derivative analysis for KIDs sweep data.

This module provides tools for computing the D21 derivative (dS21/df) from KIDs
sweep measurements. The D21 analysis is critical for identifying resonance features
and characterizing detector response.

Key Features
------------
- Two derivative methods: gradient (fast) and Savitzky-Golay (accurate)
- Optional smoothing for noise reduction
- Multi-channel support with unified frequency grid averaging
- Edge sample exclusion to avoid boundary artifacts
- Configurable frequency grid resampling

Examples
--------
High-level API using __call__ method:

>>> import numpy as np
>>> from astropy import units as u
>>> from tolteca_kidsproc.analysis import D21Analysis
>>> from tolteca_kidsproc.accessors import make_kids_dataset
>>> # Create synthetic sweep data
>>> freq = np.linspace(1e9, 1.1e9, 101)
>>> s21 = 1 / (1 + 1j * (freq - 1.05e9) / 1e6)  # Lorentzian
>>> ds = make_kids_dataset(
...     i_data=s21.real,
...     q_data=s21.imag,
...     frequency=freq << u.Hz,
... )
>>> # Compute and store D21 results in dataset
>>> analyzer = D21Analysis(method="gradient", smooth=5)
>>> ds_with_d21 = analyzer(ds)
>>> # Access results via accessor
>>> d21_matched = ds_with_d21.kids.d21.d21
>>> d21_unified = ds_with_d21.kids.d21.d21_unified

Low-level API using view methods:

>>> # Create multi-channel data
>>> n_channels = 10
>>> f_chans = np.linspace(1.0e9, 1.1e9, n_channels)
>>> f_sweep = np.linspace(-5e6, 5e6, 101)
>>> frequency = f_sweep[np.newaxis, :] + f_chans[:, np.newaxis]
>>> i_data = np.random.randn(n_channels, 101)
>>> q_data = np.random.randn(n_channels, 101)
>>> ds = make_kids_dataset(
...     i_data=i_data,
...     q_data=q_data,
...     frequency=frequency << u.Hz,
... )
>>> # Compute using sweep view
>>> analyzer = D21Analysis(smooth=5, exclude_edge=3)
>>> d21_matched = analyzer.make_matched(ds.kids.multi_sweep)
>>> d21_unified = analyzer.make_unified(ds.kids.multi_sweep)
>>> # Access coverage information
>>> coverage = d21_unified.coords["cov_unified"]

Notes
-----
The D21 derivative is computed as dS21/df, where S21 is the complex transmission
coefficient. The absolute value |D21| is particularly useful for finding resonance
features, as it peaks at the resonance frequency.

For multi-channel FDM data, the unified D21 provides a single representative curve
averaged across all channels, which is useful for quality assessment and
visualization.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import astropy.units as u
import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from tollan.accessor import Mapping, Schema
from tollan.accessor.xarray import XarrayMapper, ensure_dataset
from tollan.config import FrozenBaseModel
from tollan.config.types import FrequencyQuantityField
from tollan.utils.log import logger
from tollan.utils.np import make_complex

from ..accessors.views import KidsView

if TYPE_CHECKING:
    from tolteca_kidsproc.accessors.kids import MultiSweepView, SweepView

__all__ = [
    "D21Analysis",
    "D21Mapper",
    "D21Schema",
    "D21View",
]


D21SmoothMethod = Literal["savgol", "gradient"]


@dataclass
class D21Schema(Schema):
    """Schema for D21 analysis metadata and results.

    Attributes
    ----------
    d21 : Mapping
        Complex D21 derivative on original frequency grid.
    d21_unified : Mapping
        Unified |D21| averaged across channels.
    analysis : Mapping
        Analysis configuration (method, smooth, exclude_edge, etc.).
    f_unified : Mapping
        Unified frequency coordinate for d21_unified.
    cov_unified : Mapping
        Coverage map for unified D21.
    """

    d21: Mapping = Mapping(f"{__name__}.d21")

    d21_unified: Mapping = Mapping(f"{__name__}.d21_unified")

    analysis: Mapping = Mapping(f"{__name__}.analysis")

    # these goes with d21_unified so no __name__ prefix
    f_unified: Mapping = Mapping("f_unified")
    cov_unified: Mapping = Mapping("cov_unified")


class D21Mapper(XarrayMapper[D21Schema]):
    """Xarray mapper for D21 analysis metadata.

    Provides field mapping for D21 schema fields in xarray objects.
    """


class D21View(KidsView[D21Mapper]):
    """Data view for D21 analysis metadata and results.

    Provides convenient property-based access to D21 derivatives and
    analysis configuration stored in xarray Dataset/DataArray objects.

    Handles DataTree child node resolution automatically when instantiated
    with a DataTree - resolves to the D21 analysis namespace child node.

    Accessed via KidsAccessor: ds.kids.d21.matched, ds.kids.d21.unified, etc.

    Attributes
    ----------
    mapper : D21Mapper
        Field mapper for D21 data access
    data_source : xr.Dataset
        Dataset containing the D21 data (resolved from DataTree if needed)
    d21_analysis : D21Analysis
        D21 analysis configuration

    Examples
    --------
    Access D21 matched derivative (Dataset):

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from tolteca_kidsproc.accessors import make_kids_dataset
    >>> freq = np.linspace(1e9, 1.1e9, 101)
    >>> s21 = 1 / (1 + 1j * (freq - 1.05e9) / 1e6)
    >>> ds = make_kids_dataset(
    ...     i_data=s21.real,
    ...     q_data=s21.imag,
    ...     frequency=freq << u.Hz,
    ... )
    >>> analyzer = D21Analysis(method="gradient", smooth=5)
    >>> ds_with_d21 = analyzer(ds)
    >>> # Access via accessor
    >>> d21_data = ds_with_d21.kids.d21.d21  # Returns xr.DataArray

    Access D21 from DataTree (automatic child resolution):

    >>> dt = D21Analysis()(ds)  # Returns DataTree with child node
    >>> d21_view = dt.kids.d21  # Automatically resolves to child node
    >>> d21_data = d21_view.d21
    >>> d21_unified_data = d21_view.d21_unified

    Get analysis configuration:

    >>> config = ds_with_d21.kids.d21.d21_analysis
    >>> config.method
    'gradient'
    >>> config.smooth
    5
    """

    _data_analysis: D21Analysis

    @functools.cached_property
    def data_source(self) -> xr.Dataset:
        """Resolve data source to D21 analysis dataset.

        If _data_source is a DataTree, resolves to the D21 analysis child node.
        Otherwise returns the dataset directly.

        Returns
        -------
        xr.Dataset
            Dataset containing D21 analysis results
        """
        if isinstance(self._data_source, xr.DataTree):
            # Get namespace from schema (use class-level access)
            d21_namespace = self._get_namespace_from_schema_cls()

            # Try to find child node
            if d21_namespace in self._data_source.children:
                return self._data_source[d21_namespace].dataset

            # Fall back to root dataset
            return self._data_source.dataset

        return ensure_dataset(self._data_source)

    @classmethod
    def _get_namespace_from_schema_cls(cls) -> str:
        """Extract namespace from D21Schema field mappings.

        Returns
        -------
        str
            Namespace key (e.g., 'tolteca_kidsproc.analysis.d21')
        """
        # Access schema from mapper class (D21Mapper has D21Schema)
        schema = D21Mapper.schema

        # Get first field's mapping
        first_field = next(iter(schema.__dataclass_fields__.values()))
        mapping = first_field.default
        if isinstance(mapping, Mapping) and "." in mapping.names[0]:
            # Extract module path (
            # e.g., "tolteca_kidsproc.analysis.d21.field"
            # )
            parts = mapping.names[0].rsplit(".", 1)
            return parts[0]  # Return namespace without field name
        return "analysis"  # Fallback

    def _validate(self) -> None:
        """Validate data source has D21 analysis.

        Raises
        ------
        ValueError
            If no D21 analysis metadata found.
        """
        # Load D21 analysis metadata
        d21_analysis = D21Analysis._load_from_attr(self.data_source)
        if d21_analysis is None:
            msg = "No D21 analysis found in dataset."
            raise ValueError(msg)
        self._d21_analysis: D21Analysis = d21_analysis

    @property
    def d21_analysis(self) -> D21Analysis:
        """D21 analysis configuration.

        Returns
        -------
        D21Analysis
            D21 analysis configuration object.
        """
        return self._d21_analysis

    @property
    def d21(self) -> xr.DataArray:
        """D21 matched derivative DataArray.

        Returns
        -------
        xr.DataArray
            D21 matched derivative if present, None otherwise.
        """
        return self.mapper.get_arr(self.data_source, self.mapper.schema.d21)

    @property
    def d21_unified(self) -> xr.DataArray:
        """D21 unified derivative DataArray.

        Returns
        -------
        xr.DataArray
            D21 unified derivative if present, None otherwise.
        """
        return self.mapper.get_arr(self.data_source, self.mapper.schema.d21_unified)

    @property
    def f_unified(self) -> xr.DataArray:
        """Unified frequency coordinate from D21 unified result.

        Returns
        -------
        xr.DataArray | None
            Unified frequency coordinate if present, None otherwise.
        """
        return self.mapper.get_arr(self.d21_unified, self.mapper.schema.f_unified)

    @property
    def cov_unified(self) -> xr.DataArray:
        """D21 coverage map from unified result.

        Returns
        -------
        xr.DataArray | None
            Coverage map if present (from unified D21), None otherwise.
        """
        return self.mapper.get_arr(self.d21_unified, self.mapper.schema.cov_unified)


class D21Analysis(FrozenBaseModel):
    """D21 derivative analysis for KIDs sweep data.

    Computes dS21/df derivatives using gradient or Savitzky-Golay methods,
    with optional smoothing and unified averaging across channels.
    """

    f_lims: None | tuple[FrequencyQuantityField, FrequencyQuantityField] = Field(
        default=None,
        description="If set, the D21 is resampled to this frequency range.",
    )
    f_step: None | FrequencyQuantityField = Field(
        default=None,
        description="The step size of the frequency grid.",
    )
    resample: int = Field(
        default=1,
        description=(
            "Infer f_step by sampling the original by "
            "this factor, if f_step is not set."
        ),
    )
    exclude_edge: None | int = Field(
        default=None,
        description="Number of samples to exclude at the channel edge.",
    )
    smooth: None | int = Field(
        default=5,
        description="Number of samples used in smoothing the data.",
    )
    method: D21SmoothMethod = Field(
        default="savgol",
        description="The smoothing method",
    )

    # Private computed attributes (set by validator)
    _smooth: int = 0
    _exclude_edge: int = 0

    _mapper: ClassVar[D21Mapper] = D21Mapper.from_defaults()

    @model_validator(mode="after")
    def validate_smooth_and_edge(self) -> D21Analysis:
        """Validate and set defaults for smooth and exclude_edge.

        Stores computed values in private attributes _smooth and _exclude_edge.

        Returns
        -------
        D21Analysis
            Self with validated parameters.
        """
        exclude_edge = self.exclude_edge
        smooth = self.smooth

        if exclude_edge is None:
            exclude_edge = smooth

        # make sure they are non-negative
        smooth = smooth or 0
        if smooth < 0:
            msg = "smooth cannot be negative."
            raise ValueError(msg)

        exclude_edge = exclude_edge or 0
        if exclude_edge < 0:
            msg = "exclude_edge cannot be negative."
            raise ValueError(msg)

        if smooth == 0 and self.method != "gradient":
            msg = "no-smooth only works for gradient"
            raise ValueError(msg)

        _min_smooth_savgol = 3
        if smooth < _min_smooth_savgol and self.method == "savgol":
            msg = f"savgol requires smooth >= {_min_smooth_savgol}"
            raise ValueError(msg)

        # Store computed values in private attributes
        object.__setattr__(self, "_smooth", smooth)
        object.__setattr__(self, "_exclude_edge", exclude_edge)

        return self

    def _dump_to_attr(self, ds: xr.Dataset | xr.DataArray) -> None:
        """Save analysis configuration to dataset attributes.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            Dataset or DataArray to store configuration in.
        """
        mapper = self._mapper
        analysis_key = mapper.get_name(mapper.schema.analysis)
        ds.attrs[analysis_key] = self.model_dump_json()

    @classmethod
    def _load_from_attr(cls, ds: xr.Dataset | xr.DataArray) -> Self | None:
        """Load analysis configuration from dataset attributes.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            Dataset or DataArray to load configuration from.

        Returns
        -------
        D21Analysis | None
            Loaded configuration or None if not found.
        """
        mapper = cls._mapper
        analysis_key = mapper.get_name(mapper.schema.analysis)
        if analysis_key not in ds.attrs:
            return None
        return cls.model_validate_json(ds.attrs[analysis_key])

    def __call__(self, ds: xr.Dataset, *, force: bool = False) -> xr.Dataset:
        """Compute D21 analysis and store results in dataset.

        High-level API that computes both matched and unified D21,
        stores them in the dataset, and caches the configuration.
        If the dataset already contains D21 results with the same
        configuration, computation is skipped unless force=True.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset with I, Q, and frequency data.
        force : bool, optional
            If True, recompute even if results exist with same config.
            Default is False (use cached results).

        Returns
        -------
        xr.Dataset
            Dataset with D21 results added:

            - "d21" : matched D21 (complex)
            - "d21_unified" : unified |D21| (real)
            - attrs["d21_analysis"] : analysis configuration
            - coords["coverage"] : coverage map (from unified)

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from tolteca_kidsproc.accessors import make_kids_dataset
        >>> # Create synthetic sweep data
        >>> freq = np.linspace(1e9, 1.1e9, 101)
        >>> s21 = 1 / (1 + 1j * (freq - 1.05e9) / 1e6)
        >>> ds = make_kids_dataset(
        ...     i_data=s21.real,
        ...     q_data=s21.imag,
        ...     frequency=freq << u.Hz,
        ... )
        >>> # High-level API - compute and store
        >>> analyzer = D21Analysis(method="gradient", smooth=5)
        >>> ds_with_d21 = analyzer(ds)
        >>> # Results are stored in dataset
        >>> ds_with_d21.kids.d21.d21 is not None
        True
        >>> ds_with_d21.kids.d21.d21_unified is not None
        True
        >>> # Config is cached - second call skips computation
        >>> ds_cached = analyzer(ds_with_d21)
        >>> # Force recomputation
        >>> ds_new = analyzer(ds_with_d21, force=True)

        See Also
        --------
        make_matched : Low-level API for matched D21
        make_unified : Low-level API for unified D21
        """
        mapper = D21Mapper.from_data_source(ds)

        # Check if results exist and match current config
        if not force:
            cached_d21_analysis = self._load_from_attr(ds)
            if cached_d21_analysis == self and mapper.schema.d21 in mapper:
                # Return dataset as-is (already has results)
                return ds

        # Compute D21 results - modify dataset in-place
        # note the computation is the same for both sweep and multi_sweep
        try:
            sweep_view = ds.kids.multi_sweep
        except ValueError:
            sweep_view = ds.kids.sweep

        # Compute matched D21
        d21_matched = self.make_matched(sweep_view)
        d21_key = mapper.get_name(mapper.schema.d21)
        ds[d21_key] = d21_matched

        # Compute unified D21
        d21_unified = self.make_unified(
            sweep_view,
            # this by-pass the recomputation
            _d21_matched=d21_matched,
        )
        d21_unified_key = mapper.get_name(mapper.schema.d21_unified)
        ds[d21_unified_key] = d21_unified

        # Store analysis configuration as JSON string for cache
        self._dump_to_attr(ds)
        return ds

    def make_matched(
        self,
        sweep_view: MultiSweepView | SweepView,
    ) -> xr.DataArray:
        """Compute complex D21 derivative with matched frequency grid.

        Computes dS21/df where S21 = I + 1j*Q. The result has the same
        frequency grid as the input data.

        Works with any view that provides S21 and frequency properties.

        Parameters
        ----------
        sweep_view : MultiSweepView | SweepView
            Sweep view containing I, Q, and frequency data.

        Returns
        -------
        xr.DataArray
            Complex D21 derivative with units Hz^-1.
            Has same dimensions and coordinates as input I/Q data.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from tolteca_kidsproc.accessors import make_kids_dataset
        >>> freq = np.linspace(1e9, 1.1e9, 101)
        >>> s21 = 1 / (1 + 1j * (freq - 1.05e9) / 1e6)
        >>> ds = make_kids_dataset(
        ...     i_data=s21.real,
        ...     q_data=s21.imag,
        ...     frequency=freq << u.Hz,
        ... )
        >>> analyzer = D21Analysis(method="gradient", smooth=5)
        >>> d21 = analyzer.make_matched(ds.kids.sweep)
        >>> d21.shape
        (101,)
        """
        # Get S21 and frequency from view
        s21: npt.NDArray[np.complex128] = sweep_view.S21.values
        fs_Hz = sweep_view.frequency.u.to("Hz").values

        # Compute derivative based on method
        method = self.method
        smooth = self._smooth  # this is computed in validator

        if method == "gradient":
            if smooth is not None and smooth > 0:
                s21_smooth = uniform_filter1d(s21, size=smooth, mode="mirror", axis=-1)
            else:
                s21_smooth = s21
            # Get 1-D frequency
            if fs_Hz.ndim == 1:  # noqa: SIM108
                fs_Hz_1d = fs_Hz
            else:
                # dim is [chan, sweep]
                fs_Hz_1d = fs_Hz[0, :]

            d21 = np.gradient(s21_smooth, fs_Hz_1d, axis=-1)
        elif method == "savgol":
            # Get frequency spacing (works for both 1-D and 2-D)
            if fs_Hz.ndim == 1:
                deltaf = fs_Hz[1] - fs_Hz[0]
            else:
                # dim is [chan, sweep]
                deltaf = fs_Hz[0, 1] - fs_Hz[0, 0]

            # Savgol filter computes derivative directly
            d21_real = savgol_filter(
                s21.real,
                window_length=smooth,
                polyorder=2,
                deriv=1,
                delta=deltaf,
                axis=-1,
            )
            d21_imag = savgol_filter(
                s21.imag,
                window_length=smooth,
                polyorder=2,
                deriv=1,
                delta=deltaf,
                axis=-1,
            )
            d21 = make_complex(d21_real, d21_imag)
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)
        result = xr.DataArray(
            d21,
            dims=sweep_view.S21.dims,
            coords=sweep_view.S21.coords,
            attrs={
                "long_name": "D21 (dS21/df)",
                "units": "Hz^-1",
            },
        )
        self._dump_to_attr(result)
        return result

    def make_unified(
        self,
        sweep_view: MultiSweepView | SweepView,
        _d21_matched: None | xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute unified |D21| averaged across channels.

        For multi-channel data, this function:

        1. Computes D21 for each channel (via make_matched)
        2. Excludes edge samples from each channel
        3. Interpolates each channel to a common frequency grid
        4. Averages across channels (tracking coverage)
        5. Returns unified |D21| on the common grid

        Works with any view that provides S21 and frequency properties.

        Parameters
        ----------
        sweep_view : MultiSweepView | SweepView
            Sweep view containing I, Q, and frequency data.
            For multi-channel data, frequency should be 2-D.

        Returns
        -------
        xr.DataArray
            Unified |D21| on common frequency grid with units Hz^-1.
            Has coordinates: f_unified, cov_unified.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from tolteca_kidsproc.accessors import make_kids_dataset
        >>> # Create multi-channel sweep data
        >>> n_channels = 10
        >>> n_tones = 101
        >>> f_chans = np.linspace(1.0e9, 1.1e9, n_channels)
        >>> f_sweep = np.linspace(-5e6, 5e6, n_tones)
        >>> frequency = f_sweep[np.newaxis, :] + f_chans[:, np.newaxis]
        >>> i_data = np.random.randn(n_channels, n_tones)
        >>> q_data = np.random.randn(n_channels, n_tones)
        >>> ds = make_kids_dataset(
        ...     i_data=i_data,
        ...     q_data=q_data,
        ...     frequency=frequency << u.Hz,
        ... )
        >>> # Compute unified D21
        >>> analyzer = D21Analysis(smooth=5, exclude_edge=3)
        >>> d21_unified = analyzer.make_unified(ds.kids.multi_sweep)
        >>> # Result has unified frequency grid
        >>> "f_unified" in d21_unified.coords
        True

        Notes
        -----
        The unified grid parameters are controlled by f_lims and f_step.
        If not specified, they are inferred from the data.

        Edge exclusion is critical for multi-channel data to avoid artifacts
        at channel boundaries where the tone spacing changes.

        Coverage tracking helps identify frequency regions with sparse data.
        """
        fs = sweep_view.frequency.u.quantity
        if _d21_matched is None:
            d21_matched = self.make_matched(sweep_view).values
        else:
            d21_matched = _d21_matched.values
        # Handle 1-D vs 2-D frequency - wrap 1-D as 2-D
        if fs.ndim == 1:
            fs = fs[np.newaxis, :]
            d21_matched = d21_matched[np.newaxis, :]

        # Determine unified frequency grid
        f_lims = self.f_lims
        if f_lims is None:
            f_lims = (fs.min(), fs.max())
        f_min, f_max = f_lims

        f_step = self.f_step
        if f_step is None:
            f_step = (fs[0, 1] - fs[0, 0]) / self.resample

        # Validate edge exclusion (use validated private attribute)
        exclude_edge = self._exclude_edge

        if exclude_edge > 0:
            if 2 * exclude_edge > fs.shape[-1]:
                msg = f"insufficient number of data points for {exclude_edge=}."
                raise ValueError(msg)
            edge_slice = slice(exclude_edge, -exclude_edge)
        else:
            edge_slice = slice(None, None)

        smooth = self._smooth
        method = self.method
        logger.debug(
            f"build unified D21 with {f_min=} {f_max=} {f_step=}"
            f" {exclude_edge=} {smooth=} {method=} "
            f"from original f_min={fs.min()} f_max={fs.max()}",
        )

        # Create unified frequency grid with units
        fs_unified = (
            np.arange(
                f_min.to_value(u.Hz),
                f_max.to_value(u.Hz),
                f_step.to_value(u.Hz),
            )
            << u.Hz
        )

        # Get absolute D21 values
        ad21_matched = np.abs(d21_matched)
        ad21_unified = np.zeros(fs_unified.shape, dtype=np.double)
        ad21_unified_cov = np.zeros(fs_unified.shape, dtype=int)

        # Get channel boundaries (convert to Hz for comparison)
        chan_min = np.min(fs, axis=1).to_value(u.Hz)
        chan_max = np.max(fs, axis=1).to_value(u.Hz)
        fs_unified_Hz = fs_unified.to_value(u.Hz)

        # Interpolate each channel to unified grid
        for i in range(fs.shape[0]):
            m = (fs_unified_Hz >= chan_min[i]) & (fs_unified_Hz <= chan_max[i])
            tmp = np.interp(
                fs_unified_Hz[m],
                fs[i, edge_slice].to_value(u.Hz),
                ad21_matched[i, edge_slice],
                left=np.nan,
                right=np.nan,
            )
            cov = ~np.isnan(tmp)
            tmp[~cov] = 0
            ad21_unified[m] += tmp
            ad21_unified_cov[m] += cov.astype(dtype=int)

        # Normalize by coverage and clean up
        m = ad21_unified_cov > 0
        ad21_unified[m] /= ad21_unified_cov[m]
        ad21_unified[np.isnan(ad21_unified)] = 0

        # Get mapped coordinate names from schema
        mapper = self._mapper
        f_unified_key = mapper.get_name(mapper.schema.f_unified)
        cov_unified_key = mapper.get_name(mapper.schema.cov_unified)

        # Create result DataArray with proper units
        result = xr.DataArray(
            ad21_unified,
            dims=[f_unified_key],
            coords={
                f_unified_key: (
                    [f_unified_key],
                    fs_unified.value,
                    {"units": str(fs_unified.unit)},
                ),
                cov_unified_key: ([f_unified_key], ad21_unified_cov),
            },
            attrs={
                "long_name": "Unified |D21|",
                "units": f"{fs_unified.unit:s}^-1",
            },
        )
        self._dump_to_attr(result)
        return result
