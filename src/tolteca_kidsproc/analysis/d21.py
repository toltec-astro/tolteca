"""D21 derivative analysis for KIDs sweep data.

Compute dS21/df to inspect resonance features and detector response.

Key Features
------------
- Gradient and Savitzky-Golay derivative estimators (``method``).
- Optional smoothing (``smooth``).
- Single-channel and multi-channel sweep inputs.
- Channel averaging of |D21| on a unified frequency grid.
- Edge exclusion (``exclude_edge``) and grid selection
  (``f_lims``, ``f_step``, ``resample``).

API
---
- High-level: ``analysis(sweep_view)`` returns a DataTree containing both results
  and the input sweep. Use ``output=tree`` to install results in an existing scope.
- Low-level: ``analysis.make_matched(sweep_view)`` and
  ``analysis.make_unified(sweep_view)`` return standalone DataArrays.
- Result access: ``result.kids.d21`` or ``D21View(result)`` exposes the matched
  derivative, unified magnitude, frequency, coverage, and saved configuration.
- Detached result access: ``D21View.from_namespace(node)`` reads an already
  selected analysis namespace.

Examples
--------
1. High-level API: compute both results from a single-channel sweep.

>>> import numpy as np
>>> import xarray as xr
>>> from astropy import units as u
>>> from tolteca_kidsproc.accessors import make_kids_dataset
>>> from tolteca_kidsproc.analysis.d21 import D21Analysis, D21View
>>> frequency = np.linspace(1e9, 1.1e9, 101)
>>> signal = 1 / (1 + 1j * (frequency - 1.05e9) / 1e6)
>>> sweep = make_kids_dataset(
...     signal.real, signal.imag, frequency=frequency * u.Hz)
>>> analysis = D21Analysis(method="gradient", smooth=5)
>>> result = analysis(sweep.kids.sweep)
>>> view = result.kids.d21
>>> view.d21.shape
(101,)
>>> view.d21_unified.ndim
1
>>> view.d21_analysis == analysis
True
>>> result.kids.sweep.S21.identical(sweep.kids.sweep.S21)
True

2. Low-level API: compute standalone arrays from a multi-channel sweep.

>>> f_chans = np.linspace(1e9, 1.1e9, 10)
>>> f_sweep = np.linspace(-5e6, 5e6, 101)
>>> frequency = f_chans[:, None] + f_sweep[None, :]
>>> signal = 1 / (1 + 1j * f_sweep[None, :] / 1e6) * np.ones((10, 1))
>>> multi = make_kids_dataset(
...     signal.real, signal.imag, frequency=frequency * u.Hz)
>>> sweep_view = multi.kids.multi_sweep
>>> matched = analysis.make_matched(sweep_view)
>>> unified = analysis.make_unified(sweep_view)
>>> matched.shape
(10, 101)
>>> unified.ndim
1
>>> coverage = unified.coords["cov_unified"]
>>> frequency = unified.coords["f_unified"]
>>> coverage.dims == unified.dims
True
>>> frequency.attrs["units"]
'Hz'

3. Result access: inspect arrays, coordinates, and producing configuration.

>>> view = D21View(result)  # Also available as result.kids.d21.
>>> matched = view.d21
>>> unified = view.d21_unified
>>> frequency = view.f_unified
>>> coverage = view.cov_unified
>>> saved_config = view.d21_analysis
>>> view.data_source is result["tolteca_kidsproc.analysis.d21"]
True

4. High-level API on a block: install results beside an existing sweep.

>>> tree = xr.DataTree.from_dict({"blocks/0/sweep": sweep})
>>> block = tree["blocks/0"]
>>> analysis(block.kids.sweep, output=block) is block
True
>>> block.kids.d21.d21.identical(result.kids.d21.d21)
True

Select the same destination through its enclosing tree with ``parent_namespace``:

>>> analysis(block.kids.sweep, output=tree, parent_namespace="blocks/0") is tree
True

5. Reuse results or force recomputation after changing input values.

>>> saved = block.kids.d21
>>> analysis(block.kids.sweep, output=block) is block
True
>>> block.kids.d21.data_source is saved.data_source
True
>>> analysis(block.kids.sweep, output=block, force=True) is block
True
>>> block.kids.d21.data_source is saved.data_source
False

6. Detach, export, and reopen the results and their configuration.

>>> view = block.kids.d21
>>> detached = view.data_source.copy(deep=False)
>>> D21View.from_namespace(detached).d21_analysis == analysis
True
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as directory:
...     target = Path(directory) / "derivatives.zarr"
...     _ = detached.to_zarr(target, zarr_format=2)
...     with xr.open_datatree(target, engine="zarr") as reopened:
...         restored = D21View.from_namespace(reopened)
...         assert restored.d21_analysis == analysis
...         xr.testing.assert_identical(restored.d21, view.d21)
...         xr.testing.assert_identical(restored.d21_unified, view.d21_unified)

Notes
-----
- Both derivative methods evaluate input arrays eagerly. Sweep samples occupy
  the last axis; channels, when present, precede it.
- Matched D21 is complex dS21/df on the measured grid. Unified D21 is the real
  mean of interpolated |D21| across channels, with zero in uncovered bins.
  Both results have units of inverse Hz.
- Multi-channel sweeps must share frequency offsets. Gradient uses the first
  channel's frequency grid; Savitzky-Golay uses its first frequency interval
  and requires uniformly spaced samples.
- Reuse requires matching configuration and the required outputs. Input values
  are not compared; pass ``force=True`` after changing them. Obtain a new result
  view to read recomputed results; an existing view retains its selected results.
- Standalone high-level output retains input at ``_input`` and stores separate
  ``d21`` and ``d21_unified`` Dataset leaves under
  ``tolteca_kidsproc.analysis.d21``. Export the enclosing scope to retain input
  as well as results; export the analysis namespace for results only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import xarray as xr
from pydantic import Field, model_validator
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from tollan.accessor import NameMapping, SubpathMapping
from tollan.accessor.xarray import (
    Analysis,
    AnalysisMapper,
    AnalysisSchemaMixin,
    AnalysisView,
    ensure_analysis_scope,
)
from tollan.config.types import (
    FrequencyQuantityField,  # noqa: TC002 -- Pydantic fields.
)
from tollan.utils.log import logger
from tollan.utils.np import make_complex

if TYPE_CHECKING:
    import numpy.typing as npt
    from tollan.accessor import Mapping

    from tolteca_kidsproc.accessors.kids import MultiSweepView, SweepView

__all__ = [
    "D21Analysis",
    "D21Mapper",
    "D21Schema",
    "D21View",
]


D21SmoothMethod = Literal["savgol", "gradient"]


@dataclass
class D21Schema(AnalysisSchemaMixin):
    """Namespace, output, and variable mappings for D21 analysis.

    Attributes
    ----------
    _namespace : SubpathMapping
        Analysis namespace relative to the working scope. Defaults to
        ``tolteca_kidsproc.analysis.d21``.
    analysis : NameMapping
        Producing configuration attribute on the analysis namespace.
    d21_subpath : SubpathMapping
        Matched Dataset path within the analysis namespace; defaults to ``d21``.
    d21 : Mapping
        Complex matched derivative variable, in inverse Hz.
    d21_unified_subpath : SubpathMapping
        Unified Dataset path within the analysis namespace; defaults to
        ``d21_unified``.
    d21_unified : Mapping
        Channel-averaged derivative magnitude variable, in inverse Hz.
    f_unified : Mapping
        Unified frequency coordinate, in Hz.
    cov_unified : Mapping
        Number of contributing channels at each unified frequency.
    """

    _namespace: SubpathMapping = SubpathMapping.path(__name__)
    analysis: NameMapping = NameMapping.attr(f"{__name__}.analysis")

    d21_subpath: SubpathMapping = SubpathMapping.path("d21")
    d21: Mapping = d21_subpath / NameMapping.var(f"{__name__}.d21")

    d21_unified_subpath: SubpathMapping = SubpathMapping.path("d21_unified")
    d21_unified: Mapping = d21_unified_subpath / NameMapping.var(
        f"{__name__}.d21_unified"
    )
    f_unified: Mapping = d21_unified_subpath / NameMapping.var("f_unified")
    cov_unified: Mapping = d21_unified_subpath / NameMapping.var("cov_unified")


class D21Mapper(AnalysisMapper[D21Schema]):
    """Namespace and numerical field mappings for D21 analysis."""


class D21Analysis(Analysis[D21Mapper]):
    """Configuration for matched and unified D21 derivatives.

    Parameters
    ----------
    f_lims : tuple of frequency quantities, optional
        Bounds of the unified grid. Defaults to the measured frequency range;
        the upper bound is excluded from the generated grid.
    f_step : frequency quantity, optional
        Unified grid spacing. Defaults to the first channel's sample spacing
        divided by ``resample``.
    resample : int, default 1
        Sampling factor used when ``f_step`` is not supplied.
    exclude_edge : int, optional
        Samples excluded from each channel edge during unified interpolation.
        Defaults to ``smooth``; zero disables exclusion.
    smooth : int, optional
        Smoothing window in samples, default 5. Zero or None disables smoothing
        for the gradient method. Savitzky-Golay requires at least three samples.
    method : {"savgol", "gradient"}, default "savgol"
        Derivative estimator. Gradient uses optional uniform-filter smoothing;
        Savitzky-Golay uses a quadratic fit over the configured window.

    Notes
    -----
    - High-level API: call with a sweep view to compute and store both results
      in a DataTree. Use ``output`` to choose an existing destination.
    - Low-level API: ``make_matched`` and ``make_unified`` return standalone
      DataArrays for further numerical work.
    - Both numerical methods evaluate the input arrays eagerly.
    """

    f_lims: tuple[FrequencyQuantityField, FrequencyQuantityField] | None = Field(
        default=None,
        description="If set, the D21 is resampled to this frequency range.",
    )
    f_step: FrequencyQuantityField | None = Field(
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
    exclude_edge: int | None = Field(
        default=None,
        description="Number of samples to exclude at the channel edge.",
    )
    smooth: int | None = Field(
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

    @model_validator(mode="after")
    def validate_smooth_and_edge(self) -> D21Analysis:
        """Validate smoothing and determine the effective edge exclusion.

        Returns
        -------
        D21Analysis
            This configuration with effective smoothing and edge counts initialized.
            An omitted edge count uses the smoothing window; ``None`` smoothing
            corresponds to zero samples.

        Raises
        ------
        ValueError
            A count is negative, or Savitzky-Golay smoothing uses fewer than three
            samples. Zero smoothing is supported only with the gradient method.
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

    def __call__(
        self,
        sweep_view: SweepView,
        *,
        output: xr.DataTree | None = None,
        parent_namespace: str | None = None,
        force: bool = False,
    ) -> xr.DataTree:
        """Compute derivatives from a sweep view into an xarray result scope.

        Parameters
        ----------
        sweep_view : SweepView or MultiSweepView
            Validated sweep supplying transmission and physical frequency.
        output : xarray.DataTree, optional
            Tree in which to install the analysis namespace. Omit to prepare a
            standalone scope with the view's source stored at ``_input``.
        parent_namespace : str, optional
            Working scope in ``output``. Relative paths start at the supplied
            node; absolute paths start at its root. Omit to use that node.
        force : bool, default False
            Recompute even when configuration and required outputs match.
            Input values are not compared; use after changing the input.

        Returns
        -------
        xarray.DataTree
            Supplied output tree, or a new standalone scope. Matched and unified
            Datasets occupy the mapped analysis namespace beneath the scope.

        Raises
        ------
        TypeError
            Input is not a sweep view, output is not a DataTree, or a parent
            namespace is supplied without an output tree.
        ValueError
            Output scope or namespace is invalid, or saved configuration is invalid.
        KeyError
            The selected parent namespace does not exist.
        """
        from ..accessors.views import SweepView  # noqa: PLC0415

        if not isinstance(sweep_view, SweepView):
            msg = "D21Analysis requires a sweep view"
            raise TypeError(msg)
        if output is not None and not isinstance(output, xr.DataTree):
            msg = "D21 output must be a DataTree"
            raise TypeError(msg)
        if output is None and parent_namespace is not None:
            msg = "Selecting a parent namespace requires an output tree"
            raise TypeError(msg)
        scope = ensure_analysis_scope(
            sweep_view.data_source if output is None else output,
            parent_namespace=parent_namespace,
        )
        mapper = self._mapper_cls.from_data_source(scope)
        self._call(
            scope,
            mapper=mapper,
            compute=lambda: self._compute_result(sweep_view, mapper=mapper),
            can_reuse=lambda saved: self._can_reuse(saved, mapper=mapper),
            force=force,
        )
        return scope if output is None else output

    def _get_output_subpaths(self, *, mapper: D21Mapper) -> tuple[SubpathMapping, ...]:
        """Declare the matched and unified derivative Dataset locations."""
        return (mapper.schema.d21_subpath, mapper.schema.d21_unified_subpath)

    def _can_reuse(self, result, *, mapper):
        return all(
            mapper.has_var(result, field)
            for field in (mapper.schema.d21, mapper.schema.d21_unified)
        )

    def _compute_result(
        self, sweep_view: SweepView, *, mapper: D21Mapper
    ) -> dict[SubpathMapping, xr.Dataset]:
        matched = self.make_matched(sweep_view, mapper=mapper)
        unified = self.make_unified(sweep_view, _d21_matched=matched, mapper=mapper)
        # Tree results store producing configuration on the analysis namespace.
        config_name = mapper.mapped_fields[mapper.schema.analysis].name
        for result in (matched, unified):
            result.attrs.pop(config_name, None)
        return {
            mapper.schema.d21_subpath: matched.to_dataset(
                name=mapper.get_name(mapper.schema.d21)
            ),
            mapper.schema.d21_unified_subpath: unified.to_dataset(
                name=mapper.get_name(mapper.schema.d21_unified)
            ),
        }

    def make_matched(
        self,
        sweep_view: MultiSweepView | SweepView,
        *,
        mapper: D21Mapper | None = None,
    ) -> xr.DataArray:
        """Compute the complex derivative on the measured frequency grid.

        Parameters
        ----------
        sweep_view : SweepView or MultiSweepView
            Sweep with ``S21`` and a unit-aware ``frequency`` coordinate. Sweep
            samples occupy the last axis; channels, when present, precede it.
        mapper : D21Mapper, optional
            Output mappings identifying the configuration attribute. Omit to use
            the default D21 mappings. Input fields use ``sweep_view.mapper``.

        Returns
        -------
        xarray.DataArray
            Eager complex derivative with the dimensions and coordinates of ``S21``.
            Units are recorded as ``Hz^-1``. Attributes include the serialized
            analysis configuration.
        """
        if mapper is None:
            mapper = self._mapper_cls.from_defaults()
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
        self._dump_to_attr(result, mapper=mapper)
        return result

    def make_unified(
        self,
        sweep_view: MultiSweepView | SweepView,
        _d21_matched: xr.DataArray | None = None,
        *,
        mapper: D21Mapper | None = None,
    ) -> xr.DataArray:
        """Average channel derivative magnitudes on a common frequency grid.

        Parameters
        ----------
        sweep_view : SweepView or MultiSweepView
            Sweep with ``S21`` and a unit-aware ``frequency`` coordinate. Each channel
            is interpolated onto the unified grid after excluding its edge samples.
        _d21_matched : xarray.DataArray, optional
            Precomputed matched derivative for this sweep and configuration. When
            omitted, compute it with ``make_matched``.
        mapper : D21Mapper, optional
            Output coordinate names and configuration attribute. Omit to use the
            default D21 mappings. Input fields use ``sweep_view.mapper``.

        Returns
        -------
        xarray.DataArray
            Eager real-valued mean of the interpolated derivative magnitudes, with
            units ``Hz^-1``. ``f_unified`` gives frequencies in Hz; ``cov_unified``
            counts contributing channels. Bins with no contributions contain zero.
            Attributes include the serialized analysis configuration.

        Raises
        ------
        ValueError
            Edge exclusion leaves too few samples for interpolation.
        """
        if mapper is None:
            mapper = self._mapper_cls.from_defaults()
        fs = sweep_view.frequency.u.quantity
        if _d21_matched is None:
            d21_matched = self.make_matched(sweep_view, mapper=mapper).values
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
        self._dump_to_attr(result, mapper=mapper)
        return result


class D21View(AnalysisView[D21Mapper, D21Analysis]):
    """Matched and unified derivatives with their producing configuration.

    Parameters
    ----------
    data_source : xarray.DataTree
        Working scope containing the D21 analysis namespace. Use
        ``from_namespace`` for an already selected namespace. The namespace has
        ``d21`` and ``d21_unified`` Dataset leaves by default and becomes the
        view's ``data_source``.
    mapper : D21Mapper, optional
        Field mappings for the namespace and its outputs. Omit for the default
        D21 names and paths.

    Attributes
    ----------
    d21 : xarray.DataArray
        Complex derivative on the measured frequency grid.
    d21_unified : xarray.DataArray
        Channel-averaged derivative magnitude on the unified grid.
    f_unified : xarray.DataArray
        Unified frequency coordinate in Hz.
    cov_unified : xarray.DataArray
        Number of contributing channels at each unified frequency.
    d21_analysis : D21Analysis
        Saved configuration that produced these results.

    Raises
    ------
    ValueError
        The producing configuration is absent or invalid.
    KeyError
        The namespace path does not identify a tree node.

    Notes
    -----
    - Use ``result.kids.d21`` or ``D21View(result)`` to read results in a scope.
    - Use ``D21View.from_namespace(node)`` for a selected or detached namespace.
    - Accessing a missing output or variable raises ``KeyError``.
    """

    @property
    def d21_analysis(self) -> D21Analysis:
        """Saved D21 analysis configuration."""
        return self._analysis_snapshot

    @property
    def d21(self) -> xr.DataArray:
        """Complex D21 derivative on the measured frequency grid."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.d21)

    @property
    def d21_unified(self) -> xr.DataArray:
        """Channel-averaged magnitude of D21 on the unified frequency grid."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.d21_unified)

    @property
    def f_unified(self) -> xr.DataArray:
        """Unified frequency coordinate in Hz."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_unified)

    @property
    def cov_unified(self) -> xr.DataArray:
        """Number of contributing channels at each unified frequency."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.cov_unified)
