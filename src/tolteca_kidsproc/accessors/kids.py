"""KIDs data accessor with automatic view validation.

This module provides an xarray accessor for KIDs (Kinetic Inductance Detector) data
with automatic validation and type-specific data views.

Components
----------
**KidsAccessor** (xarray-registered accessor):
    Unified entry point providing data views as cached properties.
    Inherits standard view infrastructure from KidsAccessorMixin.

**Data Views**:
    - SweepView: Single-channel frequency sweep (1-D I, Q with frequency)
    - MultiSweepView: Multi-channel frequency sweep (2-D I, Q with frequency)
    - TimestreamView: Single-channel timestream (1-D I, Q with time)
    - MultiTimestreamView: Multi-channel timestream (2-D I, Q with time)

**Analysis Views**:
    - D21View: D21 derivative analysis results

**KidsMapper**:
    Schema-based field mapper extending tollan's XarrayMapper with validation methods

**KidsSchema**:
    Field definitions for I, Q, frequency, and time coordinates

Design
------
Views are validated on first access based on:
- Coordinate type (frequency vs time) → sweep vs timestream
- Data dimensionality (1-D vs 2-D) → single vs multi-channel
Invalid access raises ValueError with diagnostic message

Examples
--------
Access single-channel sweep data:
    >>> ds.kids.sweep.S21           # Complex S21
    >>> ds.kids.sweep.frequency     # Frequency coordinate
    >>> ds.kids.sweep.aS21_db       # Amplitude in dB

Access multi-channel sweep data:
    >>> ds.kids.multi_sweep.S21      # 2-D array
    >>> ds.kids.multi_sweep.n_chans  # Number of channels

Access D21 analysis results:
    >>> from tolteca_kidsproc.analysis import D21Analysis
    >>> d21_analysis = D21Analysis()
    >>> ds_with_d21 = d21_analysis(ds)
    >>> ds_with_d21.kids.d21.matched    # Matched D21 derivative
    >>> ds_with_d21.kids.d21.unified    # Unified across channels
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr
from tollan.accessor import Mapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper

if TYPE_CHECKING:
    from .views import (
        MultiSweepView,
        MultiTimestreamView,
        S21View,
        SweepView,
        TimestreamView,
    )

__all__ = [
    "KidsAccessor",
    "KidsMapper",
    "KidsSchema",
]


@dataclass
class KidsSchema(Schema):
    """Field mappings for KIDs data.

    Defines available field mappings. Views determine field requirements.

    Attributes
    ----------
    I : Mapping
        In-phase component
    Q : Mapping
        Quadrature component
    frequency : Mapping
        Frequency coordinate (required for sweep views)
    time : Mapping
        Time coordinate (required for timestream views)
    """

    I: Mapping = Mapping("I")
    Q: Mapping = Mapping("Q")
    frequency: Mapping = Mapping("frequency")
    time: Mapping = Mapping("time")


class KidsMapper(XarrayMapper[KidsSchema]):
    """Mapper for KIDs data.

    Extends XarrayMapper, inheriting get_arr(), get_scalar(),
    get_shape(), and validation methods (validate_has_field(), validate_is_coord(),
    validate_ndim(), validate_has_physical_type()) from base class.

    Provides standardized validation methods for view compatibility checking.
    """


@xr.register_dataset_accessor("kids")
@xr.register_datatree_accessor("kids")
class KidsAccessor(XarrayAccessorBase[KidsMapper]):
    """Accessor for KIDs data with validated data views.

    Provides data views as cached properties:
    - s21: Base S21 data view
    - sweep: Single-channel frequency sweep
    - multi_sweep: Multi-channel frequency sweep
    - timestream: Single-channel timestream
    - multi_timestream: Multi-channel timestream
    - d21: D21 derivative analysis results

    Examples
    --------
    Access sweep data:
        >>> ds.kids.sweep.S21
        >>> ds.kids.sweep.aS21_db

    Access multi-channel data:
        >>> ds.kids.multi_sweep.S21
        >>> ds.kids.multi_sweep.n_chans

    Access D21 analysis:
        >>> ds.kids.d21.matched
        >>> ds.kids.d21.unified
    """

    # Data View Properties
    # ====================

    @functools.cached_property
    def s21(self) -> S21View:
        """Base S21 data view.

        Returns
        -------
        S21View
            View for S21 properties (I, Q, S21, aS21, aS21_db)

        Raises
        ------
        ValueError
            If data is not compatible (requires I/Q fields)

        Examples
        --------
        >>> ds.kids.s21.S21          # Complex S21
        >>> ds.kids.s21.aS21_db      # Amplitude in dB
        """
        from .views import S21View  # noqa: PLC0415

        return S21View(self.data_source, self.mapper)

    @functools.cached_property
    def sweep(self) -> SweepView:
        """Single-channel sweep data view.

        Returns
        -------
        SweepView
            View for single-channel sweep properties

        Raises
        ------
        ValueError
            If data is not compatible (requires 1-D I/Q with frequency)

        Examples
        --------
        >>> ds.kids.sweep.S21
        >>> ds.kids.sweep.frequency
        >>> ds.kids.sweep.aS21_db
        """
        from .views import SweepView  # noqa: PLC0415

        return SweepView(self.data_source, self.mapper)

    @functools.cached_property
    def multi_sweep(self) -> MultiSweepView:
        """Multi-channel sweep data view.

        Returns
        -------
        MultiSweepView
            View for multi-channel sweep properties (includes n_chans)

        Raises
        ------
        ValueError
            If data is not compatible (requires 2-D I/Q with frequency)

        Examples
        --------
        >>> ds.kids.multi_sweep.S21
        >>> ds.kids.multi_sweep.n_chans
        >>> ds.kids.multi_sweep.frequency
        """
        from .views import MultiSweepView  # noqa: PLC0415

        return MultiSweepView(self.data_source, self.mapper)

    @functools.cached_property
    def timestream(self) -> TimestreamView:
        """Single-channel timestream data view.

        Returns
        -------
        TimestreamView
            View for single-channel timestream properties

        Raises
        ------
        ValueError
            If data is not compatible (requires 1-D I/Q with time)

        Examples
        --------
        >>> ds.kids.timestream.S21
        >>> ds.kids.timestream.time
        """
        from .views import TimestreamView  # noqa: PLC0415

        return TimestreamView(self.data_source, self.mapper)

    @functools.cached_property
    def multi_timestream(self) -> MultiTimestreamView:
        """Multi-channel timestream data view.

        Returns
        -------
        MultiTimestreamView
            View for multi-channel timestream properties (includes n_chans)

        Raises
        ------
        ValueError
            If data is not compatible (requires 2-D I/Q with time)

        Examples
        --------
        >>> ds.kids.multi_timestream.S21
        >>> ds.kids.multi_timestream.n_chans
        >>> ds.kids.multi_timestream.time
        """
        from .views import MultiTimestreamView  # noqa: PLC0415

        return MultiTimestreamView(self.data_source, self.mapper)

    @functools.cached_property
    def d21(self):
        """D21 derivative analysis results view.

        Returns
        -------
        D21View
            Analysis view providing:
            - d21: Matched D21 derivative
            - d21_unified: Unified D21 derivative
            - f_unified: Unified frequency coordinate
            - cov_unified: Coverage map
            - d21_analysis: Analysis configuration

        Examples
        --------
        Dataset usage:
            >>> ds.kids.d21.d21
            >>> ds.kids.d21.d21_unified

        DataTree usage:
            >>> dt = D21Analysis()(ds)  # Returns DataTree with child node
            >>> dt.kids.d21.d21         # D21View resolves to child node
            >>> dt.kids.d21.d21_unified
        """
        from ..analysis.d21 import D21View  # noqa: PLC0415

        return D21View(self._data_source)
