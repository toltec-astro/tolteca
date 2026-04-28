"""Data view classes for KIDs accessors.

Provides generic KidsView base class and specific view implementations for
different data types (sweep, timestream, single/multi-channel).
"""

from __future__ import annotations

import xarray as xr
from tollan.accessor import Mapper
from tollan.accessor.xarray import XarrayAccessorBase

from .kids import KidsMapper

__all__ = [
    "KidsView",
    "MultiSweepView",
    "MultiTimestreamView",
    "S21View",
    "SweepView",
    "TimestreamView",
]


class KidsView[MapperT: Mapper](XarrayAccessorBase[MapperT]):
    """Base class for KIDs data views.

    Provides common functionality for all KIDs data views. Inherits validation
    framework from AccessorBase - subclasses can override _validate() to add
    data compatibility checks.

    Type Parameters
    ---------------
    MapperT : Mapper
        Type of mapper used for field resolution
    """


class S21View(KidsView[KidsMapper]):
    """Base view providing common I/Q and S21 properties.

    All derived properties use standard arithmetic operations that are
    compatible with both numpy arrays and dask arrays (lazy computation).

    Attributes
    ----------
    mapper : KidsMapper
        Field mapper for data access
    data_source : xr.Dataset
        Dataset containing the data
    """

    def _validate(self) -> None:
        """Validate I and Q fields exist.

        Raises
        ------
        ValueError
            If I or Q fields are missing
        """
        self.mapper.validate_has_field(self.mapper.schema.I)
        self.mapper.validate_has_field(self.mapper.schema.Q)

    @property
    def I(self) -> xr.DataArray:
        """I component as DataArray."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.I)

    @property
    def Q(self) -> xr.DataArray:
        """Q component as DataArray."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.Q)

    @property
    def S21(self) -> xr.DataArray:
        """Complex S21 = I + 1j*Q."""
        return self.I + 1j * self.Q

    @property
    def aS21(self) -> xr.DataArray:
        """Absolute value |S21|."""
        return xr.ufuncs.absolute(self.S21)

    @property
    def aS21_db(self) -> xr.DataArray:
        """Amplitude |S21| in dB: 20*log10(|S21|)."""
        return 20 * xr.ufuncs.log10(self.aS21)


class SweepView(S21View):
    """Data view for single-channel frequency sweep.

    Provides access to I/Q data and derived S21 properties.

    Properties
    ----------
    I : xr.DataArray
        In-phase component (inherited)
    Q : xr.DataArray
        Quadrature component (inherited)
    S21 : xr.DataArray
        Complex transmission coefficient (inherited)
    aS21 : xr.DataArray
        Absolute value |S21| (inherited)
    aS21_db : xr.DataArray
        Amplitude in dB (inherited)
    frequency : xr.DataArray
        Frequency coordinate
    """

    def _validate(self) -> None:
        """Validate data source compatibility with this view.

        Raises
        ------
        ValueError
            If data is not compatible with this view
        """
        super()._validate()
        self.mapper.validate_has_field(self.mapper.schema.frequency)
        self.mapper.validate_has_physical_type(
            self.data_source,
            self.mapper.schema.frequency,
            "frequency",
        )

    @property
    def frequency(self) -> xr.DataArray:
        """Frequency coordinate."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.frequency)


class MultiSweepView(SweepView):
    """Data view for multi-channel frequency sweep.

    Extends SweepView with channel count property.
    Inherits: I, Q, S21, aS21, aS21_db, frequency

    Additional Properties
    ---------------------
    n_chans : int
        Number of channels
    """

    def _validate(self) -> None:
        """Validate 2-D sweep data.

        Raises
        ------
        ValueError
            If data is not 2-D
        """
        super()._validate()
        self.mapper.validate_ndim(self.data_source, self.mapper.schema.I, 2)

    @property
    def n_chans(self) -> int:
        """Number of channels (first axis size)."""
        i_data = self.I
        return i_data.sizes[i_data.dims[0]]


class TimestreamView(S21View):
    """Data view for single-channel timestream data.

    Provides access to I/Q data and derived S21.

    Properties
    ----------
    I : xr.DataArray
        In-phase component (inherited)
    Q : xr.DataArray
        Quadrature component (inherited)
    S21 : xr.DataArray
        Complex transmission coefficient (inherited)
    aS21 : xr.DataArray
        Absolute value |S21| (inherited)
    aS21_db : xr.DataArray
        Amplitude in dB (inherited)
    time : xr.DataArray
        Time coordinate
    """

    def _validate(self) -> None:
        """Validate data source compatibility with this view.

        Raises
        ------
        ValueError
            If data is not compatible with this view
        """
        super()._validate()
        self.mapper.validate_has_field(self.mapper.schema.time)
        self.mapper.validate_has_physical_type(
            self.data_source,
            self.mapper.schema.time,
            "time",
        )

    @property
    def time(self) -> xr.DataArray:
        """Time coordinate."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.time)


class MultiTimestreamView(TimestreamView):
    """Data view for multi-channel timestream data.

    Extends TimestreamView with channel count property.
    Inherits: I, Q, S21, aS21, aS21_db, time

    Additional Properties
    ---------------------
    n_chans : int
        Number of channels
    """

    def _validate(self) -> None:
        """Validate 2-D timestream data.

        Raises
        ------
        ValueError
            If data is not 2-D
        """
        super()._validate()
        self.mapper.validate_ndim(self.data_source, self.mapper.schema.I, 2)

    @property
    def n_chans(self) -> int:
        """Number of channels (first axis size)."""
        i_data = self.I
        return i_data.sizes[i_data.dims[0]]
