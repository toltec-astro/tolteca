"""Data view classes for KIDs accessors.

Data views expose I/Q signals and their physical coordinates. Construct a view
from a Dataset or a tree node containing data. ``data_source`` exposes the Dataset
or current node's payload; ``_data_source`` retains the supplied container. Use
``scope.kids.sweep`` or ``scope.kids.timestream`` to select data within a scope.
A supplied mapper must describe the payload.
"""

from __future__ import annotations

import xarray as xr
from astropy import units as u
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


def _validate_multi_channel(view: S21View) -> None:
    """Validate multi-channel I/Q data and its mapped channel dimension."""
    mapper = view.mapper
    mapper.validate_ndim(view.data_source, mapper.schema.I, 2)
    mapper.validate_has_field(mapper.schema.chan_dim)
    chan_dim = mapper.get_name(mapper.schema.chan_dim)
    if chan_dim not in view.I.dims or chan_dim not in view.Q.dims:
        msg = "Channel dimension must label an I/Q axis"
        raise ValueError(msg)


class KidsView[MapperT: Mapper](XarrayAccessorBase[MapperT]):
    """Base class for KIDs data views.

    The supplied Dataset or current tree node provides the payload. Subclasses
    validate their required fields in ``_validate``.

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
        """Validate 2-D sweep data with a mapped channel dimension.

        Raises
        ------
        ValueError
            Data is not 2-D or the mapped channel dimension is absent from I/Q.
        """
        super()._validate()
        _validate_multi_channel(self)

    @property
    def n_chans(self) -> int:
        """Length of the mapped channel dimension."""
        return self.mapper.get_dim(self.data_source, self.mapper.schema.chan_dim)


class TimestreamView(S21View):
    """Data view for single-channel timestream data.

    Provides access to I/Q data and derived S21. Time is one-dimensional and
    labels the sample dimension, shared by all channels in multi-channel data.

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
        One-dimensional time coordinate on the shared sample dimension
    f_smp : astropy.units.Quantity
        Recorded sampling frequency, returned in Hz.
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
        self.mapper.validate_has_field(self.mapper.schema.f_smp)
        self.mapper.validate_has_physical_type(
            self.data_source,
            self.mapper.schema.time,
            "time",
        )
        self.mapper.validate_ndim(self.data_source, self.mapper.schema.time, 1)
        time_dim = self.time.dims[0]
        if time_dim not in self.I.dims or time_dim not in self.Q.dims:
            msg = "Time must label an I/Q sample dimension"
            raise ValueError(msg)

    @property
    def time(self) -> xr.DataArray:
        """One-dimensional time coordinate shared by all channels."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.time)

    @property
    def f_smp(self) -> u.Quantity:
        """Recorded sampling frequency in Hz, without evaluating timestamps."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.f_smp) << u.Hz


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
        """Validate 2-D timestream data with a mapped channel dimension.

        Raises
        ------
        ValueError
            Data is not 2-D or the mapped channel dimension is absent from I/Q.
        """
        super()._validate()
        _validate_multi_channel(self)

    @property
    def n_chans(self) -> int:
        """Length of the mapped channel dimension."""
        return self.mapper.get_dim(self.data_source, self.mapper.schema.chan_dim)
