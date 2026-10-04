"""Data view classes for KIDs accessors.

Data views expose I/Q signals and their physical coordinates. Construct a view
from a Dataset or a tree node containing data. ``data_source`` exposes the Dataset
or current node's payload. Use
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
    """Base class for KIDs views over a Dataset payload.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Dataset or tree node whose current payload supplies the fields.
    mapper : Mapper, optional
        Resolved field mappings for this payload. Omit to resolve default names.

    Type Parameters
    ---------------
    MapperT : Mapper
        Mapper type used for field resolution.
    """


class S21View(KidsView[KidsMapper]):
    """I/Q transmission and amplitude, preserving lazy array operations.

    I and Q must describe the same samples with matching dimensions and
    coordinates. Construction checks that both fields exist.

    Attributes
    ----------
    I, Q : xarray.DataArray
        In-phase and quadrature samples.
    S21 : xarray.DataArray
        Complex transmission, ``I + 1j*Q``.
    aS21, aS21_db : xarray.DataArray
        Transmission magnitude and ``20*log10(abs(S21))``.
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
        """Absolute value ``abs(S21)``."""
        return xr.ufuncs.absolute(self.S21)

    @property
    def aS21_db(self) -> xr.DataArray:
        """Amplitude ``abs(S21)`` in dB: 20*log10(``abs(S21)``)."""
        return 20 * xr.ufuncs.log10(self.aS21)


class SweepView(S21View):
    """I/Q frequency sweep with one or more channels.

    Requires I/Q fields and a frequency coordinate with frequency units.
    Frequency ordering and sample alignment must be checked by the caller.

    Attributes
    ----------
    frequency : xarray.DataArray
        Physical frequency coordinate for the I/Q samples.
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
    """Two-dimensional frequency sweep with a mapped channel axis.

    Attributes
    ----------
    n_chans : int
        Number of channels. Transmission and frequency properties are inherited.
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
    """I/Q timestream with one or more channels and a recorded rate.

    Requires a one-dimensional time coordinate with time units that labels an
    axis of both I and Q. The view does not evaluate timestamps or check sample
    continuity; callers must supply complete, uniformly sampled input for PSD.

    Attributes
    ----------
    time : xarray.DataArray
        Shared time coordinate on the sample dimension.
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
    """Two-dimensional timestream with a mapped channel axis.

    Attributes
    ----------
    n_chans : int
        Number of channels. Transmission, time, and rate properties are inherited.
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
