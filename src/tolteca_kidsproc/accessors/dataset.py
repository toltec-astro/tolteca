"""Construction and file loading of xarray KIDs data.

``make_kids_dataset`` assigns I/Q variables and physical coordinates using a
``KidsMapper``. ``open_datatree`` opens an exported Dataset or DataTree with the
``kids`` accessor available on each node.

Examples
--------
Construct a sweep from I/Q samples and a frequency Quantity:

>>> import numpy as np
>>> from astropy import units as u
>>> from tolteca_kidsproc.accessors import make_kids_dataset, open_datatree
>>> from tolteca_kidsproc.analysis import D21Analysis
>>> frequency = np.linspace(1e9, 1.1e9, 101)
>>> signal = 1 / (1 + 1j * (frequency - 1.05e9) / 1e6)
>>> source = make_kids_dataset(
...     signal.real, signal.imag, frequency=frequency * u.Hz)
>>> source.kids.sweep.S21.shape
(101,)

Export the analysis tree to Zarr and reopen its input and result views:

>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> analysis = D21Analysis(method="gradient", smooth=5)
>>> result = analysis(source.kids.sweep)
>>> with TemporaryDirectory() as directory:
...     target = Path(directory) / "sweep.zarr"
...     _ = result.to_zarr(target, zarr_format=2)
...     with open_datatree(target, engine="zarr") as reopened:
...         assert reopened.kids.sweep.S21.shape == (101,)
...         assert reopened.kids.d21.d21_analysis == analysis
...         assert reopened.kids.d21.d21.shape == (101,)

Multi-channel timestreams share one time coordinate. The sampling frequency is
derived from its uniform spacing:

>>> time = np.arange(128) / 64 * u.s
>>> source = make_kids_dataset(
...     np.ones((3, 128)), np.zeros((3, 128)), time=time)
>>> source.I.dims, source.time.dims
(('chan', 'time'), ('time',))
>>> source.kids.multi_timestream.time.shape
(128,)
>>> source.kids.multi_timestream.f_smp
<Quantity 64. Hz>

Supply only a rate to construct time starting at zero, or both to check agreement:

>>> from_rate = make_kids_dataset(
...     np.ones((3, 128)), np.zeros((3, 128)), f_smp=64 << u.Hz)
>>> from_rate.identical(source)
True
>>> checked = make_kids_dataset(
...     np.ones((3, 128)), np.zeros((3, 128)), time=time, f_smp=64 << u.Hz)
>>> checked.identical(source)
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors.kids import KidsMapper

__all__ = [
    "KidsDataTree",
    "KidsDataset",
    "make_kids_dataset",
    "open_datatree",
]

if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy.units import Quantity

    from tolteca_kidsproc.accessors.kids import KidsAccessor

    class KidsDataset(xr.Dataset):
        """Typed wrapper for xarray Dataset with KIDs accessor."""

        kids: KidsAccessor

    class KidsDataTree(xr.DataTree):
        """Typed wrapper for xarray DataTree with KIDs accessor."""

        kids: KidsAccessor


def open_datatree(
    *args,
    **kwargs,
) -> KidsDataTree:
    """Open an xarray DataTree with the KIDs accessor available.

    Parameters
    ----------
    *args
        Positional arguments for ``xarray.open_datatree``, starting with the
        filename, path, or backend-specific store.
    **kwargs
        Keyword arguments for ``xarray.open_datatree``, including ``engine`` and
        ``chunks``. Array loading and decoding follow the selected backend.

    Returns
    -------
    KidsDataTree
        Opened tree with ``kids`` accessors on its root and children. A file
        containing one Dataset places that Dataset at the root. The stored
        hierarchy is preserved; this function does not prepare an analysis scope.
        Close the tree after use or open it in a context manager.

    See Also
    --------
    xarray.open_datatree : Backend options and supported input formats.
    """
    return cast("KidsDataTree", xr.open_datatree(*args, **kwargs))


def make_kids_dataset(  # noqa: PLR0913 -- I/Q, coordinates, sampling rate, and mapper.
    i_data: npt.NDArray[np.floating],
    q_data: npt.NDArray[np.floating],
    frequency: Quantity | None = None,
    time: Quantity | None = None,
    *,
    f_smp: Quantity | None = None,
    mapper: KidsMapper | None = None,
) -> KidsDataset:
    """Create an I/Q Dataset with optional physical coordinates.

    Parameters
    ----------
    i_data : numpy.ndarray
        In-phase samples. Shape is ``(sample,)`` for one channel or
        ``(chan, sample)`` for multiple channels.
    q_data : numpy.ndarray
        Quadrature samples with the same shape as ``i_data``.
    frequency : astropy.units.Quantity, optional
        Frequency coordinates with the same shape as ``i_data``. Any frequency
        unit is accepted and retained in the coordinate's ``units`` attribute.
    time : astropy.units.Quantity, optional
        One-dimensional time coordinate shared by all channels, with length
        equal to the final I/Q dimension. Any time unit is accepted and retained.
        Samples must be finite, increasing, and uniformly spaced. The spacing
        must agree with the sampling period to relative tolerance ``1e-5``.
        When only ``f_smp`` is supplied, construct elapsed seconds starting at zero.
    f_smp : astropy.units.Quantity, optional
        Positive finite scalar sampling frequency. When ``time`` is supplied,
        the rate must agree with its spacing. When omitted, derive it from ``time``;
        this requires at least two samples. Stored in Hz under the mapped
        attribute name whenever ``time`` or ``f_smp`` is supplied.
    mapper : KidsMapper, optional
        Field mappings used to name the variables, coordinates, attributes, and
        channel dimension. Defaults to ``KidsMapper.from_defaults()``.

    Returns
    -------
    KidsDataset
        Dataset with I/Q variables and supplied or generated coordinates. The
        final dimension is ``time`` when time or sampling frequency is supplied,
        ``sweep`` when only frequency is supplied, and ``sample`` otherwise.
        Two-dimensional inputs also have a leading channel dimension, named
        ``chan`` by default. Time uses only the final dimension; frequency uses
        the full I/Q dimensions, including when both coordinates are supplied.

    Raises
    ------
    ValueError
        I/Q shapes differ, arrays are not one- or two-dimensional, frequency does
        not match the I/Q shape, time is not one-dimensional with the sample
        count, coordinate units have the wrong physical type, or time and
        sampling frequency do not satisfy the sampling requirements above.
    astropy.units.UnitConversionError
        The sampling frequency has a unit incompatible with Hz.
    """
    # Validate inputs
    if i_data.shape != q_data.shape:
        msg = f"I and Q data must have same shape: {i_data.shape} != {q_data.shape}"
        raise ValueError(msg)

    # Use provided mapper or create default
    if mapper is None:
        mapper = KidsMapper.from_defaults()

    schema = mapper.schema
    i_key = mapper.get_name(schema.I)
    q_key = mapper.get_name(schema.Q)
    f_key = mapper.get_name(schema.frequency)
    t_key = mapper.get_name(schema.time)

    ndim = i_data.ndim
    if ndim not in (1, 2):
        msg = f"Data must be 1-D or 2-D, got {ndim}-D"
        raise ValueError(msg)

    rate = None
    if time is not None or f_smp is not None:
        time, rate = _prepare_time(time, f_smp, n_samples=i_data.shape[-1])

    sample_dim = (
        "time" if time is not None else "sweep" if frequency is not None else "sample"
    )
    dims = (
        (sample_dim,) if ndim == 1 else (mapper.get_name(schema.chan_dim), sample_dim)
    )
    coords = {}
    attrs = {}

    if time is not None:
        attrs[mapper.get_name(schema.f_smp)] = rate
        coords[t_key] = ((sample_dim,), time.value, {"units": str(time.unit)})

    if frequency is not None:
        if frequency.unit is not None and frequency.unit.physical_type != "frequency":
            msg = "frequency must have frequency units"
            raise ValueError(msg)

        if frequency.shape != i_data.shape:
            msg = f"Frequency must match I/Q shape: {frequency.shape} != {i_data.shape}"
            raise ValueError(msg)
        coords[f_key] = (dims, frequency.value, {"units": str(frequency.unit)})

    return cast(
        "KidsDataset",
        xr.Dataset(
            {
                i_key: (dims, i_data),
                q_key: (dims, q_data),
            },
            coords=coords,
            attrs=attrs,
        ),
    )


def _prepare_time(
    time: Quantity | None, f_smp: Quantity | None, *, n_samples: int
) -> tuple[Quantity, float]:
    """Prepare a shared time coordinate and its consistent sampling rate in Hz."""
    if time is not None:
        if time.unit is not None and time.unit.physical_type != "time":
            msg = "time must have time units"
            raise ValueError(msg)
        if time.shape != (n_samples,):
            msg = "Time must be 1-D with length matching the I/Q sample dimension"
            raise ValueError(msg)
    rate = _get_sampling_frequency(time, f_smp)
    if time is None:
        time = np.arange(n_samples) / rate << u.s
    return time, rate


def _get_sampling_frequency(time: Quantity | None, f_smp: Quantity | None) -> float:
    """Return the sampling rate in Hz, checking any supplied time spacing."""
    intervals = None
    if time is not None:
        time_s = time.to_value("s")
        if not np.isfinite(time_s).all():
            msg = "Time samples must be finite"
            raise ValueError(msg)
        intervals = np.diff(time_s)
        if np.any(intervals <= 0):
            msg = "Time samples must be strictly increasing"
            raise ValueError(msg)
    if f_smp is None:
        if intervals is None or not intervals.size:
            msg = "Deriving f_smp requires at least two time samples"
            raise ValueError(msg)
        rate = float(1 / intervals[0])
    else:
        rate = float(f_smp.to_value("Hz"))
    if not np.isfinite(rate) or rate <= 0:
        msg = "Sampling frequency must be finite and positive"
        raise ValueError(msg)
    if intervals is not None and not np.allclose(
        intervals, 1 / rate, rtol=1e-5, atol=0
    ):
        msg = "Time samples must be uniformly spaced and agree with f_smp"
        raise ValueError(msg)
    return rate
