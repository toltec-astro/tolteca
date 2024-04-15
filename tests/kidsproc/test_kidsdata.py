import astropy.units as u
import numpy as np
import pytest

from tolteca_kidsproc.kidsdata.sweep import (
    FrequencySweep,
    MultiFrequencySweep,
    MultiSweep,
    Sweep,
)
from tolteca_kidsproc.kidsdata.timestream import MultiTimeStream
from tolteca_kidsproc.kidsdata.utils import (
    ExtendedNDDataRef,
    FrequencyDivisionMultiplexingMixin,
    FrequencyQuantityType,
    validate_quantity,
)


def test_validate_quantity():
    f1 = 1 << u.Hz

    assert validate_quantity(f1, physical_type=FrequencyQuantityType) == f1
    assert validate_quantity(f1, physical_type="frequency") == f1

    f2 = 1 << u.m

    with pytest.raises(ValueError, match="should have physical_type=frequency"):
        validate_quantity(f2, physical_type=FrequencyQuantityType)

    f3 = 1.0j << u.W

    assert validate_quantity(f3, dtype=complex, physical_type="power") == f3

    with pytest.raises(ValueError, match="dtype"):
        assert validate_quantity(f3, dtype=float, physical_type="power") == f3


def test_entdataref():
    dr = ExtendedNDDataRef(
        data=range(10),
        meta={
            "extra": np.arange(10),
        },
        slice_meta_keys=["extra"],
    )
    np.testing.assert_array_equal(dr.data, np.arange(10))
    np.testing.assert_array_equal(dr[-3:].data, np.arange(10)[-3:])
    np.testing.assert_array_equal(dr[-3:].meta["extra"], np.arange(10)[-3:])
    assert dr[2].meta["extra"] == 2

    d = np.arange(12).reshape((3, 4))
    m = np.arange(3)
    dr = ExtendedNDDataRef(
        data=d,
        meta={
            "extra": m,
        },
        slice_meta_keys=[("extra", lambda s: s[0])],
    )
    np.testing.assert_array_equal(dr.data, d)
    np.testing.assert_array_equal(dr[-2:].data, d[-2:])
    np.testing.assert_array_equal(dr[:, :3].meta["extra"], m)
    assert dr[2, :].meta["extra"] == 2


def test_fdm():
    class FDM(FrequencyDivisionMultiplexingMixin, ExtendedNDDataRef):
        pass

    data = np.arange(12).reshape(3, 4)
    f_chans = np.arange(3) << u.Hz
    dr = FDM(
        data=data,
        f_chans=f_chans,
    )
    np.testing.assert_array_equal(dr.data, data)
    np.testing.assert_array_equal(dr[-3:].data, data[-3:])
    np.testing.assert_array_equal(dr[-3:].f_chans, f_chans[-3:])
    np.testing.assert_array_equal(dr[:, :3].f_chans, f_chans)
    np.testing.assert_array_equal(dr[1, :3].f_chans, f_chans[1])

    with pytest.raises(ValueError, match="physical_type"):
        FDM(data=data, f_chans=np.arange(3) << u.s)

    with pytest.raises(ValueError, match="shape"):
        FDM(data=data, f_chans=np.arange(5) << u.Hz)


def test_fswp():
    data = np.arange(12).reshape(3, 4)
    fs = np.arange(4) << u.Hz
    dr = FrequencySweep(
        data=data,
        frequency=fs,
    )
    np.testing.assert_array_equal(dr.data, data)
    np.testing.assert_array_equal(dr[-3:].data, data[-3:])
    np.testing.assert_array_equal(dr[-3:].frequency, fs[-3:])
    np.testing.assert_array_equal(dr[:, :3].frequency, fs[:3])
    np.testing.assert_array_equal(dr[1, :3].frequency, fs[:3])


def test_mfswp():
    data = np.arange(12).reshape(3, 4)
    f_chans = np.arange(3) << u.Hz
    f_sweep = np.arange(4) << u.Hz
    dr = MultiFrequencySweep(
        data=data,
        f_chans=f_chans,
        f_sweep=f_sweep,
    )

    fs = MultiFrequencySweep.make_frequency_grid(f_chans, f_sweep)

    np.testing.assert_array_equal(dr.data, data)
    np.testing.assert_array_equal(dr.frequency, fs)
    np.testing.assert_array_equal(dr[-3:].data, data[-3:])
    np.testing.assert_array_equal(dr[-3:].frequency, fs[-3:])
    np.testing.assert_array_equal(dr[:, :3].frequency, fs[:, :3])
    np.testing.assert_array_equal(dr[1, :3].frequency, fs[1, :3])
    np.testing.assert_array_equal(dr[:, :3].f_chans, f_chans)
    np.testing.assert_array_equal(dr[:, :3].f_sweep, f_sweep[:3])
    np.testing.assert_array_equal(dr[1, :3].frequency, fs[1, :3])
    np.testing.assert_array_equal(dr[1, :3].f_chans, f_chans[1])
    np.testing.assert_array_equal(dr[1, :3].f_sweep, f_sweep[:3])


def test_swp():
    S21 = np.arange(4) * 1.0j
    fs = np.arange(4) << u.Hz
    dr = Sweep(
        S21=S21,
        frequency=fs,
    )
    np.testing.assert_array_equal(dr.data, S21)
    np.testing.assert_array_equal(dr.S21, S21)
    np.testing.assert_array_equal(dr.aS21, np.abs(S21))

    np.testing.assert_array_equal(dr[1].data, S21[1])
    np.testing.assert_array_equal(dr[1].S21, S21[1])
    np.testing.assert_array_equal(dr[1].aS21, np.abs(S21)[1])


def test_mswp():
    S21 = np.arange(12).reshape(3, 4) * 1.0j + 1
    f_chans = np.arange(3) << u.Hz
    f_sweep = np.arange(4) << u.Hz
    dr = MultiSweep(
        S21=S21,
        f_chans=f_chans,
        f_sweep=f_sweep,
    )
    np.testing.assert_array_equal(dr.data, S21)
    np.testing.assert_array_equal(dr.S21, S21)
    np.testing.assert_array_equal(dr.aS21, np.abs(S21))
    np.testing.assert_array_equal(
        dr.frequency,
        MultiSweep.make_frequency_grid(f_chans, f_sweep),
    )
    np.testing.assert_array_equal(dr[1].data, S21[1])
    np.testing.assert_array_equal(dr[1].S21, S21[1])
    np.testing.assert_array_equal(dr[1].aS21, np.abs(S21)[1])

    dr = MultiSweep(
        S21=S21,
        frequency=dr.frequency,
    )
    np.testing.assert_array_equal(dr.S21, S21)
    np.testing.assert_array_equal(
        dr.frequency,
        MultiSweep.make_frequency_grid(dr.f_chans, dr.f_sweep),
    )


def test_mts():
    S21 = np.arange(12).reshape(3, 4) * 1.0j
    f_chans = np.arange(3) << u.Hz
    times = np.arange(4) << u.s
    dr = MultiTimeStream(
        S21=S21,
        f_chans=f_chans,
        times=times,
    )
    np.testing.assert_array_equal(dr.S21, S21)
    np.testing.assert_array_equal(dr.I, S21.real)
    np.testing.assert_array_equal(dr.Q, S21.imag)
    np.testing.assert_array_equal(dr.times, times)
    np.testing.assert_array_equal(dr.index, None)
    np.testing.assert_array_equal(dr[1].S21, S21[1])
    # np.testing.assert_array_equal(dr[1].I, S21[1].real)
    # np.testing.assert_array_equal(dr[1].X, None)


def test_mts_cached():

    X = np.arange(12).reshape(3, 4) * 1.0j
    f_chans = np.arange(3) << u.Hz
    index = np.arange(4)
    f_smp = 0.5 << u.Hz

    dr = MultiTimeStream(r=X.real, x=X.imag, f_chans=f_chans, index=index, f_smp=f_smp)
    np.testing.assert_array_equal(dr.S21, None)
    np.testing.assert_array_equal(dr.I, None)
    np.testing.assert_array_equal(dr.Q, None)
    np.testing.assert_array_equal(dr.times, index / f_smp)
    np.testing.assert_array_equal(dr.index, index)
    np.testing.assert_array_equal(dr.f_smp, f_smp)
    np.testing.assert_array_equal(dr.meta["f_smp"], f_smp)
    np.testing.assert_array_equal(dr.X, X)
    assert "_X_computed" in dr.__dict__

    np.testing.assert_array_equal(dr[1].S21, None)
    np.testing.assert_array_equal(dr[1].I, None)
    # the cached data get sliced directly
    assert "_X_computed" in dr[1].__dict__
    np.testing.assert_array_equal(dr[1, 2].X, X[1, 2])
    np.testing.assert_array_equal(dr[1, 2].times, index[2] / f_smp)
    np.testing.assert_array_equal(dr[1, 2].index, index[2])
    np.testing.assert_array_equal(dr[1, 2].f_smp, f_smp)
    np.testing.assert_array_equal(dr[1, 2].meta["f_smp"], f_smp)

    # reset cache
    dr.r[0] = 1000
    # bad
    assert dr.X[0, 0].real == 0
    dr.reset_cache()
    assert "_X_computed" not in dr.__dict__
    assert dr.X[0, 0].real == 1000
