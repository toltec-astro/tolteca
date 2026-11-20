"""Existing native tune parameters retain their effective sweep interpretation."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import MaskedColumn, Table

from tolteca_kidsproc.models import KidsSweepGainWithLinTrend


def native_table():
    return Table(
        {
            "fp": [500e6],
            "Qr": [10000.0],
            "Qc": [20000.0],
            "fr": [500e6],
            "A": [0.2],
            "normI": [2.0],
            "normQ": [3.0],
            "slopeI": [1e-4],
            "slopeQ": [-2e-4],
            "interceptI": [4.0],
            "interceptQ": [5.0],
            "chan": [37],
            "flag": [0],
        }
    )


def test_native_coupling_asymmetry_and_frequency_units():
    table = native_table()
    table["fp"] = [500.0] * u.MHz
    table["fr"] = [500.0] * u.MHz
    model = KidsSweepGainWithLinTrend.from_table(table)
    # Independently evaluated at resonance and at half a linewidth:
    # canonical native response .5+.1j and .3-.2j, respectively.
    np.testing.assert_allclose(
        model.eval([[500e6], [500.025e6]]), [[4.7 + 6.7j], [7.7 + 0.5j]], atol=1e-13
    )
    assert table["chan"].tolist() == [37]
    assert table["fr"].unit == u.MHz


def test_ecsv_rows_masking_and_broadcasting(tmp_path):
    table = native_table()
    table.add_row(table[0])
    table.add_row(table[0])
    table["chan"] = [37, 2, 91]
    table["flag"] = [0, 8192, 0]
    table["normI"] = MaskedColumn([2.0, 9.0, 2.0], mask=[False, True, False])
    path = tmp_path / "tune.txt"
    table.write(path, format="ascii.ecsv")
    restored = Table.read(path, format="ascii.ecsv")
    model = KidsSweepGainWithLinTrend.from_table(restored)
    predictions = model.eval(np.array([[500e6], [500.025e6]]))
    assert predictions.shape == (2, 3)
    assert np.isnan(predictions[:, 1]).all()
    np.testing.assert_allclose(
        predictions[:, [0, 2]], [[4.7 + 6.7j] * 2, [7.7 + 0.5j] * 2]
    )
    assert restored["chan"].tolist() == [37, 2, 91]
    assert restored["flag"].tolist() == [0, 8192, 0]


def test_historical_circle_derotation_preserves_complex_orientation():
    model = KidsSweepGainWithLinTrend(
        fr=500e6,
        Qr=10000,
        g0=2,
        g1=3,
        f0=500e6,
        k0=1e-4,
        k1=-2e-4,
        m0=4,
        m1=5,
    )
    f = np.array([500e6, 500.025e6, 499.975e6])
    circle = model.derotate(model.eval(f), f)
    np.testing.assert_allclose(circle, [10000, 5000 - 5000j, 5000 + 5000j])
    np.testing.assert_allclose(model.rotate(circle, f), model.eval(f))


def test_missing_column_bad_frequency_unit_and_noninvertible_gain():
    table = native_table()
    table.remove_column("Qc")
    with pytest.raises(KeyError):
        KidsSweepGainWithLinTrend.from_table(table)
    table = native_table()
    table["fr"].unit = u.s
    with pytest.raises(u.UnitConversionError):
        KidsSweepGainWithLinTrend.from_table(table)
    table["fr"].unit = u.Hz
    table["normI"] = table["normQ"] = [0.0]
    model = KidsSweepGainWithLinTrend.from_table(table)
    with pytest.raises(ValueError, match="zero gain"):
        model.derotate([1j], [500e6])
