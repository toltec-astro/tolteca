#! /usr/bin/env python


from ..toltec.lmt import get_lmt_atm_models
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose


def test_lmt_atm_model():

    T_25, tx_25 = get_lmt_atm_models(name='am_q25')
    # T_50, tx_50 = get_lmt_atm_models(name='am_q50')
    # m75 = LmtAtmosphereModel(name='am_q75')

    assert_quantity_allclose(
            T_25(1.1 << u.mm, 50. << u.deg),
            22.18836569 << u.K,
            rtol=1e-4)
    assert_quantity_allclose(
            tx_25(1.1 << u.mm, 50. << u.deg),
            0.91186965,
            rtol=1e-4)

    # print(m75)


def test_lmt_atm_model_reimplement():
    import sys
    from pathlib import Path
    _toltec_sensitivity_module_path = Path(
            '~/Codes/toltec/trunk/toltec_sensitivity').expanduser()
    sys.path.insert(
        0, _toltec_sensitivity_module_path.parent.as_posix())
    from toltec_sensitivity.LMTAtmosphere import LMTAtmosphere

    m50_orig = LMTAtmosphere(
            quartile=50., elevation=70.,
            path=_toltec_sensitivity_module_path)
    T_50, tx_50 = get_lmt_atm_models(name='am_q50')

    f = (2.0 << u.mm).to(u.GHz, equivalencies=u.spectral())
    alt = 70. << u.deg

    T_orig = m50_orig.T(f.to_value(u.GHz)) << u.K
    tx_orig = m50_orig.tx(f.to_value(u.GHz))

    T = T_50(f, alt)
    tx = tx_50(f, alt)

    assert_quantity_allclose(T_orig, T, rtol=1e-5)
    assert_quantity_allclose(tx_orig, tx, rtol=1e-5)
