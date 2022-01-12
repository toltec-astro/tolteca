#! /usr/bin/env python


from ..toltec import ArrayLoadingModel
import astropy.units as u
# import pytest
from astropy.tests.helper import assert_quantity_allclose


def test_array_loading_model():

    alm_a1100 = ArrayLoadingModel(array_name='a1100', atm_model_name='am_q50')
    assert ArrayLoadingModel(array_name='a1400', atm_model_name='am_q50')
    assert ArrayLoadingModel(array_name='a2000', atm_model_name='am_q50')

    alt = 70. << u.deg

    assert_quantity_allclose(
            alm_a1100._get_T(alt, return_avg=True), 59.75238531 << u.K)

    # assert alm_a1400_get_P(70. << u.deg) == 0
    # assert alm_a2000_get_P(70. << u.deg) == 0


def test_array_loading_reimplement():
    import sys
    from pathlib import Path
    _toltec_sensitivity_module_path = Path(
            '~/Codes/toltec/trunk/toltec_sensitivity').expanduser()
    sys.path.insert(
        0, _toltec_sensitivity_module_path.parent.as_posix())
    from toltec_sensitivity.Detector import Detector

    alt = 70. << u.deg
    alm_a1100_orig = Detector(
            atmQuartile=25,
            elevation=alt.to_value(u.deg),
            band=1.1,
            )
    # to compare the per-freq values, we need to
    # patch the passband so it uses the same passband as array loading model
    alm_a1100_orig.getPassband = lambda *a, **k: None
    alm_a1100 = ArrayLoadingModel(array_name='a1100', atm_model_name='am_q25')
    alm_a1100_orig.f_GHz = alm_a1100._f.to_value(u.GHz)
    alm_a1100_orig.passband = alm_a1100._throughput
    alm_a1100_orig.update()

    assert_quantity_allclose(
            alm_a1100_orig.primaryOpticalEfficiency(),
            alm_a1100._tel_primary_surface_optical_efficiency
            )
    assert_quantity_allclose(
            alm_a1100_orig.effectiveOpticsTemperature() << u.K,
            alm_a1100._get_T(
                alt=alt, return_avg=False)
            )
    assert_quantity_allclose(
            alm_a1100_orig.getEffectiveTemperatureAtDetectors() << u.K,
            alm_a1100._get_T_det(
                alt=alt, return_avg=False)
            )

    # integrated quantities. no need to patch the passband
    alm_a1100_orig = Detector(
            atmQuartile=25,
            elevation=alt.to_value(u.deg),
            band=1.1,
            )
    alm_a1100_orig.summary()
    print(alm_a1100.make_summary_table())
    assert_quantity_allclose(
            alm_a1100_orig.getToptics() << u.K,
            alm_a1100._get_T(alt=alt, return_avg=True),
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.getTatDetectors() << u.K,
            # alm_a1100._get_T_det(alt=alt, return_avg=True),
            # note that the Detector.py get this weight sum again over
            # the passband
            alm_a1100._wsum(
                alm_a1100._get_T_det(alt=alt, return_avg=False),
                alm_a1100._throughput),
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.getTatDetectors() << u.K,
            # alm_a1100._get_T_det(alt=alt, return_avg=True),
            # note that the Detector.py get this weight sum again over
            # the passband
            alm_a1100._wsum(
                alm_a1100._get_T_det(alt=alt, return_avg=False),
                alm_a1100._throughput),
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.P0 << u.W,
            alm_a1100._get_P(alt),
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.nep << u.W * u.Hz ** -0.5,
            alm_a1100._get_noise(alt)['nep'],
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.net << u.uK * u.Hz ** -0.5,
            alm_a1100._get_noise(alt)['net_cmb'],
            rtol=1e-3
            )
    assert_quantity_allclose(
            alm_a1100_orig.nefd << u.mJy * u.Hz ** -0.5,
            alm_a1100._get_noise(alt)['nefd'],
            rtol=1e-3
            )
