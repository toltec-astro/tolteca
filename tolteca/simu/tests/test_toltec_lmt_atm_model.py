#! /usr/bin/env python


from ..toltec.lmt_loading_models import (
        LmtAtmosphereModel, ArrayLoadingModel)
import astropy.units as u
import pytest


def test_lmt_atm_model():

    with pytest.raises(
            Exception,
            match='altitude has to be within'
            ):
        m25 = LmtAtmosphereModel(name='am_q25', alt=100 << u.deg)
    m25 = LmtAtmosphereModel(name='am_q25', alt=50 << u.deg)
    m50 = LmtAtmosphereModel(name='am_q50')
    # m75 = LmtAtmosphereModel(name='am_q75')

    print(m25(1.1 << u.mm))
    print(m50(1.1 << u.mm))
    # print(m75)


def test_detector_loading_model():

    alm_a1100 = ArrayLoadingModel(array_name='a1100', atm_model_name='am_q50')
    alm_a1400 = ArrayLoadingModel(array_name='a1400', atm_model_name='am_q50')

    print(alm_a1100(70. << u.deg))
    print(alm_a1400(70. << u.deg))
    # print(m75)
