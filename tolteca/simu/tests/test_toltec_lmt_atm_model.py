#! /usr/bin/env python


from ..toltec.lmt_loading_models import LmtAtmosphereModel
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

    print(m25)
    print(m50)
