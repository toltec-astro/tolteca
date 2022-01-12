#!/usr/bin/env python


from copy import deepcopy
from tollan.utils import rupdate
import astropy.units as u
from .lmt import lmt_info
from ...common.toltec import toltec_info as _toltec_info


__all__ = ['toltec_info']


def _make_extended_toltec_info(toltec_info):
    """Extend the toltec_info dict with array properties related to
    the simulator.
    """
    toltec_info = deepcopy(toltec_info)

    # array props
    def get_fwhm(array_name):
        a_fwhm_a1100 = 5. << u.arcsec
        b_fwhm_a1100 = 5. << u.arcsec
        if array_name == 'a1100':
            return {
                'a_fwhm': a_fwhm_a1100,
                'b_fwhm': b_fwhm_a1100,
                }
        scale = (
            toltec_info[array_name]['wl_center']
            / toltec_info['a1100']['wl_center'])
        return {
                'a_fwhm': a_fwhm_a1100 * scale,
                'b_fwhm': b_fwhm_a1100 * scale,
                }
    rupdate(toltec_info, {
        array_name: get_fwhm(array_name)
        for array_name in toltec_info['array_names']
        })
    # these are generated from Grant's Mapping-Speed-Calculator code
    # The below is for elev 45 deg, atm 25 quantiles
    rupdate(toltec_info, {
        'a1100': {
            'background': 10.01 * u.pW,
            'bkg_temp': 9.64 * u.K,
            'responsivity': 5.794e-5 / u.pW,
            'passband': 65 * u.GHz,
            },
        'a1400': {
            'background': 7.15 * u.pW,
            'bkg_temp': 9.43 * u.K,
            'responsivity': 1.1e-4 / u.pW,
            'passband': 50 * u.GHz,
            },
        'a2000': {
            'background': 5.29 * u.pW,
            'bkg_temp': 8.34 * u.K,
            'responsivity': 1.1e-4 / u.pW,
            'passband': 42 * u.GHz,
            },
        })
    # add lmt info as toltec site_info
    toltec_info['site'] = lmt_info
    return toltec_info


toltec_info = _make_extended_toltec_info(_toltec_info)
"""The TolTEC info dict with additional items related to simulator."""
