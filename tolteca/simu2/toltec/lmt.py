#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from pytz import timezone
from astroplan import Observer
from astropy.utils.data import download_file
import astropy.units as u
from astropy import coordinates as coord

from tollan.utils.log import get_logger

from astropy.modeling import Model
from ...common.lmt import lmt_info as _lmt_info


__all__ = [
        'lmt_info',
        'LmtAtmosphereModel', 'LmtAtmosphereTxModel',
        'get_lmt_atm_models']


lmt_location = coord.EarthLocation.from_geodetic(**_lmt_info['location'])
"""The local of LMT."""


lmt_timezone_local = timezone(_lmt_info['timezone_local'])
"""The local time zone of LMT."""


lmt_observer = Observer(
        name=_lmt_info['name_long'],
        location=lmt_location,
        timezone=lmt_timezone_local,
        )
"""The observer at LMT."""


lmt_info = dict(
    _lmt_info,
    location=lmt_location,
    timezone_local=lmt_timezone_local,
    observer=lmt_observer
    )
"""The LMT info dict with additional items related to simulator."""


class LmtAtmosphereData(object):
    """
    A data class of the LMT atmosphere model.

    This is adapted from the ``toltec-astro/Mapping-Speed-Calculator``.

    Parameters
    ----------
    name : {"am_q25", "am_q50", "am_q75"}, default="am_q50"
        The source dataset used in the model.
        The "am_q*" data are constructed with the am code
        https://www.cfa.harvard.edu/~spaine/am/, with
        quartiles set to 25 and 50, respectively.

    *args, **kwargs
        Arguments passed to the `_Model` constructor.

    """
    logger = get_logger()

    def __init__(self, name='am_q50'):
        self._data = self._load_data(name)

    def get_interp2d(self, key='T'):
        if key not in ('T', 'tx'):
            raise ValueError(f"invalid data interp key {key}")
        df = self._data
        interp = RectBivariateSpline

        el = df['el']
        f_GHz = df['atmFreq'][:, 0]
        if key == 'T':
            z = df['atmTRJ']
        elif key == 'tx':
            z = df['atmTtx']
        return interp(f_GHz, el, z)

    def get_interp(self, alt=None, key='T'):
        # we use alt here to follow the convention of astropy
        if alt is None:
            alt = self.alt
        alt = alt.to_value(u.deg)
        if key not in ('T', 'tx'):
            raise ValueError(f"invalid data interp key {key}")
        df = self._data
        interp = interp1d

        el = df['el']
        f_GHz = df['atmFreq'][:, 0]
        T_atm = df['atmTRJ']
        atm_tx = df['atmTtx']

        # deal with elevation match first
        i, = np.searchsorted(el, alt)
        if(abs(el[i] - alt) < 0.001):
            if key == 'T':
                return interp(f_GHz, T_atm[:, i])
            if key == 'tx':
                return interp(f_GHz, atm_tx[:, i])
        # got to manually interpolate over elevation
        if key == 'T':
            Thigh = T_atm[:, i]
            Tlow = T_atm[:, i - 1]
            TatEl = (Tlow * (el[i] - alt) + Thigh * (alt - el[i-1])) \
                / (el[i] - el[i-1])
            return interp(self.f_GHz, TatEl)
        if key == 'tx':
            txhigh = atm_tx[:, i]
            txlow = atm_tx[:, i - 1]
            txatEl = (txlow * (el[i] - alt) + txhigh * (alt - el[i-1])) \
                / (el[i] - el[i-1])
            return interp(self.f_GHz, txatEl)
        raise Exception  # should not happen

    # @classmethod
    # def _load_data(cls, name):
    #     am_data_url_fmt = (
    #         'https://github.com/toltec-astro/'
    #         'Mapping-Speed-Calculator/raw/master/amLMT{quartile:d}.npz')
    #     if name == 'am_q25':
    #         url = am_data_url_fmt.format(quartile=25)
    #     elif name == 'am_q50':
    #         url = am_data_url_fmt.format(quartile=50)
    #     else:
    #         raise ValueError(f"invalid dataset name {name}")
    #     cls.logger.debug(f"download data {name} from url {url}")
    #     return np.load(download_file(url, cache=True))

    @classmethod
    def _load_data(cls, name):
        base_url = 'https://drive.google.com/uc?export=download&id={id_}'
        id_ = {
                'am_q75': '1pPwjYVzP-PhH0wjH0lIR8CbCPudfuJ8q',
                'am_q50': '19dn0ZrHegW7NL8nIZ5ahspYTE83ykjPA',
                'am_q25': '1ZiM9KU0TfChKi1m8gCbuIztFJWLVqKUy',
                }[name]
        url = base_url.format(id_=id_)
        # cls.logger.debug(f"download data {name} from url {url}")
        return dict(np.load(download_file(url, cache=True)))


class LmtAtmosphereModel(Model):
    """
    A model of the LMT atmosphere.

    Parameters
    ----------
    name : {"am_q25", "am_q50", "am_q75"}, default="am_q50"
        The source dataset used in the model.
        The "am_q*" data are constructed with the am code
        https://www.cfa.harvard.edu/~spaine/am/, with
        quartiles set to 25 and 50, respectively.

    *args, **kwargs
        Arguments passed to the `_Model` constructor.

    """
    logger = get_logger()

    n_inputs = 2
    n_outputs = 1

    input_units_equivalencies = {'f': u.spectral()}

    def __init__(self, name='am_q50', *args, **kwargs):
        self._data = LmtAtmosphereData(name=name)
        self._interp = self._data.get_interp2d(key='T')
        super().__init__(*args, **kwargs)
        self._inputs = ('f', 'alt')
        self._outputs = ('T', )

    @property
    def input_units(self):
        return {
                self.inputs[0]: u.Hz,
                self.inputs[1]: u.deg,
                }

    @property
    def return_units(self):
        return {
                self.outputs[0]: u.K,
                }

    def evaluate(self, f, alt):
        T = self._interp(
                f.to_value(u.GHz), alt.to_value(u.deg), grid=False) << u.K
        return T


class LmtAtmosphereTxModel(Model):
    """
    A model of the LMT atmosphere transmission.

    Parameters
    ----------
    name : {"am_q25", "am_q50", "am_q75"}, default="am_q50"
        The source dataset used in the model.
        The "am_q*" data are constructed with the am code
        https://www.cfa.harvard.edu/~spaine/am/, with
        quartiles set to 25 and 50, respectively.

    *args, **kwargs
        Arguments passed to the `Model` constructor.

    """
    logger = get_logger()

    n_inputs = 2
    n_outputs = 1

    input_units_equivalencies = {'f': u.spectral()}

    def __init__(self, name='am_q50', *args, **kwargs):
        self._data = LmtAtmosphereData(name=name)
        self._interp = self._data.get_interp2d(key='tx')
        super().__init__(*args, **kwargs)
        self._inputs = ('f', 'alt')
        self._outputs = ('tx', )

    @property
    def input_units(self):
        return {
                self.inputs[0]: u.Hz,
                self.inputs[1]: u.deg,
                }

    def evaluate(self, f, alt):
        tx = self._interp(f.to_value(u.GHz), alt.to_value(u.deg), grid=False)
        return tx


def get_lmt_atm_models(name, *args, **kwargs):
    """Return the LMT atmosphere model and the transmission model.

    Parameters
    ----------
    name : {"am_q25", "am_q50", "am_q75"}, default="am_q50"
        The source dataset used in the model.
        The "am_q*" data are constructed with the am code
        https://www.cfa.harvard.edu/~spaine/am/, with
        quartiles set to 25 and 50, respectively.
    *args, **kwargs
        Arguments passed to the `_Model` constructor.
    """
    return (
            LmtAtmosphereModel(name=name, *args, **kwargs),
            LmtAtmosphereTxModel(name=name, *args, **kwargs)
            )
