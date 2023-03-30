#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from pytz import timezone
from astroplan import Observer
from astropy.utils.data import download_file, compute_hash
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

    # @classmethod
    # def _load_data(cls, name):
    #     base_url = 'https://drive.google.com/uc?export=download&id={id_}'
    #     id_ = {
    #             'am_q75': '1pPwjYVzP-PhH0wjH0lIR8CbCPudfuJ8q',
    #             'am_q50': '19dn0ZrHegW7NL8nIZ5ahspYTE83ykjPA',
    #             'am_q25': '1ZiM9KU0TfChKi1m8gCbuIztFJWLVqKUy',
    #             }[name]
    #     url = base_url.format(id_=id_)
    #     # cls.logger.debug(f"download data {name} from url {url}")
    #     return dict(np.load(download_file(url, cache=True)))

    @classmethod
    def _load_data(cls, name):
        # base_url = 'http://toltec1.unity.rc.umass.edu/api/access/datafile/{id}'
        base_url = 'https://dp.lmtgtm.org/api/access/datafile/{id}'
        dl_info = {
            'am_q95': {
                'id': '461',
                'md5': '0ca7b331823237767d26016d19bffb3d',
                },
            'am_q75': {
                'id': '456',
                'md5': 'd6cf4bb27008179ec491864388deac58',
                },
            'am_q50': {
                'id': '455',
                'md5': '6ec393672be8af4dfa06a3f4cf9aa32e',
                },
            'am_q25': {
                'id': '454',
                'md5': '008d7fa69aff187a9edf419f3d961b4c',
                },
            # per season data
            'am_djf_q05': {
                'id': '463',
                'md5': '91545dca93d0e9300718b049893b8eea',
                },
            'am_djf_q25': {
                'id': '466',
                'md5': '3abe83329e39baa734b62f0e87db5a9c',
                },
            'am_djf_q50': {
                'id': '465',
                'md5': '004cb342896210fd23d81b329d0246f0',
                },
            'am_djf_q75': {
                'id': '462',
                'md5': 'e2478719dd67fdbe6ea0d1fb753ab267',
                },
            'am_djf_q95': {
                'id': '464',
                'md5': 'dc8e9e15e5df3238d9e5ecdb39e17dd4',
                },

            'am_jja_q05': {
                'id': '474',
                'md5': '5eae399cba2948630164230c461e24e6',
                },
            'am_jja_q25': {
                'id': '483',
                'md5': '0e677b7ce7f52718584c25c0fbd801c1',
                },
            'am_jja_q50': {
                'id': '478',
                'md5': 'c866c558d1c6c1d20e323927dde1df6c',
                },
            'am_jja_q75': {
                'id': '471',
                'md5': 'e0139e6760b2e45f2768bea572b3c081',
                },
            'am_jja_q95': {
                'id': '468',
                'md5': '91473b3fbb2fd50fa8ae8afb76fea8c0',
                },

            'am_mam_q05': {
                'id': '480',
                'md5': 'b0347815c968aea4873fdff8d1d0258e',
                },
            'am_mam_q25': {
                'id': '482',
                'md5': '074009197665f6c642ca8a9659f6f650',
                },
            'am_mam_q50': {
                'id': '469',
                'md5': '1c0b684bb540ddc13bef67e27a8c15bc',
                },
            'am_mam_q75': {
                'id': '473',
                'md5': '72d39cbcd474622f2ac3874efe0b882b',
                },
            'am_mam_q95': {
                'id': '467',
                'md5': 'd24a7e89e50033de830fe9fbb4627bbf',
                },

            'am_son_q05': {
                'id': '472',
                'md5': 'e587b429e123adde3c77f524d59e71e2',
                },
            'am_son_q25': {
                'id': '475',
                'md5': '6c4384ff61fa00efc59e624946f4e6b6',
                },
            'am_son_q50': {
                'id': '485',
                'md5': '5d8b89c054e8b9dd6279cad6d1f33854',
                },
            'am_son_q75': {
                'id': '481',
                'md5': 'd047c7f6bebfc01242ee9a7898af3165',
                },
            'am_son_q95': {
                'id': '477',
                'md5': '072078db4d8ad0fc2f56e51c39c42034',
                },
            }[name]
        url = base_url.format(id=dl_info['id'])
        filepath = download_file(url, cache=True)
        hash_ = compute_hash(filepath)
        if hash_ != dl_info['md5']:
            raise ValueError(f"MD5 missmatch for file {url}")
        # cls.logger.debug(f"download data {name} from url {url}")
        return dict(np.load(filepath))


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
