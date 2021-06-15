#! /usr/bin/env python

import numpy as np
from astropy.utils.data import download_file
from astropy.modeling.parameters import Parameter, InputParameterError
import astropy.units as u
from astropy import constants as const
from scipy.interpolate import interp1d


from ..base import _Model
from ...utils import get_pkg_data_path

from tollan.utils.log import get_logger


def get_passbands():
    from ...cal.toltec import ToltecPassband
    calobj = ToltecPassband.from_indexfile(get_pkg_data_path().joinpath(
        'cal/toltec_passband/index.yaml'
        ))
    result = dict()
    for array_name in calobj.array_names:
        result[array_name] = calobj.get(array_name=array_name)
    return result


class LmtAtmosphereModel(_Model):
    """
    A model of the LMT atmosphere.

    This model is adapted from the ``toltec-astro/Mapping-Speed-Calculator``.

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

    alt = Parameter(
            default=70., unit=u.deg, min=30., max=70.,
            description='Altitude')

    n_inputs = 1
    n_outputs = 2

    input_units_equivalencies = {'f': u.spectral()}

    @alt.validator
    def alt(self, val):
        """ Ensure `alt` is within range."""
        if isinstance(val, u.Quantity):
            val = val.to_value(self.alt.unit)
        if val < self.alt.min or val > self.alt.max:
            raise InputParameterError(
                f"altitude has to be within [{self.alt.min}:{self.alt.max}]")

    def __init__(self, name='am_q50', *args, **kwargs):
        self._data = self._load_data(name)
        super().__init__(*args, **kwargs)
        self._inputs = ('f', )
        self._outputs = ('T', 'tx')
        self._passbands = get_passbands()

    @property
    def passbands(self):
        return self._passbands

    @property
    def input_units(self):
        return {self.inputs[0]: u.Hz}

    @property
    def return_units(self):
        return {
                self.outputs[0]: u.K,
                self.outputs[1]: u.K,
                }

    def _get_interp(self, alt=None):
        # we use alt here to follow the convention of astropy
        if alt is None:
            alt = self.alt
        alt = alt.to_value(u.deg)
        df = self._data
        interp = interp1d

        el = df['el']
        f_GHz = df['atmFreq'][:, 0]
        T_atm = df['atmTRJ']
        atm_tx = df['atmTtx']

        # deal with elevation match first
        i, = np.searchsorted(el, alt)
        if(abs(el[i] - alt) < 0.001):
            T = interp(f_GHz, T_atm[:, i])
            tx = interp(f_GHz, atm_tx[:, i])
        else:
            # got to manually interpolate over elevation
            Thigh = T_atm[:, i]
            Tlow = T_atm[:, i - 1]
            txhigh = atm_tx[:, i]
            txlow = atm_tx[:, i - 1]
            TatEl = (Tlow * (el[i] - alt) + Thigh * (alt - el[i-1])) \
                / (el[i] - el[i-1])
            txatEl = (txlow * (el[i] - alt) + txhigh * (alt - el[i-1])) \
                / (el[i] - el[i-1])
            T = interp(self.f_GHz, TatEl)
            tx = interp(self.f_GHz, txatEl)
        return {'T': T, 'tx': tx}

    def evaluate(self, f, alt):
        interp = self._get_interp(alt=alt)
        T = interp['T'](f.to_value(u.GHz)) << u.K
        tx = interp['tx'](f.to_value(u.GHz)) << u.K
        return T, tx
        # return alt.value << u.pW

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
        return np.load(download_file(url, cache=True))


class ArrayLoadingModel(_Model):
    """
    A model of the LMT optical loading at the TolTEC arrays.

    This is based on the Mapping-speed-caluator
    """

    logger = get_logger()

    n_inputs = 1
    n_outputs = 2

    T_coldbox = Parameter(
            default=5.75, unit=u.K, min=1.,
            description='The cold box temperature')
    detector_eff = Parameter(
            default=0.8,
            description='The detector optical efficiency'
            )
    noise_factor = Parameter(
            default=0.334,
            description='The detector noise factor'
            )
    horn_aper_eff = Parameter(
            default=0.35,
            description='The horn aperture efficiency'
            )
    tel_surface_rms = Parameter(
            default=76., unit=u.um,
            description='The telescope surface RMS'
            )
    tel_emis = Parameter(
            default=0.06,
            description='The telescope emissivity'
            )

    @property
    def input_units(self):
        return {self.inputs[0]: u.deg}

    def __init__(self, array_name, atm_model_name='am_q50', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs = ('alt', )
        self._outputs = ('P', 'dP')
        self._array_name = array_name
        self._passband = get_passbands()[array_name]
        self._f = self._passband['f'].quantity
        self._norm_throughput = (
            self._passband['throughput']
            / np.sum(self._passband['throughput'])
            )
        self._atm_model = LmtAtmosphereModel(
                name=atm_model_name, alt=70. << u.deg)

    def _calc_consts(self, atm_model, *args):
        f = self._f
        pb = self._norm_throughput
        T_coldbox = self.T_coldbox
        detector_eff = self.detector_eff
        horn_aper_eff = self.horn_aper_eff
        tel_rms = self.tel_surface_rms
        tel_emis = self.tel_emis
        T_atm, tx_atm = atm_model(f)

        # optical efficiency of LMT
        # this is just the Ruze formula
        def primaryOpticalEfficiency():
            return np.exp(-((4.0 * np.pi * tel_rms)/(const.c / f)) ** 2)

        tel_eff = primaryOpticalEfficiency()

        # effective temperature due to optics
        # this is the loading incident on the cryostat window
        def effectiveOpticsTemperature(self):
            # include atmosphere, telescope and coupling optics
            T_tot = T_atm + tel_emis * (273. << u.K) \
                + 3. * (290. << u.K) * 0.01
            return T_tot

        # system efficiency
        def getSystemEfficiency():
            sys_eff = detector_eff * pb * horn_aper_eff * tel_eff
            return sys_eff

        # calculates a weighted average of the loading in Temperature units
        # incident on the cryostat window
        def getToptics():
            to = effectiveOpticsTemperature()
            se = getSystemEfficiency()
            T_opt = np.sum(to*se)/np.sum(se)
            return T_opt

        T_eff = getToptics()

        # effective temperature for loading calculation at detector
        # note that the "horn aperture efficiency" is actually the
        # internal system aperture efficiency since it includes the
        # truncation of the lyot stop and the loss to the cold optics
        def getEffectiveTemperatureAtDetectors():
            Twindow = effectiveOpticsTemperature() \
                    * detector_eff * pb * horn_aper_eff
            TcoldBox = T_coldbox * detector_eff * pb * (1. - horn_aper_eff)
            return Twindow+TcoldBox

        # calculates a weighted average of the loading in Temperature units
        # at the detectors
        # TODO - this isn't really right ... needs work.
        def getTatDetectors():
            td = getEffectiveTemperatureAtDetectors()
            se = pb
            Teff = np.sum(td*se)/np.sum(se)
            return Teff

        T_det = getTatDetectors()
        sys_eff = getSystemEfficiency()

        return locals()
        # noise and sensitivity
        # self.P0, self.net, self.nefd, self.nep = self.getNetAndNep()
        # self.mappingSpeed = self.getMappingSpeed()

    def evaluate(
            self, alt,
            T_coldbox, detector_eff, noise_factor, horn_aper_eff,
            tel_surface_rms, tel_emis
            ):
        m = self._atm_model
        m.alt = alt
        f = self._f
        T, tx = m(f)
        return (
            np.nansum(T * self._norm_throughput),
            np.nansum(tx * self._norm_throughput)
            )
