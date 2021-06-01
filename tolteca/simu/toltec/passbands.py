#! /usr/bin/env python

import numpy as np
from astropy.utils.data import download_file
from astropy.modeling.parameters import Parameter, InputParameterError
import astropy.units as u
from ..base import _Model

from tollan.utils.log import get_logger


class Passband():
    """
    A model of the LMT atmosphere.

    This model is adapted from the ``toltec-astro/Mapping-Speed-Calculator``.

    Parameters
    ----------
    name : {"am_q25", "am_q50"}, default="am_q50"
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
    n_outputs = 1

    input_units_equivalencies = {'f': u.spectral()}

    @alt.validator
    def alt(self, val):
        """ Ensure `alt` is within range."""
        if val < self.alt.min or val > self.alt.max:
            raise InputParameterError(
                f"altitude has to be within [{self.alt.min}:{self.alt.max}]")

    def __init__(self, name='am_q50', *args, **kwargs):
        self._data = self._load_data(name)
        super().__init__(*args, **kwargs)
        self._inputs = ('f', )
        self._outputs = ('P', )

    @property
    def input_units(self):
        return {self.inputs[0]: u.Hz}

    @property
    def return_units(self):
        return {self.outputs[0]: u.pW}

    def evaluate(self, alt):
        return alt.value << u.pW

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
                'am_q50': '19dn0ZrHegW7NL8nIZ5ahspYTE83ykjPA',
                'am_q25': '1ZiM9KU0TfChKi1m8gCbuIztFJWLVqKUy',
                }[name]
        url = base_url.format(id_=id_)
        # cls.logger.debug(f"download data {name} from url {url}")
        return np.load(download_file(url, cache=True))


class LmtLoadingModel(_Model):
    """
    A model of the LMT optical loading at the TolTEC detectors.

    This is based on the Mapping-speed-caluator
    """
    pass
