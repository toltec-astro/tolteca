#!/usr/bin/env python

from .models import (ToltecArrayProjModel, ToltecSkyProjModel)
from .toltec_info import toltec_info
from tollan.utils.log import get_logger
from kidsproc.kidsmodel.simulator import KidsSimulator
from kidsproc.kidsmodel import ReadoutGainWithLinTrend

import astropy.units as u
from astropy.table import Column, QTable
import numpy as np


__all__ = ['ToltecObsSimulator']


class ToltecObsSimulator(object):

    logger = get_logger()

    info = toltec_info
    site_info = info['site']
    observer = ToltecSkyProjModel.observer
    _m_array_proj = ToltecArrayProjModel()
    _m_sky_proj_cls = ToltecSkyProjModel
    _kids_readout_model_cls = ReadoutGainWithLinTrend

    def __init__(self, array_prop_table, polarized=False):

        apt = self._array_prop_table = self._prepare_array_prop_table(
            array_prop_table)
        self._polarized = polarized

        # create low level models
        self._kids_simulator = KidsSimulator(
            fr=apt['fr'],
            Qr=apt['Qr'],
            background=apt['background'],
            responsivity=apt['responsivity']
            )
        self._kids_readout_model = self._kids_readout_model_cls(
            n_models=len(apt),
            **{
                c: apt[c]
                for c in self._kids_readout_model_cls.param_names
                }
            )
        self.logger.debug(f"kids_simulator: {self.kids_simulator}")
        self.logger.debug(f"kids_readout_model: {self.kids_readout_model}")

    @property
    def array_prop_table(self):
        """The table containing all detector properties."""
        return self._array_prop_table

    @property
    def polarized(self):
        """True if to simulate polarized signal."""
        return self._polarized

    @property
    def kids_simulator(self):
        """The KIDs signal simulator to convert optical loading to
        KIDs timestream (I, Q)."""
        return self._kids_simulator

    @property
    def kids_readout_model(self):
        """The model to simulate specialties of the KIDs data readout system.
        """
        return self._kids_readout_model

    # these are some fiducial kids model params
    _default_kids_props = {
        'fp': 'f',  # column name of apt if string
        'fr': 'f',
        'Qr': 1e4,
        'g0': 200,
        'g1': 0,
        'g': 200,
        'phi_g': 0,
        'f0': 'f',
        'k0': 0 / u.Hz,
        'k1': 0 / u.Hz,
        'm0': 0,
        'm1': 0
            }

    @classmethod
    def _prepare_array_prop_table(cls, array_prop_table):
        """This function populates the `array_prop_table` with sensible
        defaults required to run the simulator"""
        tbl = array_prop_table.copy()
        # note that the apt passed to the function maybe a small portion
        # (both of row-wise and column-wise) of the full array_prop_table
        # of the TolTEC instrument. We rely on the meta dict to loop over
        # the groups
        # array props
        ap_to_cn_map = {
            'wl_center': 'wl_center',
            'a_fwhm': 'a_fwhm',
            'b_fwhm': 'b_fwhm',
            'background': 'background',
            'bkg_temp': 'bkg_temp',
            'responsivity': 'responsivity',
            'passband': 'passband',
            }
        for array_name in tbl.meta['array_names']:
            m = tbl['array_name'] == array_name
            props = {
                c: toltec_info[array_name][k]
                for k, c in ap_to_cn_map.items()
                }
            for c in props.keys():
                if c not in tbl.colnames:
                    tbl.add_column(Column(
                                np.empty((len(tbl), ), dtype='d'),
                                name=c, unit=props[c].unit))
                tbl[c][m] = props[c]

        # kids props
        for c, v in cls._default_kids_props.items():
            if c in tbl.colnames:
                continue
            cls.logger.debug(f"create kids prop column {c}")
            if isinstance(v, str) and v in tbl.colnames:
                tbl[c] = tbl[v]
                continue
            if isinstance(v, u.Quantity):
                value = v.value
                unit = v.unit
            else:
                value = v
                unit = None
            if np.isscalar(value):
                tbl.add_column(
                        Column(np.full((len(tbl),), value), name=c, unit=unit))
            else:
                raise ValueError('invalid kids prop')

        # calibration related
        # TODO need to revisit these assumptions
        if 'flxscale' not in tbl.colnames:
            tbl['flxscale'] = (1. / tbl['responsivity']).quantity.value

        # kids readout noise
        if 'sigma_readout' not in tbl.colnames:
            tbl['sigma_readout'] = 10.

        # detector locations in toltec frame
        if not {'x_t', 'y_t', 'pa_t'}.issubset(tbl.colnames):
            x_t, y_t, pa_t = cls._m_array_proj(
                tbl['x'].quantity,
                tbl['y'].quantity,
                tbl['array'], tbl['fg']
                )
            if not {"x_t", "y_t"}.issubset(tbl.colnames):
                tbl['x_t'] = x_t
                tbl['y_t'] = y_t
            if 'pa_t' not in tbl.colnames:
                tbl['pa_t'] = pa_t
        return QTable(tbl)

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_detectors={len(self.array_prop_table)}, '
            f'polarized={self.polarized})'
            )
