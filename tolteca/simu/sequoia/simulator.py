#!/usr/bin/env python

from astropy.table import QTable
import numpy as np

from tollan.utils.log import get_logger

from .sequoia_info import sequoia_info
from .models import SequoiaSkyProjModel
from ..mapping.utils import rotation_matrix_2d


__all__ = ['SequoiaObsSimulator']


class SequoiaObsSimulator(object):

    logger = get_logger()

    info = sequoia_info
    site_info = info['site']
    observer = SequoiaSkyProjModel.observer
    _m_sky_proj_cls = SequoiaSkyProjModel

    def __init__(self, mode):

        self._array_prop_table = self._prepare_array_prop_table()
        self._mode = mode

    @property
    def array_prop_table(self):
        """The table containing all detector properties."""
        return self._array_prop_table

    @property
    def mode(self):
        return self._mode

    @classmethod
    def _prepare_array_prop_table(cls):
        """Create the array property table for SEQUOIA."""
        tbl = QTable()
        pixel_space = cls.info['pixel_space']
        pa0 = cls.info['pa0']
        xx = np.arange(0, 4) * pixel_space
        yy = np.arange(0, 4) * pixel_space
        xx, yy = (
            np.ravel(v)
            for v in np.meshgrid(xx - np.mean(xx), yy - np.mean(yy)))
        mat_rot_pa0 = rotation_matrix_2d(pa0)
        x_t = mat_rot_pa0[0, 0] * xx + mat_rot_pa0[0, 1] * yy
        y_t = mat_rot_pa0[1, 0] * xx + mat_rot_pa0[1, 1] * yy
        tbl['x_t'] = x_t
        tbl['y_t'] = y_t
        tbl['pa_t'] = pa0
        return tbl

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_detectors={len(self.array_prop_table)}, '
            f'mode={self.mode})'
            )

    def output_context(self, dirpath, **kwargs):
        return NotImplemented
