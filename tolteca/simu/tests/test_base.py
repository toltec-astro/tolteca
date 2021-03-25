#! /usr/bin/env python


from ..base import SourceCatalogModel, SourceImageModel
from astropy.table import Table, Column
import astropy.units as u
import numpy as np
from astropy.modeling.models import Gaussian2D


def test_point_source_list_model():
    t = Table([
        Column(np.arange(0, 10, 1), unit=u.deg, name='ra'),
        Column(np.arange(1, 11, 1), unit=u.deg, name='dec'),
        Column(np.arange(2, 12, 1), unit=u.mJy, name='flux'),
        ])
    psfmodel = Gaussian2D(fwhm)
    m = SourceListModel(lon=t['ra'], lat=t['dec'], flux=t['flux'])

    print(m)


def test_source_image_model():
    return
