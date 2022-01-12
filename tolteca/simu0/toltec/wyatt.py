#! /usr/bin/env python

import numpy as np

import astropy.units as u
from astropy.modeling import models, Parameter
from astropy import coordinates as coord

from gwcs import wcs
from gwcs import coordinate_frames as cf

from tollan.utils.log import timeit

from ..base import (
        ProjModel,
        SkyMapModel,
        RasterScanModelMeta,
        LissajousModelMeta)
from . import ArrayProjModel


class WyattRasterScanModel(SkyMapModel, metaclass=RasterScanModelMeta):
    frame = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))


class WyattLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))


class WyattProjModel(ProjModel):
    """A projection model for the Wyatt robot arm.

    This model is an affine transformation that projects the designed positions
    of detectors on the toltec frame to a plane in front of the cryostat
    window, on which the Wyatt robot arm moves.

    Parameters
    ----------
    rot: `astropy.units.Quantity`
        Rotation angle between the Wyatt frame and the TolTEC frame.
    scale: 2-tuple of `astropy.units.Quantity`
        Scale between the Wyatt frame and the TolTEC frame
    ref_coord: 2-tuple of `astropy.units.Quantity`
        The coordinate of the TolTEC frame origin on the Wyatt frame.
    """

    input_frame = ArrayProjModel.output_frame
    output_frame = cf.Frame2D(
                name='wyatt',
                axes_names=("x", "y"),
                unit=(u.cm, u.cm))
    _name = f'{output_frame.name}_proj'

    n_inputs = 2
    n_outputs = 2

    crval0 = Parameter(default=0., unit=output_frame.unit[0])
    crval1 = Parameter(default=0., unit=output_frame.unit[1])

    def __init__(self, rot, scale, ref_coord=None, **kwargs):
        if not scale[0].unit.is_equivalent(u.m / u.deg):
            raise ValueError("invalid unit for scale.")
        if ref_coord is not None:
            if 'crval0' in kwargs or 'crval1' in kwargs:
                raise ValueError(
                        "ref_coord cannot be specified along with crvals")
            if isinstance(ref_coord, coord.SkyCoord):
                ref_coord = (
                        ref_coord.ra.degree, ref_coord.dec.degree) * u.deg
            kwargs['crval0'] = ref_coord[0]
            kwargs['crval1'] = ref_coord[1]
            kwargs['n_models'] = np.asarray(ref_coord[0]).size

        m_rot = models.Rotation2D._compute_matrix(angle=rot.to_value('rad'))

        self._t2w_0 = models.AffineTransformation2D(
                m_rot * u.deg,
                translation=(0., 0.) * u.deg) | (
                    models.Multiply(scale[0]) & models.Multiply(scale[1])
                    )
        super().__init__(**kwargs)

    @timeit(_name)
    def evaluate(self, x, y, crval0, crval1):
        c0, c1 = self._t2w_0(x, y)
        return c0 + crval0, c1 + crval1

    def get_map_wcs(self, pixscale, ref_coord=None):

        """Return a WCS object that describes a Wyatt map of given pixel scale.

        Parameters
        ----------
        pixscale: 2-tuple of `astropy.units.Quantity`
            Pixel scale of Wyatt map on the Wyatt frame, specified as
            value per pix.
        ref_coord: 2-tuple of `astropy.units.Quantity`, optional
            The coordinate of pixel (0, 0).
        """

        # transformation to go from Wyatt coords to pix coords
        w2m = (
            models.Multiply(1. / pixscale[0]) &
            models.Multiply(1. / pixscale[1])
            )

        # the coord frame used in the array design.
        af = cf.Frame2D(
                name=self.array_name, axes_names=("x", "y"),
                unit=(u.um, u.um))
        # the coord frame on the Wyatt plane
        wf = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))
        # the coord frame on the wyatt map. It is aligned with wyatt.
        mf = cf.Frame2D(
                name='wyattmap', axes_names=("x", "y"),
                unit=(u.pix, u.pix))

        if ref_coord is None:
            ref_coord = self.crval0, self.crval1
        a2w = self._a2w_0 | (
                models.Shift(ref_coord[0]) & models.Shift(ref_coord[1]))
        pipeline = [
                (af, a2w),
                (wf, w2m),
                (mf, None)
                ]
        return wcs.WCS(pipeline)
