#!/usr/bin/env python

import numpy as np
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates import SkyCoord
from astroquery.utils import parse_coordinates
# this is not public API so be careful for future changes.
from astropy.coordinates.sky_coordinate_parsers import _get_frame_class \
    as _astropy_get_frame_class


__all__ = ['rotation_matrix_2d', ]


def rotation_matrix_2d(angle):
    """The rotation matrix for 2d."""
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]
                     ], dtype=np.float64)


def _get_skyoffset_frame(c):
    """This function creates a skyoffset_frame and ensures
    the cached origin frame attribute is the correct instance.
    """
    frame = c.skyoffset_frame()
    frame_transform_graph._cached_frame_attributes['origin'] = \
        frame.frame_attributes['origin']
    return frame


def _resolve_target(target, target_frame='icrs'):
    """Return an `astropy.coordinates.SkyCoord` form `target` and its frame."""

    if target_frame == 'icrs':
        return parse_coordinates(target)
    return SkyCoord(target, frame=target_frame)


def _get_frame_class(frame_name):
    return _astropy_get_frame_class(frame_name)
