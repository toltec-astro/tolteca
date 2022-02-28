#!/usr/bin/env python

import numpy as np
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates import SkyCoord
from astroquery.utils import parse_coordinates
# this is not public API so be careful for future changes.
from astropy.coordinates.sky_coordinate_parsers import _get_frame_class \
    as _astropy_get_frame_class
from astropy.coordinates import BaseCoordinateFrame


__all__ = ['rotation_matrix_2d', 'resolve_sky_coords_frame']


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

    if _check_frame_by_name(target_frame, 'icrs'):
        return parse_coordinates(target)
    return SkyCoord(target, frame=target_frame)


def _get_frame_class(frame_name):
    return _astropy_get_frame_class(frame_name)


def _check_frame_by_name(frame, frame_name):
    if isinstance(frame, str):
        return frame == frame_name
    return frame.name == frame_name


def _ensure_frame_class(frame):
    if isinstance(frame, str):
        frame_cls = _get_frame_class(frame)
    elif isinstance(frame, BaseCoordinateFrame):
        frame_cls = frame.__class__
    elif issubclass(frame, BaseCoordinateFrame):
        frame_cls = frame
    else:
        raise ValueError(f"unknown frame {frame}")
    return frame_cls


def resolve_sky_coords_frame(frame, observer=None, time_obs=None):
    """
    Return frame with appropriate attributes set for `observer` at `time_obs`.
    """
    if _check_frame_by_name(frame, 'altaz'):
        return observer.altaz(time=time_obs)
    # fallback to return a frame instance
    if isinstance(frame, BaseCoordinateFrame):
        return frame
    return _ensure_frame_class(frame)()
