#!/usr/bin/env python


from astropy.modeling import Model
from gwcs import coordinate_frames as cf
import astropy.units as u
from astropy.coordinates import SkyCoord
from tollan.utils.log import timeit
from astropy.table import Table
from astropy.utils import indent
from enum import Flag, auto

from .utils import _get_skyoffset_frame, resolve_sky_coords_frame


__all__ = ['OffsetMappingModel', 'TargetedOffsetMappingModel']


class PatternKind(Flag):
    """The available pattern kinds."""
    raster = auto()
    lissajous = auto()
    rastajous = auto()
    raster_like = raster | rastajous


class MappingModelBase(Model):
    """The base class for all mapping models."""

    def pattern_kind(self):
        return self._pattern_kind


class OffsetMappingModel(Model):
    """The base class for mapping models defined in offset coordinates."""

    n_inputs = 1
    n_outputs = 2
    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.pattern_name
        self.inputs = ('t', )
        self.outputs = self.frame.axes_names

    def get_traj_model(self, **kwargs):
        return TargetedOffsetMappingModel(
            offset_mapping_model=self, **kwargs)


class TrajMappingModel(Model):
    """The base class for mapping models in absolute coordinates."""

    n_inputs = 1
    n_outputs = 2
    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('t', )


class TargetedOffsetMappingModel(TrajMappingModel):
    """The class for trajectories rendered using `OffsetMappingModel`
    around a target.

    """
    def __init__(
            self,
            offset_mapping_model,
            target,
            ref_frame=None,
            t0=None,
            observer=None
            ):
        if ref_frame is None:
            ref_frame = target.frame
        # TODO maybe we need not hard-code the units here.
        frame = cf.CelestialFrame(
                name=ref_frame.name,
                reference_frame=ref_frame,
                unit=(u.deg, u.deg)
                )
        self._offset_mapping_model = offset_mapping_model
        self._target = target
        self._ref_frame = ref_frame
        self._frame = frame
        self._t0 = t0
        self._observer = observer
        super().__init__(name=f'targeted_{offset_mapping_model.name}')
        self.outputs = self.frame.axes_names

    @property
    def offset_mapping_model(self):
        """The offset model."""
        return self._offset_mapping_model

    @property
    def target(self):
        """The target coordinates."""
        return self._target

    @property
    def ref_frame(self):
        """The reference frame with respect to which the offsets
        are interpreted."""
        return self._ref_frame

    @property
    def frame(self):
        """The WCS coordinate frame for `ref_frame`."""
        return self._frame

    @property
    def t0(self):
        """The start time (in UT) of the mapping."""
        return self._t0

    @property
    def observer(self):
        """The observer of the mapping."""
        return self._observer

    @property
    def t_pattern(self):
        return self.offset_mapping_model.t_pattern

    @property
    def pattern_kind(self):
        return self.offset_mapping_model.pattern_kind

    @timeit
    def evaluate_coords(self, t):
        """Return the mapping pattern coordinates as evaluated at target.
        """
        time_obs = self.t0 + t
        ref_frame = resolve_sky_coords_frame(
                self.ref_frame,
                observer=self.observer,
                time_obs=time_obs)
        target_in_ref_frame = self.target.transform_to(ref_frame)
        frame = _get_skyoffset_frame(target_in_ref_frame)
        lon, lat = self.offset_mapping_model(t)
        return SkyCoord(lon, lat, frame=frame).transform_to(ref_frame)

    @timeit
    def evaluate(self, t):
        coords = self.evaluate_coords(t)
        return coords.data.lon.to(u.deg), coords.data.lat.to(u.deg)

    @timeit
    def evaluate_holdflag(self, t):
        return self.offset_mapping_model.evaluate_holdflag(t)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.offset_mapping_model}, '
            f'target={self.target}, t0={self.t0})'
            )

    def __str__(self):
        default_keywords = [
            ('Model', self.__class__.__name__),
            ('Name', self.name),
            ('Inputs', self.inputs),
            ('Outputs', self.outputs),
            ('Model set size', len(self)),
            ("Target", self.target),
            ("Reference frame", self.ref_frame.__class__.__name__),
            ("Time_0", self.t0),
        ]
        parts = [f'{keyword}: {value}'
                 for keyword, value in default_keywords
                 if value is not None]
        parts.append('Offset Model Parameters:')
        m = self.offset_mapping_model
        if len(m) == 1:
            columns = [[getattr(m, name).value]
                       for name in m.param_names]
        else:
            columns = [getattr(m, name).value
                       for name in m.param_names]
        if columns:
            param_table = Table(columns, names=m.param_names)
            # Set units on the columns
            for name in m.param_names:
                param_table[name].unit = getattr(m, name).unit
            parts.append(indent(str(param_table), width=4))
        return '\n'.join(parts)
