#!/usr/bin/env python


from astropy.modeling import Model
from gwcs import coordinate_frames as cf
import astropy.units as u
from astropy.coordinates import SkyCoord
from tollan.utils.log import timeit
from cached_property import cached_property
from astropy.table import Table
from astropy.utils import indent


from .utils import _get_skyoffset_frame


__all__ = ['OffsetMappingModel', 'TargetedOffsetMappingModel']


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
    def __init__(self, offset_mapping_model, target, ref_frame=None, t0=None):
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
        """The coordinate frame for `ref_frame`."""
        return self._frame

    @property
    def t0(self):
        """The start time (in UT) of the mapping."""
        return self._t0

    @property
    def t_pattern(self):
        return self.offset_mapping_model.t_pattern

    @cached_property
    def target_in_ref_frame(self):
        """The target coordinates in `ref_frame`."""
        return self._target.transform_to(self.ref_frame)

    @timeit
    def evaluate_coords(self, t):
        """Return the mapping pattern coordinates as evaluated at target.
        """
        frame = _get_skyoffset_frame(self.target_in_ref_frame)
        lon, lat = self.offset_mapping_model(t)
        return SkyCoord(lon, lat, frame=frame).transform_to(
               self.ref_frame)

    @timeit
    def evaluate(self, t):
        coords = self.evaluate_coords(t)
        return coords.data.lon.to(u.deg), coords.data.lat.to(u.deg)

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
