#!/usr/bin/env python


from astropy.modeling import Model
import astropy.units as u


__all__ = ['ProjModel', 'LabelFrame']


class ProjModel(Model):
    """
    Base class for models that transform properties from one frame to another.
    """

    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = self.input_frame.axes_names
        self.outputs = self.output_frame.axes_names
        self.name = f'{self.output_frame.name}_proj'


class LabelFrame(object):
    """
    An coordinate frame for describing discrete detector attributes.

    """

    def __init__(self, axes_names, axes_order=None, name=None):
        self._naxes = len(axes_names)
        if axes_order is None:
            axes_order = tuple(range(self._naxes))
        self._axes_order = axes_order
        self._axes_names = axes_names
        self._axes_type = ('LABEL', ) * self._naxes
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self._unit = (u.dimensionless_unscaled, ) * self._naxes
        self._axis_physical_types = ('meta.id', ) * self._naxes

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}(name="{self.name}", '
            f'axes_names={self.axes_names}, '
            f'axes_order={self.axes_order})>')

    def __str__(self):
        return f'{self.__class__.__name__}({self.axes_names[0]})'

    @property
    def name(self):
        """ A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """ A custom name of this frame."""
        self._name = val

    @property
    def naxes(self):
        """ The number of axes in this frame."""
        return self._naxes

    @property
    def axes_names(self):
        """ Names of axes in the frame."""
        return self._axes_names

    @property
    def axes_order(self):
        """ A tuple of indices which map inputs to axes."""
        return self._axes_order

    @property
    def axes_type(self):
        return self._axes_type

    def coordinates(self, *args):
        """ Create world coordinates object"""
        # reorder the args with axes order
        return args

    @property
    def unit(self):
        return self._unit

    @property
    def axis_physical_types(self):
        return self._axis_physical_types
