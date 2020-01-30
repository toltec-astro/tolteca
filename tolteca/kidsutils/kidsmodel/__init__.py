#! /usr/bin/env python

from astropy.modeling import Parameter, Model
from astropy.modeling.core import (
        _prepare_inputs_model_set, _prepare_inputs_single_model,
        _validate_input_shapes)
import numpy as np
from astropy import units as u
import inspect


class _Model(Model):
    """Subclass of astropy.modeling.Model that support complex type."""

    # code snippet from `astropy.modeling.Model`
    def prepare_inputs(self, *inputs, model_set_axis=None, equivalencies=None,
                       **kwargs):
        """
        This method is used in `~astropy.modeling.Model.__call__` to ensure
        that all the inputs to the model can be broadcast into compatible
        shapes (if one or both of them are input as arrays), particularly if
        there are more than one parameter sets. This also makes sure that (if
        applicable) the units of the input will be compatible with the evaluate
        method.
        """

        # When we instantiate the model class, we make sure that __call__ can
        # take the following two keyword arguments: model_set_axis and
        # equivalencies.

        if model_set_axis is None:
            # By default the model_set_axis for the input is assumed to be the
            # same as that for the parameters the model was defined with
            # TODO: Ensure that negative model_set_axis arguments are respected
            model_set_axis = self.model_set_axis

        n_models = len(self)

        params = [getattr(self, name) for name in self.param_names]
        inputs = [np.asanyarray(_input, dtype=None) for _input in inputs]

        _validate_input_shapes(inputs, self.inputs, n_models,
                               model_set_axis, self.standard_broadcasting)

        if 'inputs_map' in kwargs:
            inputs_map = kwargs['inputs_map']
        else:
            inputs_map = None
        inputs = self._validate_input_units(inputs, equivalencies, inputs_map)

        # The input formatting required for single models versus a multiple
        # model set are different enough that they've been split into separate
        # subroutines
        if n_models == 1:
            return _prepare_inputs_single_model(self, params, inputs,
                                                **kwargs)
        else:
            return _prepare_inputs_model_set(self, params, inputs, n_models,
                                             model_set_axis, **kwargs)


def _get_func_args(func):
    return inspect.getfullargspec(func).args


def _set_mutual_inversion(cls1, cls2):

    cls1.inverse = property(lambda self: cls2())
    cls2.inverse = property(lambda self: cls1())


class _ResonanceCircleTransformMixin(object):
    """Mixin class that defines basic resonance circle transform.

    The transform reads

    .. code-block:: text

        v' = 0.5 / v

    """

    n_inputs = 1
    n_outputs = 1
    _separable = True

    @staticmethod
    def evaluate(value):
        return 0.5 / value


class ResonanceCircleComplex(_ResonanceCircleTransformMixin, _Model):
    """Model that describes the resonance circle of KIDs in complex plane.

    The model reads

    .. code-block:: text

        S = 0.5 / X

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.inputs = ('X', )
        self.outputs = ('S', )


class ResonanceCircleComplexInv(_ResonanceCircleTransformMixin, _Model):
    """Inversion of `ResonanceCircleComplex`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('S', )
        self.outputs = ('X', )


class ResonanceCircleQr(_ResonanceCircleTransformMixin, _Model):
    """Model that describes the relation of `r` and `Qr`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('r', )
        self.outputs = ('Qr', )


class ResonanceCircleQrInv(_ResonanceCircleTransformMixin, _Model):
    """Inversion of `ResonanceCircleQr`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('Qr', )
        self.outputs = ('r', )


class _ResonanceCircleTransform2Mixin(object):
    """Mixin class that defines basic resonance circle transform for
    real and imaginary parts separately.

    The transform reads

    .. code-block:: text

        I + iQ = 0.5 / (r + ix)

    """

    n_inputs = 2
    n_outputs = 2
    _separable = False

    @staticmethod
    def evaluate(v1, v2):
        f = 0.5 / (v1 ** 2 + v2 ** 2)
        return v1 * f, - v2 * f


class ResonanceCircle(_ResonanceCircleTransform2Mixin, _Model):
    """Same as `ResonanceCircleComplex`, but with separate real and imaginary
    parts as inputs and outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('r', 'x')
        self.outputs = ('I', 'Q')


class ResonanceCircleInv(_ResonanceCircleTransform2Mixin, _Model):
    """Inversion of `ResonanceCircle`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('I', 'Q')
        self.outputs = ('r', 'x')


_set_mutual_inversion(ResonanceCircleComplex, ResonanceCircleComplexInv)
_set_mutual_inversion(ResonanceCircleQr, ResonanceCircleQrInv)
_set_mutual_inversion(ResonanceCircle, ResonanceCircleInv)


class OpticalDetune(_Model):
    """Model that describes detuning of KIDs in response to incident
    optical power.

    The model reads

    .. code-block:: text

        x = (p - background) * responsivity

    """

    n_inputs = 1
    n_outputs = 1
    _separable = True

    background = Parameter(default=5.0 * u.pW, min=0.)
    responsivity = Parameter(default=1e-17, unit=1. / u.W, min=0.)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('p', )
        self.outputs = ('x', )

    @staticmethod
    def evaluate(p, background, responsivity):
        return (p - background) * responsivity


class InstrumentalDetune(_Model):
    """Model that describes the detuning in terms of the probe tone and
    detector intrinsics.

    The model reads

    .. code-block:: text

        x = (fp / fr - 1.)

    """

    n_inputs = 2
    n_outputs = 1
    _separable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ('fp', 'fr')
        self.outputs = ('x', )

    @staticmethod
    def evaluate(fp, fr):
        return fp / fr - 1.


class _ComposableModelBase(_Model):
    """Base class that setup itself with mixin classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_inputs()
        self._set_outputs()


class _ReadoutReprComplexMixin(object):
    """Mixin class that sets the output to use complex S21."""

    n_outputs = 1
    _separable = True

    def _set_outputs(self):
        self.outputs = ('S', )

    @staticmethod
    def _repr_apply(a, b):
        if a.shape == (1,):
            a = a[0]
        if b.shape == (1,):
            b = b[0]
        return a + 1.j * b

    def __call__(self, x, y, **kwargs):
        # make sure they are the same shape
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        if x.shape == ():
            x = np.full_like(y, np.asscalar(x))
        elif y.shape == ():
            y = np.full_like(x, np.asscalar(y))
        return super().__call__(x, y, **kwargs)


class _ReadoutRepr2Mixin(object):
    """Mixin class that sets the output to use separate I and Q."""

    n_outputs = 2
    _separable = False

    def _set_outputs(self):
        self.outputs = ('I', 'Q')

    @staticmethod
    def _repr_apply(c):
        return c.real, c.imag


class ReadoutIQToComplex(_ReadoutReprComplexMixin, _ComposableModelBase):
    """Utility model to convert from (I, Q) to complex S21."""
    n_inputs = 2

    def _set_inputs(self):
        self.inputs = ('I', 'Q')

    @staticmethod
    def evaluate(I, Q):
        return super(
                ReadoutIQToComplex, ReadoutIQToComplex)._repr_apply(I, Q)


class ReadoutComplexToIQ(_ReadoutRepr2Mixin, _ComposableModelBase):
    """Utility model to convert from complex S21 to (I, Q)."""
    n_inputs = 1

    def _set_inputs(self):
        self.inputs = ('S', )

    @staticmethod
    def evaluate(S):
        return super(
                _ReadoutRepr2Mixin, ReadoutComplexToIQ)._repr_apply(S)


class _ResonanceCircleSweepMixin(object):
    """Mixin class that sets up the frequency sweep model."""

    n_inputs = 1
    # fr = Parameter(default=1e9, unit=u.Hz, min=0.)
    # Qr = Parameter(default=2e4)

    def _set_inputs(self):
        self.inputs = ('f', )

    @staticmethod
    def evaluate(f, fr, Qr):
        r = ResonanceCircleQrInv(Qr)
        x = InstrumentalDetune.evaluate(f, fr)
        return ResonanceCircle.evaluate(r, x)


class ResonanceCircleSweep(
        _ResonanceCircleSweepMixin, _ReadoutRepr2Mixin, _ComposableModelBase):
    """Model that describes the frequency sweep of The resonance circle."""
    # already separate in I, Q.
    pass


class ResonanceCircleSweepComplex(
        _ResonanceCircleSweepMixin,
        _ReadoutReprComplexMixin, _ComposableModelBase):
    """The same as `ResonanceCircleSweep`, but the result is the complex S21.
    """

    fr = Parameter(default=1e9, unit=u.Hz, min=0.)
    Qr = Parameter(default=2e4)

    # make it return complex
    @staticmethod
    def evaluate(f, fr, Qr):
        return super()._repr_apply(
                super().evaluate(f, fr, Qr)
                )


class _ResonanceCircleProbeMixin(object):
    """Mixin class that sets up the probing model."""

    n_inputs = 2

    fp = Parameter(default=1e9, unit=u.Hz, min=0.)

    def _set_inputs(self):
        self.inputs = ('fr', 'Qr')

    @staticmethod
    def evaluate(fr, Qr, fp):
        r = ResonanceCircleQrInv(Qr)
        x = InstrumentalDetune.evaluate(fp, fr)
        return ResonanceCircle.evaluate(r, x)


class ResonanceCircleProbe(
        _ResonanceCircleProbeMixin, _ReadoutRepr2Mixin, _ComposableModelBase):
    """Model that describes the probing of The resonance circle."""
    # already separate in I, Q.
    pass


class ResonanceCircleProbeComplex(
        _ResonanceCircleProbeMixin, _ReadoutReprComplexMixin,
        _ComposableModelBase):
    """The same as `ResonanceCircleProbe`, but the result is the complex S21.
    """

    @staticmethod
    def evaluate(fr, Qr, fp):
        return super()._repr_apply(
                super().evaluate(fr, Qr, fp)
                )


class _ReadoutTransformComplexMixin(object):
    """Mixin class that setup the transformation of (I + jQ) due to readout."""

    n_inputs = 1

    def _set_inputs(self):
        self.inputs = ('S', )

    @classmethod
    def _get_transform_params(cls):
        exclude_args = ('S', 'f')
        return [a for a in _get_func_args(cls._transform)
                if a not in exclude_args]

    @classmethod
    def _get_inverse_transform_params(cls):
        exclude_args = ('S', 'f')
        return [a for a in _get_func_args(cls._inverse_transform)
                if a not in exclude_args]


class _ReadoutGeometryParamsMixin(object):
    """Mixin class that defines the KIDs parameters related to the
    readout circuit geometry."""

    tau = Parameter(default=0., unit=u.s)
    Qc = Parameter(default=4e4)  # optimal coupling
    Qi = Parameter(tied=lambda m: m.Qr * m.Qc / (m.Qc - m.Qr))
    phi_c = Parameter(default=0.)


class _ReadoutGainParamsMixin(object):
    """Mixin class that defines some general gain parameters.

    Note that these parameters could mean different in different concrete
    classes.
    """

    g0 = Parameter(default=1.)
    g1 = Parameter(default=0.)
    g = Parameter(tied=lambda m: np.hypot(m.g0, m.g1))
    phi_g = Parameter(tied=lambda m: np.arctan2(m.g1, m.g0))


class _ReadoutLinTrendParamsMixin(object):
    """Mixin class that defines parameters that describes a
    linear baseline trend."""

    f0 = Parameter(default=1e9, unit=u.Hz, min=0.)
    k0 = Parameter(default=0.)
    k1 = Parameter(default=0.)
    m0 = Parameter(default=0.)
    m1 = Parameter(default=0.)


class _ReadoutGainWithLinTrendMixin(
        _ReadoutGainParamsMixin, _ReadoutLinTrendParamsMixin,
        _ReadoutTransformComplexMixin,
        ):
    """Mixin class that defines readout transform of S21 using an effective
    complex gain and a linear baseline trend."""

    @staticmethod
    def _transform(S, f, g0, g1, f0, k0, k1, m0, m1):
        gg = g0 + 1.j * g1
        kk = k0 + 1.j * k1
        mm = m0 + 1.j * m1
        return gg * S + kk * (f - f0) + mm

    @staticmethod
    def _inverse_transform(S, f, g0, g1, f0, k0, k1, m0, m1):
        gg = g0 + 1.j * g1
        kk = k0 + 1.j * k1
        mm = m0 + 1.j * m1
        return (S - mm - kk * (f - f0)) / gg


class _ReadoutGainWithGeometryMixin(
        _ReadoutGainParamsMixin, _ReadoutGeometryParamsMixin,
        _ReadoutTransformComplexMixin,
        ):
    """Mixin class that defines readout transform of S21 using the
    geometry parameters."""

    @staticmethod
    def _transform(S, f, g0, g1, tau, Qc, phi_c):
        gg = g0 + 1.j * g1
        cc = 1. / (Qc * np.exp(1.j * phi_c))
        aa = gg * np.exp(-2.j * np.pi * f * tau)
        return aa * (1 - S * cc)

    @staticmethod
    def _inverse_transform(S, f, g0, g1, tau, Qc, phi_c):
        gg = g0 + 1.j * g1
        cc = 1. / (Qc * np.exp(1.j * phi_c))
        aa = gg * np.exp(-2.j * np.pi * f * tau)
        return (1. - S / aa) / cc


class _KidsReadoutMixin(object):
    """Mixin class that defines some utility functions for readout models."""

    def derotate(self, S, f):
        args = (getattr(self, a) for a in self._get_inverse_transform_params())
        return super()._inverse_transform(S, f, *args)

    @staticmethod
    def _apply_transform(args, locals_, cls):
        args += tuple(locals_[a] for a in cls._get_transform_params())
        return cls._transform(*args)


class KidsSweepGainWithLinTrend(
        _KidsReadoutMixin, _ReadoutGainWithLinTrendMixin,
        ResonanceCircleSweepComplex):
    """Model that describes the S21 for frequency sweep with
    an effective gain and a linear baseline trend.
    """

    @staticmethod
    def evaluate(f, fr, Qr, g0, g1, g, phi_g, f0, k0, k1, m0, m1):
        S = super().evaluate(f, fr, Qr)
        return super()._apply_transform(S, f, locals(), super())


class KidsSweepGainWithGeometry(
        _KidsReadoutMixin, _ReadoutGainWithGeometryMixin,
        ResonanceCircleSweepComplex):
    """Model that describes the S21 for frequency sweep with the geometry
    parameters."""

    @staticmethod
    def evaluate(f, fr, Qr, g0, g1, g, phi_g, tau, Qc, Qi, phi_c):
        S = super().evaluate(f, fr, Qr)
        return super()._apply_transform(S, f, locals(), super())


class KidsProbeGainWithLinTrend(
        _KidsReadoutMixin, _ReadoutGainWithLinTrendMixin,
        ResonanceCircleProbeComplex):
    """Model that describes the S21 probed at fixed frequencies with
    an effective gain and a linear baseline trend.
    """

    @staticmethod
    def evaluate(fr, Qr, fp, g0, g1, g, phi_g, f0, k0, k1, m0, m1):
        S = super().evaluate(fr, Qr, fp)
        return super()._apply_transform(S, fp, locals(), super())


class KidsProbeGainWithGeometry(
        _KidsReadoutMixin, _ReadoutGainWithGeometryMixin,
        ResonanceCircleProbeComplex):
    """Model that describes the S21 probed at fixed frequencies with
    the geometry parameters.
    """

    @staticmethod
    def evaluate(fr, Qr, fp, g0, g1, g, phi_g, tau, Qc, Qi, phi_c):
        S = super().evaluate(fr, Qr, fp)
        return super()._apply_transform(S, fp, locals(), super())
