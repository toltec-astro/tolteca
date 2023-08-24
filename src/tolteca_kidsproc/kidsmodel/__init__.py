from astropy.modeling import Parameter, Model
import numpy as np
from astropy import units as u
import inspect


class _Model(Model):
    """Subclass of astropy.modeling.Model that support complex type."""

    # code snippet from `astropy.modeling.Model`
    def prepare_inputs(
        self, *inputs, model_set_axis=None, equivalencies=None, **kwargs
    ):
        """
        This method is used in `~astropy.modeling.Model.__call__` to ensure
        that all the inputs to the model can be broadcast into compatible
        shapes (if one or both of them are input as arrays), particularly if
        there are more than one parameter sets. This also makes sure that (if
        applicable) the units of the input will be compatible with the evaluate
        method.
        """  # noqa: D404, D401, D205
        # When we instantiate the model class, we make sure that __call__ can
        # take the following two keyword arguments: model_set_axis and
        # equivalencies.
        if model_set_axis is None:
            # By default the model_set_axis for the input is assumed to be the
            # same as that for the parameters the model was defined with
            # TODO: Ensure that negative model_set_axis arguments are respected
            model_set_axis = self.model_set_axis

        params = [getattr(self, name) for name in self.param_names]
        inputs = [np.asanyarray(_input, dtype=None) for _input in inputs]

        self._validate_input_shapes(inputs, self.inputs, model_set_axis)

        inputs_map = kwargs.get("inputs_map", None)

        inputs = self._validate_input_units(inputs, equivalencies, inputs_map)

        # The input formatting required for single models versus a multiple
        # model set are different enough that they've been split into separate
        # subroutines
        if self._n_models == 1:
            return self._prepare_inputs_single_model(params, inputs, **kwargs)
        return self._prepare_inputs_model_set(params, inputs, model_set_axis, **kwargs)


def _get_func_args(func):
    return inspect.getfullargspec(func).args


def _set_mutual_inversion(cls1, cls2):
    cls1.inverse = property(
        lambda self: cls2(n_models=len(self), model_set_axis=self.model_set_axis)
    )
    cls2.inverse = property(
        lambda self: cls1(n_models=len(self), model_set_axis=self.model_set_axis)
    )


class _ResonanceCircleTransformMixin:
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
        self.inputs = ("X",)
        self.outputs = ("S",)


class ResonanceCircleComplexInv(_ResonanceCircleTransformMixin, _Model):
    """Inversion of `ResonanceCircleComplex`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("S",)
        self.outputs = ("X",)


class ResonanceCircleQr(_ResonanceCircleTransformMixin, _Model):
    """Model that describes the relation of `r` and `Qr`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("r",)
        self.outputs = ("Qr",)


class ResonanceCircleQrInv(_ResonanceCircleTransformMixin, _Model):
    """Inversion of `ResonanceCircleQr`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("Qr",)
        self.outputs = ("r",)


class _ResonanceCircleTransform2Mixin(object):
    """A resonance circle transform with separate real and imaginary part.

    The transform reads

    .. code-block:: text

        I + iQ = 0.5 / (r + ix)

    """

    n_inputs = 2
    n_outputs = 2
    _separable = False

    @staticmethod
    def evaluate(v1, v2):
        f = 0.5 / (v1**2 + v2**2)
        return v1 * f, -v2 * f


class ResonanceCircle(_ResonanceCircleTransform2Mixin, _Model):
    """A resonance circle model with separate real and imaginary part."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("r", "x")
        self.outputs = ("I", "Q")


class ResonanceCircleInv(_ResonanceCircleTransform2Mixin, _Model):
    """Inversion of `ResonanceCircle`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("I", "Q")
        self.outputs = ("r", "x")


_set_mutual_inversion(ResonanceCircleComplex, ResonanceCircleComplexInv)
_set_mutual_inversion(ResonanceCircleQr, ResonanceCircleQrInv)
_set_mutual_inversion(ResonanceCircle, ResonanceCircleInv)


class OpticalDetune(_Model):
    """A model to describe optical detuning of KIDs.

    The model reads

    .. code-block:: text

        x = (p - background) * responsivity

    """

    n_inputs = 1
    n_outputs = 1
    _separable = True

    background = Parameter(default=5.0 * u.pW, min=0.0)
    responsivity = Parameter(default=1e-17, unit=1.0 / u.W, min=0.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("p",)
        self.outputs = ("x",)

    @staticmethod
    def evaluate(p, background, responsivity):
        """Evaluate the optical detuning."""
        return (p - background) * responsivity


class InstrumentalDetune(_Model):
    """A model to describe the detuning term from the placement of probing tone.

    The model reads

    .. code-block:: text

        x = (fp / fr - 1.)

    """

    n_inputs = 2
    n_outputs = 1
    _separable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = ("fp", "fr")
        self.outputs = ("x",)

    @staticmethod
    def evaluate(fp, fr):
        """Evaluate the instrumental detuning."""
        return fp / fr - 1.0


class _ComposableModelBase(_Model):
    """Base class that setup itself with mixin classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_inputs()
        self._set_outputs()


class _ReadoutReprComplexMixin:
    """Mixin class that sets the output to use complex S21."""

    n_outputs = 1
    _separable = True

    def _set_outputs(self):
        self.outputs = ("S",)

    @staticmethod
    def IQ_to_S21(a, b):
        if a.shape == (1,):
            a = a[0]
        if b.shape == (1,):
            b = b[0]
        return a + 1.0j * b


class _ReadoutRepr2Mixin:
    """Mixin class that sets the output to use separate I and Q."""

    n_outputs = 2
    _separable = False

    def _set_outputs(self):
        self.outputs = ("I", "Q")

    @staticmethod
    def S21_to_IQ(c):
        return c.real, c.imag


class ReadoutIQToComplex(_ReadoutReprComplexMixin, _ComposableModelBase):
    """Utility model to convert from (I, Q) to complex S21."""

    n_inputs = 2

    def _set_inputs(self):
        self.inputs = ("I", "Q")

    @staticmethod
    def evaluate(I, Q):  # noqa: E741
        """Return the complex S21 from I and Q."""
        return super().IQ_to_S21(I, Q)

    def __call__(self, x, y, **kwargs):
        """Return the complex S21 from I and Q."""
        # make sure they are the same shape
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        if x.shape == ():
            x = np.full_like(y, x.item())
        elif y.shape == ():
            y = np.full_like(x, y.item())
        return super().__call__(x, y, **kwargs)


class ReadoutComplexToIQ(_ReadoutRepr2Mixin, _ComposableModelBase):
    """Utility model to convert from complex S21 to (I, Q)."""

    n_inputs = 1

    def _set_inputs(self):
        self.inputs = ("S",)

    @staticmethod
    def evaluate(S):
        """Return the I and Q from complex S21."""
        return super().S21_to_IQ(S)


class _ResonanceCircleSweepMixin:
    """Mixin class that sets up the frequency sweep model."""

    n_inputs = 1

    def _set_inputs(self):
        self.inputs = ("f",)

    @staticmethod
    def evaluate(f, fr, Qr):
        r = ResonanceCircleQrInv.evaluate(Qr)
        x = InstrumentalDetune.evaluate(f, fr)
        return ResonanceCircle.evaluate(r, x)


class ResonanceCircleSweep(
    _ResonanceCircleSweepMixin,
    _ReadoutRepr2Mixin,
    _ComposableModelBase,
):
    """A model to describe the frequency sweep of The resonance circle."""

    fr = Parameter(default=1e9, unit=u.Hz, min=0.0)
    Qr = Parameter(default=2e4)


class ResonanceCircleSweepComplex(
    _ResonanceCircleSweepMixin, _ReadoutReprComplexMixin, _ComposableModelBase
):
    """The same as `ResonanceCircleSweep`, but the result is the complex S21."""

    fr = Parameter(default=1e9, unit=u.Hz, min=0.0)
    Qr = Parameter(default=2e4)

    @staticmethod
    def evaluate(f, fr, Qr):
        """Return the frequency sweep in complex S21."""
        return _ReadoutReprComplexMixin.IQ_to_S21(
            *_ResonanceCircleSweepMixin.evaluate(f, fr, Qr),
        )


class _ResonanceCircleProbeMixin:
    """A mixin class to set up the probing model."""

    n_inputs = 2

    def _set_inputs(self):
        self.inputs = ("fr", "Qr")

    @staticmethod
    def evaluate(fr, Qr, fp):
        r = ResonanceCircleQrInv.evaluate(Qr)
        x = InstrumentalDetune.evaluate(fp, fr)
        return ResonanceCircle.evaluate(r, x)


class ResonanceCircleProbe(
    _ResonanceCircleProbeMixin, _ReadoutRepr2Mixin, _ComposableModelBase
):
    """A model to describe the probing of The resonance circle."""

    fp = Parameter(default=1e9, unit=u.Hz, min=0.0)


class ResonanceCircleProbeComplex(
    _ResonanceCircleProbeMixin, _ReadoutReprComplexMixin, _ComposableModelBase
):
    """The same as `ResonanceCircleProbe`, but the result is the complex S21."""

    fp = Parameter(default=1e9, unit=u.Hz, min=0.0)

    @staticmethod
    def evaluate(fr, Qr, fp):
        """Return the probed S21."""
        return _ReadoutReprComplexMixin.IQ_to_S21(
            *_ResonanceCircleProbeMixin.evaluate(fr, Qr, fp),
        )


# --------------
# Readout models
# --------------


class _ReadoutTransformComplexMixin:
    """Mixin class that setup the transformation of resonance circle S21 to readout."""

    n_inputs = 1

    def _set_inputs(self):
        self.inputs = ("S",)

    @classmethod
    def _get_transform_params(cls):
        exclude_args = ("S", "f")
        return [a for a in _get_func_args(cls.transform_S21) if a not in exclude_args]

    @classmethod
    def _get_inverse_transform_params(cls):
        exclude_args = ("S", "f")
        return [
            a
            for a in _get_func_args(cls.inverse_transform_S21)
            if a not in exclude_args
        ]


class _ReadoutGainWithLinTrendMixin(
    _ReadoutTransformComplexMixin,
):
    """A readout model with effective complex gain and a linear baseline trend."""

    @staticmethod
    def transform_S21(S, f, g0, g1, f0, k0, k1, m0, m1):  # noqa: PLR0913
        """Return the readout S21."""
        gg = g0 + 1.0j * g1
        kk = k0 + 1.0j * k1
        mm = m0 + 1.0j * m1
        return gg * S + kk * (f - f0) + mm

    @staticmethod
    def inverse_transform_S21(S, f, g0, g1, f0, k0, k1, m0, m1):  # noqa: PLR0913
        """Return the resonance circle S21."""
        gg = g0 + 1.0j * g1
        kk = k0 + 1.0j * k1
        mm = m0 + 1.0j * m1
        return (S - mm - kk * (f - f0)) / gg


class _ReadoutGainWithGeometryMixin(
    _ReadoutTransformComplexMixin,
):
    """A readout model using the geometry parameters."""

    @staticmethod
    def transform_S21(S, f, g0, g1, tau, Qc, phi_c):  # noqa: PLR0913
        """Return the readout S21."""
        gg = g0 + 1.0j * g1
        cc = 1.0 / (Qc * np.exp(1.0j * phi_c))
        aa = gg * np.exp(-2.0j * np.pi * f * tau)
        return aa * (1 - S * cc)

    @staticmethod
    def inverse_transform_S21(S, f, g0, g1, tau, Qc, phi_c):  # noqa: PLR0913
        """Return the resonance circle S21."""
        gg = g0 + 1.0j * g1
        cc = 1.0 / (Qc * np.exp(1.0j * phi_c))
        aa = gg * np.exp(-2.0j * np.pi * f * tau)
        return (1.0 - S / aa) / cc


class _KidsReadoutMixin:
    """Mixin class that defines some utility functions for readout models."""

    def derotate(self, S, f):
        args = (getattr(self, a) for a in self._get_inverse_transform_params())
        return super().inverse_transform_S21(S.T, f.T, *args).T

    def rotate(self, S_derot, f):
        args = (getattr(self, a) for a in self._get_transform_params())
        return super().transform_S21(S_derot.T, f.T, *args).T

    @staticmethod
    def apply_transform(args, locals_, cls):
        args += tuple(locals_[a] for a in cls._get_transform_params())
        return cls.transform_S21(*args)


def _tied_g_amp(m):
    return np.hypot(m.g0, m.g1)


def _tied_g_phase(m):
    return np.arctan2(m.g1, m.g0)


def _tied_Qi(m):
    return m.Qr * m.Qc / (m.Qc - m.Qr)


class ReadoutGainWithLinTrend(
    _ReadoutReprComplexMixin,
    _KidsReadoutMixin,
    _ReadoutGainWithLinTrendMixin,
    _ComposableModelBase,
):
    """A readout model with an effective gain and a linear baseline trend."""

    g0 = Parameter(default=1.0)
    g1 = Parameter(default=0.0)
    g = Parameter(tied=_tied_g_amp)
    phi_g = Parameter(tied=_tied_g_phase)
    f0 = Parameter(default=1e9, unit=u.Hz, min=0.0)
    k0 = Parameter(default=0.0, unit=u.s)
    k1 = Parameter(default=0.0, unit=u.s)
    m0 = Parameter(default=0.0)
    m1 = Parameter(default=0.0)

    n_inputs = 2

    def _set_inputs(self):
        self.inputs = ("S", "f")

    @classmethod
    def evaluate(  # noqa: PLR0913
        cls, S, f, g0, g1, g, phi_g, f0, k0, k1, m0, m1  # noqa: ARG003, COM812
    ):
        """Return readout S21 from resoanance circle S21."""
        return _KidsReadoutMixin.apply_transform(
            (S, f),
            locals(),
            cls,
        )


class KidsSweepGainWithLinTrend(
    _ResonanceCircleSweepMixin,
    _ReadoutReprComplexMixin,
    _KidsReadoutMixin,
    _ReadoutGainWithLinTrendMixin,
    _ComposableModelBase,
):
    """A frequency sweep model with an effective gain and a linear baseline trend."""

    fr = Parameter(default=1e9, unit=u.Hz, min=0.0)
    Qr = Parameter(default=2e4)
    g0 = Parameter(default=1.0)
    g1 = Parameter(default=0.0)
    g = Parameter(tied=_tied_g_amp)
    phi_g = Parameter(tied=_tied_g_phase)
    f0 = Parameter(default=1e9, unit=u.Hz, min=0.0)
    k0 = Parameter(default=0.0, unit=u.s)
    k1 = Parameter(default=0.0, unit=u.s)
    m0 = Parameter(default=0.0)
    m1 = Parameter(default=0.0)

    @classmethod
    def evaluate(  # noqa: PLR0913
        cls, f, fr, Qr, g0, g1, g, phi_g, f0, k0, k1, m0, m1  # noqa: ARG003, COM812
    ):
        """Return readout S21 for resonator."""
        S = ResonanceCircleSweepComplex.evaluate(f, fr, Qr)
        return _KidsReadoutMixin.apply_transform(
            (S, f),
            locals(),
            cls,
        )


class KidsSweepGainWithGeometry(
    _ResonanceCircleSweepMixin,
    _ReadoutReprComplexMixin,
    _KidsReadoutMixin,
    _ReadoutGainWithGeometryMixin,
    _ComposableModelBase,
):
    """A frequency sweep model with geometry parameters."""

    fr = Parameter(default=1e9, unit=u.Hz, min=0.0)
    Qr = Parameter(default=2e4)
    g0 = Parameter(default=1.0)
    g1 = Parameter(default=0.0)
    g = Parameter(tied=_tied_g_amp)
    phi_g = Parameter(tied=_tied_g_phase)
    tau = Parameter(default=0.0, unit=u.s)
    Qc = Parameter(default=4e4)  # optimal coupling
    Qi = Parameter(tied=_tied_Qi)
    phi_c = Parameter(default=0.0)

    @classmethod
    def evaluate(  # noqa: PLR0913
        cls, f, fr, Qr, g0, g1, g, phi_g, tau, Qc, Qi, phi_c  # noqa: ARG003, COM812
    ):
        """Return readout S21 for resonator."""
        S = ResonanceCircleSweepComplex.evaluate(f, fr, Qr)
        return _KidsReadoutMixin.apply_transform(
            (S, f),
            locals(),
            cls,
        )


class KidsProbeGainWithLinTrend(
    _ResonanceCircleProbeMixin,
    _ReadoutReprComplexMixin,
    _KidsReadoutMixin,
    _ReadoutGainWithLinTrendMixin,
    _ComposableModelBase,
):
    """A probing model with an effective gain and a linear baseline trend."""

    fp = Parameter(default=1e9, unit=u.Hz, min=0.0)
    g0 = Parameter(default=1.0)
    g1 = Parameter(default=0.0)
    g = Parameter(tied=_tied_g_amp)
    phi_g = Parameter(tied=_tied_g_phase)
    f0 = Parameter(default=1e9, unit=u.Hz, min=0.0)
    k0 = Parameter(default=0.0, unit=u.s)
    k1 = Parameter(default=0.0, unit=u.s)
    m0 = Parameter(default=0.0)
    m1 = Parameter(default=0.0)

    @classmethod
    def evaluate(  # noqa: PLR0913
        cls, fr, Qr, fp, g0, g1, g, phi_g, f0, k0, k1, m0, m1  # noqa: ARG003, COM812
    ):
        """Return probed S21."""
        S = ResonanceCircleProbeComplex.evaluate(fr, Qr, fp)
        return _KidsReadoutMixin.apply_transform((S, fp), locals(), cls)


class KidsProbeGainWithGeometry(
    _ResonanceCircleProbeMixin,
    _ReadoutReprComplexMixin,
    _KidsReadoutMixin,
    _ReadoutGainWithGeometryMixin,
    _ComposableModelBase,
):
    """A probing model with the geometry parameters."""

    fp = Parameter(default=1e9, unit=u.Hz, min=0.0)
    g0 = Parameter(default=1.0)
    g1 = Parameter(default=0.0)
    g = Parameter(tied=_tied_g_amp)
    phi_g = Parameter(tied=_tied_g_phase)
    tau = Parameter(default=0.0, unit=u.s)
    Qc = Parameter(default=4e4)  # optimal coupling
    Qi = Parameter(tied=_tied_Qi)
    phi_c = Parameter(default=0.0)

    @staticmethod
    def evaluate(  # noqa: PLR0913
        cls, fr, Qr, fp, g0, g1, g, phi_g, tau, Qc, Qi, phi_c  # noqa: ARG004, COM812
    ):
        """Return probed S21."""
        S = ResonanceCircleProbeComplex.evaluate(fr, Qr, fp)
        return _KidsReadoutMixin.apply_transform((S, fp), locals(), cls)
