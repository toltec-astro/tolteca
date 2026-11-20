"""Complex KID responses with physical and historical effective parameters."""

from __future__ import annotations

import numpy as np

from ._gainlintrend import KidsSweepGainWithLinTrend
from ._nonlinear import (
    NONLINEARITY_UNIQUE_MAX,
    nonlinear_detuning,
    validate_nonlinearity,
)
from ._physical_sweep import KidsSweepWithAffineGain, KidsSweepWithDelayGain

__all__ = [
    "NONLINEARITY_UNIQUE_MAX",
    "KidsSweepGainWithLinTrend",
    "KidsSweepModel",
    "KidsSweepWithAffineGain",
    "KidsSweepWithDelayGain",
    "resonance_feature_frequency",
    "resonance_kernel",
]

MIN_WINDOW_SAMPLES = 2


def resonance_feature_frequency(fr, qr, a_nl=0.0, *, nonlinear_branch="unique"):
    """Locate the largest canonical resonance contribution in a declared sweep.

    Parameters
    ----------
    fr, qr, a_nl : array_like
        Broadcastable zero-drive frequencies (Hz), loaded Q and stationary
        nonlinearity. Zero nonlinearity returns fr.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Declared stationary sweep history, as in ``KidsSweepModel``.

    Returns
    -------
    frequency : numpy.ndarray
        Frequency in Hz of maximum ``abs(c/(r*(1+2j*y)))``. This is a
        component response landmark, not the minimum of measured abs(S21),
        which also depends on gain, coupling asymmetry and neighbours.
    kind : numpy.ndarray of str
        ``resonance_center`` if y=0 is reached; otherwise ``upward_jump``.
        For an increasing sweep with a_nl>1, the jump skips y=0 and the
        largest contribution occurs immediately after the jump.
    """
    from scipy.optimize import brentq  # noqa: PLC0415 -- Lazy fold calculation.

    fr, qr, a = np.broadcast_arrays(fr, qr, a_nl)
    validate_nonlinearity(a, nonlinear_branch)
    detuning = -a.astype(float)
    jump = (a > 1) & (nonlinear_branch == "increasing")
    for value in np.unique(a[jump]):

        def derivative(y, strength=value):
            return 1 + 8 * strength * y / (1 + 4 * y**2) ** 2

        y = brentq(derivative, -(value + 1), -1 / (2 * np.sqrt(3)))
        detuning = np.where(jump & (a == value), y - value / (1 + 4 * y * y), detuning)
    return fr * (1 + detuning / qr), np.where(jump, "upward_jump", "resonance_center")


def resonance_kernel(frequency, fr, r, coupling):
    """Evaluate a physical multi-resonance transmission kernel.

    Parameters
    ----------
    frequency : array_like
        Probe frequencies in Hz, with any shape.
    fr, r, coupling : array_like
        One frequency in Hz, positive loss ``1/(2*Qr)``, and complex coupling
        ``(1+1j*A)/(2*Qc)`` per resonance. Detector states may vary independently
        of coupling when evaluating a timestream.

    Returns
    -------
    numpy.ndarray
        Complex transmission ``1-sum(c/(r+1j*(f/fr-1)))``.
    """
    f = np.asarray(frequency, float)
    return 1 - np.sum(
        np.asarray(coupling) / (np.asarray(r) + 1j * (f[..., None] / fr - 1)),
        axis=-1,
    )


class KidsSweepModel:
    """Physical resonances shared across channels, with one gain per channel.

    Parameters
    ----------
    fr_reference, qr_reference : array_like
        Fixed positive frequency (Hz) and loaded-Q scales, one per resonance.
        These set optimizer coordinates and do not vary during fitting.
    frequency : sequence of array_like
        One selected frequency vector in Hz per channel. Used to fix gain
        origins and spans. Evaluation accepts other probe frequencies.
    associations : array_like of bool, optional
        Shape ``(channel, resonance)``; true includes a component in a channel.
        The default includes every component in every channel.
    gain_family : {"affine", "delay"}, default "affine"
        Complex affine gain or exponential amplitude slope plus delay.
    gain_origin : array_like, optional
        Fixed gain expansion frequencies in Hz, one per channel. Defaults to
        each selected window's midpoint; these are not actual-probe bindings.
    signal_scale : float, default 1
        Positive fixed signal normalization used only for optimizer scaling.
    nonlinear : bool, default False
        Fit one a_nl per resonance using the stationary softening response.
        Zero recovers the linear model. With this option,
        fr is the zero-drive resonance; the observed minimum can be displaced.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Unique response below bifurcation, or the stable response reached by a
        monotonic sweep from below/above resonance. Direction is declared by
        the caller, independent of evaluation point order. Requires nonlinear
        mode for a directional branch. Arbitrary history and resets are excluded.

    Notes
    -----
    State coordinates are ``xi=Qref*(fref/fr-1)`` and ``rho=2*Qref*r``.
    Coupling coordinates are ``chi=2*Qref*c``. Gain coordinates use a fixed
    measured-frequency span. The model applies gain once to the whole kernel.
    Programmed input phase is not factored out. Only the two adopted gain
    families are available. Derivatives and covariance are local to the selected
    stable branch; they do not include uncertainty in the jump location.
    """

    def __init__(  # noqa: PLR0913 -- Fixed state and gain coordinates plus association mask.
        self,
        fr_reference,
        qr_reference,
        frequency,
        *,
        associations=None,
        gain_family="affine",
        gain_origin=None,
        signal_scale=1.0,
        nonlinear=False,
        nonlinear_branch="unique",
    ):
        self.nonlinear = bool(nonlinear)
        validate_nonlinearity(0, nonlinear_branch)
        if not self.nonlinear and nonlinear_branch != "unique":
            msg = "A sweep branch requires nonlinear=True"
            raise ValueError(msg)
        self.nonlinear_branch = nonlinear_branch
        self.fr_reference = np.asarray(fr_reference, float)
        self.qr_reference = np.asarray(qr_reference, float)
        if (
            self.fr_reference.ndim != 1
            or not len(self.fr_reference)
            or self.qr_reference.shape != self.fr_reference.shape
            or not np.isfinite([self.fr_reference, self.qr_reference]).all()
            or np.any(self.fr_reference <= 0)
            or np.any(self.qr_reference <= 0)
        ):
            msg = "Reference frequencies and quality factors must be positive vectors"
            raise ValueError(msg)
        fs = [np.asarray(f, float) for f in frequency]
        if not fs or any(
            f.ndim != 1
            or len(f) < MIN_WINDOW_SAMPLES
            or not np.isfinite(f).all()
            or np.ptp(f) <= 0
            or np.any(f <= 0)
            for f in fs
        ):
            msg = "Each channel needs a finite positive frequency window"
            raise ValueError(msg)
        self.n_channels, self.n_resonances = len(fs), len(self.fr_reference)
        shape = (self.n_channels, self.n_resonances)
        self.associations = (
            np.ones(shape, bool) if associations is None else np.asarray(associations)
        )
        if (
            self.associations.shape != shape
            or self.associations.dtype != bool
            or not self.associations.any(axis=0).all()
            or not self.associations.any(axis=1).all()
        ):
            msg = "Every channel and resonance needs a boolean association"
            raise ValueError(msg)
        if gain_family not in ("affine", "delay"):
            msg = "gain_family must be affine or delay"
            raise ValueError(msg)
        self.gain_family = gain_family
        self.gain_origin = (
            np.array([(f.min() + f.max()) / 2 for f in fs])
            if gain_origin is None
            else np.asarray(gain_origin, float)
        )
        if (
            self.gain_origin.shape != (len(fs),)
            or not np.isfinite(self.gain_origin).all()
        ):
            msg = "gain_origin must contain one finite frequency per channel"
            raise ValueError(msg)
        self.gain_span = np.array([np.ptp(f) for f in fs])
        self.signal_scale = float(signal_scale)
        if not np.isfinite(self.signal_scale) or self.signal_scale <= 0:
            msg = "signal_scale must be finite and positive"
            raise ValueError(msg)
        self.names = tuple(
            [
                f"r{j}_{name}"
                for j in range(self.n_resonances)
                for name in (("xi", "rho", "a_nl") if self.nonlinear else ("xi", "rho"))
            ]
            + [
                f"c{k}_r{j}_{name}"
                for k, j in zip(*np.nonzero(self.associations), strict=True)
                for name in ("re", "im")
            ]
            + [
                f"c{k}_{name}"
                for k in range(self.n_channels)
                for name in ("h0_re", "h0_im", "h1_re", "h1_im")
            ]
        )
        self._indices = {n: i for i, n in enumerate(self.names)}

    def make_params(
        self,
        *,
        fr_bounds=None,
        qr_bounds=(500.0, 100000.0),
        a_nl_max=NONLINEARITY_UNIQUE_MAX,
    ):
        """Create scaled lmfit parameters with physical state bounds.

        Parameters
        ----------
        fr_bounds : pair of array_like, optional
            Frequency bounds in Hz. Default: one initial linewidth on each side.
        qr_bounds : pair of float, default (500, 100000)
            Positive increasing loaded-Q bounds, translated to loss coordinates.
        a_nl_max : float, default 0.75
            Positive finite upper bound. Values above 0.75 require an explicit
            directional branch. Supplied frequency bounds apply to zero-drive fr.

        Returns
        -------
        lmfit.Parameters
            One independent state per resonance and gain/coupling per association.
            Bounds and fixed values can be changed before fitting.
        """
        # Keep configuration imports renderer-free.
        from lmfit import Parameters  # noqa: PLC0415

        validate_nonlinearity(a_nl_max, self.nonlinear_branch)
        if a_nl_max <= 0:
            msg = "a_nl_max must be positive"
            raise ValueError(msg)
        fr, qr = self.fr_reference, self.qr_reference
        lo, hi = (
            (fr - fr / qr, fr + fr / qr)
            if fr_bounds is None
            else (np.broadcast_to(a, fr.shape) for a in fr_bounds)
        )
        qlo, qhi = qr_bounds
        if (
            not np.isfinite([lo, hi]).all()
            or np.any(lo <= 0)
            or np.any(lo >= fr)
            or np.any(hi <= fr)
            or not 0 < qlo < qhi
            or not np.isfinite(qhi)
            or np.any(qr < qlo)
            or np.any(qr > qhi)
        ):
            msg = "Bounds must contain the reference frequencies and quality factors"
            raise ValueError(msg)
        p = Parameters()
        for j in range(self.n_resonances):
            p.add(
                f"r{j}_xi",
                value=0,
                min=qr[j] * (fr[j] / hi[j] - 1),
                max=qr[j] * (fr[j] / lo[j] - 1),
            )
            p.add(f"r{j}_rho", value=1, min=qr[j] / qhi, max=qr[j] / qlo)
            if self.nonlinear:
                p.add(f"r{j}_a_nl", value=0.05, min=0, max=a_nl_max)
        for k, j in zip(*np.nonzero(self.associations), strict=True):
            p.add(f"c{k}_r{j}_re", value=0.25, min=1e-8, max=20)
            p.add(f"c{k}_r{j}_im", value=0, min=-20, max=20)
        for k in range(self.n_channels):
            for name, value in (("h0_re", 1), ("h0_im", 0), ("h1_re", 0), ("h1_im", 0)):
                p.add(f"c{k}_{name}", value=value)
        return p

    def physical(self, params):
        """Return physical state, coupling and gain arrays from lmfit parameters.

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters belonging to this model.

        Returns
        -------
        dict
            ``fr,r,a_nl`` (resonance); ``c`` (channel,resonance); ``H0,H1,a,tau``
            (channel). Missing associations and inactive gain fields are NaN.
            Frequencies are Hz, gain slopes per Hz, and delay seconds.
        """
        v = params.valuesdict()
        fr = self.fr_reference / (
            1
            + np.array([v[f"r{j}_xi"] for j in range(self.n_resonances)])
            / self.qr_reference
        )
        r = np.array([v[f"r{j}_rho"] for j in range(self.n_resonances)]) / (
            2 * self.qr_reference
        )
        c = np.full(self.associations.shape, np.nan + 1j * np.nan)
        for k, j in zip(*np.nonzero(self.associations), strict=True):
            c[k, j] = (v[f"c{k}_r{j}_re"] + 1j * v[f"c{k}_r{j}_im"]) / (
                2 * self.qr_reference[j]
            )
        h0 = self.signal_scale * np.array(
            [v[f"c{k}_h0_re"] + 1j * v[f"c{k}_h0_im"] for k in range(self.n_channels)]
        )
        h1 = (
            np.array(
                [
                    v[f"c{k}_h1_re"] + 1j * v[f"c{k}_h1_im"]
                    for k in range(self.n_channels)
                ]
            )
            / self.gain_span
        )
        return {
            "fr": fr,
            "r": r,
            "a_nl": np.array(
                [
                    v[f"r{j}_a_nl"] if self.nonlinear else 0.0
                    for j in range(self.n_resonances)
                ]
            ),
            "c": c,
            "H0": h0,
            "H1": self.signal_scale * h1
            if self.gain_family == "affine"
            else np.full_like(h0, np.nan),
            "a": h1.real if self.gain_family == "delay" else np.full(len(h0), np.nan),
            "tau": -h1.imag / (2 * np.pi)
            if self.gain_family == "delay"
            else np.full(len(h0), np.nan),
        }

    def pair_model(self, params, channel, resonator):
        """Select a channel-resonator model without assigning a primary.

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters belonging to this group, with expressions updated.
        channel, resonator : int
            Explicit zero-based positions in the group's input order.
            The association must be included in this group.

        Returns
        -------
        KidsSweepWithAffineGain or KidsSweepWithDelayGain
            Detached physical-parameter snapshot with ``eval``, ``rotate`` and
            ``derotate``. Evaluation describes the selected resonance alone;
            derotation retains all contributions in the supplied measurement.
            Later group-parameter changes do not update this snapshot. Joint
            uncertainty remains with the parent fit result.

        Raises
        ------
        IndexError
            A position is outside the group.
        ValueError
            The channel-resonator association is not included.
        """
        if not 0 <= channel < self.n_channels or not 0 <= resonator < self.n_resonances:
            msg = "Channel or resonator position is outside the group"
            raise IndexError(msg)
        if not self.associations[channel, resonator]:
            msg = "Channel-resonator association is not included"
            raise ValueError(msg)
        physical = self.physical(params)
        values = {
            "fr": physical["fr"][resonator],
            "r": physical["r"][resonator],
            "c": physical["c"][channel, resonator],
            "f_g": self.gain_origin[channel],
            "H0": physical["H0"][channel],
            "a_nl": physical["a_nl"][resonator],
            "nonlinear_branch": self.nonlinear_branch,
        }
        if self.gain_family == "affine":
            return KidsSweepWithAffineGain(**values, H1=physical["H1"][channel])
        return KidsSweepWithDelayGain(
            **values, a=physical["a"][channel], tau=physical["tau"][channel]
        )

    def eval(self, params, frequency, *, jacobian=False):  # noqa: C901 -- Gain and branch derivatives share channel assembly.
        """Evaluate channels or their analytic derivatives in optimizer coordinates.

        Parameters
        ----------
        params : lmfit.Parameters
            Model parameters; expressions must already be updated.
        frequency : sequence of array_like
            One frequency vector in Hz per channel, including fixed-probe vectors.
        jacobian : bool, default False
            Return complex derivative matrices with columns ordered by ``names``.
            These partial derivatives treat coordinates as independent. The fitter
            applies parameter ties before computing its weighted Jacobian.

        Returns
        -------
        tuple of numpy.ndarray
            Complex predictions, or complex sample-by-parameter matrices.
        """
        if len(frequency) != self.n_channels:
            msg = "frequency must have one vector per channel"
            raise ValueError(msg)
        v = params.valuesdict()
        out = []
        xi = np.array([v[f"r{j}_xi"] for j in range(self.n_resonances)])
        rho = np.array([v[f"r{j}_rho"] for j in range(self.n_resonances)])
        for k, frequencies in enumerate(frequency):
            f = np.asarray(frequencies, float)
            ids = np.flatnonzero(self.associations[k])
            u = (f - self.gain_origin[k]) / self.gain_span[k]
            ratio = f[:, None] / self.fr_reference[ids]
            den = rho[ids] + 2j * (
                self.qr_reference[ids] * (ratio - 1) + ratio * xi[ids]
            )
            d_xi = 2j * ratio
            d_rho = np.ones_like(den)
            if self.nonlinear:
                y0 = den.imag / (2 * rho[ids])
                if self.nonlinear_branch != "unique":
                    # A rounding difference at a fold can select the other
                    # response. Use the exact persisted physical coordinates,
                    # as calibration and pair-model evaluation do.
                    fr = self.fr_reference[ids] / (1 + xi[ids] / self.qr_reference[ids])
                    loss = rho[ids] / (2 * self.qr_reference[ids])
                    y0 = (f[:, None] / fr - 1) / (2 * loss)
                a_nl = np.array([v[f"r{j}_a_nl"] for j in ids])
                y, implicit = nonlinear_detuning(y0, a_nl, branch=self.nonlinear_branch)
                den = rho[ids] * (1 + 2j * y)
                d_xi = 2j * ratio / implicit
                d_rho = 1 + 2j * y - 2j * y0 / implicit
                d_a = 2j * rho[ids] / ((1 + 4 * y**2) * implicit)
            chi = np.array([v[f"c{k}_r{j}_re"] + 1j * v[f"c{k}_r{j}_im"] for j in ids])
            h0 = v[f"c{k}_h0_re"] + 1j * v[f"c{k}_h0_im"]
            h1 = v[f"c{k}_h1_re"] + 1j * v[f"c{k}_h1_im"]
            if self.gain_family == "affine":
                gain = h0 + h1 * u
                d0 = np.ones_like(u)
                d1 = u
            else:
                d0 = np.exp(h1 * u)
                gain = h0 * d0
                d1 = gain * u
            kernel = 1 - np.sum(chi / den, axis=1)
            if not jacobian:
                out.append(self.signal_scale * gain * kernel)
                continue
            derivative = np.zeros((len(f), len(self.names)), complex)
            for column, j in enumerate(ids):
                for name, values in (
                    (
                        f"r{j}_xi",
                        gain * chi[column] / den[:, column] ** 2 * d_xi[:, column],
                    ),
                    (
                        f"r{j}_rho",
                        gain * chi[column] / den[:, column] ** 2 * d_rho[:, column],
                    ),
                    (f"c{k}_r{j}_re", -gain / den[:, column]),
                    (f"c{k}_r{j}_im", -1j * gain / den[:, column]),
                ):
                    derivative[:, self._indices[name]] = values
                if self.nonlinear:
                    derivative[:, self._indices[f"r{j}_a_nl"]] = (
                        gain * chi[column] / den[:, column] ** 2 * d_a[:, column]
                    )
            for name, values in (
                ("h0_re", d0),
                ("h0_im", 1j * d0),
                ("h1_re", d1),
                ("h1_im", 1j * d1),
            ):
                derivative[:, self._indices[f"c{k}_{name}"]] = values * kernel
            out.append(self.signal_scale * derivative)
        return tuple(out)
