"""Physical single-resonator views of a channel's measured IQ coordinates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ._nonlinear import nonlinear_detuning


@dataclass(frozen=True)
class _PhysicalSweepModel(ABC):
    fr: np.ndarray | float
    r: np.ndarray | float
    c: np.ndarray | complex
    f_g: np.ndarray | float
    H0: np.ndarray | complex
    a_nl: np.ndarray | float = field(default=0.0, kw_only=True)

    nonlinear_branch: str = field(default="unique", kw_only=True)

    @abstractmethod
    def gain(self, frequency):
        """Evaluate the channel gain at probe frequencies in Hz."""

    def eval(self, frequency):
        """Evaluate the selected resonance alone through the channel gain.

        Parameters
        ----------
        frequency : array_like
            Probe frequencies in Hz, broadcastable with the model parameters.

        Returns
        -------
        numpy.ndarray
            Complex ``H*(1-c/(r+1j*x))`` in measured signal units. This is the
            single-resonance prediction. Use the parent group's prediction for
            the complete response, including neighbours.
        """
        f = np.asarray(frequency, float)
        with np.errstate(invalid="ignore"):
            if np.any(self.a_nl):
                y, _ = nonlinear_detuning(
                    (f / self.fr - 1) / (2 * np.asarray(self.r)),
                    self.a_nl,
                    branch=self.nonlinear_branch,
                )
                circle = 0.5 / (np.asarray(self.r) * (1 + 2j * y))
            else:
                circle = 0.5 / (np.asarray(self.r) + 1j * (f / self.fr - 1))
        return self.rotate(circle, f)

    def rotate(self, circle, frequency):
        """Transform selected-resonator coordinates to measured IQ.

        Parameters
        ----------
        circle : array_like
            Complex values in the selected resonator's coordinates. These can
            include neighbouring features and noise, or an isolated canonical
            response ``0.5/(r+1j*x)``.
        frequency : array_like
            Probe frequencies in Hz, broadcastable with circle and the model.

        Returns
        -------
        numpy.ndarray
            Complex ``H*(1-2*c*circle)`` in measured signal units.
        """
        return self.gain(frequency) * (1 - 2 * np.asarray(self.c) * np.asarray(circle))

    def derotate(self, s21, frequency):
        """Express measured IQ about the selected resonator, retaining neighbours.

        Parameters
        ----------
        s21 : array_like
            Measured complex signal in the model's normalization. Neighbouring
            resonances and noise remain in the transformed measurement.
        frequency : array_like
            Probe frequencies in Hz, broadcastable with s21 and the model.

        Returns
        -------
        numpy.ndarray
            ``(1-s21/H)/(2*c)``. No resonance state is evaluated or subtracted.
            This coordinate transformation does not solve detector states.

        Raises
        ------
        ValueError
            Gain or selected coupling is zero. Select usable rows first.
        """
        gain = self.gain(frequency)
        coupling = np.asarray(self.c)
        if np.any(gain == 0) or np.any(coupling == 0):
            msg = "Cannot derotate a model with zero gain or coupling"
            raise ValueError(msg)
        with np.errstate(invalid="ignore"):
            return (1 - np.asarray(s21) / gain) / (2 * coupling)


@dataclass(frozen=True)
class KidsSweepWithAffineGain(_PhysicalSweepModel):
    """Evaluate one physical resonance through a complex affine channel gain.

    Parameters
    ----------
    fr, r : array_like
        Resonance frequency in Hz and loss ``1/(2*Qr)``.
    c : array_like
        Selected channel-resonator coupling ``(1+1j*A)/(2*Qc)``.
    f_g : array_like
        Fixed gain expansion frequency in Hz.
    H0, H1 : array_like
        Complex gain at f_g in signal units and slope in signal units per Hz.
    a_nl : array_like, optional
        Shared stationary softening parameter, default zero.
        Nonzero values make fr the zero-drive resonance frequency.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Unique response requires a_nl <= 0.75. Directional responses assume a
        monotonic sweep starting below/above resonance; evaluation order is free.

    Notes
    -----
    Inputs use NumPy broadcasting. For N models, frequencies shaped
    ``(sample,N)`` evaluate all models. Nonfinite parameters propagate as NaN.
    No primary selection, fit acceptance or covariance independence is implied.
    """

    H1: np.ndarray | complex

    def gain(self, frequency):
        """Evaluate the affine channel gain.

        Parameters
        ----------
        frequency : array_like
            Probe frequencies in Hz, broadcastable with gain parameters.

        Returns
        -------
        numpy.ndarray
            Complex ``H0 + H1*(frequency-f_g)`` in signal units.
        """
        return np.asarray(self.H0) + np.asarray(self.H1) * (
            np.asarray(frequency, float) - self.f_g
        )


@dataclass(frozen=True)
class KidsSweepWithDelayGain(_PhysicalSweepModel):
    """Evaluate one physical resonance through amplitude slope and cable delay.

    Parameters
    ----------
    fr, r : array_like
        Resonance frequency in Hz and loss ``1/(2*Qr)``.
    c : array_like
        Selected channel-resonator coupling ``(1+1j*A)/(2*Qc)``.
    f_g : array_like
        Fixed gain expansion frequency in Hz.
    H0 : array_like
        Complex gain at f_g in signal units.
    a, tau : array_like
        Log-amplitude slope per Hz and delay in seconds. Positive delay gives
        a phase slope of ``-2*pi*tau`` radians per Hz.
    a_nl : array_like, optional
        Shared stationary softening parameter, default zero.
        This parameter is distinct from the amplitude slope a.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Unique response requires a_nl <= 0.75. Directional responses assume a
        monotonic sweep starting below/above resonance; evaluation order is free.

    Notes
    -----
    Inputs use NumPy broadcasting. For N models, frequencies shaped
    ``(sample,N)`` evaluate all models. Nonfinite parameters propagate as NaN.
    No primary selection, fit acceptance or covariance independence is implied.
    """

    a: np.ndarray | float
    tau: np.ndarray | float

    def gain(self, frequency):
        """Evaluate the amplitude-slope and delay channel gain.

        Parameters
        ----------
        frequency : array_like
            Probe frequencies in Hz, broadcastable with gain parameters.

        Returns
        -------
        numpy.ndarray
            Complex ``H0*exp((a-2*pi*1j*tau)*(frequency-f_g))`` in signal units.
        """
        return np.asarray(self.H0) * np.exp(
            (np.asarray(self.a) - 2j * np.pi * np.asarray(self.tau))
            * (np.asarray(frequency, float) - self.f_g)
        )
