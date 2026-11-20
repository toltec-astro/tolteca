"""Historical effective sweep model for reading existing tune calibrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u


def _column(table, name, unit=None):
    column = table[name]
    value = np.ma.asarray(column, dtype=float).filled(np.nan)
    declared_unit = getattr(column, "unit", None)
    if unit is not None and declared_unit is not None:
        value = (value * declared_unit).to_value(unit)
    return value


@dataclass
class KidsSweepGainWithLinTrend:
    """Evaluate the historical effective resonance plus linear background.

    Parameters
    ----------
    fr, Qr : array_like
        Resonance frequency in Hz and loaded quality factor.
    g0, g1 : array_like
        Real and imaginary canonical resonance coefficient G, in signal units.
    f0 : array_like
        Fixed background expansion frequency in Hz; native tables call it fp.
    k0, k1 : array_like
        Real and imaginary background slopes, in signal units per Hz.
    m0, m1 : array_like
        Real and imaginary background at f0, in signal units.

    Notes
    -----
    The sweep is ``M + K*(f-f0) + G*Qr/(1+2j*Qr*(f-fr)/fr)``.
    This preserves the familiar kidsproc parameter names and numerical equation,
    without requiring Astropy Model internals. Inputs follow NumPy broadcasting:
    for N model rows use frequencies shaped ``(sample,N)`` or one frequency per
    row. Nonfinite stored parameters remain nonfinite; evaluation does not
    establish fit acceptance. Source flags and channel identities stay in the
    caller's table.

    G, K and M are effective sweep coefficients. Freezing them during detector
    changes does not in general give the physical fixed-probe response. Use the
    physical gain/coupling interpretation for new timestream calibration.
    """

    fr: np.ndarray | float
    Qr: np.ndarray | float
    g0: np.ndarray | float
    g1: np.ndarray | float
    f0: np.ndarray | float
    k0: np.ndarray | float
    k1: np.ndarray | float
    m0: np.ndarray | float
    m1: np.ndarray | float

    @classmethod
    def from_table(cls, table):
        """Load effective coefficients from an existing native gainlintrend table.

        Parameters
        ----------
        table : astropy.table.Table
            Rows with ``fp,Qr,Qc,fr,A,normI,normQ,slopeI,slopeQ,interceptI,
            interceptQ``. Read ECSV with ``Table.read(path, format="ascii.ecsv")``
            before calling. Undeclared fp/fr units mean Hz, as in historical
            native output; declared frequency units are converted. Other values
            use the native signal normalization and per-Hz slope convention.
            Masked entries become NaN. Row order is preserved.

        Returns
        -------
        KidsSweepGainWithLinTrend
            Vectorized model with ``G=(normI+1j*normQ)*(1+1j*A)/Qc``.
            This follows kidscpp's S21WithGainLinTrend equation. The older Python
            adapter omitted Qc and A; its results agree for Qc=1 and A=0.
            These table fields do not independently establish physical coupling.

        Raises
        ------
        KeyError
            A required model column is missing.
        astropy.units.UnitConversionError
            Declared frequency units cannot convert to Hz.
        """
        norm = _column(table, "normI") + 1j * _column(table, "normQ")
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = norm * (1 + 1j * _column(table, "A")) / _column(table, "Qc")
        return cls(
            fr=_column(table, "fr", u.Hz),
            Qr=_column(table, "Qr"),
            g0=gain.real,
            g1=gain.imag,
            f0=_column(table, "fp", u.Hz),
            k0=_column(table, "slopeI"),
            k1=_column(table, "slopeQ"),
            m0=_column(table, "interceptI"),
            m1=_column(table, "interceptQ"),
        )

    def eval(self, frequency):
        """Evaluate the stored stationary sweep in its measured IQ frame.

        Parameters
        ----------
        frequency : array_like
            Probe frequency in Hz, broadcastable with the model arrays.

        Returns
        -------
        numpy.ndarray
            Complex signal in the stored normalization. One frequency per model
            returns one prediction per model; ``(sample,N)`` evaluates N models
            at several frequencies.
        """
        f = np.asarray(frequency, float)
        qr = np.asarray(self.Qr)
        circle = qr / (1 + 2j * qr * (f - self.fr) / self.fr)
        return self.rotate(circle, f)

    def rotate(self, circle, frequency):
        """Transform effective resonator coordinates to measured IQ.

        Parameters
        ----------
        circle : array_like
            Complex values in the selected resonator's coordinates, including
            any neighbouring features or noise. An isolated canonical response
            is ``0.5/(r+1j*x)``.
        frequency : array_like
            Known probe frequencies in Hz, broadcastable with circle and model.

        Returns
        -------
        numpy.ndarray
            Signal ``G*circle + K*(frequency-f0) + M`` in recorded IQ units.
        """
        gain = np.asarray(self.g0) + 1j * np.asarray(self.g1)
        slope = np.asarray(self.k0) + 1j * np.asarray(self.k1)
        offset = np.asarray(self.m0) + 1j * np.asarray(self.m1)
        return (
            gain * np.asarray(circle)
            + slope * (np.asarray(frequency) - self.f0)
            + offset
        )

    def derotate(self, s21, frequency):
        """Remove the stored effective background and gain for circle inspection.

        Parameters
        ----------
        s21 : array_like
            Measured complex signal in the model's normalization.
        frequency : array_like
            Known probe frequencies in Hz, broadcastable with s21 and model.

        Returns
        -------
        numpy.ndarray
            Values in the effective resonator coordinates, retaining neighbours
            and noise. No fitted resonance is subtracted. This reproduces
            historical derotation, not a validated physical timestream inverse.

        Raises
        ------
        ValueError
            A model has zero gain; select usable rows before derotation.
        """
        gain = np.asarray(self.g0) + 1j * np.asarray(self.g1)
        if np.any(gain == 0):
            msg = "Cannot derotate a gainlintrend model with zero gain"
            raise ValueError(msg)
        slope = np.asarray(self.k0) + 1j * np.asarray(self.k1)
        offset = np.asarray(self.m0) + 1j * np.asarray(self.m1)
        # Keep the historical subtraction order for large cancelling baselines.
        return (
            np.asarray(s21) - offset - slope * (np.asarray(frequency) - self.f0)
        ) / gain
