"""Measured coordinate spectra and fixed-drive noise propagation after solving."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import csd

from .analysis import PSDAnalysis

__all__ = ["coordinate_spectra", "spectral_matrix"]
_MATRIX_NDIM = 2
_MIN_WINDOW = 2


def spectral_matrix(values, f_smp, *, nperseg=1024):
    """Measure a one-sided Welch cross-spectral matrix of complete real series.

    Parameters
    ----------
    values : array_like, shape (sample, coordinate)
        Finite, uniformly sampled real signals. The caller validates sample
        continuity. Rejected samples must not be removed or gap-filled here.
    f_smp : float
        Positive sampling frequency in Hz.
    nperseg : int, optional
        Hann window length, default 1024, with half-window overlap, constant
        detrending and density scaling. At least two window lengths are required.

    Returns
    -------
    frequency, cross : numpy.ndarray
        Frequencies in Hz and complex matrix (frequency, coordinate, coordinate).
        Entry ij is conj(FFT_i)*FFT_j averaged over windows, with units
        coordinate_i*coordinate_j/Hz. Diagonal entries are real PSDs.

    Raises
    ------
    ValueError
        Samples, sampling rate or window length are invalid or insufficient.
    """
    if np.iscomplexobj(values):
        msg = "Require complete real series, not complex input"
        raise ValueError(msg)
    values = np.asarray(values, float)
    if (
        values.ndim != _MATRIX_NDIM
        or not values.shape[1]
        or not np.isfinite(values).all()
        or not np.isfinite(f_smp)
        or f_smp <= 0
        or int(nperseg) != nperseg
        or nperseg < _MIN_WINDOW
        or len(values) < 2 * nperseg
    ):
        msg = "Require complete real series, positive sample rate and two windows"
        raise ValueError(msg)
    options = {
        "window": "hann",
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "detrend": "constant",
        "scaling": "density",
        "return_onesided": True,
        "axis": 0,
    }
    frequency, power = PSDAnalysis.compute_psd(values, f_smp=f_smp, **options)
    count = values.shape[1]
    matrix = np.zeros((len(frequency), count, count), complex)
    for i in range(count):
        matrix[:, i, i] = power[:, i]
        for j in range(i + 1, count):
            _, cross = csd(values[:, i], values[:, j], fs=f_smp, **options)
            matrix[:, i, j], matrix[:, j, i] = cross, cross.conj()
    return frequency, matrix


def coordinate_spectra(x_prime, x_opt, r, f_smp, *, D, mu, nperseg=1024):  # noqa: N803, PLR0913 -- Physical coordinate names.
    """Compare measured detuning spectra with local fixed-drive propagation.

    Parameters
    ----------
    x_prime, x_opt, r : array_like, shape (sample,)
        Effective detuning, corrected intrinsic detuning and loaded loss, at
        one declared probe. These are dimensionless solved/model coordinates.
    f_smp : float
        Recorded sampling frequency in Hz; samples must be complete and uniform.
    D, mu : float
        Local derivatives from the same fitted group model, evaluated at a
        declared reference state: dx_opt = D*dx_prime + mu*dr at fixed b.
        They are constant only for this first-order spectral comparison.
    nperseg : int, optional
        Welch window length, default 1024; see ``spectral_matrix``.

    Returns
    -------
    xarray.Dataset
        Full complex cross-spectrum as real/imaginary arrays, coherence and
        signed first-order PSD terms. The exact correction identity is retained
        separately from the local linear prediction. All spectral densities
        have units Hz^-1. No optical attribution or detector selection is made.

    Raises
    ------
    ValueError
        Coordinates differ in shape, derivatives are nonfinite or spectral
        input is invalid. No interpolation or exclusion is performed.
    """
    xp, xo, loss = [np.asarray(v, float) for v in (x_prime, x_opt, r)]
    if xp.ndim != 1 or xp.shape != xo.shape or xp.shape != loss.shape:
        msg = "Coordinates must be equal one-dimensional sample arrays"
        raise ValueError(msg)
    if not np.isfinite([D, mu]).all():
        msg = "Require finite local response derivatives"
        raise ValueError(msg)
    frequency, matrix = spectral_matrix(
        np.column_stack([xp, xo, loss, xp - xo]), f_smp, nperseg=nperseg
    )
    power = matrix.diagonal(axis1=1, axis2=2).real
    denominator = power[:, :, None] * power[:, None, :]
    coherence = np.divide(
        abs(matrix) ** 2,
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
    terms = np.column_stack(
        [D**2 * power[:, 0], mu**2 * power[:, 2], 2 * D * mu * matrix[:, 0, 2].real]
    )
    _, linear_power = spectral_matrix(
        (D * xp + mu * loss)[:, None], f_smp, nperseg=nperseg
    )
    _, remainder = spectral_matrix(
        (xo - D * xp - mu * loss)[:, None], f_smp, nperseg=nperseg
    )
    coordinates = ["x_prime", "x_opt", "r", "correction"]
    dims = ("f_psd", "coordinate", "coordinate2")
    return xr.Dataset(
        {
            "cross_real": (dims, matrix.real, {"units": "Hz^-1"}),
            "cross_imag": (dims, matrix.imag, {"units": "Hz^-1"}),
            "coherence": (dims, coherence),
            "linear_terms": (("f_psd", "term"), terms, {"units": "Hz^-1"}),
            "linear_predicted": ("f_psd", terms.sum(axis=1), {"units": "Hz^-1"}),
            "linear_measured": ("f_psd", linear_power[:, 0, 0].real),
            "linear_remainder": ("f_psd", remainder[:, 0, 0].real),
            "exact_reconstruction": (
                "f_psd",
                power[:, 0] + power[:, 3] - 2 * matrix[:, 0, 3].real,
            ),
        },
        coords={
            "f_psd": ("f_psd", frequency, {"units": "Hz"}),
            "coordinate": coordinates,
            "coordinate2": coordinates,
            "term": ["x_prime", "loss", "cross"],
        },
        attrs={
            "D": float(D),
            "mu": float(mu),
            "f_smp_hz": float(f_smp),
            "nperseg": nperseg,
            "noverlap": nperseg // 2,
            "window": "hann",
            "detrend": "constant",
            "scaling": "density",
            "cross_convention": "conj(first)*second",
            "correction": "x_prime - x_opt",
            "interpretation": "measured fluctuations; no optical attribution",
        },
    )
