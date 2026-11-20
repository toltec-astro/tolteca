"""Research helpers for a dissipation-referenced complex noise budget."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def rx_noise_budget(pr, px, beta=1.0):
    """Infer excess x power assuming its readout PSD is beta times the r PSD.

    Inputs are power spectral densities, not amplitudes. A nonpositive excess
    remains signed; its amplitude ratio and penalty are undefined (NaN).
    The returned excess is not automatically identified with photon noise.
    """
    pr, px, beta = np.broadcast_arrays(
        np.asarray(pr, float), np.asarray(px, float), np.asarray(beta, float)
    )
    if np.any(~np.isfinite([pr, px, beta])):
        msg = "Noise powers and readout transfer must be finite."
        raise ValueError(msg)
    if np.any(pr <= 0) or np.any(px < 0) or np.any(beta <= 0):
        msg = (
            "r power and readout transfer must be positive; x power cannot be negative."
        )
        raise ValueError(msg)
    readout = beta * pr
    excess = px - readout
    resolved = excess > 0
    kappa = np.full(excess.shape, np.nan)
    penalty = kappa.copy()
    np.divide(excess, readout, out=kappa, where=resolved)
    np.sqrt(kappa, out=kappa, where=resolved)
    np.divide(px, excess, out=penalty, where=resolved)
    np.sqrt(penalty, out=penalty, where=resolved)
    return {
        "readout_x": readout,
        "excess_x": excess,
        "power_ratio": px / readout,
        "kappa": kappa,
        "penalty": penalty,
    }


def rx_periodograms(w, fs, nperseg=1024, step=None):
    """Return frequency and one-sided Hermitian r/x spectra for each segment.

    Leading input dimensions are retained. Segment and frequency axes precede
    the final 2x2 matrix axes. Default segments do not overlap; incomplete final
    segments are discarded. Detrending removes each segment's mean.
    """
    w = np.asarray(w)
    step = nperseg if step is None else step
    min_length = 2
    if nperseg < min_length or step < 1 or fs <= 0 or w.shape[-1] < nperseg:
        msg = (
            "Require positive sample rate and step, and at least one "
            "segment of length 2 or more."
        )
        raise ValueError(msg)
    blocks = np.lib.stride_tricks.sliding_window_view(w, nperseg, axis=-1)[
        ..., ::step, :
    ]
    v = np.stack((blocks.real, blocks.imag), axis=-1)
    v -= v.mean(axis=-2, keepdims=True)
    window = get_window("hann", nperseg)
    fft = np.fft.rfft(v * window[:, None], axis=-2)
    spectral = fft[..., :, :, None] * fft[..., :, None, :].conj()
    spectral /= fs * np.sum(window**2)
    spectral[..., 1 : (-1 if nperseg % 2 == 0 else None), :, :] *= 2
    return np.fft.rfftfreq(nperseg, 1 / fs), spectral
