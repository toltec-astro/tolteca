"""Fixed-total DAC power fractions on Tlaloc's emitted frequency bins.

This approximation distributes total power by squared amplitude. It does not
model the phase-dependent peak normalization of the synthesized DAC waveform.
"""

import numpy as np

POWER_MODEL = "fixed_total_unique_dac_bins"


def dac_bin_owners(frequency):
    """Return the last channel assigned to each row's Tlaloc DAC bin.

    Parameters
    ----------
    frequency : array-like
        Finite baseband frequencies in Hz, in tone-file order.

    Returns
    -------
    numpy.ndarray
        Index of the last row writing each input row's quantized DAC bin.

    Raises
    ------
    ValueError
        Frequencies are not a finite one-dimensional array.
    """
    frequency = np.asarray(frequency, dtype=float)
    if frequency.ndim != 1 or not np.isfinite(frequency).all():
        msg = "DAC frequencies must be a finite one-dimensional array in Hz"
        raise ValueError(msg)
    # Tlaloc: DAC_FREQ_RES = 2 * 512 MHz / 2**21; C round is away from zero.
    bins = (
        2 * np.copysign(np.floor(np.abs(frequency) / 488.28125 + 0.5), frequency)
    ).astype(np.int64) % (2**21)
    last = {int(value): index for index, value in enumerate(bins)}
    return np.array([last[int(value)] for value in bins], dtype=int)


def tone_power_fractions(amplitudes, frequency, mask=None):
    """Distribute unit total power over the full emitted comb, then map to rows.

    Masked tones have zero power; a later row overwrites an earlier row in the
    same DAC bin. Uniform amplitude scaling leaves the fractions unchanged.

    Parameters
    ----------
    amplitudes, frequency : array-like
        Full-comb amplitude coefficients and baseband frequencies in Hz.
    mask : array-like of bool or None, optional
        Enabled tone rows. None enables every row.

    Returns
    -------
    numpy.ndarray
        Fraction of the total emitted power associated with each recorded row.
        Sum over unique controlling rows to obtain one, not over duplicate rows.

    Raises
    ------
    ValueError
        Inputs are misaligned, amplitudes are invalid, or no positive power is
        assigned to an emitted DAC bin.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    owners = dac_bin_owners(frequency)
    mask = np.ones(len(owners), dtype=bool) if mask is None else np.asarray(mask, bool)
    if amplitudes.shape != owners.shape or mask.shape != owners.shape:
        msg = "Tone amplitudes, frequencies and masks must have equal lengths"
        raise ValueError(msg)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        msg = "Tone amplitudes must be finite and nonnegative"
        raise ValueError(msg)
    active = amplitudes * mask
    scale = active.max(initial=0)
    if scale <= 0:
        msg = "The emitted comb has no positive tone amplitudes"
        raise ValueError(msg)
    power = (active / scale) ** 2
    total = power[np.unique(owners)].sum()
    if total <= 0:
        msg = "The emitted DAC bins have no positive tone amplitudes"
        raise ValueError(msg)
    return power[owners] / total
