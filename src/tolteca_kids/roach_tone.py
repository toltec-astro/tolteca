"""ROACH tone bins, baseline calibration and DAC waveform power."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path  # noqa: TC003 -- Resolve public dataclass annotations.
from typing import Annotated

import astropy.units as u
import numpy as np
from pydantic import Field, FiniteFloat, field_validator
from scipy.fft import ifft
from scipy.interpolate import CubicSpline
from tollan.config import FrozenBaseModel
from tollan.config.types import FrequencyQuantityField

POWER_MODEL = "dac_fourier_coefficient_power"
DAC_SAMPLE_RATE = 512e6
DAC_LUT_SIZE = 2**21


class RoachDacConfig(FrozenBaseModel):
    """DAC sampling and waveform length for the ROACH firmware.

    Parameters
    ----------
    sample_rate : astropy.units.Quantity, optional
        DAC sample rate, default 512 MHz. Must be finite and positive.
    lut_size : int, optional
        Even waveform length, default 2**21. The emitted frequency spacing is
        twice sample_rate/lut_size, following the ROACH controller.
    """

    sample_rate: FrequencyQuantityField = DAC_SAMPLE_RATE * u.Hz
    lut_size: int = Field(default=DAC_LUT_SIZE, ge=4, multiple_of=2)

    @field_validator("sample_rate")
    @classmethod
    def _positive_rate(cls, value):
        if not np.isfinite(value.to_value(u.Hz)) or value <= 0 * u.Hz:
            msg = "DAC sample rate must be finite and positive"
            raise ValueError(msg)
        return value


class RoachTonePowerConfig(FrozenBaseModel):
    """Explicit calibration for detector-input power predictions.

    Parameters
    ----------
    dac_single_tone_power_dbm : float
        Measured DAC output power for one full-scale complex sinusoid, in dBm.
        This is not the total power of an arbitrary many-tone comb.
    drive_path_loss_db : float
        Fixed net loss from the DAC output to detector input, in dB, excluding
        the variable drive attenuator. Negative values represent net gain.
    calibration : str
        Identifier or description of the calibration used for these values.
        The preparation result retains it with the prediction.
    """

    dac_single_tone_power_dbm: FiniteFloat
    drive_path_loss_db: FiniteFloat
    calibration: str = Field(min_length=1)


READOUT_ELEMENTS = (
    "dac",
    "drive_atten",
    "cryo_atten",
    "kids",
    "lna",
    "if_amp",
    "if_board",
    "sense_atten",
    "adc",
)


class RoachReadoutPowerConfig(FrozenBaseModel):
    """Calibration and plotted limits for an acquired ROACH sweep.

    The defaults reproduce the production-v2 link budget. DAC power is the
    calibrated total comb power, distinct from the single-tone calibration in
    :class:`RoachTonePowerConfig`. Drive and sense attenuation come from the
    acquisition. Changing these settings does not change instrument settings.
    """

    dac_power_dbm: FiniteFloat = Field(
        default=-12.6, description="Total DAC comb power in dBm (2025 SA calibration)."
    )
    cryo_atten_db: FiniteFloat = Field(
        default=0.0, description="Cryogenic cable attenuation in dB."
    )
    kids_loss_db: FiniteFloat = Field(
        default=2.0, description="KIDs insertion loss in dB."
    )
    lna_gain_db: FiniteFloat = Field(default=30.0, description="LNA gain in dB.")
    if_amp_gain_db: FiniteFloat = Field(
        default=27.0, description="IF amplifier gain in dB."
    )
    if_board_gains_db_by_roach: dict[
        Annotated[int, Field(strict=False)], FiniteFloat
    ] = Field(
        default={
            0: 16.8,
            1: 5.7,
            2: 9.2,
            3: 17.1,
            4: 10.7,
            5: 17.1,
            6: 10.8,
            7: 14.0,
            8: 17.0,
            9: 17.0,
            10: 0.0,
            11: 17.0,
            12: 17.0,
        },
        description="IF-board gains in dB by network (2023-01-28 calibration).",
    )
    adc_conversion_offset_db: FiniteFloat = Field(
        default=162.0,
        description="Offset subtracted from 10 log10(I² + Q²) to obtain dBm.",
    )
    lna_input_dbm_max: FiniteFloat = Field(
        default=-52.0, description="Plotted maximum LNA input power in dBm."
    )
    lna_output_dbm_max: FiniteFloat = Field(
        default=-15.0, description="Plotted maximum LNA output power in dBm."
    )
    if_board_input_dbm_max: FiniteFloat = Field(
        default=-5.0, description="Plotted maximum IF-board input power in dBm."
    )
    adc_snap_frac_min: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Plotted minimum ADC full-scale usage fraction.",
    )

    def gains(
        self, roach: int, *, atten_drive: float, atten_sense: float
    ) -> np.ndarray:
        """Return signed stage gains in signal-path order.

        Parameters
        ----------
        roach : int
            Network selecting the IF-board calibration.
        atten_drive, atten_sense : float
            Acquired attenuator settings in dB.

        Returns
        -------
        numpy.ndarray
            Gains in dB, ordered by READOUT_ELEMENTS.

        Raises
        ------
        ValueError
            The network has no IF-board calibration.
        """
        if roach not in self.if_board_gains_db_by_roach:
            msg = f"No IF-board gain calibration for ROACH {roach}"
            raise ValueError(msg)
        return np.array(
            [
                0.0,
                -atten_drive,
                -self.cryo_atten_db,
                -self.kids_loss_db,
                self.lna_gain_db,
                self.if_amp_gain_db,
                self.if_board_gains_db_by_roach[roach],
                -atten_sense,
                0.0,
            ]
        )


def readout_power(  # noqa: PLR0913 -- explicit calibrated inputs
    tone_amplitudes,
    I,  # noqa: E741, N803 -- physical I/Q names
    Q,  # noqa: N803
    adc_snap,
    *,
    dac_power_dbm,
    gains_db,
    adc_conversion_offset_db,
):
    """Evaluate the production-v2 acquired-comb power budget.

    Parameters
    ----------
    tone_amplitudes : array-like
        Recorded voltage coefficients, one per channel. Power is distributed
        over these rows by amplitude squared; no DAC-bin deduplication occurs.
    I, Q : array-like
        Reduced, optionally despiked samples in ADC units, shape (chan, probe).
    adc_snap : array-like or None
        Two arrays of recorded 12-bit ADC codes in 16-bit words. None omits
        ADC full-scale usage when the acquisition did not record a snapshot.
    dac_power_dbm : float
        Calibrated total DAC output power in dBm.
    gains_db : array-like
        Signed gains in READOUT_ELEMENTS order, including acquired attenuation.
    adc_conversion_offset_db : float
        Offset converting mean squared I/Q magnitude to dBm.

    Returns
    -------
    dict
        Predicted/inferred input/output powers for every stage, per channel and
        in total, relative amplitude attenuation, and ADC full-scale usage.
        Powers use dBm and relative attenuation uses dB. Zero amplitudes retain
        negative-infinite power and positive-infinite relative attenuation.
    """
    amps = np.asarray(tone_amplitudes)
    gains = np.asarray(gains_db)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = (
            dac_power_dbm + 10 * np.log10(amps**2) - 10 * np.log10(np.sum(amps**2))
        )
        inferred = (
            10 * np.log10((np.asarray(I) ** 2 + np.asarray(Q) ** 2).mean(axis=-1))
            - adc_conversion_offset_db
        )
        inferred_total = 10 * np.log10((10 ** (inferred / 10)).sum())
        relative_atten = 20 * np.log10(amps.max() / amps)
    result = {"tone_amps_db": relative_atten, "gain": gains}
    for name, start, backward in (
        ("predicted_per_tone", predicted, False),
        ("predicted_total", dac_power_dbm, False),
        ("inferred_per_tone", inferred, True),
        ("inferred_total", inferred_total, True),
    ):
        inputs = np.empty((len(gains), *np.shape(start)))
        outputs = np.empty_like(inputs)
        power = start
        for i in range(len(gains) - 1, -1, -1) if backward else range(len(gains)):
            if backward:
                outputs[i] = power
                power = inputs[i] = power - gains[i]
            else:
                inputs[i] = power
                power = outputs[i] = power + gains[i]
        result[name + "_input"] = inputs
        result[name + "_output"] = outputs
    # Header codes are already 12-bit counts, not left-shifted int16 values.
    if adc_snap is not None:
        snapshot = np.asarray(adc_snap).view(np.int16)
        result["adc_snap_frac"] = np.mean(
            [(row.max() - row.min()) / 2**12 for row in snapshot]
        )
    return result


@dataclass
class RoachToneLut:
    """RF amplitude baseline with the controller's interpolation convention.

    Parameters
    ----------
    frequency : astropy.units.Quantity
        Strictly increasing RF frequencies; at least three finite samples.
    amplitude : array-like
        Finite, nonnegative relative voltage coefficients at those frequencies.
    path : pathlib.Path or None, optional
        Calibration source, populated by from_file.
    digest : str or None, optional
        SHA-256 of the source bytes, populated by from_file.
    """

    frequency: u.Quantity
    amplitude: np.ndarray
    path: Path | None = None
    digest: str | None = None
    _spline: CubicSpline = field(init=False, repr=False)

    def __post_init__(self):
        x = np.array(self.frequency.to_value(u.Hz), dtype=float, copy=True)
        y = np.array(self.amplitude, dtype=float, copy=True)
        if (
            x.ndim != 1
            or y.shape != x.shape
            or len(x) < 3  # noqa: PLR2004 -- tk::spline minimum.
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0)
            or np.any(y < 0)
        ):
            msg = f"Invalid amplitude LUT: {self.path}"
            raise ValueError(msg)
        self.frequency = x * u.Hz
        self.amplitude = y
        self._spline = CubicSpline(x, y, bc_type="natural")

    @classmethod
    def from_file(cls, path: Path) -> RoachToneLut:
        """Read a headerless CSV baseline.

        Parameters
        ----------
        path : pathlib.Path
            Two columns: absolute RF frequency in Hz and relative amplitude.

        Returns
        -------
        RoachToneLut
            Baseline retaining the path and content digest.

        Raises
        ------
        ValueError
            Columns or calibration samples are invalid.
        """
        content = path.read_bytes()
        data = np.loadtxt(BytesIO(content), delimiter=",", ndmin=2)
        if data.shape[1] != 2:  # noqa: PLR2004 -- frequency and amplitude.
            msg = f"Invalid amplitude LUT: {path}"
            raise ValueError(msg)
        return cls(data[:, 0] * u.Hz, data[:, 1], path, sha256(content).hexdigest())

    def __call__(self, frequency: u.Quantity) -> np.ndarray:
        """Evaluate the natural spline with linear endpoint extrapolation.

        Parameters
        ----------
        frequency : astropy.units.Quantity
            Absolute RF frequencies, in any frequency unit.

        Returns
        -------
        numpy.ndarray
            Positive relative voltage coefficients with the input shape.

        Raises
        ------
        ValueError
            Interpolation produces nonpositive or nonfinite coefficients.
        """
        f = np.asarray(frequency.to_value(u.Hz), dtype=float)
        x = self.frequency.to_value(u.Hz)
        values = self._spline(f)
        for edge, outside in [(0, f < x[0]), (-1, f > x[-1])]:
            values = np.where(
                outside,
                self.amplitude[edge] + self._spline(x[edge], 1) * (f - x[edge]),
                values,
            )
        if not np.isfinite(values).all() or np.any(values <= 0):
            msg = "LUT values must be positive at fallback and anchor frequencies"
            raise ValueError(msg)
        return values


def dac_bins(frequency, *, sample_rate=DAC_SAMPLE_RATE, lut_size=DAC_LUT_SIZE):
    """Quantize baseband frequencies to periodic ROACH DAC-bin indices.

    Parameters
    ----------
    frequency : array-like
        Finite one-dimensional baseband frequencies in Hz, in tone-file order.
    sample_rate : float, optional
        DAC sample rate in Hz, default 512e6.
    lut_size : int, optional
        Even waveform length, default 2**21.

    Returns
    -------
    numpy.ndarray
        Nonnegative FFT-bin indices. Rounding uses C's half-away-from-zero rule
        on the controller's two-bin frequency grid.

    Raises
    ------
    ValueError
        Frequencies or sampling parameters are invalid.
    """
    frequency = np.asarray(frequency, dtype=float)
    if frequency.ndim != 1 or not np.isfinite(frequency).all():
        msg = "DAC frequencies must be a finite one-dimensional array in Hz"
        raise ValueError(msg)
    if (
        not np.isfinite(sample_rate)
        or sample_rate <= 0
        or lut_size < 4  # noqa: PLR2004 -- At least two controller frequency bins.
        or lut_size % 2
    ):
        msg = "DAC sampling requires a positive rate and even waveform length"
        raise ValueError(msg)
    spacing = 2 * sample_rate / lut_size
    return (
        2 * np.copysign(np.floor(np.abs(frequency) / spacing + 0.5), frequency)
    ).astype(np.int64) % lut_size


def _owners(bins):
    last = {int(value): index for index, value in enumerate(bins)}
    return np.array([last[int(value)] for value in bins], dtype=int)


def dac_bin_owners(frequency, *, sample_rate=DAC_SAMPLE_RATE, lut_size=DAC_LUT_SIZE):
    """Return the last tone row assigned to each row's emitted DAC bin.

    Parameters
    ----------
    frequency : array-like
        Baseband frequencies in Hz, in tone-file order.
    sample_rate : float, optional
        DAC sample rate in Hz, default 512e6.
    lut_size : int, optional
        Even waveform length, default 2**21.

    Returns
    -------
    numpy.ndarray
        Last writer's row index for every input row, including masked writers.
    """
    return _owners(dac_bins(frequency, sample_rate=sample_rate, lut_size=lut_size))


def _active_amplitudes(amplitudes, mask, owners):
    amplitudes = np.asarray(amplitudes, dtype=float)
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
    active = active / scale
    if np.sum(active[np.unique(owners)] ** 2) <= 0:
        msg = "The emitted DAC bins have no positive tone amplitudes"
        raise ValueError(msg)
    return active


def tone_power_fractions(
    amplitudes,
    frequency,
    mask=None,
    *,
    sample_rate=DAC_SAMPLE_RATE,
    lut_size=DAC_LUT_SIZE,
):
    """Distribute unit total power over emitted bins, then map to tone rows.

    Parameters
    ----------
    amplitudes, frequency : array-like
        Full-comb voltage coefficients and baseband frequencies in Hz.
    mask : array-like of bool or None, optional
        Enabled rows; None enables all rows. A later masked row overwrites an
        earlier row in the same bin with zero amplitude.
    sample_rate : float, optional
        DAC sample rate in Hz, default 512e6.
    lut_size : int, optional
        Even waveform length, default 2**21.

    Returns
    -------
    numpy.ndarray
        Fixed-total power fractions. Sum over unique controlling rows to obtain
        one, not over duplicated rows. Divide by the waveform power crest factor
        to obtain powers relative to a full-scale sinusoid.

    Raises
    ------
    ValueError
        Inputs are invalid or the emitted bins contain no positive power.
    """
    owners = dac_bin_owners(frequency, sample_rate=sample_rate, lut_size=lut_size)
    power = _active_amplitudes(amplitudes, mask, owners) ** 2
    return power[owners] / power[np.unique(owners)].sum()


def tone_waveform_attenuation(  # noqa: PLR0913 -- Explicit waveform and hardware inputs.
    power_target_db,
    frequency,
    phases,
    mask,
    *,
    phase_trials=0,
    seed=None,
    sample_rate=DAC_SAMPLE_RATE,
    lut_size=DAC_LUT_SIZE,
):
    """Realize tone powers using the available DAC range and drive attenuation.

    Parameters
    ----------
    power_target_db : array_like
        Target powers in dB relative to a full-scale DAC sinusoid, after attenuation.
    frequency, phases, mask : array_like
        Baseband Hz, initial phases in radians, and enabled tone rows.
    phase_trials : int, optional
        Number of candidate phase sets, default zero.
    seed : int or None, optional
        Random phase-search seed.
    sample_rate : float, optional
        DAC sample rate in Hz, default 512 MHz.
    lut_size : int, optional
        Waveform sample count, default 2**21.

    Returns
    -------
    dict
        Relative coefficients, explicit Fourier-sum gain, phase-search result,
        selected attenuation and predicted powers before integer quantization.
        Attenuation is the highest setting in [0, 31] dB on the 0.25 dB grid
        that keeps the complex waveform peak within DAC full scale.

    Raises
    ------
    ValueError
        Targets are invalid, collide inconsistently, or exceed DAC headroom.
    """
    targets = np.asarray(power_target_db, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if (
        targets.shape != mask.shape
        or not mask.any()
        or not np.isfinite(targets[mask]).all()
    ):
        msg = "Every enabled tone requires a finite power target"
        raise ValueError(msg)
    amplitude = np.zeros(len(targets))
    target_max = float(np.max(targets[mask]))
    amplitude[mask] = 10 ** ((targets[mask] - target_max) / 20)
    if np.any(amplitude[mask] == 0):
        msg = "Tone power range exceeds floating-point amplitude resolution"
        raise ValueError(msg)
    waveform = optimize_tone_phases(
        frequency,
        amplitude,
        phases,
        mask,
        trials=phase_trials,
        seed=seed,
        sample_rate=sample_rate,
        lut_size=lut_size,
    )
    owners = waveform.owners
    if np.any(~mask[owners][mask]):
        msg = "An enabled output tone is suppressed by a masked DAC-bin owner"
        raise ValueError(msg)
    if not np.allclose(targets[mask], targets[owners][mask], rtol=0, atol=1e-10):
        msg = "Output tones sharing a DAC bin request different powers"
        raise ValueError(msg)
    active = amplitude[owners] * mask[owners]
    writers = np.unique(owners)
    peak_unscaled = np.sqrt(waveform.crest_factor * np.sum(amplitude[writers] ** 2))
    attenuation_limit = -target_max - 20 * np.log10(peak_unscaled)
    roundoff_db = 1e-10
    if not np.isfinite(attenuation_limit) or attenuation_limit < -roundoff_db:
        msg = "Requested tone powers exceed DAC headroom even at zero attenuation"
        raise ValueError(msg)
    # Quantize attenuation downward; the digital gain retains the exact targets.
    atten_drive = float(
        min(31.0, np.floor((attenuation_limit + roundoff_db) / 0.25) * 0.25)
    )
    gain = float(10 ** ((target_max + atten_drive) / 20))
    peak = gain * peak_unscaled
    if not np.isfinite(peak) or peak > 1 + 1e-12:
        msg = (
            "Requested tone powers exceed DAC headroom at selected drive "
            f"attenuation {atten_drive:g} dB (peak {peak:g})"
        )
        raise ValueError(msg)
    normalized = (gain * active) ** 2
    with np.errstate(divide="ignore"):
        delivered = 10 * np.log10(normalized) - atten_drive
    return {
        "amplitudes": amplitude,
        "mask": mask,
        "waveform": waveform,
        "dac_normalization": gain,
        "dac_peak": peak,
        "power_normalized": normalized,
        "atten_drive": float(atten_drive),
        "atten_drive_limit": float(attenuation_limit),
        "power_output_db": delivered,
        "power_error_db": np.where(mask, delivered - targets, np.nan),
    }


@dataclass(frozen=True)
class RoachPhaseResult:
    """Chosen phases and waveform diagnostics from an evaluated comb.

    Attributes
    ----------
    phases : numpy.ndarray
        Final phases in radians, in original row order.
    bins, owners : numpy.ndarray
        Emitted bins and their controlling tone rows.
    crest_factor_initial, crest_factor : float
        Initial and selected peak power divided by mean waveform power.
    selected_trial : int
        Zero retains the supplied phases; positive values select a search trial.
    power_fraction : numpy.ndarray
        Power fractions mapped to rows through emitted-bin ownership.
    """

    phases: np.ndarray
    bins: np.ndarray
    owners: np.ndarray
    crest_factor_initial: float
    crest_factor: float
    selected_trial: int
    power_fraction: np.ndarray


def optimize_tone_phases(  # noqa: PLR0913 -- Complete comb and explicit search parameters.
    frequency,
    amplitudes,
    phases,
    mask=None,
    *,
    trials=32,
    seed=None,
    sample_rate=DAC_SAMPLE_RATE,
    lut_size=DAC_LUT_SIZE,
):
    """Search phases for a lower complex-waveform crest factor.

    Parameters
    ----------
    frequency, amplitudes, phases : array-like
        Final ordered baseband frequencies in Hz, voltage coefficients and
        initial phases in radians. Amplitudes and frequencies remain fixed.
    mask : array-like of bool or None, optional
        Enabled tone rows; None enables all rows.
    trials : int, optional
        Number of random candidates after the initial vector, default 32.
        Zero evaluates the supplied phases without changing them.
    seed : int or None, optional
        NumPy generator seed. Repeatability also requires identical initial phases.
    sample_rate : float, optional
        DAC sample rate in Hz, default 512e6.
    lut_size : int, optional
        Even waveform length, default 2**21.

    Returns
    -------
    RoachPhaseResult
        Best phases and initial/final crest factors. The initial candidate is
        retained unless a trial strictly improves the score. Masked and
        overwritten row phases remain unchanged. Peak normalization is common
        to I and Q, matching the controller before integer sample encoding.

    Raises
    ------
    ValueError
        Inputs are invalid, phases are nonfinite, or no positive bins remain.
    """
    bins = dac_bins(frequency, sample_rate=sample_rate, lut_size=lut_size)
    owners = _owners(bins)
    active = _active_amplitudes(amplitudes, mask, owners)
    phases = np.array(phases, dtype=float, copy=True)
    if phases.shape != active.shape or not np.isfinite(phases).all():
        msg = "Tone phases must be finite and aligned with frequencies"
        raise ValueError(msg)
    if not isinstance(trials, int) or trials < 0:
        msg = "Phase trials must be a nonnegative integer"
        raise ValueError(msg)
    writers = np.unique(owners)
    writers = writers[active[writers] > 0]
    power = active**2
    total = power[writers].sum()
    # A single inverse FFT buffer; no trial-by-sample array or forward FFT.
    spectrum = np.empty(lut_size, dtype=complex)

    def score(candidate):
        spectrum.fill(0)
        spectrum[bins[writers]] = active[writers] * np.exp(1j * candidate[writers])
        wave = ifft(spectrum, overwrite_x=True)
        return float(np.max(np.abs(wave)) ** 2 * lut_size**2 / total)

    initial = best = score(phases)
    selected = 0
    rng = np.random.default_rng(seed)
    candidate = phases.copy()
    for trial in range(1, trials + 1):
        candidate[writers] = rng.uniform(0, 2 * np.pi, len(writers))
        value = score(candidate)
        if value < best:
            best = value
            phases = candidate.copy()
            selected = trial
    return RoachPhaseResult(
        phases, bins, owners, initial, best, selected, power[owners] / total
    )


def detector_tone_power_dbm(
    normalized_power, *, dac_single_tone_power_dbm, drive_path_loss_db, atten_drive_db
):
    """Predict detector-input powers at the acquired drive attenuation.

    Parameters
    ----------
    normalized_power : array-like
        Tone powers relative to one full-scale sinusoid: power_fraction/crest_factor.
    dac_single_tone_power_dbm : float
        Calibrated full-scale single-tone DAC power, in dBm.
    drive_path_loss_db : float
        Fixed net DAC-to-detector loss in dB, excluding drive attenuation.
    atten_drive_db : float
        Acquired drive attenuation in dB; required, with no inferred default.

    Returns
    -------
    numpy.ndarray
        Predicted detector-input dBm; zero-power bins are negative infinity.

    Raises
    ------
    ValueError
        A calibration/attenuation is nonfinite or powers are invalid.
    """
    power = np.asarray(normalized_power, dtype=float)
    if (
        not np.isfinite(
            [dac_single_tone_power_dbm, drive_path_loss_db, atten_drive_db]
        ).all()
        or not np.isfinite(power).all()
        or np.any(power < 0)
    ):
        msg = (
            "Power prediction requires finite calibration, attenuation "
            "and nonnegative powers"
        )
        raise ValueError(msg)
    with np.errstate(divide="ignore"):
        return (
            dac_single_tone_power_dbm
            + 10 * np.log10(power)
            - drive_path_loss_db
            - atten_drive_db
        )


def recover_dac_spectrum(i, q, *, sample_rate=DAC_SAMPLE_RATE, full_scale=32767):
    """Recover the emitted complex DAC coefficients from recorded integer samples.

    Parameters
    ----------
    i, q : array_like
        One chronological, signed-integer DAC waveform period.
    sample_rate : float, optional
        Recorded DAC sampling frequency in Hz.
    full_scale : int, optional
        Recorded positive full-scale count, default 32767.

    Returns
    -------
    frequency, coefficient : numpy.ndarray
        Signed baseband frequencies in Hz and complex Fourier coefficients in
        full-scale units, in FFT order. Amplitude is abs(coefficient); phase is
        angle(coefficient). Quantization spurs are retained. Add the recorded LO
        to obtain RF frequencies. Masked requests and duplicate channel identities
        cannot be recovered from the waveform alone.

    Raises
    ------
    ValueError
        Samples or sampling parameters are invalid.
    """
    from scipy.fft import fft, fftfreq  # noqa: PLC0415 -- inverse of recorded synthesis

    i, q = np.asarray(i), np.asarray(q)
    if (
        i.ndim != 1
        or i.size == 0
        or i.shape != q.shape
        or i.dtype.kind not in "iu"
        or q.dtype.kind not in "iu"
        or not np.isfinite(sample_rate)
        or sample_rate <= 0
        or not np.isfinite(full_scale)
        or full_scale <= 0
    ):
        msg = (
            "Expected equal nonempty integer DAC arrays and positive sampling constants"
        )
        raise ValueError(msg)
    samples = (i.astype(float) + 1j * q.astype(float)) / full_scale
    return fftfreq(i.size, 1 / sample_rate), fft(samples) / i.size
