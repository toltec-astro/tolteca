"""Calculation of TolTEC readout tone powers."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from astropy.table import Table
from loguru import logger
from pydantic import BaseModel, Field
from scipy import fftpack
from scipy.interpolate import interp1d

__all__ = [
    "DACConfig",
    "RoachTonePower",
    "RoachTonePowerChain",
    "RoachTonePowerConfig",
    "RoachTonePowerElement",
    "dac_config_default",
    "tone_amps_phases_to_powers",
    "tone_powers_to_amps",
    "tone_powers_to_amps_phases",
    "transfer_func_lut",
    "transfer_func_null",
]


class DACConfig(BaseModel):
    """Constants of the DAC."""

    V_FS_Volt: float = Field(default=1.2, description="Full-scale voltage in Volt.")
    R_LOAD_Ohm: float = Field(default=50.0, description="Load resistance in Ohm.")
    FFT_size: int = Field(default=2**21, description="FFT size.")
    SAMPLE_FREQ_Hz: float = Field(default=512e6, description="Sample frequency in Hz.")
    BIT_DEPTH: int = Field(default=16, description="DAC bit depth.")
    # FIXED_ATTENUATION_dB: float = Field(
    #    default=35.0, description="Fixed cryogenic attenuation in dB"
    # )
    FIXED_ATTENUATION_dB: float = Field(
        default=0.0,
        description="Fixed cryogenic attenuation in dB",
    )
    # TOTAL_POWER_dBm: float = Field(
    #    default=-8.6, description='Total power output, according to the Link Budget.'
    # )
    TOTAL_POWER_dBm: float = Field(
        default=-12.6,
        description="Total power output, according to the Link Budget.",
    )


dac_config_default = DACConfig.model_validate({})


@dataclass
class RoachTonePowerElement:
    """A single element (device) in the ROACH power chain.

    This represents a node in the power chain. Each element has an operation
    (gain or attenuation) and tracks power levels at its input and output.
    The element is agnostic about whether it's used for predicted or inferred
    calculations.

    Attributes
    ----------
    label : str
        Human-readable label for this element (e.g., "dac", "drive_atten", "lna")
    op_db : float
        Operation in dB. Positive for gain, negative for attenuation.
        output_dbm = input_dbm + op_db
    input_dbm : float | None
        Input power in dBm
    output_dbm : float | None
        Output power in dBm
    prev : RoachTonePowerElement | None
        Previous element in the chain (None for DAC)
    next : RoachTonePowerElement | None
        Next element in the chain (None for ADC)
    """

    label: str
    op_db: float = 0.0
    input_dbm: float | None = None
    output_dbm: float | None = None
    prev: "RoachTonePowerElement | None" = None
    next: "RoachTonePowerElement | None" = None

    @staticmethod
    def get_factory(label: str):
        """Get a factory function for creating elements with a specific label.

        Parameters
        ----------
        label : str
            The label for the element

        Returns
        -------
        callable
            A factory function that creates RoachTonePowerElement with the given label
        """
        from functools import partial

        return partial(RoachTonePowerElement, label=label)

    def calculate_output(self):
        """Calculate output from input using op_db.

        For forward propagation (DAC → ADC).
        """
        if self.input_dbm is not None:
            self.output_dbm = self.input_dbm + self.op_db

    def calculate_input(self):
        """Calculate input from output by reversing op_db.

        For backward propagation (ADC → DAC).
        """
        if self.output_dbm is not None:
            self.input_dbm = self.output_dbm - self.op_db

    def __repr__(self):
        """Return string representation of the element."""
        return (
            f"Element({self.label}: op={self.op_db:+.1f}dB, "
            f"in={self.input_dbm:.1f if self.input_dbm is not None else 'N/A'}, "
            f"out={self.output_dbm:.1f if self.output_dbm is not None else 'N/A'})"
        )


@dataclass
class RoachTonePowerChain:
    """A complete power chain from DAC to ADC.

    This class manages the linked list of RoachTonePowerElement instances
    representing the entire readout chain. The chain can be calculated in
    either direction:
    - Forward (predicted): DAC → ADC
    - Backward (inferred): ADC → DAC

    Power Chain Diagram
    -------------------
    dac → drive_atten → cryo_atten → kids → lna → if_amp → if_board → sense_atten → adc

    Use the `from_config` factory method to create a chain with operation values
    set from a RoachTonePowerConfig instance.

    Attributes
    ----------
    dac : RoachTonePowerElement
        DAC element
    drive_atten : RoachTonePowerElement
        Drive attenuation element
    cryo_atten : RoachTonePowerElement
        Cryo cable attenuation element
    kids : RoachTonePowerElement
        KIDs element
    lna : RoachTonePowerElement
        LNA element
    if_amp : RoachTonePowerElement
        IF amplifier element
    if_board : RoachTonePowerElement
        IF board element
    sense_atten : RoachTonePowerElement
        Sense attenuation element
    adc : RoachTonePowerElement
        ADC element
    elements : list[RoachTonePowerElement]
        List of all elements in the chain, in order from DAC to ADC
    """

    # Individual element instances (created via default_factory)
    dac: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("dac"),
    )
    drive_atten: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("drive_atten"),
    )
    cryo_atten: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("cryo_atten"),
    )
    kids: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("kids"),
    )
    lna: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("lna"),
    )
    if_amp: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("if_amp"),
    )
    if_board: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("if_board"),
    )
    sense_atten: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("sense_atten"),
    )
    adc: RoachTonePowerElement = field(
        default_factory=RoachTonePowerElement.get_factory("adc"),
    )

    # List composed from the element instances
    elements: list[RoachTonePowerElement] = field(init=False)

    def __post_init__(self):
        """Link elements into chain."""
        # Compose the list from element instances
        self.elements = [
            self.dac,
            self.drive_atten,
            self.cryo_atten,
            self.kids,
            self.lna,
            self.if_amp,
            self.if_board,
            self.sense_atten,
            self.adc,
        ]

        # Set up prev/next pointers
        for i, elem in enumerate(self.elements):
            if i > 0:
                elem.prev = self.elements[i - 1]
            if i < len(self.elements) - 1:
                elem.next = self.elements[i + 1]

    @classmethod
    def from_config(
        cls,
        config: "RoachTonePowerConfig",
        roach: int,
    ) -> "RoachTonePowerChain":
        """Create a power chain from configuration.

        Parameters
        ----------
        config : RoachTonePowerConfig
            Configuration with all power chain parameters
        roach : int
            Roach/network number (needed to get IF board gain)

        Returns
        -------
        RoachTonePowerChain
            A new power chain instance with operation values set from config
        """
        # Create instance with default elements
        chain = cls()

        # Set operation values for each element
        # DAC and ADC have op_db = 0 (already set by default)
        chain.drive_atten.op_db = -config.drive_atten_db  # Attenuation is negative
        chain.cryo_atten.op_db = -config.cryo_atten_db  # Attenuation is negative
        chain.kids.op_db = -config.kids_loss_db  # Loss is negative
        chain.lna.op_db = config.lna_gain_db  # Gain is positive
        chain.if_amp.op_db = config.if_amp_gain_db  # Gain is positive
        chain.if_board.op_db = config.get_if_board_gain_db(roach)  # Gain is positive
        chain.sense_atten.op_db = -config.sense_atten_db  # Attenuation is negative

        return chain

    def propagate_forward(self, dac_output_dbm: float):
        """Propagate power forward from DAC to ADC.

        Parameters
        ----------
        dac_output_dbm : float
            DAC output power in dBm (starting point)
        """
        # Set DAC output
        self.dac.output_dbm = dac_output_dbm

        # Propagate through chain
        for elem in self.elements:
            # Set input from previous element's output
            if elem.prev is not None:
                elem.input_dbm = elem.prev.output_dbm
            else:
                # DAC is the starting point
                elem.input_dbm = elem.output_dbm

            # Calculate output
            elem.calculate_output()
        return self

    def propagate_backward(self, adc_input_dbm: float):
        """Propagate power backward from ADC to DAC.

        Parameters
        ----------
        adc_input_dbm : float
            ADC input power in dBm (starting point)
        """
        # Set ADC input
        self.adc.input_dbm = adc_input_dbm

        # Propagate backwards through chain
        for elem in reversed(self.elements):
            # Set output from next element's input
            if elem.next is not None:
                elem.output_dbm = elem.next.input_dbm
            else:
                # ADC is the end point
                elem.output_dbm = elem.input_dbm

            # Calculate input
            elem.calculate_input()
        return self

    def __repr__(self):
        """Return string representation of the power chain."""
        return f"RoachTonePowerChain(n_elements={len(self.elements)})"

    def pformat(self, title=None):
        """Pretty format of the chain elements."""
        lines = [f"RoachTonePowerChain: {title}" if title else "RoachTonePowerChain:"]
        lines.extend(
            f"  {elem.label:12s}: "
            f"op={elem.op_db:+6.1f} dB, "
            f"in={elem.input_dbm if elem.input_dbm is not None else ' N/A':>6}, "
            f"out={elem.output_dbm if elem.output_dbm is not None else ' N/A':>6}"
            for elem in self.elements
        )
        return "\n".join(lines)


class RoachTonePowerConfig(BaseModel):
    """Configuration constants for ROACH tone power calculations.

    This class holds the fixed constants and default values for the TolTEC readout
    power chain. The chain follows the signal path:

    dac → drive_atten → cryo_atten → kids → lna → if_amp → if_board → sense_atten → adc

    Power propagates forward (predicted) from DAC to ADC, or backward (inferred)
    from measured ADC values to estimate power at each stage.
    """

    # DAC power (measured value from observations)
    dac_power_dbm: float = Field(
        # default=-15.1,
        default=-12.6,  # according to SA results 2025
        description="DAC output power in dBm",
    )

    # User-controllable drive attenuation
    drive_atten_db: float = Field(
        default=0.0,
        description="Default drive attenuation in dB (user-controllable)",
    )

    # Cryogenic system constants
    cryo_atten_db: float = Field(
        # default=35.0,
        default=0.0,  # according to SA results 2025
        description="Cryogenic cable attenuation in dB",
    )
    kids_loss_db: float = Field(
        default=2.0,
        description="Loss at KIDs in dB",
    )

    # Amplifier gains (in forward order)
    lna_gain_db: float = Field(
        default=30.0,
        description="Low-noise amplifier gain in dB",
    )
    if_amp_gain_db: float = Field(
        default=27.0,
        description="IF amplifier gain in dB (updated from 30 dB)",
    )

    # IF board gains for roaches 0-12 (measured 2023-01-28)
    if_board_gains_db_by_roach: dict[int, float] = Field(
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
        description="IF board gains in dB for roaches 0-12",
    )

    # User-controllable sense attenuation
    sense_atten_db: float = Field(
        default=0.0,
        description="Default sense attenuation in dB (user-controllable)",
    )

    # ADC conversion (private - use adc2_to_adc_dbm() method)
    adc_conversion_offset_db: float = Field(
        default=162.0,
        description="ADC units² to dBm conversion offset in dB",
    )

    # Safety limits
    lna_input_dbm_max: float = Field(
        default=-52.0,
        description="Maximum safe LNA input power in dBm",
    )
    lna_output_dbm_max: float = Field(
        default=-15.0,
        description="Maximum safe LNA output power in dBm",
    )
    if_board_input_dbm_max: float = Field(
        default=-5.0,
        description="Maximum safe IF board input power in dBm",
    )
    adc_snap_frac_min: float = Field(
        default=0.30,
        description="Minimum ADC snap fraction (0-1) for good dynamic range",
    )

    def get_if_board_gain_db(self, roach: int) -> float:
        """Get IF board gain for a specific roach/network number.

        Parameters
        ----------
        roach : int
            Roach/network number (0-12)

        Returns
        -------
        float
            IF board gain in dB
        """
        if roach not in self.if_board_gains_db_by_roach:
            raise ValueError(
                f"Roach number {roach} not found. "
                f"Available roaches: {sorted(self.if_board_gains_db_by_roach.keys())}",
            )
        return self.if_board_gains_db_by_roach[roach]

    def adc2_to_adc_dbm(self, adc2: float | np.ndarray) -> float | np.ndarray:
        """Convert ADC units² to power in dBm.

        This performs the unit conversion from raw ADC measurements (in ADC units²)
        to calibrated power levels (in dBm).

        Parameters
        ----------
        adc2 : float or np.ndarray
            ADC signal power in ADC units² (I² + Q²)

        Returns
        -------
        float or np.ndarray
            Power in dBm
        """
        return 10.0 * np.log10(adc2) - self.adc_conversion_offset_db


def transfer_func_null():
    """Return a function that adjust the power per tone.

    This does not do adjustment.
    """

    def func(*_kwargs):
        return 0.0

    return func


def transfer_func_lut(flo_Hz, lut_file):
    """Return a function that adjust the power per tone.

    This uses the LUT to adjust the power so the appear unifrom
    at the IF.
    """
    lut = Table.read(
        lut_file,
        format="ascii.no_header",
        delimiter=",",
        names=["f_Hz", "amplitude"],
    )
    interp_lut = interp1d(lut["f_Hz"], lut["amplitude"], kind="cubic")

    def func(tone_comb_freq_Hz, **_kwargs):
        amp = interp_lut(tone_comb_freq_Hz + flo_Hz)
        offset = 20 * np.log10(amp)
        # make sure the offsets adds up to 0
        return offset - np.sum(offset) / len(offset)

    return func


def tone_powers_to_amps(  # noqa: PLR0915
    tone_comb_freqs_Hz,
    tone_powers_dBm,
    tone_phases_rad,
    transfer_func=None,
    drive_attens=None,
    dac_config=dac_config_default,
):
    """Calculate DAC amplitudes and expected KID tone powers.

    Parameters
    ----------
    tone_comb_freqs_Hz : array-like
        The tone comb frequencies in Hz (not LO scaled).
    tone_powers_dbm : array-like
        The tone powers requested at the KIDS in dBm.
    tone_phases_rad : array-like
        The tone phases in radians.
    transfer_func: callable, optional
        The transfer function to assume. This evaluates to dB offset.
    drive_attens_dB : array-like
        Additional driving attenuations for each tone.
    dac_config : DACConfig
        The DAC config.

    Returns
    -------
    tone_prop_table : astropy.table.Table
        A table contains the properties of the tones.
    atten_global_dB : float
        A overall attenuation value to set for all tones.
    """
    logger.debug(f"use DAC config: {dac_config}")

    # Generate a list of dictionaries of tone properties.
    tpt = Table()
    tpt["comb_freq_Hz"] = tone_comb_freqs_Hz
    tpt["power_dBm_requested"] = tone_powers_dBm
    tpt["phase_rad"] = tone_phases_rad
    if drive_attens is None:
        drive_attens = 0.0
        # drive_attens_norm = 0
    else:
        # normalized offsets
        # drive_atten_norm = np.sum(drive_attens) / len(drive_attens)
        pass
    # drive_offsets = drive_attens - drive_attens_norm
    tpt["drive_atten_dB"] = drive_attens
    # tpt['drive_offset_dB'] = drive_offsets
    if transfer_func is None:
        transfer_func = transfer_func_null()
    tpt["transfer_offset_dB"] = transfer_func(tone_comb_freqs_Hz, dac_config)

    # tpt['dac_atten_fixed_dB'] = dac_config.FIXED_ATTENUATION_dB

    # Calculate required amplitude for each tone, taking into account
    # the attenuations and various offsets.
    dac2det_atten = (
        dac_config.FIXED_ATTENUATION_dB
        + tpt["drive_atten_dB"]
        + tpt["transfer_offset_dB"]
    )

    power_dBm_dac_req = tpt["power_dBm_requested"] + dac2det_atten
    power_mW_dac_req = 10 ** (power_dBm_dac_req / 10.0)
    amps = tpt["amplitude"] = np.sqrt(
        2 * power_mW_dac_req / 1000 * dac_config.R_LOAD_Ohm,
    )
    tpt["amplitude_scaled"] = amps / np.max(amps)
    amplitude_scale_offset = 20 * np.log10(np.max(amps))
    logger.debug(f"{amplitude_scale_offset=}")

    Pdac_total_mW_req = np.sum(power_mW_dac_req)
    Pdac_total_dBm_req = 10 * np.log10(Pdac_total_mW_req)

    # This shouldn't ever be the case, but let's add a total power
    # check to make sure we're not asking for more power than the DAC
    # can deliver.
    # max_power_watt = 28e-3  # 28mW (assumption by GW)
    Pdac_max_dBm = dac_config.TOTAL_POWER_dBm
    logger.debug(f"{Pdac_total_dBm_req=:3.2f} dBm {Pdac_max_dBm=:3.2f} dBm")
    if Pdac_total_dBm_req > Pdac_max_dBm:
        P_offset_factor = (Pdac_max_dBm - Pdac_total_dBm_req) / len(tpt)
        logger.warning(
            f"total requested power {Pdac_total_dBm_req:3.2f} dBm "
            f"exceeds max DAC power: {Pdac_max_dBm:3.2f} dBm, "
            f"offset power by {P_offset_factor} dB per tone.",
        )
        # return the function with new power setting
        return tone_powers_to_amps(
            tone_comb_freqs_Hz,
            tone_powers_dBm + P_offset_factor,
            tone_phases_rad,
            transfer_func=None,
            drive_attens=None,
            dac_config=dac_config_default,
        )

    # A helper function for the waveform construction that follows
    def fft_bin_idx(freq):
        return int(round(freq / dac_config.SAMPLE_FREQ_Hz * dac_config.FFT_size))
        # return int(freq / dac_config.SAMPLE_FREQ_Hz * dac_config.FFT_size)

    # Generate the time domain waveforms (both I and Q)
    spec = np.zeros(dac_config.FFT_size, dtype=complex)

    for f, a, ph in tpt.iterrows("comb_freq_Hz", "amplitude", "phase_rad"):
        spec[fft_bin_idx(f)] = a * np.exp(1.0j * ph)
    wave = np.fft.ifft(spec)
    waveform_I = wave.real
    waveform_Q = wave.imag

    V_FS_Volt = 2.8 * np.sqrt(
        2 * 10 ** (dac_config.TOTAL_POWER_dBm / 10) / 1000 * dac_config.R_LOAD_Ohm,
    )
    logger.debug(f"use {V_FS_Volt=}")
    # Rescale the waveforms so that their peak matches the full-scale DAC voltage
    waveform_I *= V_FS_Volt / np.max(np.abs(waveform_I))
    waveform_Q *= V_FS_Volt / np.max(np.abs(waveform_Q))

    # Compute the FFT to get back to the frequency domain
    spectrum = fftpack.fft(waveform_I + 1.0j * waveform_Q, n=dac_config.FFT_size)

    # Compute power of each output tone in dBm
    Pdac_total_mW_actual = 0.0
    dac_powers_dBm_actual = []
    powers_dBm_actual = []
    for i, f in enumerate(tpt["comb_freq_Hz"]):
        atten = dac2det_atten[i]
        idx = fft_bin_idx(f)
        a = np.abs(spectrum[idx]) / dac_config.FFT_size
        p_mW = (a**2) / (2 * dac_config.R_LOAD_Ohm) * 1000
        p_dbm = 10 * np.log10(p_mW)
        ap_dbm = p_dbm - atten
        Pdac_total_mW_actual += p_mW
        dac_powers_dBm_actual.append(p_dbm)
        powers_dBm_actual.append(ap_dbm)
    tpt["DAC_power_dBm_actual"] = dac_powers_dBm_actual
    tpt["power_dBm_actual"] = powers_dBm_actual
    Pdac_total_dBm_actual = 10 * np.log10(Pdac_total_mW_actual)
    logger.debug(f"{Pdac_total_dBm_actual=}")

    # Compute required drive attenuation
    power_differences = tpt["power_dBm_actual"] - tpt["power_dBm_requested"]
    required_attenuation = min(power_differences)

    # Round up to nearest 0.25 dB increment to ensure none of the
    # signals exceed the requested power
    required_attenuation = np.ceil(required_attenuation / 0.25) * 0.25

    # correction for the amps norm
    # required_attenuation_adjusted = required_attenuation + drive_atten_offset

    # add some metadata
    tpt.meta["dac_config"] = dac_config.model_dump()
    tpt.meta["Pdac_total_mW_requested"] = Pdac_total_mW_req
    tpt.meta["Pdac_total_dBm_requested"] = Pdac_total_dBm_req
    tpt.meta["Pdac_total_mW_actual"] = Pdac_total_mW_actual
    tpt.meta["Pdac_total_dBm_actual"] = Pdac_total_dBm_actual
    tpt.meta["atten_required"] = required_attenuation
    # tpt.meta['atten_required_adjusted'] = required_attenuation_adjusted

    pdiff = np.abs(
        tpt["power_dBm_requested"] - (tpt["power_dBm_actual"] - required_attenuation),
    )
    mpdiff = pdiff.mean()
    tpt.meta["mean_power_diff_dB"] = mpdiff
    return tpt, required_attenuation


def tone_powers_to_amps_phases(
    tone_comb_freqs_Hz,
    tone_powers_dBm,
    n_waveforms=20,
    random_seed=None,
    **kwargs,
):
    """Return amps and phases for given tone powers."""
    # Let's run this for N different sets of random phases and then choose
    # the phases that result in the lowest request-actual average.
    rng = np.random.default_rng(random_seed)
    best_phases = None
    best_match = np.inf
    for _ in range(n_waveforms):
        ph = rng.uniform(0.0, 2.0 * np.pi, size=len(tone_comb_freqs_Hz))
        tpt, _ = tone_powers_to_amps(
            tone_comb_freqs_Hz=tone_comb_freqs_Hz,
            tone_powers_dBm=tone_powers_dBm,
            tone_phases_rad=ph,
            **kwargs,
        )
        m = tpt.meta["mean_power_diff_dB"]
        if m < best_match:
            best_match = m
            best_phases = ph
        logger.debug(f"mean power difference: {m} dB")
    logger.debug(f"minimum power difference: {best_match} dB")
    # run again with the best phases
    return tone_powers_to_amps(
        tone_comb_freqs_Hz=tone_comb_freqs_Hz,
        tone_powers_dBm=tone_powers_dBm,
        tone_phases_rad=best_phases,
        **kwargs,
    )


def tone_amps_phases_to_powers(
    tone_comb_freqs_Hz,  # noqa: ARG001
    tone_amps,
    tone_phases,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
):
    """Convert tone amps and phases to powers."""
    # Compute the tone powers from the amps and phases
    return 10 * np.log10((tone_amps**2) / (2 * 50)) + 30


@dataclass
class RoachTonePower:
    """Calculate and track power levels through the ROACH readout chain.

    This class encapsulates all power calculations from DAC through the cryogenic
    components to ADC, tracking attenuation, gains, and power levels at each stage.

    The power chain is:
    DAC -> Drive Atten -> Cryo Cable Atten -> KIDs -> LNA -> Cryo Cable ->
    IF Input Amp -> IF Board -> Sense Atten -> ADC

    Parameters
    ----------
    roach : int
        Roach/network number (0-12) for IF board gain lookup
    tone_f_combs_Hz : array-like
        Tone comb frequencies in Hz
    tone_amps : array-like
        Tone amplitudes (unnormalized)
    tone_phases_rad : array-like
        Tone phases in radians
    adc_snap : array-like
        ADC snap block data (shape: [2, N] for two ADC channels)
    Is : array-like
        I channel data (shape: [n_samples, n_tones])
    Qs : array-like
        Q channel data (shape: [n_samples, n_tones])
    config : RoachTonePowerConfig
        Configuration object with all power chain parameters including
        drive_atten_db, sense_atten_db, dac_power_dbm, etc.

    Attributes
    ----------
    config : RoachTonePowerConfig
        Configuration - single source of truth for all parameters
    """

    # Input parameters
    roach: int
    tone_f_combs_Hz: np.ndarray = field(repr=False)
    tone_amps: np.ndarray = field(repr=False)
    tone_phases_rad: np.ndarray = field(repr=False)
    adc_snap: np.ndarray = field(repr=False)
    Is: np.ndarray = field(repr=False)
    Qs: np.ndarray = field(repr=False)
    config: RoachTonePowerConfig = field(
        default_factory=lambda: RoachTonePowerConfig.model_validate({}),
    )

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert inputs to numpy arrays
        self.tone_f_combs_Hz = np.asarray(self.tone_f_combs_Hz)
        self.tone_amps = np.asarray(self.tone_amps)
        self.tone_phases_rad = np.asarray(self.tone_phases_rad)
        self.adc_snap = np.asarray(self.adc_snap)
        self.Is = np.asarray(self.Is, dtype=float)
        self.Qs = np.asarray(self.Qs, dtype=float)

        self._predicted_power_chain_per_tone = (
            self._build_power_chain().propagate_forward(
                self.P_dac_dbm_per_tone,
            )
        )
        self._predicted_power_chain_total = self._build_power_chain().propagate_forward(
            self.P_dac_dbm_total,
        )

        self._inferred_power_chain_per_tone = (
            self._build_power_chain().propagate_backward(self.P_adc_dbm_per_tone)
        )
        self._inferred_power_chain_total = self._build_power_chain().propagate_backward(
            self.P_adc_dbm_total,
        )

    def _build_power_chain(self) -> RoachTonePowerChain:
        """Build the power chain.

        Creates a RoachTonePowerChain instance using the factory method.

        Returns
        -------
        RoachTonePowerChain
            A new power chain instance with operation values set from config
        """
        return RoachTonePowerChain.from_config(self.config, self.roach)

    @property
    def predicted_per_tone(self):
        """Get predicted per-tone power chain."""
        return self._predicted_power_chain_per_tone

    @property
    def predicted_total(self):
        """Get predicted total power chain."""
        return self._predicted_power_chain_total

    @property
    def inferred_per_tone(self):
        """Get inferred per-tone power chain."""
        return self._inferred_power_chain_per_tone

    @property
    def inferred_total(self):
        """Get inferred total power chain."""
        return self._inferred_power_chain_total

    @cached_property
    def P_dac_dbm_per_tone(self):
        """Calculate per-tone power at the DAC in dBm.

        Returns
        -------
        np.ndarray
            Per-tone power in dBm
        """
        P_total = self.P_dac_dbm_total
        amps = self.tone_amps
        # distribute total power according to amps squared
        return P_total + 10.0 * np.log10(amps**2) - 10.0 * np.log10(np.sum(amps**2))

    @cached_property
    def P_dac_dbm_total(self):
        """Total power at the DAC in dBm.

        Returns
        -------
        float
            Total power in dBm
        """
        return self.config.dac_power_dbm

    @cached_property
    def P_adc_dbm_per_tone(self):
        """Calculate per-tone power at the ADC in dBm.

        Returns
        -------
        np.ndarray
            Per-tone power in dBm
        """
        adc2 = self.Is**2 + self.Qs**2
        adc2_mean = adc2.mean(axis=-1)
        return self.config.adc2_to_adc_dbm(adc2_mean)

    @cached_property
    def P_adc_dbm_total(self):
        """Calculate total power at the ADC in dBm.

        Returns
        -------
        float
            Total ADC power in dBm
        """
        Ps = self.P_adc_dbm_per_tone
        Ps_linear = 10.0 ** (Ps / 10.0)
        Ps_total = Ps_linear.sum()
        return 10.0 * np.log10(Ps_total)

    @cached_property
    def adc_snap_frac(self):
        """Calculate ADC snap block dynamic range usage fraction.

        Returns
        -------
        float
            Fraction of ADC full scale range used
        """
        # Convert from 16 bits to 12 bits
        x0 = self.adc_snap[0].view(np.int16) / 16
        x1 = self.adc_snap[1].view(np.int16) / 16
        r0 = (x0.max() - x0.min()) / 2**12
        r1 = (x1.max() - x1.min()) / 2**12
        return np.array([r0, r1]).mean()

    @cached_property
    def tone_amps_db(self):
        """Calculate tone amplitudes in dB.

        Returns
        -------
        np.ndarray
            Tone amplitudes in dB
        """
        tone_amps = self.tone_amps
        return 20.0 * np.log10(tone_amps.max() / tone_amps)

    def validate_power_levels(self):
        """Validate power levels against safe operating ranges.

        Returns
        -------
        dict
            Dictionary with validation results and warnings
        """
        # Use inferred power chain for validation
        chain = self.get_power_chain_adc_to_dac

        warnings = []

        # Check LNA input power
        lna_in = chain.lna.input_dbm
        if lna_in > self.config.lna_input_dbm_max:
            warnings.append(
                f"LNA input power {lna_in:.1f} dBm exceeds "
                f"{self.config.lna_input_dbm_max} dBm limit",
            )

        # Check LNA output power
        lna_out = chain.lna.output_dbm
        if lna_out > self.config.lna_output_dbm_max:
            warnings.append(
                f"LNA output power {lna_out:.1f} dBm exceeds "
                f"{self.config.lna_output_dbm_max} dBm limit",
            )

        # Check IF board input power
        if_board_in = chain.if_board.input_dbm
        if if_board_in > self.config.if_board_input_dbm_max:
            warnings.append(
                f"IF board input power {if_board_in:.1f} dBm exceeds "
                f"{self.config.if_board_input_dbm_max} dBm limit",
            )

        # Check ADC snap fraction
        if self.adc_snap_frac < self.config.adc_snap_frac_min:
            warnings.append(
                f"ADC snap fraction {self.adc_snap_frac:.1f}% is below "
                f"{self.config.adc_snap_frac_min * 100:.1f}% minimum",
            )

        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "lna_input_dbm": lna_in,
            "lna_output_dbm": lna_out,
            "if_board_input_dbm": if_board_in,
            "adc_snap_frac_pct": self.adc_snap_frac,
        }

    def __repr__(self):
        """Return string representation of the RoachTonePower instance."""
        return (
            f"RoachTonePower(roach={self.roach}, "
            f"n_dets={len(self.tone_f_combs_Hz)}, "
            f"atten_drive={self.config.drive_atten_db:.1f}dB, "
            f"atten_sense={self.config.sense_atten_db:.1f}dB, "
            f"dac_power={self.P_dac_dbm_total:.1f}dBm, "
            f"adc_power={self.P_adc_dbm_total:.1f}dBm)"
        )

    @classmethod
    def from_kidsdata(cls, kidsdata):
        """Create RoachTonePower instance from kidsdata.

        Parameters
        ----------
        kidsdata : KidsData
            KIDs data containing tone and ADC information

        Returns
        -------
        RoachTonePower
            Power calculator instance
        """
        roach = kidsdata.meta["roach"]
        atten_drive_db = kidsdata.meta["atten_drive"]
        atten_sense_db = kidsdata.meta["atten_sense"]
        tbl_chan = kidsdata.meta["chan_axis_data"]
        tone_f_combs = tbl_chan["f_tone"]
        tone_amps = tbl_chan["amp_tone"]
        tone_phases = tbl_chan["phase_tone"]
        adc_snap = kidsdata.meta["adc_snap"]

        Is = kidsdata.I.value
        Qs = kidsdata.Q.value
        # Is = kidsdata.meta["I_raw"]
        # Qs = kidsdata.meta["Q_raw"]
        config = RoachTonePowerConfig.model_validate(
            {
                "drive_atten_db": atten_drive_db,
                "sense_atten_db": atten_sense_db,
            },
        )
        return cls(
            roach=roach,
            tone_f_combs_Hz=tone_f_combs.to_value("Hz"),
            tone_amps=tone_amps,
            tone_phases_rad=tone_phases.to_value("rad"),
            adc_snap=adc_snap,
            Is=Is,
            Qs=Qs,
            config=config,
        )

    def pformat(self):
        """Pretty-format the power chain results."""
        lines = [
            f"ROACH Tone Power (roach={self.roach}):",
            f"  ADC Power: {self.P_adc_dbm_total:.2f} dBm",
            f"  ADC Snap Fraction: {self.adc_snap_frac:.2%}",
        ]
        lines.append(self.inferred_total.pformat(title="inferred total (ADC → DAC)"))
        lines.append(self.predicted_total.pformat(title="predicted total (DAC → ADC)"))
        return "\n".join(lines)

    def make_plotly_figure(self):
        """Create a Plotly figure visualizing the power chain."""
        import plotly.graph_objects as go

        labels = [elem.label for elem in self.predicted_total.elements]
        predicted_outputs = [elem.output_dbm for elem in self.predicted_total.elements]
        inferred_inputs = [elem.input_dbm for elem in self.inferred_total.elements]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=predicted_outputs,
                mode="lines+markers",
                name="Predicted Output (DAC → ADC)",
                line={"color": "blue"},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=inferred_inputs,
                mode="lines+markers",
                name="Inferred Input (ADC → DAC)",
                line={"color": "red"},
            ),
        )

        fig.update_layout(
            title="ROACH Tone Power Chain",
            xaxis_title="Power Chain Element",
            yaxis_title="Power (dBm)",
            legend_title="Legend",
        )

        return fig
