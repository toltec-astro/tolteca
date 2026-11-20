"""Prepare ROACH probe tones from matched KIDs detections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.table import QTable, vstack
from pydantic import ConfigDict, Field, model_validator
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsFilePath, FrequencyQuantityField
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
)

from .kids_find import KidsFind
from .power_setup import (
    PowerSetupConfig,
    prepare_acquired_tones,
    prepare_drive_fit_tones,
)
from .roach_tone import (
    RoachToneLut,
)
from .sweep import ReducedSweepView
from .tlaloc_output import (
    TLALOC_REPORT_COLUMNS,
    TlalocToneArtifact,
    TlalocTones,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata


class TonePhasesRandom(FrozenBaseModel):
    """Random phases in radians; a seed makes the prepared result repeatable."""

    method: Literal["random"]
    seed: int | None = None

    def __call__(self, n_chans: int) -> np.ndarray:
        """Generate one random phase per requested tone.

        Parameters
        ----------
        n_chans : int
            Number of output tones.

        Returns
        -------
        numpy.ndarray
            Phases uniformly distributed over [0, 2 pi), in radians.
        """
        return np.random.default_rng(self.seed).random(n_chans) * (2 * np.pi)


class TonePhasesFromFile(FrozenBaseModel):
    """Phases in radians from a headerless single-column file."""

    method: Literal["file"]
    path: AbsFilePath

    def __call__(self, n_chans: int) -> np.ndarray:
        """Read the first n_chans phases from the configured file.

        Parameters
        ----------
        n_chans : int
            Number of output tones.

        Returns
        -------
        numpy.ndarray
            Loaded phases in radians, in file order.

        Raises
        ------
        ValueError
            The file contains fewer phases than requested.
        """
        phases = QTable.read(self.path, format="ascii.no_header")["col1"]
        if len(phases) < n_chans:
            msg = "Too few phases in file"
            raise ValueError(msg)
        return np.asarray(phases[:n_chans])


class PlaceHolderTones(FrozenBaseModel):
    """V2 placeholder placement around the detected frequency range."""

    n: int = Field(default=0, ge=0)
    loc: Literal["inner", "outer"] = "outer"
    sep_f_lo_min: FrequencyQuantityField = 5 * u.MHz
    sep_f_lo_max: FrequencyQuantityField = 500 * u.MHz
    sep_f_det: FrequencyQuantityField = 100 * u.kHz
    f_step: FrequencyQuantityField = 50 * u.kHz

    def __call__(self, f_dets, f_lo):
        """Place placeholder tones around detections using the v2 spacing.

        Parameters
        ----------
        f_dets : astropy.units.Quantity
            Absolute detection frequencies, with detections on both sides of
            the LO when placeholders are requested.
        f_lo : astropy.units.Quantity
            Center LO frequency.

        Returns
        -------
        astropy.units.Quantity
            Absolute placeholder frequencies in Hz; empty when n is zero.
        """
        if self.n == 0:
            return np.array([]) * u.Hz
        negative, positive = f_dets[f_dets < f_lo], f_dets[f_dets >= f_lo]
        n_neg = self.n // 2
        n_pos = self.n - n_neg
        if self.loc == "inner":
            neg0 = np.max(negative) + self.sep_f_det
            neg1 = min(neg0 + (n_neg - 1) * self.f_step, f_lo - self.sep_f_lo_min)
            pos1 = np.min(positive) - self.sep_f_det
            pos0 = max(pos1 - (n_pos - 1) * self.f_step, f_lo + self.sep_f_lo_min)
        else:
            neg1 = np.min(negative) - self.sep_f_det
            neg0 = max(neg1 - (n_neg - 1) * self.f_step, f_lo - self.sep_f_lo_max)
            pos0 = np.max(positive) + self.sep_f_det
            pos1 = min(pos0 + (n_pos - 1) * self.f_step, f_lo + self.sep_f_lo_max)
        return (
            np.r_[
                np.linspace(neg0.to_value(u.Hz), neg1.to_value(u.Hz), n_neg),
                np.linspace(pos0.to_value(u.Hz), pos1.to_value(u.Hz), n_pos),
            ]
            * u.Hz
        )


class TonePowerArtifact(ArtifactSpec[TlalocTones]):
    """Prepared tone powers, matching decisions and exact drive-fit provenance."""

    def write(self, data: TlalocTones, *, path: Path) -> list[ArtifactFile]:
        """Write the retained per-tone power table as tone_power.ecsv.

        Parameters
        ----------
        data : TlalocTones
            Prepared tones with fitted targets and power diagnostics.
        path : pathlib.Path
            Artifact directory. Existing tone_power.ecsv files are rejected.

        Returns
        -------
        list of tollan.pipeline.ArtifactFile
            One ECSV receipt with per-tone decisions and reference provenance.
        """
        table = QTable()
        for name, column in data.data.data_vars.items():
            table[name] = column.values.copy()
            if "units" in column.attrs:
                table[name] = table[name] * u.Unit(column.attrs["units"])
        table.meta.update(data.data.attrs)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "tone_power.ecsv"
        table.write(target, format="ascii.ecsv")
        return [ArtifactFile(target, "ecsv", name="tone_power")]


class TonePrepArtifacts(ArtifactSpec[ArtifactContext]):
    """Prepared tone files retained for ordinary data-product publication."""

    tones: TlalocToneArtifact = Field(default_factory=TlalocToneArtifact)
    power: TonePowerArtifact = Field(default_factory=TonePowerArtifact)

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export the retained tone table to the supplied product directory.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Prepared tone artifacts with accessible table values.
        path : pathlib.Path
            Product export directory; existing tone files are rejected.

        Returns
        -------
        list of tollan.pipeline.ArtifactFile
            Frequency, amplitude, phase, mask and attenuation receipts, plus
            the power table when a drive-fit reference was used.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(artifact.data, path=path)
        ]


class TonePhaseOptimizationConfig(FrozenBaseModel):
    """Optional bounded phase search at fixed amplitudes and attenuation.

    Parameters
    ----------
    trials : int, optional
        Random candidates in addition to the supplied phases, default 32.
        The supplied phases remain eligible, so the crest factor cannot worsen.
    seed : int or None, optional
        Search seed. Use fixed initial phases as well for repeatable results.
    """

    trials: int = Field(default=32, ge=1)
    seed: int | None = None


class TonePrepConfig(PowerSetupConfig, StepConfig):
    """Select probe tones and prepare their amplitudes and phases.

    Parameters
    ----------
    enabled : bool, optional
        Enable preparation, default False. Requires a completed KidsFind step.
    n_chans_max : int, optional
        Maximum output tones, default 1000. Excess detections retain highest Qr.
    fix_n_chans : bool or None, optional
        Preserve the input count, trimming by absolute match phase distance or
        filling from missed probes. None uses v2 observing defaults: fixed for
        Tune, variable for standalone VNA and target sweeps.
    tone_amps_method : {'match', 'ones'}, optional
        Inherit matched input amplitudes (default), or assign unity amplitudes
        to detected tones. With a drive-fit reference, ``match`` instead uses
        the fitted power targets; ``ones`` conflicts with that selection.
    tone_amps_fallback : {'median', 'lut'} or None, optional
        None (default) preserves v2 amplitude assignments, including ambiguous
        matches. An explicit policy replaces every ambiguous or unavailable
        match. ``median`` uses enabled input amplitudes. ``lut`` uses
        ``tone_amps_lut_root/toltecN/amps_correction_lut.csv`` in absolute RF Hz,
        scaled by the median input/LUT amplitude ratio in dB at unambiguous probes.
        Missing LUTs or usable scale anchors fail preparation before output.
        A common coefficient normalization keeps the final amplitudes in [0, 1].
        Without a drive-fit reference, targets retain the measured comb power
        scale. Target realization jointly calculates waveform scale and drive
        attenuation for the complete new comb.
        With a reference, fallback is required for tones without a unique valid
        fit match: median uses valid fitted targets; LUT uses its RF shape,
        anchored to those targets. Identified detectors keep their fitted target.
        The complete comb determines amplitudes and common drive attenuation.
        Installing saved tones does not evaluate the policy again.
    tone_amps_lut_root : pathlib.Path or None, optional
        Calibration directory containing the per-network LUTs. Required when
        LUT fallback fills amplitudes; independent of the installation path.
    tone_phases : TonePhasesRandom or TonePhasesFromFile, optional
        Random phases by default. Phases are assigned before the common row sort.
    phase_optimization : TonePhaseOptimizationConfig or None, optional
        None (default) retains the original phases. When configured, search on
        the final quantized, masked comb for a lower peak/mean power ratio.
        The selected targets remain fixed during phase search. The result records
        predicted waveform scale and joint drive attenuation.
    dac : RoachDacConfig, optional
        Firmware sampling constants; defaults to the TolTEC ROACH configuration.
    power : RoachTonePowerConfig or None, optional
        Optional absolute power calibration. Requires recorded drive attenuation.
        None retains targets in DAC-relative dB, without an absolute dBm conversion.
    drive_fit_reference : pathlib.Path or None, optional
        Published drive fit supplying target powers for every new comb. Powers
        include waveform peak normalization; calibrated dBm requires line calibration.
        The exact reference, target powers, fallback choices and explicit DAC
        normalization are retained in tone_power.ecsv. Additional
        reference and matching fields are defined by PowerSetupConfig.
    placeholders : PlaceHolderTones, optional
        Extra masked tones with zero amplitudes; none by default.
    artifacts : TonePrepArtifacts, optional
        Tone-file and optional drive-fit power-table export specifications.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    n_chans_max: int = Field(default=1000, ge=1, le=1000)
    fix_n_chans: bool | None = None
    tone_amps_method: Literal["match", "ones"] = "match"
    tone_amps_fallback: Literal["median", "lut"] | None = None
    tone_phases: Annotated[
        TonePhasesRandom | TonePhasesFromFile, Field(discriminator="method")
    ] = Field(default_factory=lambda: TonePhasesRandom(method="random"))
    phase_optimization: TonePhaseOptimizationConfig | None = None
    placeholders: PlaceHolderTones = Field(default_factory=PlaceHolderTones)
    artifacts: TonePrepArtifacts = Field(default_factory=TonePrepArtifacts)

    @model_validator(mode="after")
    def _check_drive_fit_policy(self):
        if self.drive_fit_reference is not None and self.tone_amps_method == "ones":
            msg = (
                "Pinned drive-fit targets cannot be combined with unity tone amplitudes"
            )
            raise ValueError(msg)
        return self


class TonePrepContext(StepContext["TonePrep", TonePrepConfig]):
    """Prepared tones, preparation settings and artifacts."""

    data: TlalocTones | None = Field(default=None, exclude=True)


class TonePrep(Step[TonePrepConfig, TonePrepContext]):
    """Prepare matched probe tones for the next ROACH readout.

    The tlaloc subtree retains the final common row order. Data-product output
    exports its tone files; TlalocOutput optionally installs them afterwards.

    Parameters
    ----------
    config : TonePrepConfig or None, optional
        Tone selection, amplitude and phase settings. None uses defaults.
    **kwargs
        TonePrepConfig overrides, following the Step constructor.
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: TonePrepContext) -> None:
        """Prepare one checked and found native or reopened sweep block.

        Parameters
        ----------
        data : xarray.DataTree
            Block with recorded probe metadata and a completed KidsFind context.
        context : TonePrepContext
            Tone selection, amplitude, phase and artifact settings, and output.

        Returns
        -------
        None
            Installs a loaded tlaloc subtree and retains its typed artifact.

        Raises
        ------
        ValueError
            Finding is incomplete, metadata is missing, or no tones are prepared.
        """
        found = KidsFind.get_context(data)
        if not found.completed:
            msg = "Tone preparation requires completed KIDs finding"
            raise ValueError(msg)
        view = ReducedSweepView(data)
        if view.meta is None:
            msg = "Tone preparation requires acquisition metadata"
            raise ValueError(msg)
        table = prepare_tone_table(
            found.data.detected,
            found.data.chan_matched,
            view.chan_axis_table,
            view.meta,
            context.config,
            query_matches=found.data.detected_matched,
        )
        data["tlaloc"] = table
        context.data = TlalocTones(table)
        context.artifacts = ArtifactContext()
        context.artifacts.add(context.config.artifacts.tones, context.data)
        context.artifacts.add(context.config.artifacts.power, context.data)


def prepare_tone_table(  # noqa: PLR0913 -- science inputs and output policies
    detected: QTable,
    chan_matched: QTable,
    probes: QTable,
    meta: ToltecSweepMetadata,
    config: TonePrepConfig,
    *,
    query_matches: QTable | None = None,
) -> xr.Dataset:
    """Select and order tones using the v2 output policy.

    Parameters
    ----------
    detected, chan_matched : astropy.table.QTable
        Finding detections and input-probe matches, including d_phi and amp_tone.
    probes : astropy.table.QTable
        Recorded chan, f_chan, tone_amp and tone_mask columns in acquisition order.
    meta : tolteca_datamodels.toltec.metadata.ToltecSweepMetadata
        Acquisition identity, network, observation type and LO frequency in Hz.
    config : TonePrepConfig
        Tone selection, amplitude policy, LUT source and phase-generation settings.
    query_matches : astropy.table.QTable or None, optional
        One row per detection, in detection order, with idx_chan and connected
        match-group size. Required for explicit fallback in match mode.

    Returns
    -------
    xarray.Dataset
        Loaded v2 report and companion columns in the same FFT order.
    """
    if meta.f_lo_center is None:
        msg = "Tone preparation requires a recorded LO frequency"
        raise ValueError(msg)
    f_lo = meta.f_lo_center * u.Hz
    tones = detected.copy()
    f_dets = tones["f_chan"] = tones["f"] if "f" in tones.colnames else tones["fr"]
    tones["mask_tone"] = True
    unique_probe_ids = _amplitude_matches(tones, query_matches, config)
    probes_acquired = probes
    probes, fixed, limit = _tone_capacity(probes, meta, config)
    if len(tones) > limit:
        ranking = (
            np.abs(tones["d_phi"]).argsort()
            if fixed
            else np.flip(tones["Qr"].argsort())
        )
        keep = np.zeros(len(tones), dtype=bool)
        keep[ranking[:limit]] = True
        tones = tones[keep]
    amp_provenance = _initial_tone_amplitudes(
        tones, probes, unique_probe_ids, meta.roach, config
    )
    placeholders = config.placeholders(f_dets, f_lo)
    if fixed and len(tones) < len(probes):
        # Match indices refer to enabled probes; map their IDs back to the full
        # input table before selecting missed channels, including masked tones.
        distances = dict(
            zip(chan_matched["idx_chan"], np.abs(chan_matched["d_phi"]), strict=True)
        )
        scores = [
            distances.get(chan, np.inf * u.deg).to_value(u.deg)
            for chan in probes["chan"]
        ]
        missing = np.zeros(len(probes), dtype=bool)
        missing[np.argsort(scores)[-(len(probes) - len(tones)) :]] = True
        prior = probes[missing]
        fill = QTable(
            {
                "f_chan": prior["f_chan"],
                "mask_tone": prior["tone_mask"],
                "amp_tone": prior["tone_amp"],
                "amp_tone_matched": prior["tone_amp"],
                "amp_tone_filled": np.zeros(len(prior), dtype=bool),
                "amp_tone_ambiguous": np.zeros(len(prior), dtype=bool),
                "f": prior["f_chan"],
                "Qr": np.zeros(len(prior)),
                "bitmask": np.zeros(len(prior), dtype=int),
            }
        )
        tones = vstack([tones, fill])
    elif not fixed:
        count = min(len(placeholders), limit - len(tones))
        fill = QTable(
            {
                "f_chan": placeholders[:count],
                "mask_tone": np.zeros(count, dtype=bool),
                "amp_tone": np.zeros(count),
                "amp_tone_matched": np.zeros(count),
                "amp_tone_filled": np.zeros(count, dtype=bool),
                "amp_tone_ambiguous": np.zeros(count, dtype=bool),
            }
        )
        tones = vstack([tones, fill])
    if len(tones) == 0:
        msg = "No tones available for Tlaloc output"
        raise ValueError(msg)
    tones["f_tone"] = tones["f_chan"] - f_lo
    tones["phase_tone"] = config.tone_phases(len(tones))
    tones.sort("f_chan")
    f_tone = tones["f_tone"]
    # Preserve the production writer's FFT permutation for every column.
    offset = 3 * np.max(f_tone)
    order = sorted(
        range(len(tones)),
        key=lambda i: f_tone[i] + offset if f_tone[i] < 0 else f_tone[i],
    )
    tones = tones[order]
    return _finish_tone_dataset(tones, meta, config, amp_provenance, probes_acquired)


def _tone_capacity(probes, meta, config):
    """Reserve adjustment slots, reusing those recorded by a preceding upload."""
    fixed = config.fix_n_chans
    if fixed is None:
        if meta.obs_type not in {"tune", "vna", "targ"}:
            msg = "Automatic tone count requires a recorded sweep observation type"
            raise ValueError(msg)
        fixed = meta.obs_type == "tune"
    limit = len(probes) if fixed else config.n_chans_max
    if limit > config.n_chans_max:
        msg = "Fixed input tone count exceeds n_chans_max"
        raise ValueError(msg)
    return probes, fixed, limit


def _initial_tone_amplitudes(tones, probes, unique_probe_ids, roach, config):
    """Keep the probe fallback separate from pinned-fit power assignment."""
    if config.drive_fit_reference is None:
        tones["amp_tone"], tones["amp_tone_filled"], amp_provenance = (
            fill_tone_amplitudes(
                tones["amp_tone"],
                tones["f_chan"].to_value(u.Hz),
                probes,
                fill_mask=tones["amp_tone_ambiguous"],
                unique_probe_ids=unique_probe_ids,
                roach=roach,
                config=config,
            )
        )
    else:
        # Frequency selection precedes power assignment. Do not use a probe's
        # inherited coefficient or LUT fallback before matching the pinned fit.
        tones["amp_tone"] = 1.0
        tones["amp_tone_filled"] = False
        amp_provenance = {"tone_amps_method": "drive_fit"}
    return amp_provenance


def fill_tone_amplitudes(  # noqa: PLR0913 -- output rows and input-probe anchors
    amplitudes,
    frequencies,
    probes,
    *,
    fill_mask,
    unique_probe_ids,
    roach,
    config,
):
    """Prepare relative amplitudes at fixed global attenuation.

    Parameters
    ----------
    amplitudes : array-like
        Matched voltage coefficients in output detection order.
    frequencies : array-like
        Absolute output RF frequencies in Hz, in the same order.
    probes : astropy.table.QTable
        Acquisition columns chan, f_chan (Hz), tone_amp and tone_mask.
    fill_mask : array-like of bool
        All detections in ambiguous match groups. Nonfinite or masked amplitude
        values also require filling when a fallback policy is enabled.
    unique_probe_ids : array-like
        Original channel IDs in one-probe/one-detection match groups. Positive,
        enabled input amplitudes at these IDs anchor the LUT scale.
    roach : int
        Network number selecting the LUT below the configured calibration root.
    config : TonePrepConfig
        Amplitude policy and LUT source directory.

    Returns
    -------
    amplitudes : numpy.ndarray
        Relative coefficients. LUT-filled coefficients can exceed one; the
        caller must normalize the entire selected comb by one common factor.
    filled : numpy.ndarray
        Rows assigned fallback values; false for v2 and unity policies.
    provenance : dict
        Applied policy and, when used, LUT identity, scale and anchor count.

    Raises
    ------
    ValueError
        Inputs are invalid, or the requested fallback has no usable population.
    FileNotFoundError
        A required LUT is missing. No alternative policy is silently substituted.
    """
    amps = np.ma.asarray(amplitudes, dtype=float).filled(np.nan).copy()
    frequency = np.asarray(frequencies, dtype=float)
    filled = np.asarray(fill_mask, dtype=bool).copy()
    if amps.ndim != 1 or frequency.shape != amps.shape or filled.shape != amps.shape:
        msg = "Tone amplitudes, frequencies and fill mask must have the same 1-D shape"
        raise ValueError(msg)
    provenance = {
        "tone_amps_method": config.tone_amps_method,
        "tone_amps_fallback": config.tone_amps_fallback,
    }
    filled |= ~np.isfinite(amps)
    if config.tone_amps_method == "ones":
        amps[:] = 1.0
        filled[:] = False
    elif config.tone_amps_fallback is None:
        filled[:] = False
    elif filled.any():
        prior = np.ma.asarray(probes["tone_amp"], dtype=float).filled(np.nan)
        enabled = np.asarray(probes["tone_mask"], dtype=bool)
        if config.tone_amps_fallback == "median":
            valid = enabled & np.isfinite(prior) & (prior >= 0)
            if not valid.any():
                msg = (
                    "Median amplitude fallback requires finite enabled input amplitudes"
                )
                raise ValueError(msg)
            amps[filled] = np.median(prior[valid])
        else:
            amps[filled], lut_provenance = _lut_fallback(
                frequency[filled], probes, unique_probe_ids, roach, config
            )
            provenance.update(lut_provenance)
    if not np.isfinite(amps).all() or np.any(amps < 0) or np.any(amps[~filled] > 1):
        msg = "Invalid matched or filled tone amplitudes"
        raise ValueError(msg)
    provenance["tone_amps_filled_count"] = int(filled.sum())
    return amps, filled, provenance


def _lut_fallback(frequency, probes, unique_probe_ids, roach, config):
    """Evaluate Tlaloc's spline in the input comb's relative amplitude scale."""
    prior = np.ma.asarray(probes["tone_amp"], dtype=float).filled(np.nan)
    enabled = np.asarray(probes["tone_mask"], dtype=bool)
    if config.tone_amps_lut_root is None:
        msg = "LUT amplitude fallback requires tone_prep.tone_amps_lut_root"
        raise ValueError(msg)
    path = config.tone_amps_lut_root / f"toltec{roach}" / "amps_correction_lut.csv"
    lut = RoachToneLut.from_file(path)

    def evaluate(f):
        return lut(f * u.Hz)

    anchor = (
        enabled
        & np.isfinite(prior)
        & (prior > 0)
        & np.isin(probes["chan"], unique_probe_ids)
    )
    if not anchor.any():
        msg = "LUT amplitude scale requires positive unambiguous input probes"
        raise ValueError(msg)
    input_frequency = probes["f_chan"].to_value("Hz")
    ratio_db = 20 * np.log10(prior[anchor] / evaluate(input_frequency[anchor]))
    offset_db = float(np.median(ratio_db))
    scale = 10 ** (offset_db / 20)
    values = scale * evaluate(frequency)
    provenance = {
        "tone_amps_lut_path": str(path),
        "tone_amps_lut_sha256": lut.digest,
        "tone_amps_lut_samples": {
            "frequency_hz": lut.frequency.to_value(u.Hz).tolist(),
            "amplitude": lut.amplitude.tolist(),
        },
        "tone_amps_lut_scale": scale,
        "tone_amps_lut_offset_db": offset_db,
        "tone_amps_lut_anchor_count": int(anchor.sum()),
    }
    return values, provenance


def _amplitude_matches(tones, query_matches, config):
    """Mark ambiguity at the output boundary without changing the matcher."""
    if (
        config.tone_amps_method == "match"
        and config.tone_amps_fallback is not None
        and config.drive_fit_reference is None
        and query_matches is None
    ):
        msg = "Amplitude fallback requires detection-to-probe match groups"
        raise ValueError(msg)
    unique_probe_ids = []
    ambiguous = np.zeros(len(tones), dtype=bool)
    if query_matches is not None:
        if len(query_matches) != len(tones):
            msg = "Amplitude match rows must align with detections"
            raise ValueError(msg)
        # Connected group size counts both probe and detection nodes. Only a
        # two-node group is unambiguous; no member of a larger group inherits.
        unique = np.asarray(query_matches["size"]) == 2  # noqa: PLR2004 -- one probe and one detection
        ambiguous = ~unique
        unique_probe_ids = query_matches["idx_chan"][unique]
    tones["amp_tone_matched"] = np.ma.asarray(tones["amp_tone"], dtype=float).filled(
        np.nan
    )
    tones["amp_tone_ambiguous"] = ambiguous
    return unique_probe_ids


def _finish_tone_dataset(tones, meta, config, amp_provenance, probes):
    """Prepare per-tone power targets and evaluate their waveform realization."""
    result = _tone_dataset(tones, meta)
    result.attrs.update(amp_provenance)
    diagnostic = (
        prepare_drive_fit_tones(result, config)
        if config.drive_fit_reference is not None
        else prepare_acquired_tones(result, probes, meta, config)
    )
    result.update(diagnostic)
    result.attrs.update(diagnostic.attrs)
    return result


def _tone_dataset(tones, meta):
    """Prepare the next comb without attaching acquired-channel model coefficients."""

    def values(column):
        data = getattr(column, "value", column)
        return np.ma.asarray(data, dtype=float).filled(np.nan)

    columns = {
        "f_centered": tones["f_tone"].to_value(u.Hz),
        "f_out": tones["f_chan"].to_value(u.Hz),
        "flag": values(tones["flag"] if "flag" in tones.colnames else tones["bitmask"]),
    }
    report_columns = list(TLALOC_REPORT_COLUMNS)
    columns.update(
        {
            name: np.asarray(tones[name])
            for name in ("amp_tone", "phase_tone", "mask_tone")
        }
    )
    for name in (
        "tone_role",
        "amp_tone_matched",
        "amp_tone_filled",
        "amp_tone_ambiguous",
    ):
        if name in tones.colnames:
            columns[name] = np.asarray(tones[name])
    return xr.Dataset(
        {name: ("tone", column) for name, column in columns.items()},
        attrs={
            "kind": "tones",
            "schema": "kids.tone_prop/1",
            "block": meta.block,
            "roach": meta.roach,
            "f_lo_center": meta.f_lo_center,
            "atten_drive": meta.atten_drive,
            "report_columns": report_columns,
            "Header.Toltec.LoCenterFreq": meta.f_lo_center,
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
        },
    )
