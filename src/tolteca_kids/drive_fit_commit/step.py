"""Tone matching and amplitude commit from a saved drive fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.table import QTable
from pydantic import ConfigDict, Field
from tollan.config.types import FrequencyQuantityField  # noqa: TC002 -- Pydantic field.
from tollan.pipeline import Step, StepConfig, StepContext

from tolteca_kids.drive_fit.computation import _read_sweep
from tolteca_kids.drive_fit.power import (
    POWER_MODEL,
    dac_bin_owners,
    tone_power_fractions,
)
from tolteca_kids.drive_fit.result import DriveFitResult
from tolteca_kids.match1d import Match1D

from .artifacts import DriveFitCommitArtifacts

if TYPE_CHECKING:
    from tolteca_kids.drive_fit.computation import _Sweep


@dataclass
class DriveFitCommitResult:
    """Per-target drive settings and persisted matching diagnostics.

    Parameters
    ----------
    data : xarray.Dataset
        Commit arrays indexed by recorded target channel. Reopened arrays may
        be lazy; retain the product's storage until values are loaded.
    """

    data: xr.Dataset

    @property
    def table(self) -> QTable:
        """Loaded target-channel settings and match diagnostics, with units."""
        table = QTable({"idx_chan": self.data.chan.values})
        for name, array in self.data.data_vars.items():
            if array.dims != ("chan",):
                continue
            table[name] = array.values.copy()
            if "units" in array.attrs:
                table[name] = table[name] * u.Unit(array.attrs["units"])
        table.meta.update(self.data.attrs)
        return table

    @property
    def n_success(self) -> int:
        """Number of target channels using an accepted drive selection."""
        return int(np.sum(~self.data.amp_flag.values))


class DriveFitCommitStepConfig(StepConfig):
    """Tone-matching and amplitude-normalization settings.

    Parameters
    ----------
    match : Match1D, optional
        Shared KIDs tone matcher; defaults to ``dtw_python``.
    match_shift_max : frequency quantity, optional
        Largest global frequency shift, default 10 MHz. The shift grid spacing
        is half the target sweep's frequency step, as in the production recipe.
    adrv_max : float, optional
        Upper accepted drive attenuation in dB, default 30 (exclusive).
    adrv_fill_perc : float, optional
        Percentile used for unsuccessful channels, default 50.
    adrv_ref_perc : float, optional
        Percentile selecting the global reference attenuation, default 10.
    adrv_ref_step : float, optional
        Downward quantization of global attenuation in dB, default 0.25.
    artifacts : DriveFitCommitArtifacts, optional
        Scientific, table and instrument-settings artifact specifications.
    """

    model_config = ConfigDict(extra="forbid")
    match: Match1D = Field(default_factory=Match1D)
    match_shift_max: FrequencyQuantityField = Field(default=10 * u.MHz)
    adrv_max: float = Field(default=30.0, allow_inf_nan=False)
    adrv_fill_perc: float = Field(default=50.0, ge=0, le=100)
    adrv_ref_perc: float = Field(default=10.0, ge=0, le=100)
    adrv_ref_step: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    artifacts: DriveFitCommitArtifacts = Field(default_factory=DriveFitCommitArtifacts)


class DriveFitCommitStepContext(
    StepContext["DriveFitCommitStep", DriveFitCommitStepConfig]
):
    """Commit execution record and retained scientific result."""

    data: DriveFitCommitResult | None = Field(default=None, exclude=True)


class DriveFitCommitStep(Step[DriveFitCommitStepConfig, DriveFitCommitStepContext]):
    """Match one target sweep to a pinned drive-fit product and prepare settings."""

    @classmethod
    def view(cls, data: xr.DataTree) -> DriveFitCommitResult:
        """Access live or reopened commit arrays.

        Parameters
        ----------
        data : xarray.DataTree
            Interface containing a drive_fit_commit result.

        Returns
        -------
        DriveFitCommitResult
            Scientific view without reopening raw inputs or repeating matching.
        """
        return DriveFitCommitResult(data["drive_fit_commit"].to_dataset())

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitCommitStepContext) -> None:
        """Compute commit settings and install their scientific artifacts.

        Parameters
        ----------
        data : xarray.DataTree
            Interface with exactly one ``inputs/source`` and a loaded saved fit
            at ``reference``. The reference product identity is retained in attrs.
        context : DriveFitCommitStepContext
            Matching and normalization settings and output context.

        Returns
        -------
        None
            Results are installed under drive_fit_commit and exposed on context.

        Raises
        ------
        ValueError
            The target is ambiguous, has another network, or has no valid fitted
            attenuation for determining amplitude corrections.
        """
        if "drive_fit_commit" in data.children:
            del data["drive_fit_commit"]
        sources = data["inputs"].to_dataset()["source"].values
        if len(sources) != 1:
            msg = "Drive-fit commit requires exactly one target sweep per interface"
            raise ValueError(msg)
        target = _read_sweep(str(sources[0]), None)
        reference = DriveFitResult(data["reference"].to_dataset())
        if target.meta.roach != reference.data.attrs["roach"]:
            msg = "Target and drive-fit reference must use the same readout network"
            raise ValueError(msg)
        result = cls.compute(target, reference, context.config)
        result.data.attrs["reference_product"] = data.attrs["reference_product"]
        data["drive_fit_commit"] = result.data
        data["tlaloc"] = xr.Dataset(
            {"amp_tone": result.data.amp_best, "f_chan": result.data.f_chan},
            attrs={
                "roach": target.meta.roach,
                "f_lo_center": target.meta.f_lo_center,
                "atten_drive": result.data.attrs["atten_drive_global"],
            },
        )
        context.data = cls.view(data)
        context.artifacts = context.config.artifacts(data, context.data)

    @staticmethod
    def compute(
        target: _Sweep, reference: DriveFitResult, config: DriveFitCommitStepConfig
    ) -> DriveFitCommitResult:
        """Match channels and compute settings without managing a step context.

        Parameters
        ----------
        target : prepared sweep
            Loaded target frequencies in Hz and channel amplitudes.
        reference : DriveFitResult
            Existing fit selections and channel metadata.
        config : DriveFitCommitStepConfig
            Explicit matcher and normalization settings.

        Returns
        -------
        DriveFitCommitResult
            Loaded settings and matching diagnostics in original target order.
        """
        ref = reference.table
        if reference.data.attrs.get("power_model") != POWER_MODEL:
            msg = "Commit requires a fit with full-comb fixed-total power fractions"
            raise ValueError(msg)
        spacing = (target.frequency[0, 1] - target.frequency[0, 0]) * u.Hz
        if not np.isfinite(spacing.value) or spacing.value <= 0:
            msg = "Commit matching requires a positive target sweep frequency step"
            raise ValueError(msg)
        matched = config.match(
            query=(target.f_chan * u.Hz).to(u.MHz),
            ref=ref["f_chan"].to(u.MHz),
            shift_kw={"shift_max": config.match_shift_max, "dx": spacing / 2},
        )
        pairs = matched.data["query_matched"]
        iq, ir = pairs["idx_query"], pairs["idx_ref"]
        if not np.array_equal(iq, np.arange(len(target.chan))):
            msg = "Tone matcher did not cover every target channel"
            raise ValueError(msg)
        values = {
            "f_chan": target.f_chan,
            "amp_tone": target.amp,
            "idx_chan_adrv": np.asarray(ref["idx_chan"])[ir],
            "flag_adrv": np.asarray(ref["p_flag"])[ir],
            "f_chan_adrv": ref["f_chan"][ir].to_value(u.Hz),
            "amp_chan_adrv": np.asarray(ref["amp_chan"])[ir],
            "adrv_best_adrv": np.asarray(ref["adrv_best"])[ir],
            "a_best_adrv": np.asarray(ref["a_best"])[ir],
            "Qr_adrv": np.asarray(ref["Qr_best"])[ir],
            "dist": pairs["dist"].to_value(u.Hz),
            "dist_shifted": pairs["dist_shifted"].to_value(u.Hz),
            "idx_ref": np.asarray(ir),
            "power_fraction_adrv": reference.data.power_fraction.values[ir],
        }
        # Preserve the recipe's quantity arithmetic and operation order.
        rr = 0.5 / values["Qr_adrv"]
        xx = pairs["dist"] / pairs["ref"]
        values["d_phi"] = np.rad2deg(np.arctan2(xx, rr)).value
        axis = target.meta.chan_axis_table
        if axis is None:
            msg = "Commit requires the target's complete recorded tone axis"
            raise ValueError(msg)
        correction = compute_ampcor(
            values["adrv_best_adrv"],
            values["flag_adrv"],
            values["power_fraction_adrv"],
            output_frequency=axis["f_tone"].to_value(u.Hz),
            output_mask=axis["tone_mask"],
            adrv_max=config.adrv_max,
            fill_percentile=config.adrv_fill_perc,
            reference_percentile=config.adrv_ref_perc,
            reference_step=config.adrv_ref_step,
        )
        values.update(
            {
                "adrv_ref": np.full(len(iq), correction["adrv_ref"]),
                "adrv_ref_total": np.full(len(iq), correction["adrv_ref_total"]),
                "adrv_best": correction["adrv_filled"],
                "amp_best": correction["ampcor"],
                "amp_flag": ~correction["m_good"],
                "power_policy_db": correction["power_policy_db"],
                "power_commit_db": correction["power_commit_db"],
                "power_error_db": correction["power_error_db"],
            }
        )
        data = xr.Dataset(
            {name: ("chan", array) for name, array in values.items()},
            coords={"chan": target.chan},
            attrs={
                "schema": "tolteca_kids.drive_fit_commit",
                "schema_version": 1,
                "interface": target.meta.interface,
                "roach": target.meta.roach,
                "target_source": target.source,
                "target_uid": target.meta.uid_raw_obs_file,
                "atten_drive_global": correction["adrv_ref_total"],
                "atten_drive_exact": correction["adrv_exact"],
                "power_model": POWER_MODEL,
                "power_reference": (
                    "relative to total DAC power; after drive attenuation"
                ),
                "match_shift_hz": float(matched.shift.to_value(u.Hz)),
                "config": config.model_dump(mode="json", exclude={"artifacts"}),
            },
        )
        for name in ("f_chan", "f_chan_adrv", "dist", "dist_shifted"):
            data[name].attrs["units"] = "Hz"
        data["d_phi"].attrs["units"] = "deg"
        for name in ("power_policy_db", "power_commit_db", "power_error_db"):
            data[name].attrs["units"] = "dB"
        return DriveFitCommitResult(data)


def compute_ampcor(  # noqa: PLR0913 -- Explicit independent numerical settings.
    attenuation,
    flags,
    power_fractions,
    *,
    output_frequency,
    output_mask=None,
    adrv_max,
    fill_percentile,
    reference_percentile,
    reference_step,
):
    """Compute commit settings under the fixed-total DAC power approximation.

    Parameters
    ----------
    attenuation, flags, power_fractions : array-like
        Matched reference attenuation in dB, selection flags and reference DAC
        power fractions in target-channel order. Fractions must be normalized
        over the full reference comb before matching or channel selection.
    output_frequency : array-like
        Baseband output frequencies in Hz, in the unchanged target tone order.
    output_mask : array-like of bool or None, optional
        Output tone masks. None enables all tones.
    adrv_max : float
        Exclusive upper accepted attenuation in dB.
    fill_percentile, reference_percentile : float
        Percentiles for unsuccessful channels and the global reference.
    reference_step : float
        Downward quantization step of the final global attenuation in dB.

    Returns
    -------
    dict
        Filled attenuation, replacement amplitudes, exact and quantized global
        attenuation, accepted mask and predicted power relative to DAC total.
        Downward rounding retains a power excess smaller than reference_step.

    Raises
    ------
    ValueError
        Inputs are invalid, no accepted attenuations remain, an emitted tone
        lacks reference power, or tones sharing a DAC bin request different powers.
    """
    attenuation = np.asarray(attenuation, dtype=float)
    fractions = np.asarray(power_fractions, dtype=float)
    flags = np.asarray(flags)
    owners = dac_bin_owners(output_frequency)
    mask = (
        np.ones(len(owners), dtype=bool)
        if output_mask is None
        else np.asarray(output_mask, bool)
    )
    if not (
        attenuation.shape
        == flags.shape
        == fractions.shape
        == owners.shape
        == mask.shape
    ):
        msg = "Commit arrays must be aligned one-dimensional target-channel arrays"
        raise ValueError(msg)
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        msg = "Reference power fractions must be finite and between zero and one"
        raise ValueError(msg)
    good = (np.asarray(flags) == 0) & (attenuation < adrv_max)
    values = attenuation[good].copy()
    if not len(values) or not np.isfinite(values).all():
        msg = "No valid drive attenuations available for commit"
        raise ValueError(msg)
    emitted = mask[owners]
    if not emitted.any() or np.any(fractions[emitted] <= 0):
        msg = "Every emitted output tone requires positive reference power"
        raise ValueError(msg)
    values[values < 0] = 0.0
    reference = np.percentile(values, reference_percentile).item()
    fill = np.percentile(values, fill_percentile).item()
    filled = attenuation.copy()
    filled[good] = values
    filled[~good] = fill
    filled[filled < reference] = reference
    adjust = 10 ** ((reference - filled) / 20.0)
    correction = np.sqrt(fractions) * adjust
    # A masked row's coefficient is irrelevant to synthesis; retain a finite value.
    correction[~emitted] = 1.0
    norm = np.max(correction)
    correction = correction / norm
    with np.errstate(divide="ignore"):
        requested = 10 * np.log10(fractions) - filled
    if not np.allclose(
        requested[emitted], requested[owners][emitted], rtol=0, atol=1e-10
    ):
        msg = "Output tones sharing a DAC bin request different powers"
        raise ValueError(msg)
    delivered_fraction = tone_power_fractions(correction, output_frequency, mask)
    with np.errstate(divide="ignore"):
        dac_power = 10 * np.log10(delivered_fraction)
    exact_values = dac_power[emitted] - requested[emitted]
    exact = float(exact_values[0])
    if not np.allclose(exact_values, exact, rtol=0, atol=1e-10):
        msg = "The output comb cannot satisfy the requested relative tone powers"
        raise ValueError(msg)
    total = _quantize_attenuation(exact, reference_step)
    delivered = dac_power - total
    residual = np.full(len(owners), np.nan)
    residual[emitted] = delivered[emitted] - requested[emitted]
    return {
        "adrv_ref": reference,
        "adrv_ref_total": total,
        "adrv_exact": exact,
        "adrv_filled": filled,
        "ampcor": correction,
        "m_good": good,
        "power_policy_db": requested,
        "power_commit_db": delivered,
        "power_error_db": residual,
    }


def _quantize_attenuation(value, step):
    """Round down without a roundoff step; reject unattainable negative settings."""
    steps = value / step
    nearest = np.rint(steps)
    if np.isclose(steps, nearest, rtol=0, atol=1e-12):
        steps = nearest
    result = float(np.floor(steps) * step)
    if result < 0:
        msg = "Requested output power exceeds the fixed DAC total"
        raise ValueError(msg)
    return result
