"""Independent power references for acquired and proposed readout combs."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import xarray as xr
from pydantic import Field
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsAnyPath, FrequencyQuantityField

from .match1d import Match1D
from .roach_tone import (
    POWER_MODEL,
    RoachDacConfig,
    RoachToneLut,
    RoachTonePowerConfig,
    detector_tone_power_dbm,
    optimize_tone_phases,
    tone_waveform_attenuation,
)


class PowerSetupConfig(FrozenBaseModel):
    """Power calibration and reference selection owned by one consumer.

    Parameters
    ----------
    dac : RoachDacConfig, optional
        Recorded-comb DAC sampling constants.
    power : RoachTonePowerConfig or None, optional
        Full-scale single-tone power and drive-line loss for absolute prediction.
        This calibration is separate from the relative drive-fit power reference.
    tone_amps_lut_root : pathlib.Path or None, optional
        Directory containing toltecN/amps_correction_lut.csv. LUT comparisons
        remove a common amplitude scale; they do not measure absolute power.
    drive_fit_reference : pathlib.Path or None, optional
        Published drive-fit product, index or version group. Resolved once per
        consumer call and retained as an exact product reference.
    drive_fit_alias : str or None, optional
        Explicit fit invocation. None selects the primary fit.
    match : Match1D, optional
        Existing frequency-pattern matcher. Ambiguous matches remain unmatched.
    match_shift_max, match_shift_step : frequency quantity, optional
        Pattern-shift search limit and grid, default 10 MHz and 1 kHz, matching
        the configured ToneMatch workflow. The shift aligns reference to query.
    match_n_fwhm : float, optional
        Maximum shifted distance in reference linewidths, default 1.
    """

    dac: RoachDacConfig = Field(default_factory=RoachDacConfig)
    power: RoachTonePowerConfig | None = None
    tone_amps_lut_root: AbsAnyPath | None = None
    drive_fit_reference: AbsAnyPath | None = None
    drive_fit_alias: str | None = None
    match: Match1D = Field(default_factory=Match1D)
    match_shift_max: FrequencyQuantityField = 10 * u.MHz
    match_shift_step: FrequencyQuantityField = 1 * u.kHz
    match_n_fwhm: float = Field(default=1.0, gt=0, allow_inf_nan=False)


def _load_reference(config, roach):
    # Product and workflow modules depend on the top-level Kids configuration.
    from .drive_fit.product import (  # noqa: PLC0415 -- configuration cycle.
        open_drive_fit_product,
        read_drive_fit_context,
    )
    from .drive_fit.step import DriveFitStep  # noqa: PLC0415 -- configuration cycle.
    from .product import _product_index  # noqa: PLC0415 -- configuration cycle.

    index = _product_index(config.drive_fit_reference)
    if (
        index.product.meta.data_prod_type != "dp_drivefit"
        or index.namespace is None
        or index.run is None
    ):
        msg = "Power setup requires a published drive-fit realization"
        raise ValueError(msg)
    with open_drive_fit_product(index, interface=f"toltec{roach}") as saved:
        context = read_drive_fit_context(saved, alias=config.drive_fit_alias)
        if not context.completed:
            msg = "Power reference requires a completed drive fit"
            raise ValueError(msg)
        data = DriveFitStep.view(saved, alias=config.drive_fit_alias).data.load()
    if data.attrs["roach"] != roach or data.attrs.get("power_model") != POWER_MODEL:
        msg = (
            "Power reference must use the same network and peak-normalized power model"
        )
        raise ValueError(msg)
    return data, index.product.reference.model_dump(mode="json")


def compare_drive_fit(frequency, normalized, attenuation, reference, config):
    """Compare full-comb peak-normalized powers with an already loaded reference.

    Parameters
    ----------
    frequency, normalized : array_like
        Absolute Hz and full-comb peak-normalized powers in current channel order.
    attenuation : float
        Recorded or proposed global drive attenuation in dB.
    reference : xarray.Dataset
        DriveFitResult.data with full-comb normalized powers for fitted channels.
    config : PowerSetupConfig
        Matching settings. Only unique, close, successful matches are compared.

    Returns
    -------
    dict
        Arrays of reference identities, match distances, statuses and current
        minus reference power in dB. Invalid comparisons have NaN power deltas.
    """
    if reference.attrs.get("power_model") != POWER_MODEL:
        msg = "Drive-fit comparison requires peak-normalized reference powers"
        raise ValueError(msg)
    result = config.match(
        query=(np.asarray(frequency) * u.Hz).to(u.MHz),
        ref=(reference.f_chan.values * u.Hz).to(u.MHz),
        shift_kw={"shift_max": config.match_shift_max, "dx": config.match_shift_step},
    )
    matched = result.data["query_matched"]
    if not np.array_equal(matched["idx_query"], np.arange(len(frequency))):
        msg = "Power matcher must preserve the current channel order"
        raise ValueError(msg)
    idx = np.asarray(matched["idx_ref"], int)
    distance = matched["dist_shifted"].to_value(u.Hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        width = reference.f_chan.values[idx] / reference.Qr_best.values[idx]
    status = np.full(len(idx), "matched", dtype="U32")
    status[
        np.asarray(matched["size"]) != 2  # noqa: PLR2004 -- One query and one reference.
    ] = "ambiguous"
    status[
        ~np.isfinite(width)
        | (width <= 0)
        | (np.abs(distance) > config.match_n_fwhm * width)
    ] = "outside_match_window"
    ref_normalized = reference.power_normalized.values[idx]
    valid = (
        (reference.p_flag.values[idx] == 0)
        & np.isfinite(reference.adrv_best.values[idx])
        & np.isfinite(ref_normalized)
        & (ref_normalized > 0)
        & (np.asarray(normalized) > 0)
    )
    status[~valid] = "invalid_power_selection"
    with np.errstate(divide="ignore", invalid="ignore"):
        current = -attenuation + 10 * np.log10(normalized)
        target = -reference.adrv_best.values[idx] + 10 * np.log10(ref_normalized)
    delta = np.where(status == "matched", current - target, np.nan)
    return {
        "reference_chan": reference.chan.values[idx],
        "reference_status": status,
        "reference_distance_hz": distance,
        "reference_distance_unshifted_hz": matched["dist"].to_value(u.Hz),
        "reference_shift_hz": np.full(len(idx), result.shift.to_value(u.Hz)),
        "reference_power_db": target,
        "power_delta_db": delta,
        "reference_a_nl": reference.a_best.values[idx],
    }


def power_setup_diagnostics(  # noqa: PLR0913 -- Complete comb and consumer-owned reference.
    frequency, baseband, amplitude, phase, mask, *, meta, config, channels
):
    """Evaluate a complete comb without changing its tones or attenuation.

    Parameters
    ----------
    frequency, baseband, amplitude, phase, mask : array_like
        Absolute Hz, DAC baseband Hz, recorded coefficients, radians and enabled
        mask for the complete comb. Do not restrict to fitted channels first.
    meta : object
        Network identity (roach) and attenuation (atten_drive).
    config : PowerSetupConfig
        Consumer-owned settings and references.
    channels : array_like
        Original channel IDs, or prepared tone row IDs.

    Returns
    -------
    xarray.Dataset
        Channel diagnostics and exact reference/configuration provenance.
        Absolute waveform predictions and relative reference comparisons have
        separate columns. Neither is a measurement of delivered detector power.
    """
    attenuation = getattr(meta, "atten_drive", None)
    if attenuation is None or not np.isfinite(attenuation):
        msg = "Power diagnostics require finite recorded drive attenuation"
        raise ValueError(msg)
    frequency, amplitude = np.asarray(frequency), np.asarray(amplitude)
    waveform = optimize_tone_phases(
        baseband,
        amplitude,
        phase,
        mask,
        trials=0,
        sample_rate=config.dac.sample_rate.to_value(u.Hz),
        lut_size=config.dac.lut_size,
    )
    fraction = waveform.power_fraction
    normalized = fraction / waveform.crest_factor
    if getattr(meta, "dac_normalization", None) is not None:
        normalized = (meta.dac_normalization * amplitude[waveform.owners]) ** 2
    with np.errstate(divide="ignore"):
        current = -attenuation + 10 * np.log10(normalized)
    values = {
        "frequency": frequency,
        "power_fraction": fraction,
        "power_normalized": normalized,
        "power_level_db": current,
    }
    attrs = {
        "power_model": POWER_MODEL,
        "atten_drive": float(attenuation),
        "roach": int(meta.roach),
        "absolute_power_status": "not_calibrated",
        "crest_factor": waveform.crest_factor,
    }
    attrs["config"] = PowerSetupConfig.model_validate(
        {name: getattr(config, name) for name in PowerSetupConfig.model_fields}
    ).model_dump(mode="json")
    if config.power is not None:
        values["power_detector_dbm"] = detector_tone_power_dbm(
            normalized,
            dac_single_tone_power_dbm=config.power.dac_single_tone_power_dbm,
            drive_path_loss_db=config.power.drive_path_loss_db,
            atten_drive_db=attenuation,
        )
        attrs.update(
            crest_factor=waveform.crest_factor,
            absolute_power_status="waveform_prediction",
        )
    if config.tone_amps_lut_root is not None:
        path = (
            config.tone_amps_lut_root
            / f"toltec{meta.roach}"
            / "amps_correction_lut.csv"
        )
        lut = RoachToneLut.from_file(path)
        expected = lut(frequency * u.Hz)
        valid = np.asarray(mask, bool) & (amplitude > 0)
        if not np.any(valid):
            msg = "LUT comparison requires positive enabled amplitudes"
            raise ValueError(msg)
        with np.errstate(divide="ignore"):
            ratio = 20 * np.log10(amplitude / expected)
        offset = float(np.median(ratio[valid]))
        values["lut_residual_db"] = np.where(valid, ratio - offset, np.nan)
        attrs.update(lut_path=str(path), lut_sha256=lut.digest, lut_offset_db=offset)
    if config.drive_fit_reference is not None:
        reference, identity = _load_reference(config, meta.roach)
        values.update(
            compare_drive_fit(frequency, normalized, attenuation, reference, config)
        )
        attrs["reference_product"] = identity
    result = xr.Dataset(
        {name: ("chan", value) for name, value in values.items()},
        coords={"chan": channels},
        attrs=attrs,
    )
    for name in values:
        result[name].attrs["units"] = (
            "Hz"
            if name == "frequency" or name.endswith("_hz")
            else "dB(mW)"
            if name == "power_detector_dbm"
            else "dB"
            if name.endswith("_db")
            else "1"
        )
    return result


def prepare_drive_fit_tones(tones, config):
    """Assign fitted detector targets and close the complete proposed comb.

    Parameters
    ----------
    tones : xarray.Dataset
        Final tone order with f_out, f_centered, amp_tone, mask_tone and roach.
        The dataset is updated with prepared amplitudes and common attenuation.
    config : TonePrepConfig
        Exact fit selection, matching settings and optional LUT/median fallback.
        Fallback applies only to rows without a usable unique fit match.

    Returns
    -------
    xarray.Dataset
        Per-tone matching, target and delivered power diagnostics with exact fit
        provenance. Powers include peak normalization; dBm needs a calibration.

    Raises
    ------
    ValueError
        No valid fit population exists, fallback is required but absent, or the
        proposed comb cannot realize the requested powers.
    """
    reference, identity = _load_reference(config, tones.attrs["roach"])
    normalized = reference.power_normalized.values
    valid = (
        (reference.p_flag.values == 0)
        & np.isfinite(reference.adrv_best.values)
        & np.isfinite(normalized)
        & (normalized > 0)
        & np.isfinite(reference.Qr_best.values)
        & (reference.Qr_best.values > 0)
    )
    if not valid.any():
        msg = "Tone preparation requires at least one valid drive-fit power target"
        raise ValueError(msg)
    active = np.asarray(tones.mask_tone.values, bool)
    if not active.any():
        msg = "Tone preparation requires an enabled output tone"
        raise ValueError(msg)
    frequency = tones.f_out.values
    # Match only enabled tones; padding must not compete for detector identity.
    matched = compare_drive_fit(
        frequency[active], np.ones(active.sum()), 0.0, reference, config
    )
    values = _expand_matches(matched, active)
    good = active & (values["reference_status"] == "matched")
    target = np.full(len(active), np.nan)
    target[good] = values["reference_power_db"][good]
    source = np.full(len(active), "masked", dtype="U16")
    source[good] = "drive_fit"
    fallback = active & ~good
    attrs = {
        "power_model": POWER_MODEL,
        "reference_product": identity,
        "atten_drive_before": float(tones.attrs["atten_drive"]),
        "tone_amps_fallback": config.tone_amps_fallback,
        "config": config.model_dump(mode="json", exclude={"artifacts"}),
    }
    if fallback.any():
        targets, fallback_attrs = _drive_fit_fallback(
            frequency[fallback], reference, valid, config
        )
        target[fallback] = targets
        attrs.update(fallback_attrs)
        source[fallback] = config.tone_amps_fallback
    return _close_tone_power(
        tones,
        config,
        target,
        source,
        xr.Dataset(
            {name: ("tone", value) for name, value in values.items()}, attrs=attrs
        ),
    )


def prepare_acquired_tones(tones, probes, meta, config):
    """Preserve the acquired comb's powers in a new FILE-mode tone package.

    Parameters
    ----------
    tones : xarray.Dataset
        Proposed tones with inherited or explicitly filled input amplitudes.
    probes : astropy.table.QTable
        Complete acquired comb, including recorded phases and masks.
    meta : ToltecSweepMetadata
        Recorded LO, observation type and applied drive attenuation.
    config : TonePrepConfig
        Complete-waveform preparation settings.

    Returns
    -------
    xarray.Dataset
        Proposed powers, applied-setting proposal and reference diagnostics.
        Targets retain the acquired powers. The proposed attenuation can change.
    """
    attenuation = meta.atten_drive
    if attenuation is None or not np.isfinite(attenuation):
        msg = "FILE preparation requires recorded drive attenuation"
        raise ValueError(msg)
    waveform = optimize_tone_phases(
        probes["f_chan"].to_value(u.Hz) - meta.f_lo_center,
        probes["tone_amp"],
        probes["tone_phase"].to_value(u.rad),
        probes["tone_mask"],
        trials=0,
        sample_rate=config.dac.sample_rate.to_value(u.Hz),
        lut_size=config.dac.lut_size,
    )
    owners = np.unique(waveform.owners)
    reference_amplitudes = np.asarray(probes["tone_amp"])[owners]
    reference_masks = np.asarray(probes["tone_mask"], bool)[owners]
    peak = waveform.crest_factor * np.sum(reference_amplitudes[reference_masks] ** 2)
    recorded_gain = getattr(meta, "dac_normalization", None)
    gain = 1 / np.sqrt(peak) if recorded_gain is None else recorded_gain
    active = np.asarray(tones.mask_tone.values, bool)
    amplitude = tones.amp_tone.values
    if (
        not active.any()
        or np.any(~np.isfinite(amplitude[active]))
        or np.any(amplitude[active] <= 0)
    ):
        msg = "FILE preparation requires positive finite probe amplitudes"
        raise ValueError(msg)
    target = np.full(len(active), np.nan)
    target[active] = 20 * np.log10(amplitude[active] * gain) - attenuation
    source = np.full(len(active), "masked", dtype="U16")
    source[active] = "vna" if meta.obs_type == "vna" else "acquired"
    attrs = {
        "power_model": POWER_MODEL,
        "atten_drive_before": float(meta.atten_drive),
        "atten_drive_reference": float(attenuation),
        "reference_dac_normalization": float(gain),
        "tone_amps_normalization_divisor": float(np.max(amplitude[active])),
        "reference_crest_factor": waveform.crest_factor,
        "reference_acquisition": {
            name: getattr(meta, name)
            for name in ("master", "obsnum", "subobsnum", "scannum", "roach", "block")
        },
        "config": config.model_dump(mode="json", exclude={"artifacts"}),
    }
    return _close_tone_power(tones, config, target, source, xr.Dataset(attrs=attrs))


def _close_tone_power(tones, config, target, source, diagnostic):
    """Materialize one power target through the shared waveform solver."""
    values = {name: value.values for name, value in diagnostic.data_vars.items()}
    attrs = diagnostic.attrs
    active = source != "masked"
    frequency = tones.f_out.values
    optimization = config.phase_optimization
    initial_phases = tones.phase_tone.values.copy()
    closure = tone_waveform_attenuation(
        target,
        tones.f_centered.values,
        initial_phases,
        active,
        phase_trials=0 if optimization is None else optimization.trials,
        seed=None if optimization is None else optimization.seed,
        sample_rate=config.dac.sample_rate.to_value(u.Hz),
        lut_size=config.dac.lut_size,
    )
    if not np.isfinite(closure["power_output_db"][active]).all():
        msg = "An enabled output tone is suppressed by a masked DAC-bin owner"
        raise ValueError(msg)
    waveform = closure["waveform"]
    search = waveform
    amplitude, active = closure["amplitudes"], closure["mask"]
    signed = np.where(
        waveform.bins < config.dac.lut_size // 2,
        waveform.bins,
        waveform.bins - config.dac.lut_size,
    )
    values.update(
        phase_tone=waveform.phases,
        phase_tone_initial=initial_phases,
        tone_dac_bin=waveform.bins,
        tone_bin_owner=waveform.owners,
        tone_power_fraction=waveform.power_fraction,
        tone_power_normalized=closure["power_normalized"],
        f_tone_emitted=signed
        * config.dac.sample_rate.to_value(u.Hz)
        / config.dac.lut_size,
    )
    attrs.update(
        dac_normalization=closure["dac_normalization"],
        dac_peak=closure["dac_peak"],
        atten_drive_limit=closure["atten_drive_limit"],
        roach_dac_config=config.dac.model_dump(mode="json"),
        tone_crest_factor_initial=search.crest_factor_initial,
        tone_crest_factor_probes=search.crest_factor,
        tone_crest_factor=waveform.crest_factor,
        tone_phase_selected_trial=search.selected_trial,
        tone_phase_optimization=None
        if optimization is None
        else optimization.model_dump(mode="json"),
    )
    if config.power is not None:
        values["tone_power_detector_dbm"] = detector_tone_power_dbm(
            closure["power_normalized"],
            dac_single_tone_power_dbm=config.power.dac_single_tone_power_dbm,
            drive_path_loss_db=config.power.drive_path_loss_db,
            atten_drive_db=closure["atten_drive"],
        )
        attrs["roach_power_config"] = config.power.model_dump(mode="json")
    values.update(
        amp_tone=amplitude,
        power_target_db=target,
        power_output_db=closure["power_output_db"],
        power_error_db=closure["power_error_db"],
        power_source=source,
        frequency=frequency,
        mask_tone=active,
    )
    # Compare against the original fit target using the actual output setting.
    values["power_delta_db"] = np.where(
        source == "drive_fit", closure["power_output_db"] - target, np.nan
    )
    attrs.update(
        atten_drive=closure["atten_drive"],
        dac_normalization=closure["dac_normalization"],
    )
    tones["phase_tone"] = ("tone", waveform.phases)
    tones["amp_tone"] = ("tone", amplitude)
    tones["mask_tone"] = ("tone", active)
    tones.attrs.update(
        atten_drive=closure["atten_drive"],
        tone_power_model=POWER_MODEL,
        dac_normalization=closure["dac_normalization"],
    )
    if "reference_product" in attrs:
        tones.attrs["reference_product"] = attrs["reference_product"]
    diagnostic = xr.Dataset(
        {name: ("tone", value) for name, value in values.items()}, attrs=attrs
    )
    for name in values:
        if name in {"phase_tone", "phase_tone_initial"}:
            diagnostic[name].attrs["units"] = "rad"
        elif name == "tone_power_detector_dbm":
            diagnostic[name].attrs["units"] = "dB(mW)"
        elif name.endswith("_db"):
            diagnostic[name].attrs["units"] = "dB"
        elif name in {"frequency", "f_tone_emitted"} or name.endswith("_hz"):
            diagnostic[name].attrs["units"] = "Hz"
    return diagnostic


def _expand_matches(matched, active):
    """Restore non-probe rows without assigning a detector identity to them."""
    values = {}
    for name, column in matched.items():
        fill = (
            "masked"
            if column.dtype.kind in "US"
            else -1
            if name == "reference_chan"
            else np.nan
        )
        values[name] = np.full(
            len(active),
            fill,
            dtype=column.dtype
            if column.dtype.kind in "US" or name == "reference_chan"
            else float,
        )
        values[name][active] = column
    return values


def _drive_fit_fallback(frequency, reference, valid, config):
    """Fill unidentified tones without changing identified detector targets."""
    normalized = reference.power_normalized.values
    attrs = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        reference_power = (
            10 * np.log10(normalized[valid]) - reference.adrv_best.values[valid]
        )
    if config.tone_amps_fallback == "lut":
        if config.tone_amps_lut_root is None:
            msg = "Drive-fit LUT fallback requires tone_prep.tone_amps_lut_root"
            raise ValueError(msg)
        path = (
            config.tone_amps_lut_root
            / f"toltec{reference.attrs['roach']}"
            / "amps_correction_lut.csv"
        )
        lut = RoachToneLut.from_file(path)
        offset = float(
            np.median(
                reference_power
                - 20 * np.log10(lut(reference.f_chan.values[valid] * u.Hz))
            )
        )
        targets = 20 * np.log10(lut(frequency * u.Hz)) + offset
        attrs.update(
            lut_path=str(path),
            lut_sha256=lut.digest,
            lut_offset_db=offset,
            lut_anchor_count=int(valid.sum()),
            lut_samples={
                "frequency_hz": lut.frequency.to_value(u.Hz).tolist(),
                "amplitude": lut.amplitude.tolist(),
            },
        )
    elif config.tone_amps_fallback == "median":
        targets = float(np.median(reference_power))
    else:
        msg = (
            "Unidentified or unusable drive-fit matches require "
            "an explicit tone_amps_fallback"
        )
        raise ValueError(msg)
    return targets, attrs
