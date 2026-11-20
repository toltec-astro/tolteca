"""Configured drive fitting over a collection of native TolTEC sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.utils.fileloc import fileloc
from tollan.utils.log import logger

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.block_create import BlockCreate
from tolteca_kids.sweep_reduce import SweepReduce

from ..roach_tone import POWER_MODEL, optimize_tone_phases
from .fit import CABLE_DELAY_NS, find_best_drive, fit_channel
from .result import DriveFitResult

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata


@dataclass
class _Sweep:
    source: str
    meta: ToltecSweepMetadata
    chan: np.ndarray
    frequency: np.ndarray
    I: np.ndarray  # noqa: E741 -- physical in-phase signal.
    Q: np.ndarray
    f_chan: np.ndarray
    amp: np.ndarray


def _read_sweep(source, channels):
    """Load one coherent sweep and release the native input."""
    with open_toltec_kids(source) as raw:
        tree = BlockCreate()(ensure_datatree(raw))
        blocks = tree.toltec_kids.blocks
        if blocks.n_blocks != 1:
            msg = f"Drive fitting requires one sweep block per input: {source}"
            raise ValueError(msg)
        block = blocks[0]
        if block.meta.obs_type not in {"targ", "tune"}:
            msg = f"Drive fitting requires target or tune sweeps: {source}"
            raise ValueError(msg)
        SweepReduce(compute_uncertainty=False)(block.tree)
        view = block.sweep
        axis = block.meta.chan_axis_table
        indices = np.asarray(axis["chan"], dtype=int)
        if channels is None:
            selected = np.arange(len(indices))
        else:
            positions = {int(value): i for i, value in enumerate(indices)}
            missing = set(channels) - positions.keys()
            if missing:
                msg = f"Channels absent from {source}: {sorted(missing)}"
                raise ValueError(msg)
            selected = np.array([positions[value] for value in channels])
        chan_dim = view.I.dims[0]
        arrays = [
            arr.isel({chan_dim: selected}).transpose(chan_dim, "sweep").values.copy()
            for arr in (view.frequency, view.I, view.Q)
        ]
        return _Sweep(
            source=fileloc(source).url,
            meta=block.meta,
            chan=indices[selected],
            frequency=arrays[0],
            I=arrays[1],
            Q=arrays[2],
            f_chan=axis["f_chan"][selected].to_value(u.Hz),
            amp=np.asarray(axis["tone_amp"][selected]).copy(),
        )


def fit_sequence(sources, config):
    """Read, validate and fit one network; close sources before fitting."""
    sources = sorted(sources, key=lambda source: fileloc(source).url)
    if not sources:
        msg = "No sweeps supplied for drive fitting"
        raise ValueError(msg)
    urls = [fileloc(source).url for source in sources]
    if len(urls) != len(set(urls)):
        msg = "Duplicate drive-fit input locations"
        raise ValueError(msg)
    sweeps = [_read_sweep(source, config.channels) for source in sources]
    sweeps.sort(key=lambda item: -item.meta.atten_drive)
    first = sweeps[0]
    identity = (first.meta.master, first.meta.roach)
    for sweep in sweeps:
        if (sweep.meta.master, sweep.meta.roach) != identity:
            msg = "A drive-fit sequence must have one master and readout network"
            raise ValueError(msg)
        if (
            not np.array_equal(sweep.chan, first.chan)
            or not np.array_equal(sweep.f_chan, first.f_chan)
            or not np.array_equal(sweep.amp, first.amp)
            or any(
                not np.array_equal(
                    sweep.meta.chan_axis_table[name], first.meta.chan_axis_table[name]
                )
                for name in ("f_tone", "tone_amp", "tone_phase", "tone_mask")
            )
            or sweep.meta.dac_normalization != first.meta.dac_normalization
            or sweep.I.shape != first.I.shape
        ):
            msg = (
                "Inconsistent channel layout, amplitudes, phases, masks or sweep shape"
            )
            raise ValueError(msg)
    attenuation = np.array([sweep.meta.atten_drive for sweep in sweeps])
    if len(np.unique(attenuation)) != len(sweeps):
        msg = "Drive-fit inputs contain repeated attenuation settings"
        raise ValueError(msg)
    if len(sweeps) < config.n_adrvs_min:
        msg = (
            f"Too few attenuation settings: {len(sweeps)}; require {config.n_adrvs_min}"
        )
        raise ValueError(msg)
    return _fit(sweeps, attenuation, config)


def _fit(sweeps, attenuation, config):  # noqa: C901, PLR0915 -- Assemble aligned fit and diagnostic arrays.
    first = sweeps[0]
    # Normalize on the full acquired comb, even for a selected-channel fit.
    axis = first.meta.chan_axis_table
    waveform = optimize_tone_phases(
        axis["f_tone"].to_value(u.Hz),
        axis["tone_amp"],
        axis["tone_phase"],
        axis["tone_mask"],
        trials=0,
        sample_rate=config.dac.sample_rate.to_value(u.Hz),
        lut_size=config.dac.lut_size,
    )
    positions = {int(chan): i for i, chan in enumerate(axis["chan"])}
    fractions = waveform.power_fraction[[positions[int(chan)] for chan in first.chan]]
    normalized = fractions / waveform.crest_factor
    if first.meta.dac_normalization is not None:
        powers = (
            first.meta.dac_normalization * np.asarray(axis["tone_amp"])[waveform.owners]
        ) ** 2
        normalized = powers[[positions[int(chan)] for chan in first.chan]]
    power = -1.0 * attenuation
    frequency = np.stack([sweep.frequency for sweep in sweeps])
    I = np.stack([sweep.I for sweep in sweeps])  # noqa: E741 -- physical I/Q.
    Q = np.stack([sweep.Q for sweep in sweeps])
    signal = I + 1j * Q
    n_power, n_chan, n_sample = signal.shape
    scalar_names = {"a": "bif", "Qr": "Q", "Qc": "Qc", "fr": "f0_corr"}
    values = {name: np.full((n_power, n_chan), np.nan) for name in scalar_names}
    circles = {
        name: np.full((n_power, n_chan), np.nan)
        for name in ("I_circle_center", "Q_circle_center", "circle_radius")
    }
    fit_flags = np.ones((n_power, n_chan), dtype=np.int8)
    phase_flags = np.ones_like(fit_flags)
    solver_nfev = np.full_like(fit_flags, -1, dtype=int)
    solver_status = np.full_like(fit_flags, -1)
    solver_cost = np.full((n_power, n_chan), np.nan)
    solver_retried = np.zeros_like(fit_flags, dtype=bool)
    errors = np.full((n_power, n_chan), "", dtype=object)
    diagnostic_names = {
        "frequency_fit": "ft",
        "phase_data": "angt",
        "phase_model": "fit_result",
        "phase_residual": "pherr",
        "phase_weight": "uncertainty_angt",
    }
    diagnostic = {
        name: np.full((n_power, n_chan, n_sample), np.nan)
        for name in (*diagnostic_names, "I_model", "Q_model")
    }
    phase_full = np.full((n_power, n_chan, n_sample), np.nan)
    best = np.empty((n_chan, 4))
    for channel, index in enumerate(first.chan):
        logger.debug("Fitting {} channel {}", first.meta.interface, index)
        results, failures = fit_channel(
            frequency[:, channel],
            signal[:, channel],
            tone_frequency=first.f_chan[channel],
            tau=CABLE_DELAY_NS[first.meta.roach],
            window_Qr=config.weight_window_Qr,
            phase_fit_method=config.phase_fit_method,
            phase_fit_maxfev=config.phase_fit_maxfev,
            phase_fit_retry=config.phase_fit_retry,
        )
        errors[:, channel] = failures
        for i, result in enumerate(results):
            if result is None:
                continue
            fit_flags[i, channel] = 0
            phase_flags[i, channel] = result["flag"]
            if "phase_fit_info" in result:
                info = result["phase_fit_info"]
                solver_nfev[i, channel] = info["nfev"]
                solver_status[i, channel] = info["status"]
                solver_cost[i, channel] = info["cost"]
                solver_retried[i, channel] = info["retried"]
            for name, key in scalar_names.items():
                values[name][i, channel] = result[key]
            for name, key in diagnostic_names.items():
                value = np.atleast_1d(result[key])
                diagnostic[name][i, channel, : len(value)] = value
            model = np.asarray(result["angt_to_z1"])
            diagnostic["I_model"][i, channel, : len(model)] = model.real
            diagnostic["Q_model"][i, channel, : len(model)] = model.imag
            circles["I_circle_center"][i, channel] = result["zc"].real
            circles["Q_circle_center"][i, channel] = result["zc"].imag
            circles["circle_radius"][i, channel] = result["r"]
            phase_full[i, channel] = result["ang"]
        best[channel] = find_best_drive(
            power,
            values["a"][:, channel],
            values["Qr"][:, channel],
            fit_flags[:, channel],
            a_ref=config.a_ref,
        )
    data = xr.Dataset(
        {
            **{name: (("power", "chan"), value) for name, value in values.items()},
            **{name: (("power", "chan"), value) for name, value in circles.items()},
            **{
                name: (("power", "chan", "fit_sample"), value)
                for name, value in diagnostic.items()
            },
            "fit_flag": (("power", "chan"), fit_flags),
            "phase_flag": (("power", "chan"), phase_flags),
            "fit_error": (("power", "chan"), errors.astype(str)),
            "fit_nfev": (("power", "chan"), solver_nfev),
            "fit_status": (("power", "chan"), solver_status),
            "fit_cost": (("power", "chan"), solver_cost),
            "fit_retried": (("power", "chan"), solver_retried),
            "frequency": (("power", "chan", "sample"), frequency),
            "phase_full": (("power", "chan", "sample"), phase_full),
            "I": (("power", "chan", "sample"), I),
            "Q": (("power", "chan", "sample"), Q),
            "adrv_best": ("chan", -best[:, 0]),
            "p_best": ("chan", best[:, 0]),
            "a_best": ("chan", best[:, 1]),
            "Qr_best": ("chan", best[:, 2]),
            "p_flag": ("chan", best[:, 3].astype(np.int8)),
            "f_chan": ("chan", first.f_chan),
            "amp_chan": ("chan", first.amp),
            "power_fraction": ("chan", fractions),
            "power_normalized": ("chan", normalized),
            "power_target_db": (
                "chan",
                10 * np.log10(np.where(normalized > 0, normalized, np.nan))
                + best[:, 0],
            ),
            "reference_f_tone": ("reference_tone", axis["f_tone"].to_value(u.Hz)),
            "reference_amp_tone": ("reference_tone", np.asarray(axis["tone_amp"])),
            "reference_phase_tone": ("reference_tone", np.asarray(axis["tone_phase"])),
            "reference_mask_tone": (
                "reference_tone",
                np.asarray(axis["tone_mask"], bool),
            ),
        },
        coords={
            "power": power,
            "chan": first.chan,
            "reference_tone": np.asarray(axis["chan"]),
        },
        attrs={
            "schema": "tolteca_kids.drive_fit",
            "schema_version": 1,
            "config": config.model_dump(
                mode="json", exclude={"artifacts", "enabled", "alias"}
            ),
            "interface": first.meta.interface,
            "master": first.meta.master,
            "roach": first.meta.roach,
            "obsnum": min(sweep.meta.obsnum for sweep in sweeps),
            "cable_delay_ns": CABLE_DELAY_NS[first.meta.roach],
            "power_model": POWER_MODEL,
            "reference_comb_channels": len(axis),
            "reference_crest_factor": waveform.crest_factor,
            "reference_dac_normalization": first.meta.dac_normalization
            if first.meta.dac_normalization is not None
            else float(
                1
                / np.sqrt(
                    waveform.crest_factor
                    * np.sum(
                        np.asarray(axis["tone_amp"])[np.unique(waveform.owners)] ** 2
                    )
                )
            ),
            "roach_dac_config": config.dac.model_dump(mode="json"),
            "inputs": [
                {
                    "source": sweep.source,
                    "uid": sweep.meta.uid_raw_obs_file,
                    "uid_raw_obs": sweep.meta.uid_raw_obs,
                    "atten_drive": sweep.meta.atten_drive,
                }
                for sweep in sweeps
            ],
        },
    )
    for name in ("frequency", "frequency_fit", "f_chan", "fr", "reference_f_tone"):
        data[name].attrs["units"] = "Hz"
    for name in ("power", "adrv_best", "p_best", "power_target_db"):
        data[name].attrs["units"] = "dB"
    for name in (
        "phase_data",
        "phase_full",
        "phase_model",
        "phase_residual",
        "reference_phase_tone",
    ):
        data[name].attrs["units"] = "rad"
    for name in ("I_model", "Q_model", *circles):
        data[name].attrs["signal_frame"] = (
            "normalized to the lowest-power final sample, then cable-delay corrected"
        )
    return DriveFitResult(data)
