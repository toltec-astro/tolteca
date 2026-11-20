"""Native-consumable physical calibration reports from saved Python fits."""

from __future__ import annotations

from enum import IntFlag
from typing import TypedDict

import numpy as np
from astropy.table import Table

from tolteca_kidsproc.models import NONLINEARITY_UNIQUE_MAX

from .model_fit_quality import ModelFitBitMask, ModelFitQualityConfig, model_fit_quality
from .model_joint import joint_models


class NativeFitFlag(IntFlag):
    """Native kids.fit/1 fit/export bits, shared with kidscpp model/flags.h.

    Warnings preserve numerical results. ``fit_unusable`` forbids inversion;
    the accompanying bits retain the reason. Zero is not scientific acceptance.
    """

    fit_good = 0
    fit_first_pass_better = 1 << 0
    seed_large_offset = 1 << 1
    seed_not_converged = 1 << 2
    seed_out_of_range = 1 << 3
    seed_qr_out_of_range = 1 << 4
    fit_large_offset = 1 << 5
    fit_not_converged = 1 << 6
    fit_out_of_range = 1 << 7
    fit_qr_out_of_range = 1 << 8
    fit_low_gain = 1 << 9
    fit_insufficient_data = 1 << 11
    fit_invalid_data = 1 << 12
    fit_unusable = 1 << 13
    fit_at_bound = 1 << 14
    fit_rank_deficient = 1 << 15
    fit_condition_high = 1 << 16
    fit_residual_high = 1 << 17
    fit_initialization_fallback = 1 << 18
    fit_no_model = 1 << 19
    fit_selection_required = 1 << 20
    fit_probe_outside_window = 1 << 21
    fit_joint_required = 1 << 22
    fit_nonlinear_unsupported = 1 << 23
    fit_recovery_unresolved = 1 << 24
    fit_power_high = 1 << 25
    fit_power_low = 1 << 26
    fit_power_reference_unmatched = 1 << 27
    fit_nonlinear_limit = 1 << 28
    fit_resonance_outside_window = 1 << 29


_QUALITY_FLAGS = {
    ModelFitBitMask.not_converged: NativeFitFlag.fit_not_converged,
    ModelFitBitMask.at_bound: NativeFitFlag.fit_at_bound,
    ModelFitBitMask.rank_deficient: NativeFitFlag.fit_rank_deficient,
    ModelFitBitMask.condition_high: NativeFitFlag.fit_condition_high,
    ModelFitBitMask.residual_high: NativeFitFlag.fit_residual_high,
    ModelFitBitMask.initialization_fallback: NativeFitFlag.fit_initialization_fallback,
    ModelFitBitMask.no_fit: NativeFitFlag.fit_no_model,
    ModelFitBitMask.selection_required: NativeFitFlag.fit_selection_required,
    ModelFitBitMask.probe_outside_window: NativeFitFlag.fit_probe_outside_window,
    ModelFitBitMask.recovery_unresolved: NativeFitFlag.fit_recovery_unresolved,
    ModelFitBitMask.power_high: NativeFitFlag.fit_power_high,
    ModelFitBitMask.power_low: NativeFitFlag.fit_power_low,
    ModelFitBitMask.power_reference_unmatched: (
        NativeFitFlag.fit_power_reference_unmatched
    ),
    ModelFitBitMask.nonlinear_limit: NativeFitFlag.fit_nonlinear_limit,
    ModelFitBitMask.resonance_outside_window: (
        NativeFitFlag.fit_resonance_outside_window
    ),
}


_COMMON = [
    ("f_g", "Hz"),
    ("fr", "Hz"),
    ("r", ""),
    ("c_re", ""),
    ("c_im", ""),
    ("H0_re", ""),
    ("H0_im", ""),
]


class _ParameterSchema(TypedDict):
    name: str
    unit: str


class _ModelSchema(TypedDict):
    name: str
    version: int
    parameters: list[_ParameterSchema]


PHYSICAL_SCHEMAS: dict[str, _ModelSchema] = {
    family: {
        "name": name,
        "version": 2,
        "parameters": [
            {"name": key, "unit": unit} for key, unit in _COMMON + tail + [("b", "")]
        ],
    }
    for family, name, tail in (
        (
            "affine",
            "KidsSweepWithAffineGain",
            [("H1_re", "1 / Hz"), ("H1_im", "1 / Hz")],
        ),
        ("delay", "KidsSweepWithDelayGain", [("a", "1 / Hz"), ("tau", "s")]),
    )
}


def calibration_report(view, block, *, quality=None):
    """Export independent rows and complete selected physical groups.

    Parameters
    ----------
    view : tolteca_kidsproc.analysis.KidsModelView
        Saved grouped fit, including channel bindings and original parameters.
    block : ToltecBlockView
        Source sweep block supplying acquisition identity and normalization.
    quality : xarray.Dataset or None, optional
        Saved ModelFit quality, including configured thresholds and power checks.
        If absent, derive the default fit warnings; optional cuts stay disabled.

    Returns
    -------
    astropy.table.Table
        Versioned ``kids.fit/1`` report understood by the native solver. Channels
        requiring joint recovery retain flagged NaN independent rows and a
        labeled ``joint_models`` payload when the group selection is unique.
        Isolated unique-domain rows export fixed-b nonlinear recovery. Nonlinear
        groups and isolated fits beyond the qualified unique domain remain
        unsupported and flagged; their parameters stay in the compact fit.
    """
    fit, props = view.fit, view.chan_prop
    if quality is None:
        quality = model_fit_quality(view, ModelFitQualityConfig())
    if not np.array_equal(quality.chan, props.chan):
        msg = "Calibration quality must retain acquired channel identities"
        raise ValueError(msg)
    families = set(fit.gain_family.values.tolist())
    if len(families) != 1:
        msg = "One exported report requires one gain family"
        raise ValueError(msg)
    schema = PHYSICAL_SCHEMAS[families.pop()]
    axis = block.chan_axis_table
    n = len(axis)
    report = Table(
        {
            "chan": np.asarray(axis["chan"], dtype=np.int64),
            "f_in": axis["f_chan"].to_value("Hz"),
            "f_out": axis["f_chan"].to_value("Hz"),
            "flag": np.full(n, int(NativeFitFlag.fit_unusable), dtype=np.int64),
        }
    )
    for parameter in schema["parameters"]:
        report[parameter["name"]] = np.full(n, np.nan)
        report[parameter["name"]].unit = parameter["unit"] or None
    for i, chan in enumerate(report["chan"]):
        prop = props.sel(chan=chan)
        selected = np.flatnonzero(
            (fit.chan.values == chan) & (fit.group_id.values == prop.group_id.item())
        )
        rows = fit.isel(row=selected)
        flag = _export_flag(rows, prop, int(quality.channel_flag.sel(chan=chan)))
        report["flag"][i] = int(flag)
        if flag & NativeFitFlag.fit_unusable:
            continue
        row = rows.isel(row=0)
        for parameter in schema["parameters"]:
            if parameter["name"] == "b":
                report["b"][i] = 2 * row.a_nl.item() * row.r.item() ** 3
                continue
            report[parameter["name"]][i] = row[
                "gain_a" if parameter["name"] == "a" else parameter["name"]
            ].item()
        report["f_out"][i] = row.fr.item()
    for name in ("f_in", "f_out"):
        report[name].unit = "Hz"
    meta = block.meta
    accumlen = calibration_accumlen(block)
    report.meta.update(
        {
            "schema": "kids.fit/1",
            "model": schema["name"],
            "model_schema": schema,
            "block": meta.block,
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
            "Header.Toltec.AccumLen": accumlen,
            "acquisition_status": {
                "blocks": [{"block": meta.block, "state": "complete"}]
            },
            "unusable_flag": int(NativeFitFlag.fit_unusable),
            "flag_definitions": {f.name: int(f) for f in NativeFitFlag},
            "quality_thresholds": quality.attrs["thresholds"],
            "calibration_acceptance": "not_assessed",
            "scope": "fixed-b singlets and explicitly labeled linear joint groups",
        }
    )
    report.meta["joint_models"] = joint_models(view, quality)
    return report


def _export_flag(rows, prop, warning):
    flag = NativeFitFlag(0)
    for source, target in _QUALITY_FLAGS.items():
        if warning & source:
            flag |= target
    if not rows.sizes["row"]:
        return flag | NativeFitFlag.fit_unusable
    if len(set(rows.resonator_id.values.tolist())) != 1:
        flag |= NativeFitFlag.fit_joint_required | NativeFitFlag.fit_unusable
    if (
        len(set(rows.resonator_id.values.tolist())) != 1
        and np.any(rows.a_nl.values != 0)
    ) or np.any(rows.a_nl.values > NONLINEARITY_UNIQUE_MAX):
        flag |= NativeFitFlag.fit_nonlinear_unsupported | NativeFitFlag.fit_unusable
    if prop.fit_status.item() != "optimizer_converged":
        flag |= NativeFitFlag.fit_not_converged | NativeFitFlag.fit_unusable
    if prop.recovery_mode.item() in {"invalid", "unresolved"}:
        flag |= NativeFitFlag.fit_unusable
    return flag


def calibration_accumlen(block):
    """Read measured IQ normalization from the sweep or its open raw source."""
    accumlen = (
        block.tree["sweep"].attrs.get("iq_accumlen")
        if "sweep" in block.tree.children
        else None
    )
    if accumlen is None:
        accumlen = block.raw["Header.Toltec.AccumLen"].item()
    if not np.isfinite(accumlen) or accumlen <= 0:
        msg = "Calibration requires the recorded positive IQ accumulation length"
        raise ValueError(msg)
    return accumlen
