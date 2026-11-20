"""Native-consumable physical calibration reports from saved Python fits."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from astropy.table import Table

from tolteca_kidsproc.model_flags import ModelFitFlag

from .model_fit_quality import ModelFitBitMask, ModelFitQualityConfig, model_fit_quality
from .model_joint import joint_models

_QUALITY_FLAGS = {
    ModelFitBitMask.not_converged: ModelFitFlag.fit_not_converged,
    ModelFitBitMask.at_bound: ModelFitFlag.fit_at_bound,
    ModelFitBitMask.rank_deficient: ModelFitFlag.fit_rank_deficient,
    ModelFitBitMask.condition_high: ModelFitFlag.fit_condition_high,
    ModelFitBitMask.residual_high: ModelFitFlag.fit_residual_high,
    ModelFitBitMask.initialization_fallback: ModelFitFlag.fit_initialization_fallback,
    ModelFitBitMask.no_fit: ModelFitFlag.fit_no_model,
    ModelFitBitMask.selection_required: ModelFitFlag.fit_selection_required,
    ModelFitBitMask.probe_outside_window: ModelFitFlag.fit_probe_outside_window,
    ModelFitBitMask.recovery_unresolved: ModelFitFlag.fit_recovery_unresolved,
    ModelFitBitMask.power_high: ModelFitFlag.fit_power_high,
    ModelFitBitMask.power_low: ModelFitFlag.fit_power_low,
    ModelFitBitMask.power_reference_unmatched: (
        ModelFitFlag.fit_power_reference_unmatched
    ),
    ModelFitBitMask.nonlinear_limit: ModelFitFlag.fit_nonlinear_limit,
    ModelFitBitMask.resonance_outside_window: (
        ModelFitFlag.fit_resonance_outside_window
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
        requiring joint solving carry routing flags and a complete group model
        in the report metadata. Independent rows retain their fitted parameters,
        including rejected fits. Solvers must honor the common fit flags.
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
            "flag": np.full(n, int(ModelFitFlag.fit_unusable), dtype=np.int64),
        }
    )
    report["a_nl"] = np.full(n, np.nan)
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
        if flag & ModelFitFlag.fit_joint_required or not len(rows.row):
            continue
        row = rows.isel(row=0)
        report["a_nl"][i] = row.a_nl.item()
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
            "nonlinear_branch": view.components().attrs.get(
                "nonlinear_branch", "unique"
            ),
            "block": meta.block,
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
            "Header.Toltec.AccumLen": accumlen,
            "acquisition_status": {
                "blocks": [{"block": meta.block, "state": "complete"}]
            },
            "unusable_flag": int(ModelFitFlag.fit_unusable),
            "flag_definitions": {f.name: int(f) for f in ModelFitFlag},
            "quality_thresholds": quality.attrs["thresholds"],
            "calibration_acceptance": "not_assessed",
            "scope": "fixed-drive singlets and joint groups",
        }
    )
    report.meta["joint_models"] = joint_models(view, quality)
    return report


def _export_flag(rows, prop, warning):
    flag = ModelFitFlag(0)
    for source, target in _QUALITY_FLAGS.items():
        if warning & source:
            flag |= target
    if not rows.sizes["row"]:
        return flag | ModelFitFlag.fit_unusable
    if len(set(rows.resonator_id.values.tolist())) != 1:
        flag |= ModelFitFlag.fit_joint_required
    if prop.fit_status.item() != "optimizer_converged":
        flag |= ModelFitFlag.fit_not_converged | ModelFitFlag.fit_unusable
    if prop.recovery_mode.item() in {"invalid", "unresolved"}:
        flag |= ModelFitFlag.fit_unusable
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
