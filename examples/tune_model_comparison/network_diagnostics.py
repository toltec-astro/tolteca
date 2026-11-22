#!/usr/bin/env python
"""Separate measured model adequacy, numerical outcomes and review policy.

The existing comparison exclusions remain in force. Their categories describe
why a component is excluded; none is an inferred optical detector class.
"""

from __future__ import annotations

import numpy as np

from tolteca_kids.model_fit_quality import ModelFitBitMask
from tolteca_kidsproc.recovery import SampleFlag

RESIDUAL_MAX = 0.05


def _validity(row, numerical, flags):
    if row["native_route"] == "unselected":
        return "not_assessed"
    if numerical:
        return "flagged"
    if row["psd_status"] == "measured":
        return "complete"
    if flags & (SampleFlag.outside_domain | SampleFlag.ambiguous_solution):
        return "branch_or_domain_limited"
    return "incomplete"


def _sample_flags(values):
    flags = SampleFlag(0)
    for value in str(values).split(";"):
        if value and value != "nan":
            flags |= SampleFlag(int(value))
    return flags


def assess(row, *, residual_max=RESIDUAL_MAX):  # noqa: C901 -- Explicit independent diagnostic axes.
    """Return overlapping diagnostics and unchanged comparison exclusions."""
    fit = ModelFitBitMask(int(row["group_flag"]))
    selected = row["native_route"] != "unselected"
    flags = _sample_flags(row["native_sample_flag_values"])
    recorded_flags = ";".join(f.name for f in SampleFlag if flags & f)
    # A singlet may have several probe estimates; the comparison uses complete
    # estimates only. Failed unused probes remain recorded, not new exclusions.
    if row["psd_status"] == "measured":
        flags = SampleFlag(0)
    residual = row["residual_to_sweep_variation"]
    adequacy = (
        "undefined"
        if not np.isfinite(residual)
        else ("within_limit" if residual <= residual_max else "exceeds_limit")
    )
    numerical = [
        "fit_" + flag.name
        for flag in (ModelFitBitMask.not_converged, ModelFitBitMask.rank_deficient)
        if fit & flag
    ]
    policy = [
        flag.name
        for flag in (
            ModelFitBitMask.at_bound,
            ModelFitBitMask.initialization_fallback,
            ModelFitBitMask.nonlinear_limit,
        )
        if fit & flag
    ]
    feature = bool(fit & ModelFitBitMask.resonance_outside_window)
    numerical.extend(
        flag.name
        for flag in SampleFlag
        if flags & flag
        and flag
        not in (
            SampleFlag.condition_high,
            SampleFlag.outside_domain,
            SampleFlag.ambiguous_solution,
        )
    )
    if flags & SampleFlag.outside_domain:
        policy.append("domain_or_continuation_limit")
    if flags & SampleFlag.ambiguous_solution:
        policy.append("branch_not_selected")
    parity_status = row["parity_status"]
    if parity_status == "tolerance_exceeded":
        policy.append("python_native_tolerance")
    elif parity_status == "mask_mismatch":
        numerical.append("python_native_mask_mismatch")
    if row["bad_readout_channel"]:
        policy.append("bad_readout_channel")
    validity = _validity(row, numerical, flags)
    exclusions = []
    if adequacy != "within_limit":
        exclusions.append("sweep_residual")
    if feature:
        exclusions.append("feature_outside_fit_window")
    exclusions.extend(numerical)
    exclusions.extend(policy)
    if not selected:
        exclusions.append("route_unselected")
    elif row["psd_status"] != "measured":
        exclusions.append("incomplete_valid_samples")
    if parity_status == "not_assessed":
        exclusions.append("parity_unassessed")
    return {
        "route_status": "selected" if selected else "unselected",
        "fit_adequacy": adequacy,
        "residual_limit": residual_max,
        "feature_outside_fit_window": feature,
        "numerical_validity": validity,
        "numerical_flags": ";".join(numerical),
        "recorded_sample_flags": recorded_flags,
        "selection_policy_limits": ";".join(policy),
        "comparison_exclusions": ";".join(exclusions),
    }
