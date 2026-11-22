"""Route an exact Python fit to the common group solver without refitting."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .model_fit_quality import residual_to_sweep_variation
from .model_joint import joint_models, joint_solver


def solve_fitted_groups(signals, view, quality, *, normalization=1, chunk_samples=4096):
    """Solve acquired probes while retaining the full fitted detector ledger.

    Parameters
    ----------
    signals : xarray.Dataset
        Open I/Q arrays on chan/time, including original sample coordinates.
    view : KidsModelView
        Exact saved Python fit. No identities or memberships are inferred here.
    quality : xarray.Dataset
        Quality records saved with that fit.
    normalization : float, optional
        Target-to-sweep accumulation ratio, default 1.
    chunk_samples : int, optional
        Ordered numerical chunk length, default 4096.

    Returns
    -------
    xarray.DataTree
        Full component ledger and lazy, independently dimensioned group nodes.
        Each node retains its complete model, probe identities and sample flags.
        Detuning is named x_opt; no channel ordinal represents a detector.

    Raises
    ------
    ValueError
        The selected signals omit a required probe or the calibration is invalid.
    """
    result = prepare_fitted_groups(view, quality)
    result.attrs.update(contract="kids.solved_groups/1", iq_normalization=normalization)
    for position, node in result["groups"].children.items():
        group = node.attrs["model"]
        required = group["channels"]
        if not set(required).issubset(signals.chan.values):
            msg = f"Missing simultaneous probes for group {group['group_id']}"
            raise ValueError(msg)
        solved = (
            joint_solver(group)
            .solve(
                signals.sel(chan=required),
                normalization=normalization,
                chunk_samples=chunk_samples,
            )
            .rename(x="x_opt")
        )
        solved.attrs["model"] = group
        result[f"groups/{position}"] = xr.DataTree(dataset=solved)
    return result


def prepare_fitted_groups(view, quality):
    """Prepare the full fitted-component inventory without reading timestreams.

    Parameters
    ----------
    view : KidsModelView
        Exact saved physical model, including channel selection and samples.
    quality : xarray.Dataset
        Matching saved fit warnings.

    Returns
    -------
    xarray.DataTree
        Component ledger, acquired channels and complete group specifications.
        No fitting, solving or output writing occurs.
    """
    payload = joint_models(view, quality, include_single=True)
    groups = {g["group_id"]: g for g in payload["groups"]}
    samples = view.samples.load()
    variation = {
        gid: residual_to_sweep_variation(
            samples.isel(sample=np.flatnonzero(samples.group_id.values == gid))
        )
        for gid in np.unique(samples.group_id.values)
    }
    rows, seen = [], set()
    for gid, rid, strength in zip(
        view.fit.group_id.values,
        view.fit.resonator_id.values,
        view.fit.a_nl.values,
        strict=True,
    ):
        key = str(gid), str(rid)
        if key in seen:
            continue
        seen.add(key)
        group = groups.get(key[0])
        q = quality.where(quality.group_id == gid, drop=True)
        rows.append(
            {
                "group_id": key[0],
                "resonator_id": key[1],
                "a_fit": float(strength),
                "probe_count": 0 if group is None else len(group["channels"]),
                "fit_warning": int(q.group_flag.item()),
                "fit_residual_fraction": float(q.residual_fraction.item()),
                "residual_to_sweep_variation": variation[str(gid)],
                "model_flag": 0 if group is None else group["fit_flag"],
                "route_status": "no_selected_probe" if group is None else "routed",
            }
        )
    result = xr.DataTree(
        dataset=xr.Dataset(
            {key: ("component", [row[key] for row in rows]) for key in rows[0]},
            attrs={
                "contract": "kids.fit_groups/1",
            },
        )
    )
    result["channels"] = xr.DataTree(dataset=view.chan_prop.copy().load())
    result["groups"] = xr.DataTree()
    for position, group in enumerate(groups.values()):
        result[f"groups/{position}"] = xr.DataTree(
            dataset=xr.Dataset(attrs={"model": group})
        )
    return result
