"""Compact calibration tables and vectorized physical-response evaluation."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._nonlinear import nonlinear_detuning, validate_nonlinearity
from .models import KidsSweepWithAffineGain, KidsSweepWithDelayGain

__all__ = [
    "calibration_channels",
    "calibration_components",
    "calibration_pair_model",
    "evaluate_calibration",
    "group_covariance",
    "validate_calibration",
]


def calibration_channels(fit, channels, probe_frequency):  # noqa: C901 -- Route and measurement-count checks stay together.
    """Resolve fixed-probe routes within the measured ranges of saved fits.

    Parameters
    ----------
    fit : xarray.Dataset
        Channel/resonator/segment fit rows, including measured ``f_min,f_max``.
    channels : array_like
        Unique acquired channel identities, in acquisition order.
    probe_frequency : array_like
        Positive recorded probe frequencies in Hz, or NaN if not bound.

    Returns
    -------
    xarray.Dataset
        Channel properties without refitting or changing scientific fit rows.
        A bound probe selects a group only when exactly one fitted range contains
        it. Multiple containing groups remain unselected. Channels used only at
        sweep edges do not become fixed-probe measurements of that group.
        Selection does not certify fit adequacy or fixed-probe identifiability.

    Raises
    ------
    ValueError
        Channel identities or probe frequencies are invalid.
    """
    channels, fp = np.asarray(channels), np.asarray(probe_frequency, float)
    if (
        channels.ndim != 1
        or len(channels) == 0
        or len(set(channels.tolist())) != len(channels)
        or fp.shape != channels.shape
        or np.any(np.isinf(fp))
        or np.any(fp[np.isfinite(fp)] <= 0)
    ):
        msg = "Require unique channels and one positive frequency or NaN per channel"
        raise ValueError(msg)
    records = []
    for c, probe in zip(channels, fp, strict=True):
        rows = fit.isel(row=np.flatnonzero(fit.chan.values == c))
        groups = list(dict.fromkeys(rows.group_id.values.tolist()))
        choices = groups
        if np.isfinite(probe):
            choices = [
                g
                for g in groups
                if bool(
                    (
                        (rows.group_id == g)
                        & (rows.f_min <= probe)
                        & (rows.f_max >= probe)
                    ).any()
                )
            ]
        gid = choices[0] if len(choices) == 1 else ""
        mode, reason = "not_assessed", "fixed_probe_recovery_not_assessed"
        status = "not_selected" if groups else "not_fitted"
        if not groups:
            reason = "no_fit"
        elif not choices:
            mode, reason = "invalid", "probe_outside_fit_window"
        elif len(choices) > 1:
            reason = "multiple_group_responses_require_selection"
        else:
            selected = rows.where(rows.group_id == gid, drop=True)
            status = str(selected.fit_status.values[0])
            if status != "optimizer_converged":
                mode, reason = "invalid", "optimizer_failed"
        records.append(
            {
                "chan": c,
                "group_id": gid,
                "fp": probe,
                "probe_status": "bound" if np.isfinite(probe) else "not_bound",
                "fit_status": status,
                "recovery_mode": mode,
                "recovery_reason": reason,
            }
        )
    # Count selected simultaneous probes, not all channels with sweep samples.
    for record in records:
        gid = record["group_id"]
        if not gid or record["recovery_mode"] != "not_assessed":
            continue
        ids = set(fit.resonator_id.values[fit.group_id.values == gid])
        count = sum(r["group_id"] == gid for r in records)
        if count < len(ids):
            record.update(
                recovery_mode="unresolved",
                recovery_reason="insufficient_simultaneous_measurements",
            )
    return xr.Dataset(
        {k: ("chan", [r[k] for r in records]) for k in records[0]}
    ).set_coords("chan")


def _consistent(table, keys, fields):
    seen = {}
    for i in range(table.sizes["row"]):
        key = tuple(table[k].values[i].item() for k in keys)
        value = tuple(table[f].values[i].item() for f in fields)
        if key in seen:
            for a, b in zip(value, seen[key], strict=True):
                if not (
                    a == b
                    or (
                        isinstance(a, float)
                        and isinstance(b, float)
                        and np.isnan(a)
                        and np.isnan(b)
                    )
                ):
                    msg = f"Conflicting shared values for {key}: {fields}"
                    raise ValueError(msg)
        seen[key] = value


def _validate_nonlinearity(table):
    validate_nonlinearity(table.a_nl, table.attrs.get("nonlinear_branch", "unique"))


def validate_calibration(tree):
    """Validate shared values and packed covariance before using a calibration.

    Parameters
    ----------
    tree : xarray.DataTree
        Analysis namespace containing ``fit,samples,joint,chan_prop``.
        Validation loads their scalar columns; file-backed sources must be open.

    Returns
    -------
    None
        Successful validation has no side effects.

    Raises
    ------
    ValueError
        Tied copies, segment bounds, derived values or covariance offsets disagree.
    KeyError
        A required node or column is missing.
    """
    table = tree["fit"].to_dataset()
    _validate_nonlinearity(table)
    _validate_relations(tree, table)
    joint = tree["joint"].to_dataset()
    _consistent(
        table,
        ["group_id", "resonator_id"],
        ["fr_initial", "r_initial", "fr", "r", "a_nl"],
    )
    _consistent(
        table,
        ["group_id", "chan"],
        ["gain_family", "f_g", "H0_re", "H0_im", "H1_re", "H1_im", "gain_a", "tau"],
    )
    _consistent(table, ["group_id", "chan", "resonator_id"], ["c_re", "c_im"])
    if np.any(table.sample_stop.values <= table.sample_start.values) or np.any(
        table.sample_start.values < 0
    ):
        msg = "Segment intervals must be nonempty and half-open"
        raise ValueError(msg)
    for name, expected in [
        ("Qr", 1 / (2 * table.r.values)),
        ("Qc", 1 / (2 * table.c_re.values)),
        ("A", table.c_im.values / table.c_re.values),
    ]:
        if not np.allclose(
            table[name].values, expected, rtol=1e-12, atol=0, equal_nan=True
        ):
            msg = f"Derived column {name} disagrees with physical parameters"
            raise ValueError(msg)
    ps = cs = 0
    for i in range(joint.sizes["group"]):
        n = int(joint.parameter_count[i])
        count = int(joint.covariance_count[i])
        if (
            n < 1
            or int(joint.parameter_start[i]) != ps
            or int(joint.covariance_start[i]) != cs
            or count != n * n
        ):
            msg = "Invalid packed covariance offsets or dimensions"
            raise ValueError(msg)
        labels = joint.parameter_label.values[ps : ps + n]
        if len(set(labels)) != n:
            msg = "Covariance parameter identities must be unique within each group"
            raise ValueError(msg)
        ps += n
        cs += count
    if ps != joint.sizes.get("parameter", 0) or cs != joint.sizes.get(
        "covariance_entry", 0
    ):
        msg = "Packed covariance payload has an inconsistent length"
        raise ValueError(msg)
    samples = tree["samples"].to_dataset()
    keys = list(
        zip(
            samples.group_id.values,
            samples.chan.values,
            samples.sample_index.values,
            strict=True,
        )
    )
    if len(set(keys)) != len(keys):
        msg = "A measured sample occurs twice within one group/channel"
        raise ValueError(msg)
    channels = tree["chan_prop"].to_dataset()
    if len(set(channels.chan.values)) != channels.sizes["chan"]:
        msg = "chan_prop must contain one row per acquired channel"
        raise ValueError(msg)


def calibration_components(tree):
    """Build a padded response-by-component view for vectorized evaluation.

    Parameters
    ----------
    tree : xarray.DataTree
        Compact analysis namespace. Shared parameters are validated first.

    Returns
    -------
    xarray.Dataset
        ``response,component`` arrays with explicit validity and resonator IDs.
        One response is a ``(group_id,chan)`` pair. Gain has only the response
        axis. Several segments for one association produce one component slot.
        The nonlinear_branch attribute retains the declared sweep history.
        This detached working view does not modify the stored calibration.
    """
    validate_calibration(tree)
    table = tree["fit"].to_dataset()
    keys = list(
        dict.fromkeys(
            zip(table.group_id.values.tolist(), table.chan.values.tolist(), strict=True)
        )
    )
    contributions = {key: {} for key in keys}
    for i, (g, c, r) in enumerate(
        zip(
            table.group_id.values,
            table.chan.values,
            table.resonator_id.values,
            strict=True,
        )
    ):
        contributions[g, c].setdefault(r, i)
    width = max(map(len, contributions.values()))
    shape = (len(keys), width)
    valid = np.zeros(shape, bool)
    ids = np.full(shape, "", dtype=table.resonator_id.dtype)
    physical = {
        name: np.full(shape, np.nan) for name in ("fr", "r", "a_nl", "c_re", "c_im")
    }
    gains = {
        name: []
        for name in (
            "gain_family",
            "f_g",
            "H0_re",
            "H0_im",
            "H1_re",
            "H1_im",
            "gain_a",
            "tau",
        )
    }
    for k, key in enumerate(keys):
        for j, (rid, index) in enumerate(contributions[key].items()):
            ids[k, j] = rid
            valid[k, j] = True
            for name, values in physical.items():
                values[k, j] = table[name].values[index]
        index = next(iter(contributions[key].values()))
        for name, values in gains.items():
            values.append(table[name].values[index])
    return xr.Dataset(
        {
            "component_valid": (("response", "component"), valid),
            "resonator_id": (("response", "component"), ids),
            **{n: (("response", "component"), v) for n, v in physical.items()},
            **{n: ("response", v) for n, v in gains.items()},
        },
        coords={
            "group_id": ("response", [k[0] for k in keys]),
            "chan": ("response", [k[1] for k in keys]),
        },
        attrs={"nonlinear_branch": table.attrs.get("nonlinear_branch", "unique")},
    )


def calibration_pair_model(tree, group_id, chan, resonator_id):
    """Select one channel-resonator model from compact calibration tables.

    Parameters
    ----------
    tree : xarray.DataTree
        Analysis namespace containing ``fit,samples,joint,chan_prop``. Keep
        backing files open during selection; the returned model is detached.
    group_id, chan, resonator_id : scalar
        Explicit group, acquired-channel and resonator labels, not positions.
        Repeated segments of one association select the same model.

    Returns
    -------
    KidsSweepWithAffineGain or KidsSweepWithDelayGain
        Physical-parameter snapshot for the selected association. ``eval``
        predicts that resonance alone. ``rotate`` and ``derotate`` transform
        all supplied IQ, including neighbours and noise. Later table changes
        do not update this snapshot. Full covariance remains in ``joint``.

    Raises
    ------
    KeyError
        The association does not exist.
    ValueError
        Calibration duplicates or other stored relations disagree.
    """
    validate_calibration(tree)
    table = tree["fit"].to_dataset()
    indices = np.flatnonzero(
        (table.group_id.values == str(group_id))
        & (table.chan.values == chan)
        & (table.resonator_id.values == str(resonator_id))
    )
    if not len(indices):
        raise KeyError((group_id, chan, resonator_id))
    # Compute the selected scalars so Dask-backed tables also detach cleanly.
    row = table.isel(row=indices[0]).compute()
    values = {
        "fr": row.fr.item(),
        "r": row.r.item(),
        "c": row.c_re.item() + 1j * row.c_im.item(),
        "f_g": row.f_g.item(),
        "H0": row.H0_re.item() + 1j * row.H0_im.item(),
        "a_nl": row.a_nl.item(),
        "nonlinear_branch": table.attrs.get("nonlinear_branch", "unique"),
    }
    if row.gain_family.item() == "affine":
        return KidsSweepWithAffineGain(
            **values, H1=row.H1_re.item() + 1j * row.H1_im.item()
        )
    return KidsSweepWithDelayGain(**values, a=row.gain_a.item(), tau=row.tau.item())


def evaluate_calibration(tree, group_id, chan, frequency):
    """Evaluate selected channel responses without counting repeated segments twice.

    Parameters
    ----------
    tree : xarray.DataTree
        Compact analysis namespace, including reopened Zarr data.
    group_id, chan, frequency : array_like
        Broadcastable group labels, channel labels and probe frequencies in Hz.
        Labels must match the fit table. Input frequency must be finite/positive.

    Returns
    -------
    numpy.ndarray
        Complex predictions with the broadcast input shape. Missing/unfitted
        physical parameters yield NaN. A prediction does not imply acceptance.

    Raises
    ------
    KeyError
        A requested group/channel response does not exist.
    ValueError
        Calibration duplicates conflict or frequency is invalid.
    """
    view = calibration_components(tree)
    gg, cc, ff = np.broadcast_arrays(
        np.asarray(group_id).astype(str), chan, np.asarray(frequency, float)
    )
    if not np.isfinite(ff).all() or np.any(ff <= 0):
        msg = "Probe frequencies must be finite and positive"
        raise ValueError(msg)
    lookup = {
        key: i
        for i, key in enumerate(
            zip(view.group_id.values.tolist(), view.chan.values.tolist(), strict=True)
        )
    }
    indices = np.array(
        [lookup[g, c] for g, c in zip(gg.ravel(), cc.ravel(), strict=True)]
    )
    valid = view.component_valid.values[indices]
    fr = np.where(valid, view.fr.values[indices], 1.0)
    r = np.where(valid, view.r.values[indices], 1.0)
    c = np.where(valid, view.c_re.values[indices] + 1j * view.c_im.values[indices], 0.0)
    a_nl = np.where(valid, view.a_nl.values[indices], 0.0)
    if np.any(a_nl):
        y, _ = nonlinear_detuning(
            (ff.ravel()[:, None] / fr - 1) / (2 * r),
            a_nl,
            branch=tree["fit"].attrs.get("nonlinear_branch", "unique"),
        )
        denominator = r * (1 + 2j * y)
    else:
        denominator = r + 1j * (ff.ravel()[:, None] / fr - 1)
    kernel = 1 - np.sum(c / denominator, axis=1)
    delta = ff.ravel() - view.f_g.values[indices]
    h0 = view.H0_re.values[indices] + 1j * view.H0_im.values[indices]
    gain = np.empty(len(indices), complex)
    affine = view.gain_family.values[indices] == "affine"
    gain[affine] = (
        h0[affine]
        + (view.H1_re.values[indices][affine] + 1j * view.H1_im.values[indices][affine])
        * delta[affine]
    )
    delay = ~affine
    gain[delay] = h0[delay] * np.exp(
        (
            view.gain_a.values[indices][delay]
            - 2j * np.pi * view.tau.values[indices][delay]
        )
        * delta[delay]
    )
    return (gain * kernel).reshape(ff.shape)


def group_covariance(tree, group_id):
    """Recover a labelled physical covariance block without discarding correlations.

    Parameters
    ----------
    tree : xarray.DataTree
        Compact analysis namespace.
    group_id : str
        Joint fitting group label.

    Returns
    -------
    xarray.DataArray
        Square covariance on labelled physical parameter axes. Parameter units
        are coordinate arrays. The block excludes any unrepresented common
        uncertainty; absent cross-group blocks do not establish independence.

    Raises
    ------
    KeyError
        The requested group has no uncertainty block.
    """
    validate_calibration(tree)
    joint = tree["joint"].to_dataset()
    ii = np.flatnonzero(joint.group_id.values == str(group_id))
    if len(ii) != 1:
        raise KeyError(group_id)
    i = ii[0]
    p = int(joint.parameter_start[i])
    n = int(joint.parameter_count[i])
    c = int(joint.covariance_start[i])
    labels = joint.parameter_label.values[p : p + n]
    units = joint.parameter_units.values[p : p + n]
    return xr.DataArray(
        joint.covariance_value.values[c : c + n * n].reshape(n, n),
        dims=("parameter", "parameter_other"),
        coords={
            "parameter": labels,
            "parameter_other": labels,
            "parameter_units": ("parameter", units),
            "parameter_other_units": ("parameter_other", units),
        },
        attrs={
            "uncertainty_status": joint.uncertainty_status[i].values.item(),
            "cross_group_uncertainty": "not_represented",
        },
    )


def _validate_relations(tree, table):
    groups = tree["joint"].to_dataset().group_id.values.tolist()
    if len(groups) != len(set(groups)) or set(groups) != set(table.group_id.values):
        msg = "Joint groups must match the fit-table groups uniquely"
        raise ValueError(msg)
    if not set(table.gain_family.values).issubset({"affine", "delay"}):
        msg = "Unknown physical gain family"
        raise ValueError(msg)
    row_keys = list(
        zip(
            *(
                table[name].values.tolist()
                for name in (
                    "group_id",
                    "chan",
                    "resonator_id",
                    "sample_start",
                    "sample_stop",
                )
            ),
            strict=True,
        )
    )
    if len(row_keys) != len(set(row_keys)):
        msg = "Duplicate association segment row"
        raise ValueError(msg)
    responses = set(zip(table.group_id.values, table.chan.values, strict=True))
    samples = tree["samples"].to_dataset()
    if not set(zip(samples.group_id.values, samples.chan.values, strict=True)).issubset(
        responses
    ):
        msg = "Sample records reference an absent response"
        raise ValueError(msg)
    channels = tree["chan_prop"].to_dataset()
    if not set(table.chan.values).issubset(set(channels.chan.values)):
        msg = "Fit rows reference channels absent from chan_prop"
        raise ValueError(msg)
    for g, c in zip(channels.group_id.values, channels.chan.values, strict=True):
        if g and (g, c) not in responses:
            msg = "Selected channel response does not exist"
            raise ValueError(msg)
