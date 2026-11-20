"""Construct compact analysis tables from fixed membership and measured sweeps."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .calibration import validate_calibration
from .fitting import fit_kids_model

MULTI_DIM = 2
MIN_FIT_SAMPLES = 3


def _table(rows, dim):
    return xr.Dataset(
        {name: (dim, np.asarray([r[name] for r in rows])) for name in rows[0]}
    )


def _validate_plan(plan, channels, sample_count):  # noqa: C901 -- Validate row layout, labels and sample bounds.
    if (
        not isinstance(plan, xr.Dataset)
        or set(plan.dims) != {"row"}
        or not plan.sizes["row"]
    ):
        msg = "plan must be a nonempty Dataset with only the row dimension"
        raise ValueError(msg)
    names = (
        "group_id",
        "chan",
        "resonator_id",
        "sample_start",
        "sample_stop",
        "fr_initial",
        "r_initial",
    )
    for name in names:
        if name not in plan or plan[name].dims != ("row",):
            msg = f"Plan requires a row column {name}"
            raise ValueError(msg)
    if any(value.dims != ("row",) for value in plan.data_vars.values()):
        msg = "All plan variables must be row columns"
        raise ValueError(msg)
    plan = plan.copy(deep=True)
    for name in ("group_id", "resonator_id"):
        if plan[name].dtype.kind not in "iuUS":
            msg = "Identifiers must be strings or integers"
            raise ValueError(msg)
        plan[name] = plan[name].astype(str)
        if np.any(plan[name].values == ""):
            msg = "Group and resonator identifiers must not be empty"
            raise ValueError(msg)
    if not set(plan.chan.values).issubset(set(channels)):
        msg = "Plan references a channel absent from the sweep"
        raise ValueError(msg)
    for name in ("sample_start", "sample_stop"):
        if plan[name].dtype.kind not in "iu":
            msg = "Segment endpoints must be integer sample positions"
            raise ValueError(msg)
    if (
        np.any(plan.sample_start < 0)
        or np.any(plan.sample_stop > sample_count)
        or np.any(plan.sample_stop <= plan.sample_start)
    ):
        msg = (
            "Plan intervals must be nonempty half-open bounds within the "
            "acquired samples"
        )
        raise ValueError(msg)
    _validate_state_columns(plan)
    _validate_identities(plan, names)
    return plan


def _validate_state_columns(plan):
    if plan.fr_initial.attrs.get("units"):
        plan["fr_initial"] = plan.fr_initial.u.to("Hz")
    if plan.r_initial.attrs.get("units"):
        plan["r_initial"] = plan.r_initial.u.to("")
    if any(
        not np.isfinite(plan[n]).all() or np.any(plan[n] <= 0)
        for n in ("fr_initial", "r_initial")
    ):
        msg = "Initial frequencies and losses must be finite and positive"
        raise ValueError(msg)
    bounds = ("fr_search_min", "fr_search_max")
    if any(n in plan for n in bounds) and not all(n in plan for n in bounds):
        msg = "Supply both fr_search_min and fr_search_max"
        raise ValueError(msg)
    for name in bounds:
        if name in plan and plan[name].attrs.get("units"):
            plan[name] = plan[name].u.to("Hz")


def _validate_identities(plan, names):
    seen = {}
    owners = {}
    keys = set()
    for row in range(plan.sizes["row"]):
        g, r = plan.group_id[row].item(), plan.resonator_id[row].item()
        state = (plan.fr_initial[row].item(), plan.r_initial[row].item())
        if r in owners and owners[r] != g:
            msg = "A resonator has one owning result group per calibration"
            raise ValueError(msg)
        owners[r] = g
        if (g, r) in seen and seen[g, r] != state:
            msg = "Conflicting initial states for one resonator"
            raise ValueError(msg)
        seen[g, r] = state
        key = tuple(plan[n][row].item() for n in names[:5])
        if key in keys:
            msg = "Duplicate association segment row"
            raise ValueError(msg)
        keys.add(key)


@dataclass
class _SweepInput:
    frequency: np.ndarray
    data: np.ndarray
    channels: np.ndarray
    selected: np.ndarray
    sigma: np.ndarray | None
    fp: np.ndarray
    unit: object


def _prepare_sweep(sweep, plan, mask, sigma, probe_frequency):
    frequency = np.atleast_2d(sweep.frequency.u.to("Hz").values)
    signal = sweep.S21
    data = np.atleast_2d(signal.values)
    if signal.ndim not in (1, 2) or data.shape != frequency.shape:
        msg = (
            "The sweep requires aligned one- or two-dimensional signal and "
            "frequency arrays"
        )
        raise ValueError(msg)
    if sweep.I.u.unit != sweep.Q.u.unit:
        msg = "I and Q must have matching signal units"
        raise ValueError(msg)
    unit = sweep.I.u.unit
    channels = np.arange(len(data))
    if signal.ndim == MULTI_DIM and signal.dims[0] in signal.coords:
        channels = signal.coords[signal.dims[0]].values
    if channels.dtype.kind not in "iuUS" or len(set(channels)) != len(channels):
        msg = "Acquired channel labels must be unique integers or strings"
        raise ValueError(msg)
    plan = _validate_plan(plan, channels, data.shape[1])
    selected = np.ones(data.shape, bool) if mask is None else np.atleast_2d(mask)
    if selected.shape != data.shape or selected.dtype != bool:
        msg = "mask must be boolean with the signal shape"
        raise ValueError(msg)
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        if signal.ndim == 1:
            sigma = sigma[None]
        if sigma.shape != (*data.shape, 2):
            msg = "sigma must have signal shape followed by the I/Q axis"
            raise ValueError(msg)
    fp = (
        np.full(len(channels), np.nan)
        if probe_frequency is None
        else np.asarray(probe_frequency, float)
    )
    if (
        fp.shape != (len(channels),)
        or np.any(np.isinf(fp))
        or np.any(fp[np.isfinite(fp)] <= 0)
    ):
        msg = (
            "probe_frequency must have one positive Hz value or NaN per "
            "acquired channel"
        )
        raise ValueError(msg)
    return _SweepInput(frequency, data, channels, selected, sigma, fp, unit), plan


def _select_samples(inputs, rows, chans, gid):
    frequency, data = inputs.frequency, inputs.data
    selected, sigma, fp = inputs.selected, inputs.sigma, inputs.fp
    lookup = {c: i for i, c in enumerate(inputs.channels)}
    fs = []
    zs = []
    ss = []
    positions = []
    fit_masks = []
    unions = []
    origins = []
    for c in chans:
        ci = lookup[c]
        union = np.zeros(data.shape[1], bool)
        for row in rows:
            if row["chan"] == c:
                union[row["sample_start"] : row["sample_stop"]] = True
        ii = np.flatnonzero(union)
        valid = (
            np.isfinite(frequency[ci, ii])
            & (frequency[ci, ii] > 0)
            & np.isfinite(data[ci, ii])
        )
        fit_mask = selected[ci, ii] & valid
        if sigma is not None:
            if np.any(~np.isfinite(sigma[ci, ii][fit_mask])) or np.any(
                sigma[ci, ii][fit_mask] <= 0
            ):
                msg = "Selected I/Q uncertainties must be finite and positive"
                raise ValueError(msg)
            ss.append(sigma[ci, ii][fit_mask])
        if fit_mask.sum() < MIN_FIT_SAMPLES:
            msg = f"Group {gid}, channel {c} has fewer than three usable samples"
            raise ValueError(msg)
        fs.append(frequency[ci, ii][fit_mask])
        zs.append(data[ci, ii][fit_mask])
        positions.append(ii)
        fit_masks.append(fit_mask)
        unions.append(valid)
        origins.append(
            fp[ci]
            if np.isfinite(fp[ci]) and fs[-1].min() <= fp[ci] <= fs[-1].max()
            else (fs[-1].min() + fs[-1].max()) / 2
        )
    return fs, zs, ss, positions, fit_masks, unions, origins


def _channel_properties(channels, fp, responses):
    channel_rows = []
    for ci, c in enumerate(channels):
        choices = [g for g, ch in responses if ch == c]
        selected_group = choices[0] if len(choices) == 1 else ""
        info = responses.get((selected_group, c))
        mode = "not_assessed"
        reason = "fixed_probe_recovery_not_assessed"
        if not choices:
            reason = "no_fit"
        elif len(choices) > 1:
            reason = "multiple_group_responses_require_selection"
        elif info["status"] != "optimizer_converged":
            mode = "invalid"
            reason = "optimizer_failed"
        elif info["n_channels"] < info["n_resonators"]:
            mode = "unresolved"
            reason = "insufficient_simultaneous_measurements"
        if (
            info
            and np.isfinite(fp[ci])
            and not info["f_min"] <= fp[ci] <= info["f_max"]
        ):
            mode = "invalid"
            reason = "probe_outside_fit_window"
        channel_rows.append(
            {
                "chan": c,
                "group_id": selected_group,
                "fp": fp[ci],
                "probe_status": "bound" if np.isfinite(fp[ci]) else "not_bound",
                "fit_status": ("not_selected" if choices else "not_fitted")
                if info is None
                else info["status"],
                "recovery_mode": mode,
                "recovery_reason": reason,
            }
        )
    return _table(channel_rows, "chan").set_coords("chan")


def _pack_group(  # noqa: PLR0913, PLR0917 -- One joint fit, labels, offsets and diagnostics.
    result, gid, chans, rids, unit_text, sigma, pstart, cstart, bound_names, rank_rtol
):
    labels, units, covariance, group_rows = [], [], [], []
    status = "optimizer_converged" if result.success else "optimizer_failed"
    fr = result.model.fr_reference
    loss = 1 / (2 * result.model.qr_reference)
    origins = result.model.gain_origin.tolist()
    for label in result.parameter_names:
        parts = label.split(":")
        if parts[0] == "resonator":
            parts[1] = rids[int(parts[1])]
        elif parts[0] == "coupling":
            parts[1] = str(chans[int(parts[1])])
            parts[2] = rids[int(parts[2])]
        else:
            parts[1] = str(chans[int(parts[1])])
        labels.append(":".join(parts))
    units.extend([u.replace("signal", unit_text) for u in result.parameter_units])
    covariance.extend(result.covariance.ravel())
    n = len(result.parameter_names)
    group_rows.append(
        {
            "group_id": gid,
            "parameter_start": pstart,
            "parameter_count": n,
            "covariance_start": cstart,
            "covariance_count": n * n,
            "optimizer": "lmfit",
            "fit_status": status,
            "message": str(result.raw.message),
            "nfev": int(result.raw.nfev),
            "rank": result.rank,
            "variable_count": len(result.raw.var_names),
            "condition": result.condition,
            "profile_condition": result.profile_condition,
            "uncertainty_status": (
                "local_covariance"
                if result.model.nonlinear_branch == "unique"
                else "branch_local_covariance"
            )
            if result.rank == len(result.raw.var_names)
            else "rank_deficient",
            "covariance_scale": "absolute_sigma"
            if sigma is not None
            else "residual_variance",
            "scales_json": json.dumps(
                {
                    "fr_reference": fr.tolist(),
                    "qr_reference": (1 / (2 * loss)).tolist(),
                    "signal_scale": result.model.signal_scale,
                    "gain_span": result.model.gain_span.tolist(),
                    "gain_origin": origins,
                }
            ),
            "initialization_json": json.dumps(result.initialization),
            "singular_values_json": json.dumps(result.singular_values.tolist()),
            "bound_names_json": json.dumps(bound_names),
            "rank_rtol": rank_rtol,
        }
    )
    return group_rows[0], labels, units, covariance


def make_calibration(  # noqa: PLR0913, PLR0915 -- Explicit analysis inputs and four-table assembly.
    sweep,
    plan,
    *,
    gain_family,
    initialization,
    max_nfev,
    qr_bounds,
    rank_rtol,
    nonlinear=False,
    nonlinear_branch="unique",
    a_nl_max=0.75,
    mask=None,
    sigma=None,
    probe_frequency=None,
    initial_params=None,
):
    inputs, plan = _prepare_sweep(sweep, plan, mask, sigma, probe_frequency)
    if initial_params is not None and not set(initial_params) <= set(
        plan.group_id.values
    ):
        msg = "Initial parameters contain unknown group IDs"
        raise ValueError(msg)
    frequency, data, channels = inputs.frequency, inputs.data, inputs.channels
    sigma, fp, unit = inputs.sigma, inputs.fp, inputs.unit
    unit_text = str(unit) if unit is not None else "unspecified"
    lookup = {c: i for i, c in enumerate(channels)}
    rows = [
        {n: plan[n][i].item() for n in plan.data_vars} for i in range(plan.sizes["row"])
    ]
    sample_rows = []
    group_rows = []
    labels = []
    units = []
    covariance = []
    optimizer_rows = []
    responses = {}
    pstart = cstart = 0
    for gid in dict.fromkeys(plan.group_id.values.tolist()):
        row_ids = np.flatnonzero(plan.group_id.values == gid)
        chans = [c for c in channels if c in set(plan.chan.values[row_ids])]
        rids = list(dict.fromkeys(plan.resonator_id.values[row_ids].tolist()))
        states = {
            r: next(rows[i] for i in row_ids if rows[i]["resonator_id"] == r)
            for r in rids
        }
        fr = np.array([states[r]["fr_initial"] for r in rids])
        loss = np.array([states[r]["r_initial"] for r in rids])
        assoc = np.array(
            [
                [
                    any(
                        rows[i]["chan"] == c and rows[i]["resonator_id"] == r
                        for i in row_ids
                    )
                    for r in rids
                ]
                for c in chans
            ]
        )
        fs, zs, ss, positions, fit_masks, unions, origins = _select_samples(
            inputs, [rows[i] for i in row_ids], chans, gid
        )
        bounds = None
        if "fr_search_min" in plan and "fr_search_max" in plan:
            for r in rids:
                entries = [rows[i] for i in row_ids if rows[i]["resonator_id"] == r]
                if any(
                    (e["fr_search_min"], e["fr_search_max"])
                    != (entries[0]["fr_search_min"], entries[0]["fr_search_max"])
                    for e in entries
                ):
                    msg = "Shared initial frequency bounds disagree"
                    raise ValueError(msg)
            bounds = (
                [states[r]["fr_search_min"] for r in rids],
                [states[r]["fr_search_max"] for r in rids],
            )
        result = fit_kids_model(
            fs,
            zs,
            fr,
            1 / (2 * loss),
            associations=assoc,
            gain_family=gain_family,
            gain_origin=origins,
            sigma=None if sigma is None else ss,
            initialization=initialization,
            max_nfev=max_nfev,
            qr_bounds=qr_bounds,
            fr_bounds=bounds,
            rank_rtol=rank_rtol,
            nonlinear=nonlinear,
            nonlinear_branch=nonlinear_branch,
            a_nl_max=a_nl_max,
            params=None if initial_params is None else initial_params.get(gid),
        )
        physical = result.model.physical(result.params)
        status = "optimizer_converged" if result.success else "optimizer_failed"
        bound_names = [
            n
            for n, p in result.params.items()
            if p.vary
            and p.expr is None
            and (
                np.isclose(p.value, p.min, rtol=1e-6, atol=1e-8)
                or np.isclose(p.value, p.max, rtol=1e-6, atol=1e-8)
            )
        ]
        for i in row_ids:
            row = rows[i]
            k = chans.index(row["chan"])
            j = rids.index(row["resonator_id"])
            c = physical["c"][k, j]
            row.update(
                fr=physical["fr"][j],
                r=physical["r"][j],
                a_nl=physical["a_nl"][j],
                c_re=c.real,
                c_im=c.imag,
                gain_family=gain_family,
                f_g=origins[k],
                gain_origin_source="actual_probe"
                if origins[k] == fp[lookup[chans[k]]]
                else "window_midpoint",
                H0_re=physical["H0"][k].real,
                H0_im=physical["H0"][k].imag,
                H1_re=physical["H1"][k].real,
                H1_im=physical["H1"][k].imag,
                gain_a=physical["a"][k],
                tau=physical["tau"][k],
                f_min=fs[k].min(),
                f_max=fs[k].max(),
                fit_status=status,
                sweep_acceptance="not_assessed",
                condition=result.condition,
                profile_condition=result.profile_condition,
                cost=float(result.raw.chisqr / 2),
                at_bound=bool(bound_names),
                signal_units=unit_text,
                initialization_used=result.initialization["used"],
                initialization_fallback=result.initialization["fallback_reason"],
            )
            row.update(Qr=1 / (2 * row["r"]), Qc=1 / (2 * c.real), A=c.imag / c.real)
        for k, c in enumerate(chans):
            ci = lookup[c]
            ii = positions[k]
            f = frequency[ci, ii]
            predicted = np.full(len(f), np.nan + 1j * np.nan)
            finite = np.isfinite(f) & (f > 0)
            probes = [
                fs[channel] if channel != k else f[finite]
                for channel in range(len(chans))
            ]
            predicted[finite] = result.model.eval(result.params, probes)[k]
            for t, index in enumerate(ii):
                z = data[ci, index]
                p = predicted[t]
                s = (np.nan, np.nan) if sigma is None else sigma[ci, index]
                sample_rows.append(
                    {
                        "group_id": gid,
                        "chan": c,
                        "sample_index": index,
                        "frequency": f[t],
                        "I": z.real,
                        "Q": z.imag,
                        "I_model": p.real,
                        "Q_model": p.imag,
                        "I_residual": (z - p).real,
                        "Q_residual": (z - p).imag,
                        "fit_selected": bool(fit_masks[k][t]),
                        "data_valid": bool(unions[k][t]),
                        "sigma_I": s[0],
                        "sigma_Q": s[1],
                    }
                )
            responses[gid, c] = {
                "status": status,
                "n_resonators": len(rids),
                "n_channels": len(chans),
                "f_min": fs[k].min(),
                "f_max": fs[k].max(),
            }
        record, group_labels, group_units, group_covariance = _pack_group(
            result,
            gid,
            chans,
            rids,
            unit_text,
            sigma,
            pstart,
            cstart,
            bound_names,
            rank_rtol,
        )
        group_rows.append(record)
        labels.extend(group_labels)
        units.extend(group_units)
        covariance.extend(group_covariance)
        # Retain numerical values, bounds and caller constraints without
        # serializing lmfit's expression runtime.
        optimizer_rows.extend(
            {
                "optimizer_name": name,
                "optimizer_value": result.params[name].value,
                "optimizer_min": result.params[name].min,
                "optimizer_max": result.params[name].max,
                "optimizer_vary": result.params[name].vary,
                "optimizer_expr": result.params[name].expr or "",
            }
            for name in result.model.names
        )
        pstart += record["parameter_count"]
        cstart += record["covariance_count"]
    fit = _table(rows, "row")
    fit.attrs["nonlinear_branch"] = nonlinear_branch
    samples = _table(sample_rows, "sample")
    joint = _table(group_rows, "group")
    joint["parameter_label"] = ("parameter", np.asarray(labels))
    joint["parameter_units"] = ("parameter", np.asarray(units))
    joint.update(_table(optimizer_rows, "parameter"))
    joint["covariance_value"] = ("covariance_entry", np.asarray(covariance))
    joint.attrs.update(
        covariance_order="row_major_full_square",
        cross_group_uncertainty="not_represented",
        conditioning_coordinates="scaled_xi_rho_coupling_gain",
        uncertainty_assumptions=(
            "independent_samples_and_quadratures; fixed_membership_and_gain_family"
        ),
    )
    chan_prop = _channel_properties(channels, fp, responses)
    _assign_units(fit, samples, chan_prop, unit, unit_text)
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                attrs={"model": "physical_resonance_group", "format_revision": 1}
            ),
            "fit": fit,
            "samples": samples,
            "joint": joint,
            "chan_prop": chan_prop,
        }
    )
    validate_calibration(tree)
    return tree


def _assign_units(fit, samples, chan_prop, unit, unit_text):
    for ds in (fit, samples, chan_prop):
        for name in (
            "fr",
            "fr_initial",
            "fr_search_min",
            "fr_search_max",
            "f_g",
            "f_min",
            "f_max",
            "frequency",
            "fp",
        ):
            if name in ds:
                ds[name].attrs["units"] = "Hz"
    for name in ("r", "r_initial", "c_re", "c_im", "Qr", "Qc", "A", "a_nl"):
        fit[name].attrs["units"] = "1"
    fit.gain_a.attrs["units"] = "1/Hz"
    fit.tau.attrs["units"] = "s"
    if unit is not None:
        for name in ("H0_re", "H0_im"):
            fit[name].attrs["units"] = str(unit)
        for name in ("H1_re", "H1_im"):
            fit[name].attrs["units"] = f"{unit}/Hz" if str(unit) else "1/Hz"
        for name in (
            "I",
            "Q",
            "I_model",
            "Q_model",
            "I_residual",
            "Q_residual",
            "sigma_I",
            "sigma_Q",
        ):
            samples[name].attrs["units"] = str(unit)
    fit.attrs.update(
        interval_convention="zero_based_half_open",
        duplicate_semantics="shared_parameter_copies",
        signal_units=unit_text,
    )
