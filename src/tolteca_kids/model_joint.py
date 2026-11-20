"""Complete physical group calibrations for native timestream consumers."""

from __future__ import annotations

import numpy as np

from tolteca_kidsproc.joint_timestream import JointTimestreamSolver
from tolteca_kidsproc.model_flags import ModelFitFlag


def joint_models(view, quality, *, iq_covariances=None, include_single=False):
    """Export selected groups, retaining every channel and resonator.

    Parameters
    ----------
    view : KidsModelView
        Saved physical fit with explicit channel-to-group selections.
    quality : xarray.Dataset
        Saved group-quality flags.
    iq_covariances : mapping, optional
        Group ID to ``(covariance, source)`` in calibration IQ units, ordered
        all I then all Q in the exported channel order. Absent entries use
        unit IQ weighting, explicitly labeled as weights rather than measured
        noise. No covariance is inferred from fit residuals.
    include_single : bool, optional
        Include single-resonator groups for a complete detector-ordered APT,
        default False. Their independent calibration rows remain available.
        Ordinary channel consumers do not need duplicate group recovery.

    Returns
    -------
    dict
        ``kids.joint/1`` payload for the native report and atomic model stub.
        Ambiguously selected groups remain in the scientific product and retain
        explicit selection flags. Nonlinearity and the full gain model are retained.
    """
    components, props = view.components(), view.chan_prop
    groups = []
    for gid in dict.fromkeys(components.group_id.values.tolist()):
        rows = components.where(components.group_id == gid, drop=True)
        valid = rows.component_valid.values.astype(bool)
        ids = list(dict.fromkeys(rows.resonator_id.values[valid].tolist()))
        channels = rows.chan.values.astype(int)
        if not ids or (len(ids) == 1 and not include_single):
            continue
        if any(props.sel(chan=c).group_id.item() != gid for c in channels):
            continue
        probes = props.sel(chan=channels).fp.values
        fr, loss, strength = [], [], []
        for rid in ids:
            mask = valid & (rows.resonator_id.values == rid)
            fr.append(float(rows.fr.values[mask][0]))
            loss.append(float(rows.r.values[mask][0]))
            strength.append(float(rows.a_nl.values[mask][0]))
        family = str(rows.gain_family.values[0])
        gain_fields = ("f_g", "H0_re", "H0_im") + (
            ("H1_re", "H1_im") if family == "affine" else ("gain_a", "tau")
        )
        gain_model = {name: rows[name].values.tolist() for name in gain_fields}
        coupling = np.zeros((len(channels), len(ids)), complex)
        for k in range(len(channels)):
            for j in np.flatnonzero(valid[k]):
                coupling[k, ids.index(rows.resonator_id.values[k, j])] = (
                    rows.c_re.values[k, j] + 1j * rows.c_im.values[k, j]
                )
        covariance, source = (iq_covariances or {}).get(
            str(gid), (np.eye(2 * len(channels)), "unit_iq_weights")
        )
        flag = _group_flag(quality, gid, channels)
        groups.append(
            {
                "group_id": str(gid),
                "gain_family": str(family),
                "channels": channels.tolist(),
                "resonator_ids": list(map(str, ids)),
                "probes": np.asarray(probes).tolist(),
                "fr": fr,
                "r": loss,
                "gain_model": gain_model,
                "a_nl": strength,
                "nonlinear_branch": components.attrs.get("nonlinear_branch", "unique"),
                "coupling_re": coupling.real.tolist(),
                "coupling_im": coupling.imag.tolist(),
                "iq_covariance": np.asarray(covariance).tolist(),
                "noise_source": str(source),
                "fit_flag": int(flag),
                "rank_rtol": 1e-8,
                "condition_max": None,
            }
        )
    payload = {"schema": "kids.joint/1", "groups": groups}
    validate_joint_models(payload, props.chan.values, props.fp.values)
    return payload


def _group_flag(quality, gid, channels):
    # Imported here because model_calibration also calls this exporter.
    from .model_calibration import _QUALITY_FLAGS  # noqa: PLC0415

    flag = ModelFitFlag(0)
    q = int(quality.where(quality.group_id == gid, drop=True).group_flag.item())
    q |= int(np.bitwise_or.reduce(quality.channel_flag.sel(chan=channels).values))
    for original, exported in _QUALITY_FLAGS.items():
        if int(q) & original:
            flag |= exported
    if flag & (ModelFitFlag.fit_not_converged | ModelFitFlag.fit_probe_outside_window):
        flag |= ModelFitFlag.fit_unusable
    return flag


def validate_joint_models(payload, channels, probes):
    """Validate group identities, probe binding and numerical calibration arrays.

    Parameters
    ----------
    payload : dict
        Complete ``kids.joint/1`` calibration payload.
    channels : array_like
        Acquired channel identities in the containing report.
    probes : array_like
        Corresponding acquired probe frequencies in Hz.

    Returns
    -------
    None
        Validation does not change the payload. Rank deficiency remains a
        valid calibration with failure diagnostics at recovery.

    Raises
    ------
    ValueError
        Schema, identities, probe binding or numerical arrays are invalid.
    """
    if set(payload) != {"schema", "groups"} or payload["schema"] != "kids.joint/1":
        msg = "Unsupported joint calibration schema"
        raise ValueError(msg)
    lookup = dict(zip(map(int, channels), probes, strict=True))
    seen_groups, seen_channels = set(), set()
    for group in payload["groups"]:
        required = {
            "group_id",
            "gain_family",
            "channels",
            "resonator_ids",
            "probes",
            "fr",
            "r",
            "gain_model",
            "a_nl",
            "nonlinear_branch",
            "coupling_re",
            "coupling_im",
            "iq_covariance",
            "noise_source",
            "fit_flag",
            "rank_rtol",
            "condition_max",
        }
        if set(group) != required or group["gain_family"] not in {"affine", "delay"}:
            msg = "Invalid joint calibration fields or gain family"
            raise ValueError(msg)
        ids = group["channels"]
        if (
            not isinstance(group["group_id"], str)
            or not group["group_id"]
            or group["group_id"] in seen_groups
            or any(type(c) is not int or c not in lookup for c in ids)
            or seen_channels.intersection(ids)
            or not np.array_equal(group["probes"], [lookup[c] for c in ids])
            or any(not isinstance(r, str) or not r for r in group["resonator_ids"])
            or not isinstance(group["noise_source"], str)
            or not group["noise_source"]
            or type(group["fit_flag"]) is not int
            or not 0 <= group["fit_flag"] <= np.iinfo(np.uint32).max
        ):
            msg = "Joint calibration identities or acquired probes disagree"
            raise ValueError(msg)
        seen_groups.add(group["group_id"])
        seen_channels.update(ids)
        joint_solver(group)


def joint_solver(group, **options):
    """Prepare a solver from a complete group in a model report.

    Parameters
    ----------
    group : dict
        Selected group, including full gain parameters and drive nonlinearity.
    **options
        Additional JointTimestreamSolver options, such as frequency bounds.

    Returns
    -------
    JointTimestreamSolver
        Solver at the recorded probes with fixed drive coefficients.
    """
    if group["gain_family"] not in {"affine", "delay"}:
        msg = "Unsupported joint gain family"
        raise ValueError(msg)
    strength = np.asarray(group["a_nl"])
    if (
        strength.shape != (len(group["resonator_ids"]),)
        or not np.isfinite(strength).all()
        or np.any(strength < 0)
    ):
        msg = "Joint nonlinear strengths must match the resonators"
        raise ValueError(msg)
    gain = group["gain_model"]
    fields = {"f_g", "H0_re", "H0_im"} | (
        {"H1_re", "H1_im"} if group["gain_family"] == "affine" else {"gain_a", "tau"}
    )
    if set(gain) != fields:
        msg = "Joint gain fields differ from their family"
        raise ValueError(msg)
    if any(
        np.asarray(gain[k]).shape != (len(group["channels"]),)
        or not np.isfinite(gain[k]).all()
        for k in fields
    ):
        msg = "Joint gain arrays must match the channels"
        raise ValueError(msg)
    delta = np.asarray(group["probes"]) - gain["f_g"]
    h = np.asarray(gain["H0_re"]) + 1j * np.asarray(gain["H0_im"])
    if group["gain_family"] == "affine":
        h += (np.asarray(gain["H1_re"]) + 1j * np.asarray(gain["H1_im"])) * delta
    else:
        h *= np.exp(
            (np.asarray(gain["gain_a"]) - 2j * np.pi * np.asarray(gain["tau"])) * delta
        )
    return JointTimestreamSolver(
        group["probes"],
        group["fr"],
        group["r"],
        h,
        _complex_array(
            group, "coupling", (len(group["channels"]), len(group["resonator_ids"]))
        ),
        group["iq_covariance"],
        channels=group["channels"],
        resonator_ids=group["resonator_ids"],
        b=2 * strength * np.asarray(group["r"]) ** 3,
        nonlinear_branch=group["nonlinear_branch"],
        fit_flag=group["fit_flag"],
        rank_rtol=group["rank_rtol"],
        condition_max=group["condition_max"],
        **options,
    )


def _complex_array(group, key, shape):
    real, imag = np.asarray(group[f"{key}_re"]), np.asarray(group[f"{key}_im"])
    if real.shape != shape or imag.shape != shape:
        msg = f"Joint {key} dimensions disagree with its identities"
        raise ValueError(msg)
    return real + 1j * imag
