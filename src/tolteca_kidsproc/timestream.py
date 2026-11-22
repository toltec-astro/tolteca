"""Selected-array recovery using explicit, channel-bound sweep calibrations."""

# ruff: noqa: E741, N803 -- Preserve physical I/Q names.
from __future__ import annotations

from enum import IntFlag

import numpy as np
import xarray as xr

from .model_flags import ModelFitFlag
from .recovery import SampleFlag

__all__ = ["ModelFlag", "TimestreamSolver", "solve_timestream"]


class ModelFlag(IntFlag):
    """Native-compatible model validity, separate from source fit flags."""

    model_good = 0
    nonphysical_model = 1
    unusable_model = 2


_FIT_UNUSABLE = int(ModelFitFlag.fit_unusable | ModelFitFlag.fit_joint_required)
_FIXED_B_VERSION = 2
_LEGACY = (
    "fp",
    "Qr",
    "Qc",
    "fr",
    "A",
    "normI",
    "normQ",
    "slopeI",
    "slopeQ",
    "interceptI",
    "interceptQ",
)
_PHYSICAL = ("f_g", "fr", "r", "c_re", "c_im", "H0_re", "H0_im")
_SCHEMAS = {
    "S21WithGainLinTrend": _LEGACY,
    "KidsSweepWithAffineGain": (*_PHYSICAL, "H1_re", "H1_im"),
    "KidsSweepWithDelayGain": (*_PHYSICAL, "a", "tau"),
}
_UNITS = {
    "fp": "Hz",
    "fr": "Hz",
    "f_g": "Hz",
    "H1_re": "1 / Hz",
    "H1_im": "1 / Hz",
    "a": "1 / Hz",
    "tau": "s",
}


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)
    return float(value)


def _selected_table(table, channels):
    """Validate the generic numerical schema and select by channel identity."""
    schema = table.meta.get("model_schema", {})
    name = schema.get("name")
    version = schema.get("version")
    if (
        name not in _SCHEMAS
        or version not in (1, _FIXED_B_VERSION)
        or (name == "S21WithGainLinTrend" and version != 1)
    ):
        msg = "Require legacy schema version 1 or physical schema version 1/2"
        raise ValueError(msg)
    names = (*_SCHEMAS[name], "b") if version == _FIXED_B_VERSION else _SCHEMAS[name]
    expected = [{"name": n, "unit": _UNITS.get(n, "")} for n in names]
    if sorted(schema.get("parameters", []), key=lambda p: p["name"]) != sorted(
        expected, key=lambda p: p["name"]
    ):
        msg = "Model parameter names or units disagree with the numerical schema"
        raise ValueError(msg)
    ids, selected = np.asarray(table["chan"]), np.asarray(channels)
    if (
        ids.ndim != 1
        or selected.ndim != 1
        or ids.dtype.kind not in "iu"
        or selected.dtype.kind not in "iu"
        or len(set(ids)) != len(ids)
        or len(set(selected)) != len(selected)
        or not len(selected)
    ):
        msg = "Require unique integer calibration and selected channel IDs"
        raise ValueError(msg)
    lookup = {int(c): i for i, c in enumerate(ids)}
    try:
        result = table[[lookup[int(c)] for c in selected]].copy(copy_data=True)
    except KeyError as exc:
        msg = f"No calibration for selected channel {exc.args[0]}"
        raise ValueError(msg) from exc
    for field in ("f_in", *names):
        column = result[field]
        unit = "Hz" if field == "f_in" else _UNITS.get(field, "")
        if str(column.unit or "") != unit or np.ma.getmaskarray(column).any():
            msg = f"Model column {field} requires unmasked values with units {unit!r}"
            raise ValueError(msg)
    if "flag" in result.colnames:
        flags = np.asarray(result["flag"])
        if (
            flags.dtype.kind not in "iu"
            or np.any(flags < 0)
            or np.any(flags > 2**32 - 1)
        ):
            msg = "Fit flags must be unsigned 32-bit values"
            raise ValueError(msg)
    return result


class TimestreamSolver:
    """Prepare reusable per-channel solvers from an explicit model table.

    Parameters
    ----------
    model_params : astropy.table.Table
        ``model_schema`` version 1 for S21WithGainLinTrend, or version 1/2 for
        KidsSweepWithAffineGain and KidsSweepWithDelayGain; columns ``chan``,
        ``f_in`` (Hz), the declared parameters, and optional ``flag`` fit bits.
        This accepts the numerical portion of a kids.fit/1 report. It does not
        validate its acquisition binding. Version 2 adds the dimensionless
        fixed drive coefficient ``b = 2*a_nl*r_calibration**3``. Joint rows marked
        fit_unusable remain invalid; they are never inverted independently.
    channels : sequence of int
        Original acquired channel labels in the desired input/output order.
        Each must have exactly one model row. Extra calibration rows are allowed.

    Notes
    -----
    Preparation copies the selected calibration. Calls do not mutate inputs or
    share numerical scratch buffers. Source fit flags and model validity remain
    separate. The historical model preserves its effective r/x convention;
    the two physical models recover loss r and intrinsic detuning x. Version 1
    sets b to zero. Version 2 uses x_opt = x_prime - b/(r**2 + x_prime**2),
    holding b, gain and coupling fixed as the detector changes with loading.
    """

    def __init__(self, model_params, channels):
        self.model_params = _selected_table(model_params, channels)
        self.channels = np.asarray(self.model_params["chan"]).copy()
        self.b = (
            np.asarray(self.model_params["b"], float).copy()
            if self.model_params.meta["model_schema"]["version"] == _FIXED_B_VERSION
            else np.zeros(len(self.channels))
        )
        self.coefficients, self.flags = self._prepare()

    def _prepare(self):
        t = self.model_params
        name = t.meta["model_schema"]["name"]
        p = {n: np.asarray(t[n], float) for n in _SCHEMAS[name]}
        f = np.asarray(t["f_in"], float)
        flags = np.zeros(len(t), dtype=np.uint8)
        if "flag" in t.colnames:
            flags[np.asarray(t["flag"]) & _FIT_UNUSABLE != 0] |= int(
                ModelFlag.unusable_model
            )
        with np.errstate(all="ignore"):
            if name != "S21WithGainLinTrend":
                h = p["H0_re"] + 1j * p["H0_im"]
                df = f - p["f_g"]
                if name == "KidsSweepWithAffineGain":
                    h = h + (p["H1_re"] + 1j * p["H1_im"]) * df
                else:
                    h = h * np.exp((p["a"] - 2j * np.pi * p["tau"]) * df)
                bad = (
                    ~np.isfinite(np.array(list(p.values()))).all(axis=0)
                    | (p["r"] <= 0)
                    | (p["fr"] <= 0)
                    | (p["f_g"] <= 0)
                    | (p["c_re"] <= 0)
                    | ~np.isfinite(self.b)
                    | (self.b < 0)
                )
                flags[bad] |= int(ModelFlag.nonphysical_model) | int(
                    ModelFlag.unusable_model
                )
                p = {
                    "fp": p["f_g"],
                    "fr": p["fr"],
                    "Qr": 0.5 / p["r"],
                    "Qc": 0.5 / p["c_re"],
                    "A": p["c_im"] / p["c_re"],
                    "normI": -h.real,
                    "normQ": -h.imag,
                    "slopeI": np.zeros(len(t)),
                    "slopeQ": np.zeros(len(t)),
                    "interceptI": h.real,
                    "interceptQ": h.imag,
                }
            norm2 = p["normI"] ** 2 + p["normQ"] ** 2
            bad = (
                ~np.isfinite(np.array(list(p.values()))).all(axis=0)
                | ~np.isfinite(f)
                | (f <= 0)
                | (p["fp"] <= 0)
                | (p["Qr"] <= 0)
                | (p["Qc"] <= 0)
                | (p["fr"] <= 0)
                | ~np.isfinite(norm2)
                | (norm2 <= 0)
            )
            flags[bad] |= int(ModelFlag.nonphysical_model)
            coefficients = np.array(
                [
                    p["interceptI"] + (f - p["fp"]) * p["slopeI"],
                    p["interceptQ"] + (f - p["fp"]) * p["slopeQ"],
                    p["normI"],
                    p["normQ"],
                    norm2,
                    p["Qc"],
                    p["A"],
                ]
            )
            unusable = (
                (flags & int(ModelFlag.unusable_model) != 0)
                | ~np.isfinite(coefficients).all(axis=0)
                | (norm2 <= 0)
                | (p["Qc"] == 0)
            )
        flags[unusable] |= int(ModelFlag.unusable_model)
        coefficients[:, unusable] = np.array([0, 0, 1, 0, 1, 1, 0])[:, None]
        return coefficients, flags

    def __call__(self, I, Q, *, normalization=1.0):
        """Recover arrays in prepared channel order.

        Parameters
        ----------
        I, Q : array_like
            Real arrays shaped ``(..., channel)`` in the calibration signal
            units multiplied by normalization. Nonfinite samples are flagged.
        normalization : float, optional
            Positive I/Q divisor, default 1. The instrument binding supplies
            the data/model accumulation-length ratio for raw ADC accumulations.

        Returns
        -------
        tuple of numpy.ndarray
            ``(r, x, flags)`` with the input shape; flags use SampleFlag bits.
            Failed samples are NaN in both signals. Physical-model estimates
            with nonpositive loss or resonance frequency have outside_domain.
            A nonpositive stationary slope has unstable_response. These samples
            are excluded even when the algebraic solution is finite.
            The legacy model retains its historical effective coordinates.
            Flags use uint16; inputs are unchanged.

        Raises
        ------
        ValueError
            Shapes, channel count or normalization are invalid.
        """
        I, Q = np.asarray(I, float), np.asarray(Q, float)
        norm = _positive(normalization, "normalization")
        if I.shape != Q.shape or I.ndim < 1 or I.shape[-1] != len(self.channels):
            msg = "I/Q shape must agree and end with the prepared channel axis"
            raise ValueError(msg)
        mi, mq, gi, gq, norm2, qc, asymmetry = self.coefficients
        with np.errstate(all="ignore"):
            di, dq = I / norm - mi, Q / norm - mq
            u, v = (di * gi + dq * gq) / norm2, (dq * gi - di * gq) / norm2
            denominator = (u**2 + v**2) * qc * 2
            x = (u * asymmetry - v) / denominator
            r = (v * asymmetry + u) / denominator
            slope = np.ones((), dtype=float)
            nonlinear = self.b != 0
            if np.any(nonlinear):
                norm_state = r**2 + x**2
                slope = 1 + np.divide(
                    2 * self.b * x,
                    norm_state**2,
                    out=np.zeros_like(x),
                    where=nonlinear,
                )
                x -= np.divide(
                    self.b, norm_state, out=np.zeros_like(x), where=nonlinear
                )
        flags = np.zeros(I.shape, dtype=np.uint16)
        flags[~np.isfinite(I) | ~np.isfinite(Q)] |= int(SampleFlag.nonfinite_input)
        flags[denominator == 0] |= int(SampleFlag.singular_response)
        flags[
            ~np.isfinite(denominator)
            | ~np.isfinite(r)
            | ~np.isfinite(x)
            | ~np.isfinite(slope)
        ] |= int(SampleFlag.nonfinite_output)
        if self.model_params.meta["model_schema"]["name"] != "S21WithGainLinTrend":
            outside = np.isfinite(r) & np.isfinite(x) & ((r <= 0) | (x <= -1))
            flags[outside] |= int(SampleFlag.outside_domain)
            flags[np.isfinite(slope) & (slope <= 0)] |= int(
                SampleFlag.unstable_response
            )
        invalid = self.flags & int(ModelFlag.unusable_model) != 0
        flags[..., invalid] = (
            flags[..., invalid] & int(SampleFlag.nonfinite_input)
        ) | int(SampleFlag.invalid_model)
        return np.where(flags, np.nan, r), np.where(flags, np.nan, x), flags


def solve_timestream(signals, model_params, *, normalization=1.0, chunk_samples=4096):
    """Solve a selected channel/time Dataset without eagerly loading its samples.

    Parameters
    ----------
    signals : xarray.Dataset
        Real I/Q arrays with exactly ``chan,time`` dimensions in either order.
        ``chan`` contains original acquired labels. Apply isel/sel before this
        call to restrict channels and samples; coordinates are preserved.
    model_params : astropy.table.Table
        Explicit calibration accepted by TimestreamSolver. Selection aligns by
        chan labels. Sample loading is deferred; model validation is immediate.
    normalization : float, optional
        Positive I/Q divisor, default 1. No instrument normalization is inferred.
    chunk_samples : int, optional
        Maximum time chunk for numerical evaluation, default 4096. Disk read
        granularity also depends on source chunks. Keep source files open until
        results are computed or written.

    Returns
    -------
    xarray.Dataset
        Lazy r/x/flags on the supplied axes, calibration_flags per channel, explicit
        normalization, model name and sample-flag definitions. Signals are not
        copied into the result, and neither input is mutated.

    Raises
    ------
    ValueError
        Signal axes or calibration identities/schema are inconsistent.
    """
    if (
        signals.I.dims != signals.Q.dims
        or set(signals.I.dims) != {"chan", "time"}
        or signals.chan.dims != ("chan",)
        or not isinstance(chunk_samples, int)
        or chunk_samples < 1
    ):
        msg = "Require matching chan/time I/Q axes and a positive chunk_samples"
        raise ValueError(msg)
    normalization = _positive(normalization, "normalization")
    solver = TimestreamSolver(model_params, signals.chan.values)
    arrays = xr.apply_ufunc(
        solver,
        signals.I.chunk({"time": chunk_samples, "chan": -1}),
        signals.Q.chunk({"time": chunk_samples, "chan": -1}),
        input_core_dims=[["chan"], ["chan"]],
        output_core_dims=[["chan"]] * 3,
        kwargs={"normalization": normalization},
        dask="parallelized",
        output_dtypes=[float, float, np.uint16],
    )
    out = xr.Dataset(
        {
            n: a.transpose(*signals.I.dims)
            for n, a in zip(("r", "x", "flags"), arrays, strict=True)
        }
    )
    for name in ("r", "x"):
        out[name].attrs["units"] = ""
    out["flags"].attrs.update(
        flag_masks=[int(f) for f in SampleFlag],
        flag_meanings=" ".join(f.name for f in SampleFlag),
    )
    out["calibration_flags"] = ("chan", solver.flags)
    out.calibration_flags.attrs.update(
        flag_masks=[int(f) for f in ModelFlag],
        flag_meanings=" ".join(f.name for f in ModelFlag),
    )
    if "flag" in solver.model_params.colnames:
        out["fit_flag"] = ("chan", np.asarray(solver.model_params["flag"]))
    out["f_in"] = ("chan", np.asarray(solver.model_params["f_in"]), {"units": "Hz"})
    out.attrs.update(
        iq_normalization=normalization, model=model_params.meta["model_schema"]["name"]
    )
    return out
