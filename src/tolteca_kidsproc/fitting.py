"""Lmfit estimation of physical, fixed-membership KID resonance groups."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import KidsSweepModel

if TYPE_CHECKING:
    import lmfit.minimizer
    from lmfit import Parameters

__all__ = ["KidsModelFitResult", "fit_kids_model"]


def _stack(values):
    z = np.concatenate(values)
    return np.r_[z.real, z.imag]


def _condition(jac, rtol):
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > rtol * singular[0])) if len(singular) else 0
    condition = (
        float(singular[0] / singular[-1]) if rank == jac.shape[1] and rank else np.inf
    )
    return singular, rank, condition


def _linear_complex(design, target, sigma):
    a = np.block([[design.real, -design.imag], [design.imag, design.real]])
    s = np.r_[sigma[:, 0], sigma[:, 1]]
    v = np.linalg.lstsq(
        a / s[:, None], np.r_[target.real, target.imag] / s, rcond=None
    )[0]
    n = design.shape[1]
    return v[:n] + 1j * v[n:]


def _rational_seed(model, fs, zs):
    n = model.n_resonances
    k = model.n_channels
    low = min(f.min() for f in fs)
    high = max(f.max() for f in fs)
    origin = (low + high) / 2
    span = high - low
    matrices = []
    targets = []
    for i, (f, z) in enumerate(zip(fs, zs, strict=True)):
        u = (f - origin) / span
        a = np.zeros((len(f), k * (n + 2) + n), complex)
        a[:, i * (n + 2) : (i + 1) * (n + 2)] = u[:, None] ** np.arange(n + 2)
        a[:, -n:] = -z[:, None] * u[:, None] ** np.arange(n)
        matrices.append(a)
        targets.append(z * u**n)
    a = np.concatenate(matrices)
    b = np.concatenate(targets)
    scale = np.linalg.norm(a, axis=0)
    if np.any(scale == 0):
        msg = "Rational seed has zero columns"
        raise ValueError(msg)
    values, _, rank, _ = np.linalg.lstsq(a / scale, b, rcond=None)
    if rank != a.shape[1]:
        msg = "Rational seed is rank deficient"
        raise ValueError(msg)
    values /= scale
    roots = np.polynomial.polynomial.polyroots(np.r_[values[-n:], 1])
    fr = origin + span * roots.real
    width = 2 * span * roots.imag
    if np.any(width <= 0) or not np.isfinite(roots).all():
        msg = "Rational seed has nonpositive or nonfinite pole widths"
        raise ValueError(msg)
    # Match candidates to the supplied identities, never label them by frequency rank.
    _, order = linear_sum_assignment(
        abs(model.fr_reference[:, None] - fr)
        / (model.fr_reference / model.qr_reference)[:, None]
    )
    return fr[order], (fr / width)[order]


def _validate_seed(model, params, proposed_fr, proposed_qr):
    xi = model.qr_reference * (model.fr_reference / proposed_fr - 1)
    rho = model.qr_reference / proposed_qr
    for j in range(model.n_resonances):
        for name, value in ((f"r{j}_xi", xi[j]), (f"r{j}_rho", rho[j])):
            if not params[name].min < value < params[name].max:
                msg = "Rational seed lies outside the supplied state bounds"
                raise ValueError(msg)


def _initialize(model, params, fs, zs, sigmas, *, initialization):  # noqa: PLR0913 -- Seed selection with measured data and weights.
    info = {
        "requested": initialization,
        "used": initialization,
        "gain": "edge_affine_conditional",
        "fallback_reason": "",
    }
    fr = model.fr_reference.copy()
    qr = model.qr_reference.copy()
    if initialization == "rational":
        try:
            proposed_fr, proposed_qr = _rational_seed(model, fs, zs)
            _validate_seed(model, params, proposed_fr, proposed_qr)
            fr, qr = proposed_fr, proposed_qr
        except (ValueError, np.linalg.LinAlgError) as exc:
            info.update(used="supplied", fallback_reason=str(exc))
    for j in range(model.n_resonances):
        params[f"r{j}_xi"].value = model.qr_reference[j] * (
            model.fr_reference[j] / fr[j] - 1
        )
        params[f"r{j}_rho"].value = model.qr_reference[j] / qr[j]
    for k, (f, z, s) in enumerate(zip(fs, zs, sigmas, strict=True)):
        ids = np.flatnonzero(model.associations[k])
        u = (f - model.gain_origin[k]) / model.gain_span[k]
        design = np.column_stack((np.ones_like(u), u))
        edge = np.argsort(f)
        count = max(2, len(f) // 6)
        edge = np.unique(np.r_[edge[:count], edge[-count:]])
        h = _linear_complex(design[edge], z[edge], s[edge])
        den = model.qr_reference[ids] / qr[ids] + 2j * model.qr_reference[ids] * (
            f[:, None] / fr[ids] - 1
        )
        for _ in range(2):
            gain = design @ h
            chi = _linear_complex(-gain[:, None] / den, z - gain, s)
            chi = np.clip(chi.real, 1e-6, 19) + 1j * np.clip(chi.imag, -19, 19)
            kernel = 1 - np.sum(chi / den, axis=1)
            h = _linear_complex(design * kernel[:, None], z, s)
        for j, c in zip(ids, chi, strict=True):
            params[f"c{k}_r{j}_re"].value = c.real
            params[f"c{k}_r{j}_im"].value = c.imag
        if model.gain_family == "delay":
            slope = h[1] / h[0] if abs(h[0]) > np.finfo(float).tiny else 0j
        else:
            slope = h[1] / model.signal_scale
        for name, value in (
            ("h0_re", h[0].real / model.signal_scale),
            ("h0_im", h[0].imag / model.signal_scale),
            ("h1_re", slope.real),
            ("h1_im", slope.imag),
        ):
            params[f"c{k}_{name}"].value = value
    return info


class _Objective:
    def __init__(self, model, params, frequency, data, sigma):
        self.model = model
        self.params = params.copy()
        for parameter in self.params.values():
            parameter.stderr, parameter.correl = None, None
        self.frequency = frequency
        self.data = data
        self.names = [n for n, p in params.items() if p.vary and p.expr is None]
        self.sigma = np.r_[np.concatenate(sigma)[:, 0], np.concatenate(sigma)[:, 1]]

    def assign(self, params):
        for name in self.names:
            self.params[name].value = params[name].value
        self.params.update_constraints()

    def transform(self):
        """Differentiate model coordinates with respect to free lmfit variables."""
        transform = np.zeros((len(self.model.names), len(self.names)))
        for i, name in enumerate(self.names):
            transform[self.model.names.index(name), i] = 1
        tied = [(i, n) for i, n in enumerate(self.model.names) if self.params[n].expr]
        if tied:
            for i, name in enumerate(self.names):
                value = self.params[name].value
                step = 1e-5 * max(1.0, abs(value))
                p = self.params.copy()
                lo = max(p[name].min, value - step)
                hi = min(p[name].max, value + step)
                p[name].value = lo
                p.update_constraints()
                a = np.array([p[n].value for _, n in tied])
                p[name].value = hi
                p.update_constraints()
                b = np.array([p[n].value for _, n in tied])
                transform[[j for j, _ in tied], i] = (b - a) / (hi - lo)
        return transform

    def residual(self, params):
        self.assign(params)
        return (
            _stack(
                [
                    p - z
                    for p, z in zip(
                        self.model.eval(self.params, self.frequency),
                        self.data,
                        strict=True,
                    )
                ]
            )
            / self.sigma
        )

    def jacobian(self, params):
        self.assign(params)
        jac = np.concatenate(
            self.model.eval(self.params, self.frequency, jacobian=True)
        )
        return np.r_[jac.real, jac.imag] @ self.transform() / self.sigma[:, None]


def _physical_coordinates(model, params):
    physical = model.physical(params)
    names = []
    units = []
    values = []
    scale = []
    for j in range(model.n_resonances):
        fr = physical["fr"][j]
        ref = model.fr_reference[j]
        q = model.qr_reference[j]
        names.extend([f"resonator:{j}:fr", f"resonator:{j}:r"])
        units.extend(["Hz", "1"])
        values.extend([fr, physical["r"][j]])
        scale.extend([-(fr**2) / (ref * q), 1 / (2 * q)])
        if model.nonlinear:
            names.append(f"resonator:{j}:a_nl")
            units.append("1")
            values.append(physical["a_nl"][j])
            scale.append(1.0)
    for k, j in zip(*np.nonzero(model.associations), strict=True):
        c = physical["c"][k, j]
        names.extend([f"coupling:{k}:{j}:c_re", f"coupling:{k}:{j}:c_im"])
        units.extend(["1", "1"])
        values.extend([c.real, c.imag])
        scale.extend([1 / (2 * model.qr_reference[j])] * 2)
    for k in range(model.n_channels):
        h = physical["H0"][k]
        names.extend([f"response:{k}:H0_re", f"response:{k}:H0_im"])
        units.extend(["signal", "signal"])
        values.extend([h.real, h.imag])
        scale.extend([model.signal_scale] * 2)
        if model.gain_family == "affine":
            h = physical["H1"][k]
            suffix = ("H1_re", "H1_im")
            v = (h.real, h.imag)
            s = (model.signal_scale / model.gain_span[k],) * 2
            u = ("signal/Hz",) * 2
        else:
            suffix = ("a", "tau")
            v = (physical["a"][k], physical["tau"][k])
            s = (1 / model.gain_span[k], -1 / (2 * np.pi * model.gain_span[k]))
            u = ("1/Hz", "s")
        names.extend([f"response:{k}:{n}" for n in suffix])
        units.extend(u)
        values.extend(v)
        scale.extend(s)
    return names, units, np.array(values), np.array(scale)


@dataclass
class KidsModelFitResult:
    """Live lmfit result and physical covariance for one fixed group.

    Attributes
    ----------
    model : KidsSweepModel
        Physical response and fixed optimizer scales.
    params : lmfit.Parameters
        Fitted independent coordinates, bounds, fixed values and ties.
    raw : lmfit.minimizer.MinimizerResult
        Native optimizer result, including convergence and termination message.
    frequency, data, best_fit, residual : tuple of numpy.ndarray
        Ragged measured windows, complex data, prediction and ``data-model``.
    covariance : numpy.ndarray
        Full physical covariance ordered by ``parameter_names``. Undefined
        covariance is NaN; fixed coordinates have zero rows when identifiable.
        For directional nonlinear sweeps this is branch-local covariance and
        excludes uncertainty in jump placement or branch occupancy.
    parameter_names, parameter_units : tuple of str
        Group-local physical parameter identities and units. Channel/resonator
        indices are replaced with caller labels in the analysis result.
    jacobian, singular_values : numpy.ndarray
        Weighted derivatives and singular values in scaled optimizer coordinates.
    rank, condition : int, float
        Numerical rank and condition number at the declared cutoff.
    profile_condition : float
        State conditioning after projecting out fitted gain and coupling.
    initialization : dict
        Requested seed, actual seed and any fallback reason.
    """

    model: KidsSweepModel
    params: Parameters
    raw: lmfit.minimizer.MinimizerResult
    frequency: tuple
    data: tuple
    best_fit: tuple
    residual: tuple
    covariance: np.ndarray
    parameter_names: tuple
    parameter_units: tuple
    jacobian: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition: float
    profile_condition: float
    initialization: dict

    @property
    def success(self):
        """Optimizer convergence, separate from calibration acceptance."""
        # Lmfit attaches termination fields dynamically to MinimizerResult.
        return bool(getattr(self.raw, "success"))  # noqa: B009 -- Dynamic lmfit field.

    def pair_model(self, channel, resonator):
        """Select a fitted channel-resonator model for evaluation or derotation.

        Parameters
        ----------
        channel, resonator : int
            Explicit zero-based positions in the fit's input order.

        Returns
        -------
        KidsSweepWithAffineGain or KidsSweepWithDelayGain
            Detached model snapshot. ``eval`` describes the selected resonance
            alone. ``derotate`` retains neighbours and noise in supplied IQ.
            Full covariance and all other associations remain in this result.

        Raises
        ------
        IndexError
            A position is outside the group.
        ValueError
            The association is not included.
        """
        return self.model.pair_model(self.params, channel, resonator)

    def response_jacobian(self, probe_frequency):
        """Differentiate fixed probes with gain and coupling held constant.

        Parameters
        ----------
        probe_frequency : array_like
            One actual probe frequency in Hz per channel.

        Returns
        -------
        numpy.ndarray
            Shape ``(2*channel, 2*resonance)``. Rows are all I, then all Q;
            columns alternate ``fr`` (Hz) and ``r``. No noise whitening is applied.
            This derivative does not establish an identifiable inverse.

        Raises
        ------
        NotImplementedError
            The fit includes nonlinear sweep parameters. Their fixed-power
            dependence on detector loss is not represented by this derivative.
        ValueError
            Probes are not a positive finite Hz vector in channel order.
        """
        if self.model.nonlinear:
            msg = "Nonlinear sweep fits require a separate fixed-power response model"
            raise NotImplementedError(msg)
        fp = np.asarray(probe_frequency, float)
        if (
            fp.shape != (self.model.n_channels,)
            or not np.isfinite(fp).all()
            or np.any(fp <= 0)
        ):
            msg = "Supply one finite positive actual probe per channel"
            raise ValueError(msg)
        jac = np.concatenate(
            self.model.eval(self.params, [[f] for f in fp], jacobian=True)
        )[:, : 2 * self.model.n_resonances]
        _, _, _, scale = _physical_coordinates(self.model, self.params)
        jac /= scale[: 2 * self.model.n_resonances]
        return np.r_[jac.real, jac.imag]


def fit_kids_model(  # noqa: PLR0913 -- Explicit numerical inputs, state bounds and solver controls.
    frequency,
    s21,
    fr_initial,
    qr_initial,
    *,
    associations=None,
    gain_family="affine",
    gain_origin=None,
    sigma=None,
    initialization="rational",
    max_nfev=2000,
    fr_bounds=None,
    qr_bounds=(500.0, 100000.0),
    params=None,
    rank_rtol=1e-8,
    nonlinear=False,
    nonlinear_branch="unique",
    a_nl_max=0.75,
):
    """Fit a physical fixed group using lmfit and analytic model derivatives.

    Parameters
    ----------
    frequency, s21 : sequence of array_like
        Finite aligned vectors per channel, in Hz and signal units. Windows may
        be unequal. Each selected acquired sample must occur once.
    fr_initial, qr_initial : array_like
        Positive reference frequencies in Hz and loaded Q, one per resonance.
    associations : array_like of bool, optional
        Channel-by-resonance inclusion matrix; default includes all components.
    gain_family : {"affine", "delay"}, default "affine"
        Physical gain family. Each has four real parameters per channel.
    gain_origin : array_like, optional
        Fixed expansion frequency per channel in Hz; default window midpoint.
    sigma : sequence of array_like, optional
        Positive I/Q standard deviations of measured means, shape (sample,2)
        per channel. Omit for equal weighting and residual-scaled covariance.
    initialization : {"rational", "supplied"}, default "rational"
        Rational pole seed, or supplied frequencies/Q. Invalid rational seeds
        fall back to supplied states with a recorded reason. Both use the same
        conditional gain/coupling initialization. Supplied ``params`` bypass this.
    max_nfev : int, default 2000
        Maximum lmfit function evaluations. Exhaustion returns a failed result.
    fr_bounds : pair of array_like, optional
        Per-resonance frequency bounds in Hz. Default: +/- initial linewidth.
    qr_bounds : pair of float, default (500,100000)
        Positive loaded-Q bounds, translated to fitted loss coordinates.
    params : lmfit.Parameters, optional
        Parameters from the same model with unchanged scales and associations.
        Copied before fitting; fixed values, expressions and bounds are retained.
    rank_rtol : float, default 1e-8
        Relative singular-value threshold in optimizer coordinates.
    nonlinear : bool, default False
        Fit shared stationary softening parameters a_nl. Nonlinear sweep fits
        cannot supply a fixed-power r/x response Jacobian yet.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Unique subcritical response, or a monotonic sweep from below/above
        resonance. The caller supplies the acquisition history; sample masks
        and evaluation order do not change it. No arbitrary history or resets.
    a_nl_max : float, default 0.75
        Positive finite upper bound for newly created parameters. Above 0.75,
        supply a directional branch. Supplied params retain their own bounds.

    Returns
    -------
    KidsModelFitResult
        Physical covariance, predictions, diagnostics and live lmfit parameters.
        Model acceptance and fixed-probe recovery require separate assessment.
        Directional-branch derivatives/covariance are local: they exclude jump
        location uncertainty. Convergence does not establish a global optimum.

    Raises
    ------
    ValueError
        Data, state scales, bounds, parameters or residual degrees of freedom
        are invalid. Known uncertainties must be finite and positive.
    """
    from lmfit import Minimizer  # noqa: PLC0415 -- Load the solver only when fitting.

    fs = tuple(np.asarray(f, float) for f in frequency)
    zs = tuple(np.asarray(z, complex) for z in s21)
    if len(fs) != len(zs) or any(
        z.shape != f.shape or not np.isfinite(z).all()
        for f, z in zip(fs, zs, strict=True)
    ):
        msg = "Supply aligned finite frequency and complex signal vectors"
        raise ValueError(msg)
    if initialization not in ("rational", "supplied") or not 0 < rank_rtol < 1:
        msg = (
            "Supply a supported initialization and rank tolerance between zero and one"
        )
        raise ValueError(msg)
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        msg = "max_nfev must be a positive integer"
        raise ValueError(msg)
    scale = (
        max(float(np.median(abs(np.concatenate(zs)))), np.finfo(float).tiny)
        if zs
        else 1.0
    )
    model = KidsSweepModel(
        fr_initial,
        qr_initial,
        fs,
        associations=associations,
        gain_family=gain_family,
        gain_origin=gain_origin,
        signal_scale=scale,
        nonlinear=nonlinear,
        nonlinear_branch=nonlinear_branch,
    )
    sigmas = (
        tuple(np.full((len(f), 2), scale) for f in fs)
        if sigma is None
        else tuple(np.asarray(s, float) for s in sigma)
    )
    if len(sigmas) != len(fs) or any(
        s.shape != (len(f), 2) or not np.isfinite(s).all() or np.any(s <= 0)
        for s, f in zip(sigmas, fs, strict=True)
    ):
        msg = "sigma must supply positive finite (sample,2) standard deviations"
        raise ValueError(msg)
    if params is None:
        params = model.make_params(
            fr_bounds=fr_bounds, qr_bounds=qr_bounds, a_nl_max=a_nl_max
        )
        init = _initialize(model, params, fs, zs, sigmas, initialization=initialization)
    else:
        params = params.copy()
        init = {
            "requested": initialization,
            "used": "parameters",
            "gain": "parameters",
            "fallback_reason": "",
        }
        if set(params) != set(model.names):
            msg = "Parameters do not match the model coordinates"
            raise ValueError(msg)
    objective = _Objective(model, params, fs, zs, sigmas)
    if not objective.names or 2 * sum(map(len, fs)) <= len(objective.names):
        msg = (
            "The fit requires free parameters and positive residual degrees of freedom"
        )
        raise ValueError(msg)
    optimizer = Minimizer(objective.residual, params, scale_covar=sigma is None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning,
            module="lmfit.minimizer",
        )
        raw = optimizer.minimize(
            method="least_squares",
            max_nfev=max_nfev,
            jac=objective.jacobian,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
    names, units, physical_cov, jac, singular, rank, condition, profile = _diagnostics(
        model, objective, raw, sigma, rank_rtol
    )
    best = model.eval(raw.params, fs)
    return KidsModelFitResult(
        model,
        raw.params.copy(),
        raw,
        fs,
        zs,
        best,
        tuple(z - p for z, p in zip(zs, best, strict=True)),
        physical_cov,
        tuple(names),
        tuple(units),
        jac,
        singular,
        rank,
        condition,
        profile,
        init,
    )


def _diagnostics(model, objective, raw, sigma, rank_rtol):
    residual = objective.residual(raw.params)
    jac = objective.jacobian(raw.params)
    singular, rank, condition = _condition(jac, rank_rtol)
    cov = np.full((len(objective.names),) * 2, np.nan)
    if rank == len(objective.names):
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        cov = (vt.T / s**2) @ vt
        if sigma is None:
            cov *= residual @ residual / (len(residual) - len(objective.names))
    names, units, _, scales = _physical_coordinates(model, objective.params)
    transform = scales[:, None] * objective.transform()
    physical_cov = transform @ cov @ transform.T
    state = [i for i, n in enumerate(objective.names) if n.startswith("r")]
    nuisance = [i for i in range(len(objective.names)) if i not in state]
    projected = jac[:, state]
    if nuisance:
        u, s, _ = np.linalg.svd(jac[:, nuisance], full_matrices=False)
        u = u[:, s > rank_rtol * s[0]]
        projected -= u @ (u.T @ projected)
    profile = _condition(projected, rank_rtol)[2] if state else np.nan
    raw.covar = cov if rank == len(objective.names) else None
    raw.errorbars = rank == len(objective.names)
    raw.params = objective.params.copy()
    for i, name in enumerate(objective.names):
        raw.params[name].stderr = float(np.sqrt(cov[i, i])) if raw.errorbars else None
    if raw.errorbars:
        raw.params.create_uvars(covar=cov)
    return names, units, physical_cov, jac, singular, rank, condition, profile
