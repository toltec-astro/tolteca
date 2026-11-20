"""Full-model joint detector recovery at fixed acquired probe frequencies."""

# ruff: noqa: E741, N803 -- Preserve physical I/Q names.
from __future__ import annotations

from itertools import pairwise
from typing import TypedDict

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from scipy.stats import chi2

from .calibration import calibration_components
from .recovery import SampleFlag, recovery_diagnostics
from .timestream import _positive

__all__ = ["JointTimestreamSolver"]

_MATRIX_NDIM = 2
_WARNING_MASK = int(SampleFlag.condition_high)
_FAILURE_MASK = np.iinfo(np.uint16).max ^ _WARNING_MASK
_NUMERICAL_FAILURE_MASK = _FAILURE_MASK ^ int(SampleFlag.residual_high)
_DISTINCT_SOLUTION_SIGMA = 5
_DAMPING_MAX = 1e12


class _BlockResult(TypedDict):
    """Recovered arrays and the detached state for the next ordered block."""

    fr: np.ndarray
    r: np.ndarray
    flags: np.ndarray
    residual_norm: np.ndarray
    residual_stat: np.ndarray
    step_noise: np.ndarray
    restarted: np.ndarray
    initial_flags: np.ndarray
    starts_tested: np.ndarray
    checkpoint: tuple[np.ndarray, bool] | None


class JointTimestreamSolver:
    """Prepare joint recovery with fixed gain/coupling and explicit IQ noise.

    Parameters
    ----------
    probes, fr, r : array_like
        Positive probe frequencies (Hz) per channel, reference resonance
        frequencies (Hz), and reference losses per resonator.
    gain, coupling : array_like
        Complex gain per channel and coupling matrix (channel, resonator).
    iq_covariance : array_like
        Positive definite IQ covariance in calibration signal units. Ordering is
        all I then all Q. This is supplied evidence, not inferred from residuals.
    channels, resonator_ids : sequence
        Unique channel and calibration-scoped resonator labels in array order.
    rank_rtol : float, optional
        Relative singular-value cutoff, default 1e-8.
    condition_max : float or None, optional
        Warning threshold, default None. Rank failure always invalidates output.
    continuation : bool, optional
        Use the last accepted state to initialize the next sample, default False.
        Requires explicit fr_bounds and residual_p_value. Initialization and
        restarts compare independent bounded starts. Unresolved alternatives
        produce ambiguous_solution, not an assigned detector identity.
    fr_bounds : array_like or None, optional
        Allowed resonance frequency interval in Hz, shape (resonator, 2), with
        lower then upper limits. It must include the reference frequencies.
        Supply calibration support or independently justified loading limits.
        It bounds recovery as well as the initialization search.
    noise_step_tol : float, optional
        Maximum Mahalanobis length of the remaining Gauss-Newton correction,
        default 0.001, using the conditional readout-noise covariance.
    residual_p_value : float or None, optional
        Nominal upper-tail probability for the gain-aware residual test,
        default None (disabled). It is approximate for nonlinear recovery.
        With no residual degrees of freedom, only competing-start validation
        is possible. A passed test does not certify global uniqueness.
    gain_covariance : array_like or None, optional
        Positive semidefinite covariance of all channel log amplitudes followed
        by all channel phases (radians), shape (2*channel, 2*channel). Default
        None assumes exact gain. It affects residual validation, not the fitted
        gain or the conditional detector covariance. Shared errors are not
        independent noise samples.
    restart_step_max : float, optional
        Largest continuation displacement before independent reinitialization,
        default 0.25 in dr/r_ref and dfr/(2*r_ref*fr_ref) coordinates. A restart
        candidate beyond this distance from the last accepted contiguous state
        is ambiguous. Mark actual discontinuities with reset or sample gaps.
    ambiguity_delta_chi2 : float, optional
        Maximum validation-score difference for competing acceptable solutions,
        default 9. Candidates separated by more than five conditional noise
        standard deviations are distinct. This is a diagnostic threshold,
        not a calibrated global-uniqueness probability.

    Notes
    -----
    Inverts the full ``gain*(1-sum(c/(r+1j*(probe/fr-1))))`` response using
    an analytical inverse for one probe/one resonator, or damped Gauss-Newton
    and SVD for groups. Independent samples are evaluated in arrays. Grouped
    recovery validates restarts and checks competing starting points even after
    initial convergence. Each accepted state must converge; there is no linear
    or fixed-iteration approximation fallback. Square and overdetermined groups
    use the same rank-aware SVD procedure. Positive loss/frequency and explicit
    frequency limits apply to both analytical and iterative recovery.
    Continuation carries accepted state across lazy chunks without mutating
    the solver. A finite rejection requires revalidation against the last
    accepted state. An explicit or nonfinite gap starts a new sequence.
    Initial/restart validation
    searches finitely many starts and does not prove global uniqueness.
    Drive-nonlinear or hysteretic calibrations are unsupported. No primary resonator
    is assigned. Detector covariance remains conditional on calibration.
    """

    def __init__(  # noqa: PLR0913, PLR0917 -- Explicit physical calibration and identities.
        self,
        probes,
        fr,
        r,
        gain,
        coupling,
        iq_covariance,
        *,
        channels,
        resonator_ids,
        rank_rtol=1e-8,
        condition_max=None,
        continuation=False,
        fr_bounds=None,
        noise_step_tol=1e-3,
        residual_p_value=None,
        gain_covariance=None,
        restart_step_max=0.25,
        ambiguity_delta_chi2=9.0,
    ):
        self.probes, self.fr, self.r = [
            np.array(a, float, copy=True) for a in (probes, fr, r)
        ]
        self.gain, self.coupling = [
            np.array(a, complex, copy=True) for a in (gain, coupling)
        ]
        self.channels = np.asarray(channels).copy()
        self.resonator_ids = np.asarray(resonator_ids).copy()
        m, n = self.probes.size, self.fr.size
        if (
            not m
            or not n
            or any(a.ndim != 1 for a in (self.probes, self.fr, self.r))
            or self.r.shape != (n,)
            or self.gain.shape != (m,)
            or self.coupling.shape != (m, n)
            or self.channels.shape != (m,)
            or self.resonator_ids.shape != (n,)
            or len(set(self.channels)) != m
            or len(set(self.resonator_ids)) != n
            or any(
                not np.isfinite(a).all()
                for a in (self.probes, self.fr, self.r, self.gain, self.coupling)
            )
            or any(np.any(a <= 0) for a in (self.probes, self.fr, self.r))
        ):
            msg = (
                "Joint calibration requires finite compatible arrays and unique labels"
            )
            raise ValueError(msg)
        self.rank_rtol, self.condition_max = rank_rtol, condition_max
        _, jacobian = self._response(np.zeros(2 * n))
        self.diagnostics = recovery_diagnostics(
            jacobian, iq_covariance, rank_rtol=rank_rtol, condition_max=condition_max
        )
        self._whitening = np.linalg.inv(np.linalg.cholesky(iq_covariance))
        self.continuation = bool(continuation)
        self.noise_step_tol = _positive(noise_step_tol, "noise_step_tol")
        self.restart_step_max = _positive(restart_step_max, "restart_step_max")
        self.ambiguity_delta_chi2 = _positive(
            ambiguity_delta_chi2, "ambiguity_delta_chi2"
        )
        self.residual_p_value = residual_p_value
        if residual_p_value is not None and not 0 < residual_p_value < 1:
            msg = "residual_p_value must be in (0, 1)"
            raise ValueError(msg)
        if continuation and (fr_bounds is None or residual_p_value is None):
            msg = "Continuation requires explicit fr_bounds and residual_p_value"
            raise ValueError(msg)
        self.fr_bounds = (
            None if fr_bounds is None else np.array(fr_bounds, float, copy=True)
        )
        self._bounds = np.tile([-1 + 1e-12, np.inf], (2 * n, 1))
        self._bounds[1::2, 0] = (-1 + 1e-12) / (2 * self.r)
        if self.fr_bounds is not None:
            if (
                self.fr_bounds.shape != (n, 2)
                or not np.isfinite(self.fr_bounds).all()
                or np.any(self.fr_bounds <= 0)
                or np.any(np.diff(self.fr_bounds, axis=1) <= 0)
                or np.any(self.fr < self.fr_bounds[:, 0])
                or np.any(self.fr > self.fr_bounds[:, 1])
            ):
                msg = "fr_bounds must bracket each reference frequency in positive Hz"
                raise ValueError(msg)
            self._bounds[1::2] = (self.fr_bounds / self.fr[:, None] - 1) / (
                2 * self.r[:, None]
            )
        self.gain_covariance = (
            np.zeros((2 * m, 2 * m))
            if gain_covariance is None
            else np.array(gain_covariance, float, copy=True)
        )
        if (
            self.gain_covariance.shape != (2 * m, 2 * m)
            or not np.isfinite(self.gain_covariance).all()
            or not np.allclose(
                self.gain_covariance, self.gain_covariance.T, rtol=1e-10, atol=1e-20
            )
            or np.linalg.eigvalsh(self.gain_covariance).min()
            < -1e-12 * max(np.linalg.norm(self.gain_covariance), 1e-30)
        ):
            msg = (
                "gain_covariance requires symmetric positive semidefinite "
                "log-amplitude/phase covariance"
            )
            raise ValueError(msg)
        self.residual_dof = 2 * (m - n)
        self._residual_max = (
            chi2.isf(residual_p_value, self.residual_dof)
            if residual_p_value is not None and self.residual_dof > 0
            else np.inf
        )

    @classmethod
    def from_calibration(cls, tree, group_id, probes, iq_covariance, **kwargs):
        """Prepare one saved linear group at the explicitly selected probes.

        Parameters
        ----------
        tree : xarray.DataTree
            Compact model-fit analysis containing fit/samples/joint/chan_prop.
        group_id : str
            Exact fitted group label.
        probes : xarray.DataArray
            Frequencies in Hz with original channel labels on ``chan``. A subset
            keeps all of the group's resonators and checks its resulting rank.
        iq_covariance : array_like
            Noise covariance in selected probe order and calibration units.
        **kwargs
            Recovery options passed to the constructor, including continuation
            and its explicit frequency bounds and residual threshold.

        Returns
        -------
        JointTimestreamSolver
            Detached calibration retaining channel and resonator identities.

        Raises
        ------
        ValueError
            Group/probe identities are inconsistent or the fit is nonlinear.
        """
        components = calibration_components(tree)
        rows = components.where(components.group_id == str(group_id), drop=True)
        if not rows.sizes["response"] or probes.dims != ("chan",):
            msg = "Require an existing group and probes indexed by chan"
            raise ValueError(msg)
        if probes.attrs.get("units"):
            probes = probes.u.to("Hz")
        valid = rows.component_valid.values.astype(bool)
        if np.any(rows.a_nl.values[valid] != 0):
            msg = (
                "Joint timestream recovery does not support drive-nonlinear calibration"
            )
            raise ValueError(msg)
        ids = list(dict.fromkeys(rows.resonator_id.values[valid]))
        fr, loss = [], []
        for rid in ids:
            mask = valid & (rows.resonator_id.values == rid)
            fr.append(rows.fr.values[mask][0])
            loss.append(rows.r.values[mask][0])
        gain = []
        coupling = np.zeros((len(probes), len(ids)), complex)
        lookup = {c: i for i, c in enumerate(rows.chan.values)}
        for k, (ch, frequency) in enumerate(
            zip(probes.chan.values, probes.values, strict=True)
        ):
            if ch not in lookup:
                msg = f"Probe channel {ch} is absent from group {group_id}"
                raise ValueError(msg)
            row = rows.isel(response=lookup[ch])
            h = row.H0_re.item() + 1j * row.H0_im.item()
            df = frequency - row.f_g.item()
            if row.gain_family.item() == "affine":
                h += (row.H1_re.item() + 1j * row.H1_im.item()) * df
            elif row.gain_family.item() == "delay":
                h *= np.exp((row.gain_a.item() - 2j * np.pi * row.tau.item()) * df)
            else:
                msg = "Unsupported joint gain family"
                raise ValueError(msg)
            gain.append(h)
            for j in np.flatnonzero(row.component_valid.values):
                coupling[k, ids.index(row.resonator_id.values[j])] = (
                    row.c_re.values[j] + 1j * row.c_im.values[j]
                )
        return cls(
            probes.values,
            fr,
            loss,
            gain,
            coupling,
            iq_covariance,
            channels=probes.chan.values,
            resonator_ids=ids,
            **kwargs,
        )

    def _response(self, state):
        fr = self.fr * (1 + 2 * self.r * state[..., 1::2])
        w = self.r * (1 + state[..., None, ::2]) + 1j * (
            self.probes[:, None] / fr[..., None, :] - 1
        )
        z = self.gain * (1 - np.sum(self.coupling / w, axis=-1))
        derivative = self.gain[:, None] * self.coupling / w**2
        dr = derivative * self.r
        df = (
            -1j
            * derivative
            * self.probes[:, None]
            / fr[..., None, :] ** 2
            * (2 * self.r * self.fr)
        )
        jac = np.empty((*state.shape[:-1], 2 * len(self.probes), 2 * len(self.fr)))
        jac[..., ::2] = np.concatenate((dr.real, dr.imag), axis=-2)
        jac[..., 1::2] = np.concatenate((df.real, df.imag), axis=-2)
        return np.concatenate((z.real, z.imag), axis=-1), jac

    def __call__(  # noqa: PLR0913 -- Signals, normalization, initial state and gap mask.
        self, I, Q, *, normalization=1.0, initial_fr=None, initial_r=None, reset=None
    ):
        """Recover an ordered block from sample-by-channel I/Q arrays.

        Parameters
        ----------
        I, Q : array_like
            Samples in the prepared channel order. Calls are independent.
        normalization : float, optional
            Positive I/Q divisor converting inputs to calibration units.
        initial_fr, initial_r : array_like or None, optional
            Initial frequency (Hz) and loss hints per resonator. Supply both.
            Grouped recovery validates hints independently; they are not trusted
            states. The analytical singlet inverse does not use hints.
        reset : array_like of bool or None, optional
            True requires reinitialization before that sample. Mark known gaps
            even when missing samples are absent from the supplied arrays.

        Returns
        -------
        fr, r : numpy.ndarray
            Frequency (Hz) and loss on (sample, resonator). Invalid values are NaN.
        flags : numpy.ndarray
            uint16 SampleFlag bits. condition_high alone retains recovered values.
            Distinct acceptable solutions produce ambiguous_solution and NaNs.
        residual_norm : numpy.ndarray
            Readout-whitened residual norm, retained for finite rejected candidates.

        Raises
        ------
        ValueError
            Input shapes, normalization, initial hints or reset mask are invalid.
        """
        observed = self._observations(I, Q, normalization)
        initial = self._initial_state(initial_fr, initial_r)
        reset = self._reset_mask(reset, len(observed))
        result = self._recover_block(observed, reset, None, initial)
        return tuple(result[name] for name in ("fr", "r", "flags", "residual_norm"))

    def _observations(self, I, Q, normalization):
        I, Q = np.asarray(I, float), np.asarray(Q, float)
        norm = _positive(normalization, "normalization")
        if (
            I.ndim != _MATRIX_NDIM
            or I.shape != Q.shape
            or I.shape[1] != len(self.channels)
        ):
            msg = "Joint I/Q must have matching sample/channel axes"
            raise ValueError(msg)
        return np.concatenate((I, Q), axis=1) / norm

    def _initial_state(self, fr, r):
        if fr is None and r is None:
            return np.zeros(2 * len(self.fr))
        if fr is None or r is None:
            msg = "Supply both initial_fr and initial_r"
            raise ValueError(msg)
        fr, r = np.asarray(fr, float), np.asarray(r, float)
        if any(
            a.shape != self.fr.shape or not np.isfinite(a).all() or np.any(a <= 0)
            for a in (fr, r)
        ):
            msg = (
                "Initial frequency/loss hints must be positive finite resonator arrays"
            )
            raise ValueError(msg)
        state = np.empty(2 * len(fr))
        state[::2], state[1::2] = r / self.r - 1, (fr / self.fr - 1) / (2 * self.r)
        return state

    @staticmethod
    def _reset_mask(reset, size):
        if reset is None:
            return np.zeros(size, bool)
        reset = np.asarray(reset)
        if reset.shape != (size,) or reset.dtype != np.dtype(bool):
            msg = "reset must be a boolean vector with one entry per sample"
            raise ValueError(msg)
        return reset.copy()

    def _recover_block(self, observed, reset, checkpoint, initial):  # noqa: PLR0915 -- Shared output for independent and ordered recovery.
        size, n = len(observed), len(self.fr)
        result: _BlockResult = {
            "fr": np.full((size, n), np.nan),
            "r": np.full((size, n), np.nan),
            "flags": np.full(size, self.diagnostics["flag"], np.uint16),
            "residual_norm": np.full(size, np.nan),
            "residual_stat": np.full(size, np.nan),
            "step_noise": np.full(size, np.nan),
            "restarted": np.zeros(size, bool),
            "initial_flags": np.zeros(size, np.uint16),
            "starts_tested": np.zeros(size, np.uint16),
            "checkpoint": None,
        }
        if self.diagnostics["rank"] != 2 * n:
            return result
        if not self.continuation or len(self.probes) == n == 1:
            finite = np.isfinite(observed).all(axis=-1)
            result["flags"][~finite] |= int(SampleFlag.nonfinite_input)
            state, flags, first, starts = self._recover_samples(
                observed[finite], initial
            )
            result["initial_flags"][finite] = first
            result["starts_tested"][finite] = starts
            result["restarted"][finite] = (first & _FAILURE_MASK != 0) & (starts > 1)
            extra, raw, stat, step_noise = self._validation_samples(
                observed[finite], state
            )
            result["flags"][finite] |= flags | extra
            result["residual_norm"][finite] = np.sqrt(raw)
            result["residual_stat"][finite] = stat
            result["step_noise"][finite] = step_noise
            valid = np.isin(result["flags"][finite], (0, _WARNING_MASK))
            positions = np.flatnonzero(finite)[valid]
            result["fr"][positions] = self.fr * (1 + 2 * self.r * state[valid, 1::2])
            result["r"][positions] = self.r * (1 + state[valid, ::2])
            return result
        for k, y in enumerate(observed):
            if reset[k]:
                checkpoint = None
            if not np.isfinite(y).all():
                result["flags"][k] |= int(SampleFlag.nonfinite_input)
                checkpoint = None
                continue
            anchor = checkpoint[0] if checkpoint is not None else None
            restarted = checkpoint is None or checkpoint[1]
            if restarted:
                states, flags, first, starts = self._recover_samples(
                    y[None, :], initial if anchor is None else anchor
                )
                state, flag = states[0], int(flags[0])
                result["initial_flags"][k], result["starts_tested"][k] = (
                    first[0],
                    starts[0],
                )
            else:
                state, flag = self._solve_sample(y, anchor)
                result["initial_flags"][k], result["starts_tested"][k] = flag, 1
                extra, _, _, _ = self._validation(y, state)
                flag |= extra
                result["initial_flags"][k] |= extra
                if (
                    flag & ~_WARNING_MASK
                    or np.max(abs(state - anchor)) > self.restart_step_max
                ):
                    restarted = True
                    states, flags, _, starts = self._recover_samples(y[None, :], state)
                    state, flag = states[0], int(flags[0])
                    result["starts_tested"][k] += starts[0]
            extra, raw, stat, step_noise = self._validation(y, state)
            flag |= extra
            if (
                anchor is not None
                and np.max(abs(state - anchor)) > self.restart_step_max
            ):
                flag |= SampleFlag.ambiguous_solution
            result["flags"][k] |= int(flag)
            result["residual_norm"][k] = np.sqrt(raw)
            result["residual_stat"][k] = stat
            result["step_noise"][k] = step_noise
            result["restarted"][k] = restarted
            if int(result["flags"][k]) & ~_WARNING_MASK:
                checkpoint = None if anchor is None else (anchor, True)
                continue
            result["fr"][k] = self.fr * (1 + 2 * self.r * state[1::2])
            result["r"][k] = self.r * (1 + state[::2])
            checkpoint = (state.copy(), False)
        result["checkpoint"] = checkpoint
        return result

    def _analytical_samples(self, observed):
        """Invert the physical singlet without a numerical starting state."""
        z = observed[:, 0] + 1j * observed[:, 1]
        with np.errstate(all="ignore"):
            denominator = 1 - z / self.gain[0]
            w = self.coupling[0, 0] / denominator
            fr = self.probes[0] / (1 + w.imag)
            state = np.column_stack(
                (w.real / self.r[0] - 1, (fr / self.fr[0] - 1) / (2 * self.r[0]))
            )
        flags = np.zeros(len(observed), np.uint16)
        flags[denominator == 0] |= int(SampleFlag.singular_response)
        flags[~np.isfinite(state).all(axis=-1)] |= int(SampleFlag.nonfinite_output)
        outside = np.isfinite(w.real) & np.isfinite(fr) & ((w.real <= 0) | (fr <= 0))
        flags[outside] |= int(SampleFlag.outside_domain)
        extra, _, _, _ = self._validation_samples(observed, state)
        return state, flags | extra

    def _restart_seeds(self, hint):
        """Finite independent starts in calibrated loss/linewidth coordinates."""
        n = len(self.fr)
        reference = np.zeros(2 * n)
        seeds = [reference, hint]
        for scale in (0.5, 2.0):
            seed = reference.copy()
            seed[::2] = scale - 1
            seeds.append(seed)
        for offset in (-1.0, 1.0):
            seed = reference.copy()
            seed[1::2] = offset
            seeds.append(seed)
        # Swapping starting frequencies tests nearby alternative basins; the
        # coupling columns and resonator labels never change.
        order = np.argsort(self.fr)
        for left, right in pairwise(order):
            seed = reference.copy()
            permuted = self.fr.copy()
            permuted[[left, right]] = permuted[[right, left]]
            seed[1::2] = (permuted / self.fr - 1) / (2 * self.r)
            seeds.append(seed)
        if self.fr_bounds is not None:
            for fraction in (0.1, 0.5, 0.9):
                seed = reference.copy()
                seed[1::2] = (
                    self._bounds[1::2, 0]
                    + fraction * np.diff(self._bounds[1::2], axis=1).ravel()
                )
                seeds.append(seed)
        return np.unique(np.clip(seeds, self._bounds[:, 0], self._bounds[:, 1]), axis=0)

    def _recover_samples(self, observed, initial):
        """Validate full inverses and compare independent converged candidates."""
        if len(self.probes) == len(self.fr) == 1:
            state, flags = self._analytical_samples(observed)
            return state, flags, flags.copy(), np.zeros(len(observed), np.uint16)
        state, flags = self._solve_samples(observed, initial)
        first_state = state.copy()
        extra, raw, stat, _ = self._validation_samples(observed, state)
        first = flags | extra
        candidates = [(state, first, raw, stat)]
        # Continue failed numerical attempts with a larger budget and damping.
        # These are still full-model solves, never truncated corrections.
        retry = (first & _NUMERICAL_FAILURE_MASK) != 0
        starts = np.ones(len(observed), np.uint16)
        if retry.any():
            q, f = self._solve_samples(
                observed[retry], first_state[retry], max_iterations=300, damped=True
            )
            state = first_state.copy()
            state[retry] = q
            flags = first.copy()
            flags[retry] = f
            extra, raw, stat, _ = self._validation_samples(observed, state)
            candidates.append((state, flags | extra, raw, stat))
            starts[retry] += 1
        # Check every sample, including first-attempt successes. A failed
        # alternate start is not evidence against a converged candidate.
        for seed in self._restart_seeds(initial):
            state, flags = self._solve_samples(
                observed, seed, max_iterations=300, damped=True
            )
            extra, raw, stat, _ = self._validation_samples(observed, state)
            candidates.append((state, flags | extra, raw, stat))
            starts += 1
        state, flags = self._select_candidates(candidates)
        return state, flags, first, starts

    def _select_candidates(self, candidates):
        states = np.stack([c[0] for c in candidates])
        flags = np.stack([c[1] for c in candidates])
        raw = np.stack([c[2] for c in candidates])
        score = np.stack([c[3] if self.residual_dof > 0 else c[2] for c in candidates])
        numerical = (flags & _NUMERICAL_FAILURE_MASK) == 0
        # Gain uncertainty changes validation, not the readout-weighted fit.
        # Keep the first minimum within the numerical convergence tolerance.
        minimum = np.min(np.where(numerical, raw, np.inf), axis=0)
        eligible = numerical & (raw <= minimum[None, :] + self.noise_step_tol**2)
        chosen = np.argmax(eligible, axis=0)
        none = ~numerical.any(axis=0)
        failed = np.argmin(np.where(np.isfinite(raw), raw, np.inf), axis=0)
        chosen[none] = failed[none]
        k = np.arange(states.shape[1])
        best = states[chosen, k]
        result_flags = flags[chosen, k].copy()
        _, jac = self._response(best)
        distance = np.linalg.norm(
            np.einsum("nij,knj->kni", self._whitening @ jac, states - best), axis=-1
        )
        good = (flags & _FAILURE_MASK) == 0
        competing = good & (abs(score - score[chosen, k]) <= self.ambiguity_delta_chi2)
        ambiguous = good[chosen, k] & np.any(
            competing & (distance > _DISTINCT_SOLUTION_SIGMA), axis=0
        )
        result_flags[ambiguous] |= int(SampleFlag.ambiguous_solution)
        return best, result_flags

    def _solve_sample(self, observed, initial=None):
        state, flags = self._solve_samples(observed[None, :], initial)
        return state[0], int(flags[0])

    def _solve_samples(
        self, observed, initial=None, *, max_iterations=30, damped=False
    ):
        """Converge independent full-model states with per-sample SVD checks."""
        seed = np.zeros(2 * len(self.fr)) if initial is None else initial
        state = np.broadcast_to(
            np.clip(seed, self._bounds[:, 0], self._bounds[:, 1]),
            (len(observed), 2 * len(self.fr)),
        ).copy()
        flags = np.zeros(len(observed), np.uint16)
        active = np.ones(len(observed), bool)
        damping = np.full(len(observed), 1e-3 if damped else 0.0)
        for _ in range(max_iterations):
            positions = np.flatnonzero(active)
            if not len(positions):
                break
            with np.errstate(all="ignore"):
                prediction, jac = self._response(state[positions])
            finite = np.isfinite(prediction).all(axis=-1) & np.isfinite(jac).all(
                axis=(-2, -1)
            )
            flags[positions[~finite]] |= int(SampleFlag.nonfinite_output)
            active[positions[~finite]] = False
            positions, prediction, jac = (
                positions[finite],
                prediction[finite],
                jac[finite],
            )
            if not len(positions):
                continue
            residual = (observed[positions] - prediction) @ self._whitening.T
            u, singular, vt = np.linalg.svd(self._whitening @ jac, full_matrices=False)
            rank = np.sum(singular > self.rank_rtol * singular[:, :1], axis=-1)
            valid = rank == state.shape[1]
            flags[positions[~valid]] |= int(SampleFlag.singular_response)
            active[positions[~valid]] = False
            positions, residual = positions[valid], residual[valid]
            u, singular, vt = u[valid], singular[valid], vt[valid]
            if not len(positions):
                continue
            if self.condition_max is not None:
                high = singular[:, 0] / singular[:, -1] > self.condition_max
                flags[positions[high]] |= int(SampleFlag.condition_high)
            projected = np.einsum("nji,nj->ni", u, residual)
            # Mahalanobis length of the remaining Gauss-Newton correction.
            # This is invariant to invertible linear detector-state scaling.
            converged = np.linalg.norm(projected, axis=-1) <= self.noise_step_tol
            active[positions[converged]] = False
            positions = positions[~converged]
            step = np.einsum(
                "nji,nj->ni",
                vt[~converged],
                projected[~converged]
                * singular[~converged]
                / (
                    singular[~converged] ** 2
                    + damping[positions, None] * singular[~converged, :1] ** 2
                ),
            )
            candidate, accepted = self._line_search(
                state[positions],
                step,
                observed[positions],
                np.sum(residual[~converged] ** 2, axis=-1),
            )
            state[positions[accepted]] = candidate[accepted]
            if damped:
                damping[positions[accepted]] *= 0.2
                damping[positions[~accepted]] *= 10
                stopped = ~accepted & (damping[positions] > _DAMPING_MAX)
            else:
                stopped = ~accepted
            flags[positions[stopped]] |= int(
                SampleFlag.recovery_failed | SampleFlag.stalled
            )
            active[positions[stopped]] = False
        flags[active] |= int(SampleFlag.recovery_failed | SampleFlag.iteration_limit)
        return state, flags

    def _line_search(self, state, step, observed, norm2):
        candidate = state.copy()
        accepted = np.zeros(len(state), bool)
        for scale in 2.0 ** -np.arange(11):
            positions = np.flatnonzero(~accepted)
            if not len(positions):
                break
            trial = state[positions] + scale * step[positions]
            within = np.all(
                (trial >= self._bounds[:, 0]) & (trial <= self._bounds[:, 1]), axis=-1
            )
            positions, trial = positions[within], trial[within]
            if not len(positions):
                continue
            with np.errstate(all="ignore"):
                residual = (
                    observed[positions] - self._response(trial)[0]
                ) @ self._whitening.T
            improved = np.sum(residual**2, axis=-1) < norm2[positions]
            candidate[positions[improved]] = trial[improved]
            accepted[positions[improved]] = True
        return candidate, accepted

    def _validation(self, observed, state):
        flag, raw, stat, step = self._validation_samples(
            observed[None, :], state[None, :]
        )
        return int(flag[0]), float(raw[0]), float(stat[0]), float(step[0])

    def _validation_samples(self, observed, state):
        """Retain noise-scaled residual evidence for each finite candidate."""
        size, n = state.shape
        flag = np.zeros(size, np.uint16)
        outside = np.isfinite(state).all(axis=-1) & np.any(
            (state < self._bounds[:, 0]) | (state > self._bounds[:, 1]), axis=-1
        )
        flag[outside] |= int(SampleFlag.outside_domain)
        raw, stat, step_noise = (np.full(size, np.nan) for _ in range(3))
        with np.errstate(all="ignore"):
            prediction, jac = self._response(state)
        finite = np.isfinite(prediction).all(axis=-1) & np.isfinite(jac).all(
            axis=(-2, -1)
        )
        flag[~finite] |= int(SampleFlag.nonfinite_output)
        positions = np.flatnonzero(finite)
        if not len(positions):
            return flag, raw, stat, step_noise
        prediction, jac = prediction[finite], jac[finite]
        residual = (observed[finite] - prediction) @ self._whitening.T
        u, singular, _ = np.linalg.svd(self._whitening @ jac, full_matrices=True)
        valid = np.sum(singular > self.rank_rtol * singular[:, :1], axis=-1) == n
        raw[positions] = np.sum(residual**2, axis=-1)
        flag[positions[~valid]] |= int(SampleFlag.singular_response)
        positions, residual = positions[valid], residual[valid]
        prediction, u = prediction[valid], u[valid]
        step_noise[positions] = np.linalg.norm(
            np.einsum("nji,nj->ni", u[:, :, :n], residual), axis=-1
        )
        if self.residual_dof <= 0:
            return flag, raw, stat, step_noise
        m = len(self.probes)
        gain_jac = np.zeros((len(positions), 2 * m, 2 * m))
        k = np.arange(m)
        gain_jac[:, k, k] = gain_jac[:, k + m, k + m] = prediction[:, :m]
        gain_jac[:, k + m, k] = prediction[:, m:]
        gain_jac[:, k, k + m] = -prediction[:, m:]
        null = u[:, :, n:].swapaxes(-1, -2)
        e = np.einsum("nij,nj->ni", null, residual)
        a = null @ self._whitening @ gain_jac
        covariance = np.eye(self.residual_dof) + a @ self.gain_covariance @ a.swapaxes(
            -1, -2
        )
        stat[positions] = np.sum(
            e * np.linalg.solve(covariance, e[..., None])[..., 0], axis=-1
        )
        flag[positions[stat[positions] > self._residual_max]] |= int(
            SampleFlag.residual_high
        )
        return flag, raw, stat, step_noise

    def solve(  # noqa: PLR0913, PLR0915 -- One labeled lazy recovery result.
        self,
        signals,
        *,
        normalization=1.0,
        chunk_samples=4096,
        initial_fr=None,
        initial_r=None,
        reset=None,
    ):
        """Return lazy resonator/time results and labeled recovery diagnostics.

        Parameters
        ----------
        signals : xarray.Dataset
            I/Q on chan/time with the same unique channel labels as this solver,
            in any order. Select samples before calling this method.
        normalization : float, optional
            Positive I/Q divisor, default 1, converting to calibration units.
        chunk_samples : int, optional
            Maximum numerical time chunk length, default 4096.
        initial_fr, initial_r : array_like or None, optional
            Frequency (Hz) and loss hints, both per resonator. Grouped recovery
            checks independent starts. Singlet analytical recovery ignores hints.
        reset : array_like of bool or None, optional
            True requires initialization before that sample. Additionally, a
            jump in the original integer sample coordinate triggers a reset.
            Without that coordinate, callers must mark omitted samples here.

        Returns
        -------
        xarray.Dataset
            Lazy fr/r on time/resonator, x on chan/time/resonator, sample flags,
            and labeled channel/resonator coupling and reference diagnostics.
            Original time/sample coordinates are retained. initial_flags records
            the first attempt; starts_tested counts starting points (zero for an
            analytical inverse). restarted marks recovery retries, or sequence
            initialization in continuation mode. residual_stat is the gain-aware
            residual statistic (NaN when no residual dimensions remain). step_noise is
            the remaining correction in conditional noise units. Failed states
            are NaN; finite rejected-candidate residuals remain diagnostic.

        Raises
        ------
        ValueError
            Channels, dimensions, normalization or chunk length are invalid.

        Notes
        -----
        Input files must stay open until computation or persistence completes.
        Detuning x is reported per (channel,resonator), because each probe has
        its own frequency. fr and r are resonator quantities.
        """
        if (
            signals.sizes.get("chan") != len(self.channels)
            or set(signals.chan.values) != set(self.channels)
            or chunk_samples < 1
            or int(chunk_samples) != chunk_samples
        ):
            msg = (
                "Selected channels must equal the prepared probes; use positive chunks"
            )
            raise ValueError(msg)
        normalization = _positive(normalization, "normalization")
        signals = signals.sel(chan=self.channels)
        source = (
            signals[["I", "Q"]]
            .transpose("time", "chan")
            .chunk({"time": chunk_samples, "chan": -1})
        )
        initial = self._initial_state(initial_fr, initial_r)
        mask = da.from_array(
            self._reset_mask(reset, source.sizes["time"]),
            chunks=source.I.data.chunks[0],
        )
        if "sample" in source.coords and source.sizes["time"]:
            if source.sample.dims != ("time",) or not np.issubdtype(
                source.sample.dtype, np.integer
            ):
                msg = "sample must be indexed by time for gap detection"
                raise ValueError(msg)
            samples = da.asarray(source.sample.data)
            mask = mask | da.concatenate(
                [da.zeros(1, dtype=bool), da.diff(samples) != 1]
            )
        mask = mask.rechunk(source.I.data.chunks[0])
        chunks = []
        checkpoint = None
        for i, q, reset_chunk in zip(
            source.I.data.to_delayed().ravel(),
            source.Q.data.to_delayed().ravel(),
            mask.to_delayed().ravel(),
            strict=True,
        ):
            observed = delayed(self._observations)(i, q, normalization)
            block = delayed(self._recover_block)(
                observed, reset_chunk, checkpoint, initial
            )
            chunks.append(block)
            if self.continuation:
                checkpoint = block["checkpoint"]
        variables = {}
        for name in (
            "fr",
            "r",
            "flags",
            "residual_norm",
            "residual_stat",
            "step_noise",
            "restarted",
            "initial_flags",
            "starts_tested",
        ):
            dims = ("time", "resonator") if name in {"fr", "r"} else ("time",)
            dtype = (
                np.uint16
                if name in {"flags", "initial_flags", "starts_tested"}
                else bool
                if name == "restarted"
                else float
            )
            tail = (len(self.fr),) if len(dims) == _MATRIX_NDIM else ()
            arrays = [
                da.from_delayed(block[name], shape=(size, *tail), dtype=dtype)
                for block, size in zip(chunks, source.I.data.chunks[0], strict=True)
            ]
            variables[name] = (dims, da.concatenate(arrays, axis=0))
        out = xr.Dataset(
            variables,
            coords={
                name: c for name, c in source.coords.items() if c.dims == ("time",)
            },
        )
        out = out.assign_coords(resonator=self.resonator_ids, chan=self.channels)
        out["coupling_re"] = (("chan", "resonator"), self.coupling.real)
        out["coupling_im"] = (("chan", "resonator"), self.coupling.imag)
        out["f_probe"] = ("chan", self.probes, {"units": "Hz"})
        out["x"] = out.f_probe / out.fr - 1
        out.fr.attrs["units"] = "Hz"
        out.r.attrs["units"] = out.x.attrs["units"] = ""
        out.flags.attrs.update(
            flag_masks=[int(flag) for flag in SampleFlag],
            flag_meanings=" ".join(flag.name for flag in SampleFlag),
        )
        out.initial_flags.attrs.update(out.flags.attrs)
        out.starts_tested.attrs["description"] = (
            "Full-model starting points evaluated; zero for the analytical inverse"
        )
        out["reference_covariance"] = (
            ("state", "state2"),
            self.diagnostics["covariance"],
        )
        out["noise_amplification"] = ("state", self.diagnostics["noise_amplification"])
        out["singular_values"] = ("singular_value", self.diagnostics["singular_values"])
        out = out.assign_coords(
            state_resonator=("state", np.repeat(self.resonator_ids, 2)),
            state_coordinate=(
                "state",
                np.tile(["dr/r_ref", "dfr/(2*r_ref*fr_ref)"], len(self.fr)),
            ),
            state2_resonator=("state2", np.repeat(self.resonator_ids, 2)),
            state2_coordinate=(
                "state2",
                np.tile(["dr/r_ref", "dfr/(2*r_ref*fr_ref)"], len(self.fr)),
            ),
        )
        out.attrs.update(
            {
                k: v
                for k, v in self.diagnostics.items()
                if k not in {"covariance", "noise_amplification", "singular_values"}
            }
        )
        out.attrs["iq_normalization"] = normalization
        out.attrs["continuation"] = self.continuation
        out.attrs["recovery_method"] = (
            "analytical"
            if len(self.probes) == len(self.fr) == 1
            else "validated_full_model"
        )
        out.attrs["noise_step_tol"] = self.noise_step_tol
        out.attrs["restart_step_max"] = self.restart_step_max
        out.attrs["ambiguity_delta_chi2"] = self.ambiguity_delta_chi2
        out.attrs["residual_dof"] = self.residual_dof
        out.attrs["residual_threshold"] = self._residual_max
        out.attrs["residual_gain_uncertainty"] = (
            "included" if np.any(self.gain_covariance) else "exact_gain_assumed"
        )
        out["gain_covariance"] = (("iq_gain", "iq_gain2"), self.gain_covariance)
        out["gain_covariance"].attrs["coordinate_order"] = (
            "all log amplitudes, then all phases (rad), in chan order"
        )
        if self.fr_bounds is not None:
            out["fr_bounds"] = (("resonator", "bound"), self.fr_bounds, {"units": "Hz"})
            out = out.assign_coords(bound=["lower", "upper"])
        return out
