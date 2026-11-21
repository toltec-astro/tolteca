"""Recovery behavior with independent initialization and sample discontinuities."""

import numpy as np
import pytest
import xarray as xr

from tolteca_kidsproc.joint_timestream import JointTimestreamSolver
from tolteca_kidsproc.recovery import SampleFlag


def case(*, continuation=True, symmetric=False, **kwargs):
    probes = np.array([500e6, 500.03e6, 500.06e6])
    fr = np.array([500.005e6, 500.025e6])
    r = np.array([3e-5, 4e-5])
    gain = np.array([2 + 3j, 3 - 2j, 4 + 1j])
    coupling = np.array(
        [
            [2e-5 + 1e-6j, 1e-5 - 1e-6j],
            [1e-5 + 2e-6j, 2e-5 + 1e-6j],
            [3e-5 + 2e-6j, 0.4e-5 - 1e-6j],
        ]
    )
    if symmetric:
        coupling[:] = 1e-5
        r[:] = 3e-5
    options = {
        "continuation": continuation,
        "residual_p_value": 0.001,
        "fr_bounds": fr[:, None] + np.array([-1, 1]) * 60000,
    }
    options.update(kwargs)
    solver = JointTimestreamSolver(
        probes,
        fr,
        r,
        gain,
        coupling,
        np.eye(6) * 1e-6,
        channels=[19, 3, 27],
        resonator_ids=["r7", "r2"],
        **options,
    )
    motion = np.linspace(-0.2, 0.7, 30)[:, None]
    truth_fr = fr * (1 + 2 * r * motion)
    truth_r = np.broadcast_to(r, truth_fr.shape)
    iq = gain * (
        1
        - np.sum(
            coupling
            / (
                truth_r[:, None, :]
                + 1j * (probes[None, :, None] / truth_fr[:, None, :] - 1)
            ),
            axis=-1,
        )
    )
    source = xr.Dataset(
        {"I": (("time", "chan"), iq.real), "Q": (("time", "chan"), iq.imag)},
        coords={
            "time": np.arange(30) * 0.02,
            "sample": ("time", np.arange(30) + 100),
            "chan": solver.channels,
        },
    )
    return solver, source, truth_fr, truth_r


def test_bad_hint_gaps_and_lazy_chunking_keep_same_identity_and_results():
    solver, source, fr, r = case(state_step_max=0.8)
    source["I"][10] = np.nan
    selected = np.r_[np.arange(20), np.arange(23, 30)]
    source = source.isel(time=selected)
    initial = {"initial_fr": solver.fr[::-1], "initial_r": solver.r[::-1] * 2}
    reset = np.r_[False, np.diff(selected) != 1]
    eager = solver(source.I.values, source.Q.values, reset=reset, **initial)
    assert eager[2][10] == SampleFlag.nonfinite_input
    assert (eager[2][np.arange(len(selected)) != 10] == 0).all()
    np.testing.assert_allclose(
        eager[0],
        np.where(np.arange(len(selected))[:, None] == 10, np.nan, fr[selected]),
        atol=0.02,
        rtol=0,
    )
    np.testing.assert_allclose(
        eager[1],
        np.where(np.arange(len(selected))[:, None] == 10, np.nan, r[selected]),
        atol=1e-10,
        rtol=0,
    )
    for chunk in [1, 7, 64]:
        lazy = solver.solve(source.isel(chan=[2, 0, 1]), chunk_samples=chunk, **initial)
        assert hasattr(lazy.fr.data, "dask")
        result = lazy.compute(scheduler="threads")
        for key, expected in zip(
            ["fr", "r", "flags", "residual_norm"], eager, strict=True
        ):
            np.testing.assert_array_equal(result[key], expected)
        assert np.flatnonzero(result.restarted).tolist() == [0, 11, 20]
        assert result.resonator.values.tolist() == ["r7", "r2"]
        xr.testing.assert_equal(result.sample, source.sample)
        assert np.all(result.step_noise[result.flags == 0] <= solver.noise_step_tol)
    # A second evaluation has no hidden mutable continuation state.
    np.testing.assert_array_equal(
        solver(source.I.values, source.Q.values, reset=reset, **initial)[0], eager[0]
    )


def test_indistinguishable_resonator_permutations_are_flagged_not_assigned():
    solver, source, fr, r = case(symmetric=True, state_step_max=1.5)
    ff, rr, flags, _ = solver(
        source.I.values[:1],
        source.Q.values[:1],
        initial_fr=fr[0, ::-1],
        initial_r=r[0, ::-1],
    )
    assert flags[0] & SampleFlag.ambiguous_solution
    assert np.isnan(ff).all() and np.isnan(rr).all()
    assert solver.diagnostics["rank"] == 4


def test_gain_covariance_changes_residual_validation_not_fitted_state():
    exact, source, _, _ = case(continuation=False)
    uncertain, _, _, _ = case(continuation=False, gain_covariance=np.eye(6) * 0.01**2)
    iq = (source.I.values[:1] + 1j * source.Q.values[:1]) * np.exp(
        1j * np.array([0.003, -0.003, 0.003])
    )
    a = exact(iq.real, iq.imag)
    b = uncertain(iq.real, iq.imag)
    assert a[2][0] & SampleFlag.residual_high
    assert b[2][0] == 0
    assert np.isfinite(b[0]).all()
    np.testing.assert_allclose(a[3], b[3], rtol=0, atol=0)


def test_noise_scaled_convergence_and_frequency_support():
    solver, source, fr, _ = case(continuation=False)
    result = solver.solve(source).compute()
    assert (result.flags == 0).all()
    assert result.step_noise.max() <= 0.001
    np.testing.assert_allclose(result.fr, fr, atol=0.02, rtol=0)
    # The supplied calibration interval is a real constraint, not an optimizer hint.
    narrow, source, _, _ = case(
        fr_bounds=np.array([[500.004e6, 500.006e6], [500.024e6, 500.026e6]])
    )
    ff, _, flags, _ = narrow(source.I.values[-1:], source.Q.values[-1:])
    assert np.isnan(ff).all()
    assert np.all(flags & (SampleFlag.recovery_failed | SampleFlag.residual_high))


@pytest.mark.parametrize("n_probes", [2, 3])
def test_independent_full_inverse_preserves_large_motion_and_sample_boundaries(
    n_probes,
):
    original, source, fr, r = case(continuation=False, noise_step_tol=1e-7)
    solver = JointTimestreamSolver(
        original.probes[:n_probes],
        original.fr,
        original.r,
        original.gain[:n_probes],
        original.coupling[:n_probes],
        np.eye(2 * n_probes) * 1e-6,
        channels=original.channels[:n_probes],
        resonator_ids=original.resonator_ids,
        fr_bounds=original.fr_bounds,
        residual_p_value=0.001,
        noise_step_tol=1e-7,
        continuation=False,
    )
    source = source.isel(chan=slice(0, n_probes))
    source["I"][9] = np.nan
    result = solver.solve(source, chunk_samples=30).compute()
    finite = np.arange(len(fr)) != 9
    assert np.isin(
        result.flags.values[finite], [0, SampleFlag.ambiguous_solution]
    ).all()
    valid = result.flags.values == 0
    assert valid.any()
    if n_probes == 3:
        np.testing.assert_array_equal(valid, finite)
    assert np.isnan(result.fr.values[~valid]).all()
    np.testing.assert_allclose(result.fr.values[valid], fr[valid], rtol=0, atol=1e-4)
    np.testing.assert_allclose(result.r.values[valid], r[valid], rtol=1e-8, atol=0)
    assert result.flags.values[9] == SampleFlag.nonfinite_input
    assert np.isnan(result.fr.values[9]).all()
    for chunk in [1, 7]:
        other = solver.solve(source, chunk_samples=chunk).compute()
        np.testing.assert_array_equal(other.flags, result.flags)
        np.testing.assert_allclose(other.fr, result.fr, rtol=0, atol=1e-6)
        np.testing.assert_allclose(other.r, result.r, rtol=0, atol=1e-13)
        np.testing.assert_allclose(other.step_noise, result.step_noise, atol=1e-8)
    # Samples are independent; reordering them must not introduce continuity.
    shuffled = solver.solve(source.isel(time=np.arange(30)[::-1])).compute()
    np.testing.assert_array_equal(shuffled.flags, result.flags[::-1])
    np.testing.assert_allclose(shuffled.fr, result.fr[::-1], rtol=0, atol=1e-6)
    empty = solver.solve(source.isel(time=slice(0, 0))).compute()
    assert empty.fr.shape == (0, 2)


def test_failed_sample_does_not_stop_or_fill_neighboring_full_inverses():
    solver, source, fr, r = case(continuation=False)
    source["I"][9] += [20, -10, 30]
    result = solver.solve(source).compute()
    assert result.flags.values[9] & (
        SampleFlag.recovery_failed | SampleFlag.residual_high
    )
    assert np.isnan(result.fr.values[9]).all()
    assert np.isnan(result.r.values[9]).all()
    good = np.arange(30) != 9
    assert (result.flags.values[good] == 0).all()
    np.testing.assert_allclose(result.fr.values[good], fr[good], rtol=0, atol=0.02)
    np.testing.assert_allclose(result.r.values[good], r[good], rtol=0, atol=1e-10)


def test_finite_rejection_and_large_steps_restart_without_filling_values():
    solver, source, fr, _ = case()
    selected = np.r_[np.arange(12), 29]
    source = source.isel(time=selected).drop_vars("sample")
    source["I"][10] += np.array([20.0, -10.0, 30.0])
    result = solver.solve(source, chunk_samples=5).compute()
    assert result.flags[10].item() & int(
        SampleFlag.residual_high | SampleFlag.recovery_failed
    )
    assert np.isnan(result.fr[10]).all()
    assert result.flags[11].item() == 0
    assert result.flags[12].item() & int(SampleFlag.outside_domain)
    assert np.isnan(result.fr[12]).all()
    assert result.restarted[11].item() and result.restarted[12].item()
    reset = np.zeros(len(selected), bool)
    reset[-1] = True
    resumed = solver.solve(source, reset=reset).compute()
    assert resumed.flags[-1].item() == SampleFlag.outside_domain
    assert np.isnan(resumed.fr[-1]).all()
    np.testing.assert_allclose(resumed.fr[11], fr[selected][11], atol=0.02, rtol=0)
    empty = solver.solve(source.isel(time=slice(0, 0))).compute()
    assert empty.fr.shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"state_step_max": 0}, "state_step_max"),
        ({"state_step_max": np.inf}, "state_step_max"),
        ({"noise_step_tol": 0}, "noise_step_tol"),
        ({"fr_bounds": np.ones((2, 2))}, "fr_bounds"),
        ({"gain_covariance": -np.eye(6)}, "gain_covariance"),
    ],
)
def test_invalid_validation_inputs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        case(**kwargs)


def test_independent_restarts_validate_incorrect_hint_and_record_first_failure():
    solver, source, fr, r = case(continuation=False)
    result = solver.solve(
        source, initial_fr=solver.fr + 50000, initial_r=solver.r * 100
    ).compute()
    assert (result.flags == 0).all()
    assert result.restarted.any()
    assert (result.starts_tested > 1).all()
    failed = result.initial_flags.values & int(SampleFlag.recovery_failed) != 0
    assert failed.any()
    reasons = int(SampleFlag.stalled | SampleFlag.iteration_limit)
    assert (result.initial_flags.values[failed] & reasons != 0).all()
    np.testing.assert_allclose(result.fr, fr, rtol=0, atol=0.02)
    np.testing.assert_allclose(result.r, r, rtol=0, atol=1e-10)


def test_sweep_anchor_tracks_drift_but_rejects_distant_initial_hint():
    solver, source, fr, _ = case(fr_bounds=None, residual_p_value=None)
    result = solver.solve(source, chunk_samples=7).compute()
    assert (result.flags == 0).all()
    assert result.attrs["branch_selection"] == "sweep_continuity"
    assert result.attrs["state_step_max"] == 0.25
    np.testing.assert_allclose(result.fr, fr, atol=0.02, rtol=0)
    # The endpoint is valid only after the intervening accepted trajectory.
    distant = solver(
        source.I.values[-1:],
        source.Q.values[-1:],
        initial_fr=fr[-1],
        initial_r=solver.r,
    )
    assert distant[2][0] == SampleFlag.outside_domain
    assert np.isnan(distant[0]).all()
    # A nonfinite gap also discards the branch history.
    source["I"][28] = np.nan
    result = solver.solve(source, chunk_samples=7).compute()
    assert result.flags[28].item() == SampleFlag.nonfinite_input
    assert result.flags[29].item() == SampleFlag.outside_domain
