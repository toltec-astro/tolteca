"""Fixed-drive nonlinear joint solving, derivatives and branch restrictions."""

import numpy as np
import pytest

from tolteca_kidsproc.joint_timestream import JointTimestreamSolver

from .test_joint_continuation import case


@pytest.mark.parametrize(
    ("branch", "strength"), [("unique", 0.4), ("increasing", 2.0), ("decreasing", 2.0)]
)
def test_fixed_drive_response_jacobian_and_recovered_states(branch, strength):
    reference, _, _, _ = case(continuation=False)
    solver = JointTimestreamSolver(
        reference.probes,
        reference.fr,
        reference.r,
        reference.gain,
        reference.coupling,
        np.eye(6) * 1e-6,
        channels=reference.channels,
        resonator_ids=reference.resonator_ids,
        b=2 * strength * reference.r**3,
        nonlinear_branch=branch,
        noise_step_tol=1e-8,
    )
    state = np.array([0.02, 0.03, 0.01, -0.02])
    response, jac = solver._response(state)
    for column in range(4):
        step = np.zeros(4)
        step[column] = 1e-5
        numeric = (
            solver._response(state + step)[0] - solver._response(state - step)[0]
        ) / (2e-5)
        np.testing.assert_allclose(jac[:, column], numeric, rtol=3e-5, atol=1e-7)
    # Different losses at the same drive: a_nl must vary as r^-3.
    losses = solver.r * (1 + state[::2])
    fr = solver.fr * (1 + 2 * solver.r * state[1::2])
    y0 = (solver.probes[:, None] / fr - 1) / (2 * losses)
    roots = np.empty_like(y0)
    for k in range(3):
        for j in range(2):
            a = solver.b[j] / (2 * losses[j] ** 3)
            candidates = np.roots([4.0, -4 * y0[k, j], 1.0, -y0[k, j] - a])
            real = np.sort(candidates.real[abs(candidates.imag) < 1e-9])
            roots[k, j] = real[-1] if branch == "decreasing" else real[0]
    expected = solver.gain * (
        1 - np.sum(solver.coupling / (losses * (1 + 2j * roots)), axis=-1)
    )
    np.testing.assert_allclose(
        response, np.r_[expected.real, expected.imag], rtol=1e-12, atol=1e-12
    )
    result = solver(expected.real[None], expected.imag[None])
    assert result[2].tolist() == [0]
    np.testing.assert_allclose(result[0][0], fr, atol=0.02, rtol=0)
    np.testing.assert_allclose(result[1][0], losses, atol=1e-12, rtol=1e-6)


def test_unique_response_rejects_unqualified_nonlinearity():
    reference, _, _, _ = case(continuation=False)
    with pytest.raises(ValueError, match="unique response"):
        JointTimestreamSolver(
            reference.probes,
            reference.fr,
            reference.r,
            reference.gain,
            reference.coupling,
            np.eye(6) * 1e-6,
            channels=reference.channels,
            resonator_ids=reference.resonator_ids,
            b=2 * 0.8 * reference.r**3,
        )


def test_unusable_group_flag_vetoes_eager_and_lazy_solutions():
    from tolteca_kidsproc.model_flags import ModelFitFlag
    from tolteca_kidsproc.recovery import SampleFlag

    solver, source, _, _ = case(continuation=False, fit_flag=ModelFitFlag.fit_unusable)
    eager = solver(source.I.values, source.Q.values)
    assert np.all(eager[2] & SampleFlag.invalid_model)
    assert np.isnan(eager[0]).all() and np.isnan(eager[1]).all()
    lazy = solver.solve(source, chunk_samples=7).compute()
    np.testing.assert_array_equal(lazy.flags, eager[2])
    np.testing.assert_array_equal(lazy.r, eager[1])
