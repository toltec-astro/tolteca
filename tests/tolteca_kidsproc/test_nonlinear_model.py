"""Unique nonlinear sweep roots, analytic derivatives and retained calibration."""

import numpy as np
import pytest
import xarray as xr

from tolteca_kidsproc._nonlinear import nonlinear_detuning
from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.fitting import fit_kids_model
from tolteca_kidsproc.models import KidsSweepModel


def test_unique_root_and_zero_limit():
    y0 = np.linspace(-4, 4, 1001)[:, None]
    a = np.array([0, 0.2, 0.5, 0.75])
    y, derivative = nonlinear_detuning(y0, a)
    np.testing.assert_allclose(y - y0, a / (1 + 4 * y * y), atol=3e-13)
    np.testing.assert_array_equal(y[:, 0], y0[:, 0])
    assert np.all(np.diff(y, axis=0) > 0)
    assert np.all(derivative > 0)
    with pytest.raises(ValueError, match="Nonlinearity"):
        nonlinear_detuning(y0, 0.8)


@pytest.mark.parametrize("branch", ["increasing", "decreasing"])
def test_directional_roots_and_history_independent_evaluation(branch):
    y0 = np.linspace(-6, 4, 401)
    a = 2.0
    y, derivative = nonlinear_detuning(y0, a, branch=branch)
    expected = []
    for value in y0:
        roots = np.roots([4, -4 * value, 1, -value - a])
        real = np.sort(roots.real[abs(roots.imag) < 1e-9])
        expected.append(real[0] if branch == "increasing" else real[-1])
    np.testing.assert_allclose(y, expected, atol=2e-12)
    assert np.all(derivative > 0)
    # Held-out or plotted frequencies must not reset the declared sweep history.
    subset = np.array([190, 41, 211, 188])
    np.testing.assert_allclose(
        nonlinear_detuning(y0[subset], a, branch=branch)[0], y[subset], atol=2e-12
    )
    np.testing.assert_array_equal(
        nonlinear_detuning(y0, 0.75, branch=branch)[0],
        nonlinear_detuning(y0, 0.75)[0],
    )


@pytest.mark.parametrize("branch", ["increasing", "decreasing"])
@pytest.mark.parametrize("family", ["affine", "delay"])
def test_directional_jacobian_away_from_jump(branch, family):
    f = [np.linspace(499.8e6, 500.2e6, 101)]
    model = KidsSweepModel(
        [500e6],
        [12000],
        f,
        gain_family=family,
        nonlinear=True,
        nonlinear_branch=branch,
    )
    params = model.make_params(a_nl_max=5.0)
    params["r0_a_nl"].value = 2
    analytic = model.eval(params, f, jacobian=True)[0]
    for i, name in enumerate(model.names):
        lo, hi = params.copy(), params.copy()
        lo[name].value -= 1e-6
        hi[name].value += 1e-6
        numeric = (model.eval(hi, f)[0] - model.eval(lo, f)[0]) / 2e-6
        np.testing.assert_allclose(analytic[:, i], numeric, rtol=2e-5, atol=2e-8)
    with pytest.raises(ValueError, match="unique response"):
        KidsSweepModel([500e6], [12000], f, nonlinear=True).make_params(a_nl_max=5)
    with pytest.raises(ValueError, match="nonlinear=True"):
        KidsSweepModel([500e6], [12000], f, nonlinear_branch=branch)


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_nonlinear_jacobian_and_noisy_recovery(family):
    frequency = [np.linspace(499.85e6, 500.18e6, 241)] * 2
    model = KidsSweepModel(
        [500e6, 500.07e6], [12000, 16000], frequency, gain_family=family, nonlinear=True
    )
    params = model.make_params()
    params["r0_a_nl"].value = 0.4
    params["r1_a_nl"].value = 0.2
    params["c0_h1_re"].value = 0.03
    params["c1_h0_im"].value = 0.1
    analytic = np.concatenate(model.eval(params, frequency, jacobian=True))
    for i, name in enumerate(model.names):
        lo, hi = params.copy(), params.copy()
        lo[name].value -= 1e-6
        hi[name].value += 1e-6
        numeric = (
            np.concatenate(model.eval(hi, frequency))
            - np.concatenate(model.eval(lo, frequency))
        ) / 2e-6
        np.testing.assert_allclose(analytic[:, i], numeric, atol=2e-8, rtol=3e-5)
    rng = np.random.default_rng(56)
    measured = [
        z + 2e-5 * (rng.normal(size=z.size) + 1j * rng.normal(size=z.size))
        for z in model.eval(params, frequency)
    ]
    fit = fit_kids_model(
        frequency,
        measured,
        model.fr_reference,
        model.qr_reference,
        gain_family=family,
        nonlinear=True,
    )
    assert fit.success
    np.testing.assert_allclose(
        fit.model.physical(fit.params)["a_nl"], [0.4, 0.2], atol=0.003
    )
    with pytest.raises(NotImplementedError, match="fixed-power"):
        fit.response_jacobian([500e6, 500.07e6])


@pytest.mark.parametrize(("a_nl", "branch"), [(0.45, "unique"), (2.0, "increasing")])
def test_nonlinear_saved_calibration_and_pair_transform(tmp_path, a_nl, branch):
    f = np.linspace(499.9e6, 500.1e6, 161)
    model = KidsSweepModel(
        [500e6], [12000], [f], nonlinear=True, nonlinear_branch=branch
    )
    upper = 0.75 if branch == "unique" else 5.0
    params = model.make_params(a_nl_max=upper)
    params["r0_a_nl"].value = a_nl
    z = model.eval(params, [f])[0]
    source = xr.Dataset(
        {
            "I": (("chan", "sweep"), z.real[None]),
            "Q": (("chan", "sweep"), z.imag[None]),
        },
        coords={
            "chan": [4],
            "sweep": np.arange(len(f)),
            "frequency": (("chan", "sweep"), f[None]),
        },
    )
    source.frequency.attrs["units"] = "Hz"
    plan = xr.Dataset(
        {
            key: ("row", [value])
            for key, value in {
                "group_id": "g",
                "chan": 4,
                "resonator_id": "r",
                "sample_start": 0,
                "sample_stop": len(f),
                "fr_initial": 500e6,
                "r_initial": 1 / 24000,
            }.items()
        }
    )
    result = KidsModelAnalysis(
        nonlinear=True,
        nonlinear_branch=branch,
        a_nl_max=upper,
    )(source.kids.sweep, plan)
    result.to_zarr(tmp_path / "fit.zarr", zarr_format=2)
    with xr.open_datatree(tmp_path / "fit.zarr", engine="zarr") as saved:
        view = saved.kids.model_fit
        view.validate()
        assert float(view.fit.a_nl[0]) == pytest.approx(a_nl, abs=1e-5)
        assert view.fit.attrs["nonlinear_branch"] == branch
        assert view.components().attrs["nonlinear_branch"] == branch
        assert view.joint.uncertainty_status.item() == (
            "local_covariance" if branch == "unique" else "branch_local_covariance"
        )
        np.testing.assert_allclose(view.predict("g", 4, f), z, atol=1e-8)
        pair = view.pair_model("g", 4, "r")
        np.testing.assert_allclose(pair.eval(f), z, atol=1e-8)
        np.testing.assert_allclose(pair.rotate(pair.derotate(z, f), f), z)
        assert "resonator:r:a_nl" in view.covariance("g").parameter.values


def test_calibration_requires_explicit_direction_above_default_domain():
    # A persisted coefficient above the unique domain must not be interpreted
    # using an unrecorded sweep history.
    from tolteca_kidsproc.calibration import validate_calibration

    tree = xr.DataTree.from_dict({"fit": xr.Dataset({"a_nl": ("row", [2.0])})})
    with pytest.raises(ValueError, match="unique response"):
        validate_calibration(tree)


def test_directional_physical_replay_at_real_sample_on_fold():
    # Obs 142440 scan 3, channel 549: scaled vs physical detuning formerly
    # selected opposite roots at this measured frequency (IQ error 0.429).
    probe = 453692687.5
    f = np.array([probe - 24000, probe + 22000])
    model = KidsSweepModel(
        [453691791.9921875],
        [38708.73842188849],
        [f],
        gain_origin=[453691687.5],
        signal_scale=1.0058833033872838,
        nonlinear=True,
        nonlinear_branch="increasing",
    )
    params = model.make_params(fr_bounds=([453600000], [453800000]), a_nl_max=20)
    values = {
        "r0_xi": -1.9845796736476289,
        "r0_rho": 0.9564743405667337,
        "r0_a_nl": 5.40276390995923,
        "c0_r0_re": 0.41897652470679925,
        "c0_r0_im": 0.06550824218177599,
        "c0_h0_re": -0.549515617486342,
        "c0_h0_im": 0.9113810891929519,
        "c0_h1_re": -0.0406207559088921,
        "c0_h1_im": -0.06468011464449835,
    }
    for name, value in values.items():
        params[name].value = value
    frequencies = probe + np.arange(-10, 11) * 2000
    predicted = model.eval(params, [frequencies])[0]
    pair = model.pair_model(params, 0, 0)
    np.testing.assert_allclose(predicted, pair.eval(frequencies), rtol=0, atol=2e-14)
    np.testing.assert_allclose(predicted[10], pair.eval(probe), rtol=0, atol=2e-14)


@pytest.mark.parametrize("branch", ["increasing", "decreasing"])
def test_directional_roots_independent_of_batch_convergence(branch):
    from scipy.optimize import brentq

    strength = 2.0
    left = brentq(lambda y: 1 + 16 * y / (1 + 4 * y * y) ** 2, -3, -0.3)
    threshold = left - strength / (1 + 4 * left * left)
    y0 = np.r_[threshold + np.array([-1e-12, -1e-14, 1e-14, 1e-12]), -2.1, 1.1]
    batch = nonlinear_detuning(y0, strength, branch=branch)
    singles = [nonlinear_detuning(v, strength, branch=branch) for v in y0]
    np.testing.assert_array_equal(batch[0], [s[0] for s in singles])
    np.testing.assert_array_equal(batch[1], [s[1] for s in singles])
