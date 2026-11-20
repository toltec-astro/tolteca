"""Selected-array solving preserves calibration identities and failure semantics."""

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table
from dask import array as da
from dask import delayed

from tolteca_kidsproc.joint_timestream import JointTimestreamSolver
from tolteca_kidsproc.recovery import SampleFlag
from tolteca_kidsproc.timestream import (
    ModelFlag,
    TimestreamSolver,
    solve_timestream,
)


def model_table(family):
    """Independent forward-model parameters and explicit labeled report."""
    common = {
        "f_g": 500e6,
        "fr": 500.01e6,
        "r": 5e-5,
        "c_re": 2e-5,
        "c_im": 3e-6,
        "H0_re": 2.0,
        "H0_im": 3.0,
    }
    if family == "legacy":
        name = "S21WithGainLinTrend"
        params = {
            "fp": 500e6,
            "Qr": 1e4,
            "Qc": 2.5e4,
            "fr": 500.01e6,
            "A": 0.15,
            "normI": -2.0,
            "normQ": -3.0,
            "slopeI": 1e-7,
            "slopeQ": -1e-7,
            "interceptI": 2.0,
            "interceptQ": 3.0,
        }
    elif family == "affine":
        name = "KidsSweepWithAffineGain"
        params = dict(common, H1_re=1e-7, H1_im=-1e-7)
    else:
        name = "KidsSweepWithDelayGain"
        params = dict(common, a=1e-7, tau=2e-8)
    units = {
        "fp": "Hz",
        "f_g": "Hz",
        "fr": "Hz",
        "H1_re": "1 / Hz",
        "H1_im": "1 / Hz",
        "a": "1 / Hz",
        "tau": "s",
    }
    table = Table({k: [v, v, v] for k, v in params.items()})
    table["chan"] = [17, 2, 93]
    table["flag"] = [0, 0, 0]
    table["f_in"] = [500.012e6, 500.016e6, 500.022e6]
    table["f_in"].unit = "Hz"
    for n in params:
        table[n].unit = units.get(n, "")
    table.meta["model_schema"] = {
        "name": name,
        "version": 1,
        "parameters": [{"name": n, "unit": units.get(n, "")} for n in params],
    }
    return table


def measured(table, family, r, x):
    f = np.asarray(table["f_in"])
    w = r + 1j * x
    if family == "legacy":
        offset = table["interceptI"] + 1j * table["interceptQ"]
        slope = table["slopeI"] + 1j * table["slopeQ"]
        norm = table["normI"] + 1j * table["normQ"]
        return (
            np.asarray(offset + slope * (f - table["fp"]))
            + np.asarray(norm * (1 + 1j * table["A"]) / (2 * table["Qc"])) / w
        )
    h = np.asarray(table["H0_re"] + 1j * table["H0_im"])
    if family == "affine":
        h += np.asarray(table["H1_re"] + 1j * table["H1_im"]) * (f - table["f_g"])
    else:
        h *= np.exp(
            np.asarray(table["a"] - 2j * np.pi * table["tau"]) * (f - table["f_g"])
        )
    return h * (1 - np.asarray(table["c_re"] + 1j * table["c_im"]) / w)


@pytest.mark.parametrize("family", ["legacy", "affine", "delay"])
def test_inverse_recovers_injected_states_and_label_selected_order(family):
    table = model_table(family)
    phase = np.arange(37)[:, None] * 0.1
    r = np.broadcast_to(5e-5 * (1 + 0.01 * np.sin(phase)), (37, 3))
    x = (np.asarray(table["f_in"]) / 500.01e6 - 1) + 1e-6 * np.cos(phase)
    iq = measured(table, family, r, x) * 19.5
    solver = TimestreamSolver(table[[1, 2, 0]], [93, 17])
    rr, xx, flags = solver(iq.real[:, [2, 0]], iq.imag[:, [2, 0]], normalization=19.5)
    np.testing.assert_allclose(rr, r[:, [2, 0]], rtol=2e-14)
    np.testing.assert_allclose(xx, x[:, [2, 0]], rtol=2e-14)
    assert not flags.any()
    # Prepared calibration is a snapshot, not a reference to mutable input rows.
    table["flag"][:] = 1 << 13
    np.testing.assert_array_equal(
        solver(iq.real[:, [2, 0]], iq.imag[:, [2, 0]], normalization=19.5)[0], rr
    )


def test_flags_distinguish_warnings_invalid_models_and_nonfinite_samples():
    table = model_table("legacy")
    table["flag"] = [1 << 13, 1 << 16, 0]
    table["Qr"][2] = -1  # Historical Qr is diagnostic, not an inverse operand.
    solver = TimestreamSolver(table, [17, 2, 93])
    assert solver.flags.tolist() == [2, 0, ModelFlag.nonphysical_model]
    iq = measured(table, "legacy", np.full((4, 3), 5e-5), np.full((4, 3), 1e-5))
    iq[1, 0] = np.nan
    iq[1, 1] = np.inf
    r, x, flags = solver(iq.real, iq.imag)
    assert flags[0, 0] == SampleFlag.invalid_model
    assert flags[1, 0] == SampleFlag.invalid_model | SampleFlag.nonfinite_input
    assert flags[1, 1] == SampleFlag.nonfinite_input | SampleFlag.nonfinite_output
    assert np.isfinite(r[0, 1:]).all() and np.isfinite(x[:, 2]).all()
    assert np.isnan(r[:, 0]).all()
    offsets = solver.coefficients[:2].T
    rr, xx, ff = solver(offsets[:, 0][None, :], offsets[:, 1][None, :])
    assert np.isnan(rr).all() and np.isnan(xx).all()
    assert ff[0, 1] == SampleFlag.singular_response | SampleFlag.nonfinite_output
    with pytest.raises(ValueError, match="normalization"):
        solver(iq.real, iq.imag, normalization=0)


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_fixed_b_singlet_inverse_loading_domain_and_calibration_identity(family):
    table = model_table(family)
    a_cal = np.array([0.1, 0.25, 0.35])
    table["b"] = 2 * a_cal * np.asarray(table["r"]) ** 3
    table.meta["model_schema"]["version"] = 2
    table.meta["model_schema"]["parameters"].append({"name": "b", "unit": ""})
    phase = np.linspace(0, 2 * np.pi, 83)[:, None]
    r = np.broadcast_to(5e-5 * (1 + 0.15 * np.sin(phase)), (83, 3))
    expected = np.broadcast_to(4e-4 * np.cos(phase), (83, 3))
    driven = np.empty_like(expected)
    # Generate IQ from the independent sweep equation y = y0 + a/(1+4*y**2).
    # The optical state is known; a changes as r changes at fixed drive power.
    for i, j in np.ndindex(driven.shape):
        y0 = expected[i, j] / (2 * r[i, j])
        a = a_cal[j] * (table["r"][j] / r[i, j]) ** 3
        roots = np.roots([4, -4 * y0, 1, -(y0 + a)])
        [y] = roots.real[np.abs(roots.imag) < 1e-12]
        driven[i, j] = 2 * r[i, j] * y
    b = np.asarray(table["b"])
    slope = 1 + 2 * b * driven / (r**2 + driven**2) ** 2
    z = measured(table, family, r, driven)
    solver = TimestreamSolver(table[[1, 2, 0]], [93, 17])
    rr, xx, flags = solver(
        4 * z.real[:, [2, 0]], 4 * z.imag[:, [2, 0]], normalization=4
    )
    valid = slope[:, [2, 0]] > 0
    np.testing.assert_allclose(rr[valid], r[:, [2, 0]][valid], atol=1e-18)
    np.testing.assert_allclose(xx[valid], expected[:, [2, 0]][valid], atol=1e-18)
    assert not flags[valid].any()
    assert np.all(flags[~valid] & SampleFlag.unstable_response)
    np.testing.assert_array_equal(solver.model_params["chan"], [93, 17])
    # The same finite IQ algebra can produce an unstable or nonphysical state.
    r = np.full((2, 3), 2e-5)
    driven = -r.copy()
    r[1] = -r[1]
    z = measured(table, family, r, driven)
    rr, xx, flags = TimestreamSolver(table, table["chan"])(z.real, z.imag)
    assert flags[0, 1] & SampleFlag.unstable_response
    assert flags[1, 1] & SampleFlag.outside_domain
    assert np.isnan(rr[:, 1]).all() and np.isnan(xx[:, 1]).all()
    table["b"][1] = -1e-15
    rr, xx, flags = TimestreamSolver(table, table["chan"])(z.real, z.imag)
    assert np.all(flags[:, 1] == SampleFlag.invalid_model)


def test_lazy_selection_computes_only_requested_source_chunks_and_preserves_axes():
    loaded = []

    @delayed
    def read(part):
        loaded.append(part)
        return np.full((40, 3), 0.3 + part)

    data = da.concatenate(
        [da.from_delayed(read(p), shape=(40, 3), dtype=float) for p in range(5)]
    )
    signals = xr.Dataset(
        {"I": (("time", "chan"), data), "Q": (("time", "chan"), data * 2)},
        coords={
            "chan": [17, 2, 93],
            "time": np.arange(200) / 10,
            "sample": ("time", np.arange(200) + 250),
        },
    )
    selected = signals.isel(time=slice(45, 57), chan=[2, 0])
    out = solve_timestream(selected, model_table("affine"), chunk_samples=5)
    assert not loaded and out.r.chunks is not None
    out = out.compute(scheduler="synchronous")
    assert loaded == [1]
    xr.testing.assert_equal(out.sample, selected.sample)
    xr.testing.assert_equal(out.time, selected.time)
    assert out.chan.values.tolist() == [93, 17]
    whole = solve_timestream(
        selected, model_table("affine"), chunk_samples=100
    ).compute()
    xr.testing.assert_equal(out, whole)


def test_schema_identity_errors_fail_before_sample_loading():
    table = model_table("affine")
    with pytest.raises(ValueError, match="No calibration"):
        TimestreamSolver(table, [4])
    with pytest.raises(ValueError, match="unique"):
        TimestreamSolver(table[[0, 0]], [17])
    table["fr"].unit = "MHz"
    with pytest.raises(ValueError, match="units"):
        TimestreamSolver(table, [17])


def joint_case(**kwargs):
    probes, fr = np.array([500e6, 500.03e6]), np.array([500.005e6, 500.025e6])
    r = np.array([3e-5, 4e-5])
    gain = np.array([2 + 3j, 3 - 2j])
    coupling = np.array([[2e-5 + 1e-6j, 1e-5 - 1e-6j], [1e-5 + 2e-6j, 2e-5 + 1e-6j]])
    solver = JointTimestreamSolver(
        probes,
        fr,
        r,
        gain,
        coupling,
        np.eye(4) * 1e-6,
        channels=[19, 3],
        resonator_ids=["r7", "r2"],
        **kwargs,
    )
    phase = np.arange(20)[:, None] * 0.2
    truth_r = r * (1 + 0.01 * np.sin(phase + np.array([0, 1])))
    truth_fr = fr * (1 + 2 * r * 0.01 * np.cos(phase + np.array([0, 1])))
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
    return solver, iq, truth_fr, truth_r


def test_joint_recovery_moves_both_resonators_and_preserves_condition_warnings():
    solver, iq, fr, r = joint_case(
        condition_max=1.01,
        noise_step_tol=1e-7,
        fr_bounds=np.array([[500.003e6, 500.007e6], [500.023e6, 500.027e6]]),
    )
    ff, rr, flags, residual = solver(iq.real * 7, iq.imag * 7, normalization=7)
    np.testing.assert_allclose(rr, r, rtol=1e-9)
    np.testing.assert_allclose(ff, fr, atol=0.01, rtol=0)
    assert (flags == SampleFlag.condition_high).all()
    assert np.max(residual) < 1e-6
    source = xr.Dataset(
        {"I": (("time", "chan"), iq.real), "Q": (("time", "chan"), iq.imag)},
        coords={
            "chan": [19, 3],
            "time": np.arange(20),
            "sample": ("time", np.arange(20) + 30),
        },
    )
    actual = solver.solve(source.isel(chan=[1, 0]), chunk_samples=7).compute()
    np.testing.assert_array_equal(actual.fr, ff)
    assert actual.resonator.values.tolist() == ["r7", "r2"]
    assert actual.chan.values.tolist() == [19, 3]
    xr.testing.assert_equal(actual.sample, source.sample)
    assert actual.x.sizes == {"chan": 2, "time": 20, "resonator": 2}


def test_joint_unresolved_subset_and_bad_samples_do_not_invent_states():
    solver, iq, _, _ = joint_case()
    partial = JointTimestreamSolver(
        solver.probes[:1],
        solver.fr,
        solver.r,
        solver.gain[:1],
        solver.coupling[:1],
        np.eye(2),
        channels=[19],
        resonator_ids=solver.resonator_ids,
    )
    fr, r, flags, _ = partial(iq.real[:, :1], iq.imag[:, :1])
    assert partial.diagnostics["rank"] < 4
    assert (flags == SampleFlag.singular_response).all()
    assert np.isnan(fr).all() and np.isnan(r).all()
    assert np.isnan(partial.diagnostics["covariance"]).all()
    iq[1, 0] = np.nan
    fr, r, flags, _ = solver(iq.real, iq.imag)
    assert flags[1] == SampleFlag.nonfinite_input
    assert np.isnan(fr[1]).all() and np.isnan(r[1]).all()
    assert flags[0] == SampleFlag.ambiguous_solution
    assert np.isnan(fr[0]).all() and np.isnan(r[0]).all()


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_physical_analytical_domain_and_large_excursions_match_group_dispatch(family):
    table = model_table(family)[[0]]
    loss = np.array([5e-5, 3e-6, 1e-3, -2e-6, 5e-5])[:, None]
    detuning = np.array([0, 0.01, -0.01, 1e-4, -1.5])[:, None]
    iq = measured(table, family, loss, detuning)
    direct = TimestreamSolver(table, [17])
    r, x, flags = direct(iq.real, iq.imag)
    np.testing.assert_allclose(r[:3], loss[:3], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(x[:3], detuning[:3], atol=1e-14)
    assert (flags[3:] == SampleFlag.outside_domain).all()
    assert np.isnan(r[3:]).all() and np.isnan(x[3:]).all()
    probe = float(table["f_in"][0])
    fg = float(table["f_g"][0])
    h = complex(table["H0_re"][0], table["H0_im"][0])
    if family == "affine":
        h += complex(table["H1_re"][0], table["H1_im"][0]) * (probe - fg)
    else:
        h *= np.exp((table["a"][0] - 2j * np.pi * table["tau"][0]) * (probe - fg))
    grouped = JointTimestreamSolver(
        [probe],
        [table["fr"][0]],
        [table["r"][0]],
        [h],
        [[complex(table["c_re"][0], table["c_im"][0])]],
        np.eye(2),
        channels=[17],
        resonator_ids=["a"],
    )
    signals = xr.Dataset(
        {"I": (("time", "chan"), iq.real), "Q": (("time", "chan"), iq.imag)},
        coords={"time": np.arange(len(iq)), "chan": [17]},
    )
    result = grouped.solve(
        signals, initial_fr=[probe * 1.01], initial_r=[0.1]
    ).compute()
    np.testing.assert_array_equal(result.flags, flags[:, 0])
    np.testing.assert_allclose(result.r, r, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result.x.values[0], x, atol=1e-14)
    assert not result.starts_tested.any()
    assert result.flags.dtype == np.uint16
    assert int(SampleFlag.outside_domain) in result.flags.attrs["flag_masks"]
