"""Physical response, derivative, uncertainty and compact-analysis contracts."""

import json

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from tolteca_kidsproc.accessors import make_kids_dataset
from tolteca_kidsproc.analysis import KidsModelAnalysis, KidsModelView
from tolteca_kidsproc.calibration import (
    calibration_pair_model,
    evaluate_calibration,
    validate_calibration,
)
from tolteca_kidsproc.fitting import fit_kids_model
from tolteca_kidsproc.models import KidsSweepModel


def case(family="affine"):
    fr = np.array(
        [500.025e6, 500e6]
    )  # Identity order deliberately differs from frequency.
    qr = np.array([14000.0, 11000.0])
    r = 1 / (2 * qr)
    f = [np.linspace(499.85e6, 500.16e6, 101), np.linspace(499.9e6, 500.2e6, 93)]
    coupling = np.array(
        [[8e-6 - 1e-6j, 1.5e-5 + 2e-6j], [7e-6 + 2e-6j, 1.4e-5 - 1e-6j]]
    )
    z = []
    for k, ff in enumerate(f):
        d = (ff - 500e6) / 3e5
        h = (1 + 0.2j) * (k + 1)
        h = (
            h + (0.06 + 0.04j) * d
            if family == "affine"
            else h * np.exp((0.06 + 0.04j) * d)
        )
        z.append(
            h * (1 - np.sum(coupling[k] / (r + 1j * (ff[:, None] / fr - 1)), axis=1))
        )
    return f, z, fr, qr, coupling


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_shared_physical_state_and_covariance(family):
    f, z, fr, qr, coupling = case(family)
    fit = fit_kids_model(
        f,
        z,
        fr + np.array([500, -800]),
        qr * 0.95,
        gain_family=family,
        sigma=[np.full((len(ff), 2), 1e-4) for ff in f],
    )
    assert fit.success
    p = fit.model.physical(fit.params)
    np.testing.assert_allclose(p["fr"], fr, rtol=0, atol=0.05)
    np.testing.assert_allclose(p["r"], 1 / (2 * qr), rtol=1e-5)
    np.testing.assert_allclose(p["c"], coupling, rtol=1e-5)
    for a, b in zip(fit.best_fit, z, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8)
    assert fit.rank == len(fit.raw.var_names)
    assert np.isfinite(fit.covariance).all()
    np.testing.assert_allclose(fit.covariance, fit.covariance.T, atol=1e-18)
    assert abs(fit.covariance[0, 2]) > 0  # Joint states retain cross-covariance.
    for k in range(2):
        for j in range(2):
            selected = fit.pair_model(k, j)
            expected = np.sum(
                (p["c"][k] / p["c"][k, j])
                * 0.5
                / (p["r"] + 1j * (f[k][:, None] / p["fr"] - 1)),
                axis=1,
            )
            np.testing.assert_allclose(
                selected.derotate(fit.best_fit[k], f[k]), expected, rtol=1e-10
            )


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_analytic_jacobian_and_physical_fixed_probe_derivative(family):
    f, z, fr, qr, _ = case(family)
    result = fit_kids_model(f, z, fr, qr, gain_family=family)
    model, params = result.model, result.params
    analytic = np.concatenate(model.eval(params, f, jacobian=True))
    for i, name in enumerate(model.names):
        a = params.copy()
        b = params.copy()
        step = 1e-5
        a[name].value -= step
        b[name].value += step
        numeric = (
            np.concatenate(model.eval(b, f)) - np.concatenate(model.eval(a, f))
        ) / (2 * step)
        np.testing.assert_allclose(analytic[:, i], numeric, atol=2e-8, rtol=2e-5)
    probes = np.array([500.002e6, 500.022e6])
    p = model.physical(params)
    h = []
    for k, fp in enumerate(probes):
        d = fp - model.gain_origin[k]
        h.append(
            p["H0"][k] + p["H1"][k] * d
            if family == "affine"
            else p["H0"][k] * np.exp((p["a"][k] - 2j * np.pi * p["tau"][k]) * d)
        )

    def forward(fr, r):
        return np.array(h) * (
            1 - np.sum(p["c"] / (r + 1j * (probes[:, None] / fr - 1)), axis=1)
        )

    matrix = []
    for j in range(2):
        for name, step in (("fr", 0.1), ("r", 1e-9)):
            a = {n: p[n].copy() for n in ("fr", "r")}
            b = {n: p[n].copy() for n in ("fr", "r")}
            a[name][j] -= step
            b[name][j] += step
            deriv = (forward(**b) - forward(**a)) / (2 * step)
            matrix.append(np.r_[deriv.real, deriv.imag])
    np.testing.assert_allclose(
        result.response_jacobian(probes), np.array(matrix).T, rtol=5e-6, atol=1e-10
    )


def test_fixed_and_tied_coordinates_retain_covariance_constraints():
    f, z, fr, qr, _ = case()
    result = fit_kids_model(f, z, fr, qr)
    params = result.params.copy()
    params["r0_rho"].vary = False
    params["r1_rho"].expr = "r0_rho"
    fitted = fit_kids_model(f, z, fr, qr, params=params)
    assert not fitted.params["r0_rho"].vary
    assert fitted.params["r1_rho"].expr == "r0_rho"
    np.testing.assert_array_equal(fitted.covariance[1], 0)
    np.testing.assert_array_equal(fitted.covariance[3], 0)
    assert params["r0_rho"].value == result.params["r0_rho"].value


def test_failure_and_coincident_poles_remain_reportable():
    f, z, fr, qr, _ = case()
    failed = fit_kids_model(
        f,
        z,
        fr + np.array([2000, -1000]),
        qr * 0.8,
        initialization="supplied",
        max_nfev=1,
    )
    assert not failed.success
    model = KidsSweepModel([500e6] * 2, [1e4] * 2, [f[0]])
    params = model.make_params()
    for name in ("r0_xi", "r0_rho", "r1_xi", "r1_rho"):
        params[name].vary = False
    signal = model.eval(params, [f[0]])
    result = fit_kids_model([f[0]], signal, [500e6] * 2, [1e4] * 2, params=params)
    assert result.rank < len(result.raw.var_names)
    assert np.isnan(result.covariance).all()


def analysis_case():
    f, z, fr, qr, _ = case()
    ff = np.stack([f[0]] * 3)
    # Reevaluate the second channel on the first channel's measured grid.
    p = fit_kids_model(f, z, fr, qr)
    zz = np.stack([*p.model.eval(p.params, [f[0], f[0]]), np.ones(len(f[0]))])
    source = make_kids_dataset(
        zz.real * u.adu, zz.imag * u.adu, frequency=ff / 1e6 * u.MHz
    ).assign_coords(chan=[41, 7, 99])
    source.I.attrs["units"] = "adu"
    source.Q.attrs["units"] = "adu"
    rows = []
    for c in [41, 7]:
        for j, rid in enumerate(["res_high", "res_low"]):
            rows.append(
                {
                    "group_id": "pair",
                    "chan": c,
                    "resonator_id": rid,
                    "sample_start": 0,
                    "sample_stop": len(f[0]),
                    "fr_initial": fr[j],
                    "r_initial": 1 / (2 * qr[j]),
                }
            )
    rows.append(dict(rows[0], sample_start=10, sample_stop=25))
    plan = xr.Dataset({n: ("row", [r[n] for r in rows]) for n in rows[0]})
    return source, plan


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_compact_analysis_unique_samples_units_and_zarr(tmp_path, family):
    source, plan = analysis_case()
    mask = np.ones(source.I.shape, bool)
    mask[:, ::5] = False
    sigma = np.full((*source.I.shape, 2), 1e-4)
    analysis = KidsModelAnalysis(gain_family=family)
    result = analysis(source.kids.multi_sweep, plan, mask=mask, sigma=sigma)
    view = result.kids.model_fit
    view.validate()
    assert len(view.data_source.children) == 4
    np.testing.assert_array_equal(view.chan_prop.chan, [41, 7, 99])
    assert view.samples.sizes["sample"] == 202
    assert int(view.samples.fit_selected.sum()) == 160
    assert view.fit.sizes["row"] == 5
    assert view.fit.fr.attrs["units"] == "Hz"
    assert view.fit.H0_re.attrs["units"] == "adu"
    assert view.chan_prop.fp.isnull().all()
    assert view.chan_prop.sel(chan=99).fit_status.item() == "not_fitted"
    assert set(view.chan_prop.recovery_mode.values) == {"not_assessed"}
    assert view.components().sizes == {"response": 2, "component": 2}
    s = view.samples
    expected = s.I_model.values + 1j * s.Q_model.values
    np.testing.assert_allclose(
        view.predict(s.group_id, s.chan, s.frequency), expected, atol=2e-11
    )
    path = tmp_path / "fit.zarr"
    result.to_zarr(path, zarr_format=2)
    with xr.open_datatree(path, engine="zarr", chunks={}) as reopened:
        restored = reopened.kids.model_fit
        restored.validate()
        assert restored.fit_analysis == analysis
        xr.testing.assert_identical(
            restored.covariance("pair"), view.covariance("pair")
        )
        np.testing.assert_allclose(
            restored.predict(s.group_id, s.chan, s.frequency), expected, atol=2e-11
        )
        detached = KidsModelView.from_namespace(restored.data_source.copy(deep=True))
        assert detached.fit.sizes["row"] == 5
        models = {
            (chan, rid): restored.pair_model("pair", chan, rid)
            for chan in (41, 7)
            for rid in ("res_high", "res_low")
        }
        assert len(models) == 4  # The repeated segment is not another model.
        with pytest.raises(KeyError):
            restored.pair_model("pair", 99, "res_high")
        xr.testing.assert_identical(restored.chan_prop, view.chan_prop)
        _check_optimizer_record(restored, family)
    # Pair snapshots remain usable after the file-backed source closes.
    for (chan, rid), model in models.items():
        measured = (
            s.I.values[s.chan.values == chan] + 1j * s.Q.values[s.chan.values == chan]
        )
        frequency = s.frequency.values[s.chan.values == chan]
        coordinate = model.derotate(measured, frequency)
        np.testing.assert_allclose(model.rotate(coordinate, frequency), measured)
        np.testing.assert_allclose(
            model.eval(frequency), view.pair_model("pair", chan, rid).eval(frequency)
        )


def _check_optimizer_record(view, family):
    from lmfit import Parameters

    joint = view.joint.load()
    assert "parameters_json" not in joint
    scales = json.loads(joint.scales_json[0].item())
    params = Parameters()
    for name, value, low, high in zip(
        joint.optimizer_name.values,
        joint.optimizer_value.values,
        joint.optimizer_min.values,
        joint.optimizer_max.values,
        strict=True,
    ):
        assert low <= value <= high
        params.add(str(name), value=value, min=low, max=high)
    stored = view.samples.load()
    samples = [
        stored.isel(sample=np.flatnonzero(stored.chan.values == c)) for c in (41, 7)
    ]
    selected = [s.frequency.values[s.fit_selected.values] for s in samples]
    model = KidsSweepModel(
        scales["fr_reference"],
        scales["qr_reference"],
        selected,
        gain_family=family,
        gain_origin=scales["gain_origin"],
        signal_scale=scales["signal_scale"],
    )
    assert tuple(params) == model.names
    predicted = model.eval(params, [s.frequency.values for s in samples])
    for value, sample in zip(predicted, samples, strict=True):
        np.testing.assert_allclose(
            value, sample.I_model.values + 1j * sample.Q_model.values, atol=2e-11
        )


def test_row_permutation_and_conflicting_parameter_copies():
    source, plan = analysis_case()
    analysis = KidsModelAnalysis()
    a = analysis.make_fit(source.kids.multi_sweep, plan)
    b = analysis.make_fit(source.kids.multi_sweep, plan.isel(row=[3, 0, 4, 1, 2]))
    s = a["samples"].to_dataset()
    np.testing.assert_allclose(
        evaluate_calibration(a, s.group_id, s.chan, s.frequency),
        evaluate_calibration(b, s.group_id, s.chan, s.frequency),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        calibration_pair_model(a, "pair", 41, "res_high").eval(s.frequency),
        calibration_pair_model(b, "pair", 41, "res_high").eval(s.frequency),
        atol=2e-9,
    )
    broken = a.copy(deep=True)
    broken["fit"]["fr"].values[0] += 1
    with pytest.raises(ValueError, match="Conflicting shared"):
        validate_calibration(broken)
    invalid = plan.copy(deep=True)
    invalid.fr_initial.values[-1] += 10
    with pytest.raises(ValueError, match="Conflicting initial"):
        analysis.make_fit(source.kids.multi_sweep, invalid)


def test_unresolved_pair_keeps_fit_and_missing_channels():
    source, plan = analysis_case()
    plan = plan.isel(row=np.flatnonzero(plan.chan.values == 41))
    result = KidsModelAnalysis()(
        source.kids.multi_sweep, plan, probe_frequency=[500e6, np.nan, np.nan]
    )
    view = result.kids.model_fit
    assert view.chan_prop.sel(chan=41).recovery_mode.item() == "unresolved"
    assert view.chan_prop.sel(chan=41).probe_status.item() == "bound"
    assert view.fit.fr.notnull().all()
    assert view.covariance("pair").shape == (12, 12)


def test_recompute_destination_and_rejected_obsolete_configuration():
    source, plan = analysis_case()
    analysis = KidsModelAnalysis()
    output = xr.DataTree.from_dict({"scope": xr.Dataset()})
    result = analysis(
        source.kids.multi_sweep, plan, output=output, parent_namespace="scope"
    )
    assert result is output
    altered = source.copy(deep=True)
    altered["I"] *= 1.1
    analysis(altered.kids.multi_sweep, plan, output=output, parent_namespace="scope")
    np.testing.assert_allclose(
        output["scope"].kids.model_fit.samples.I.values[:101], altered.I.values[0]
    )
    for kwargs in ({"backend": "ceres"}, {"backend": "lmfit"}, {"baseline_degree": 2}):
        with pytest.raises(ValueError):
            KidsModelAnalysis(**kwargs)


@pytest.mark.parametrize("sigma", [0.0, np.nan, -1.0])
def test_invalid_uncertainties_are_rejected(sigma):
    f, z, fr, qr, _ = case()
    with pytest.raises(ValueError, match="sigma"):
        fit_kids_model(f, z, fr, qr, sigma=[np.full((len(ff), 2), sigma) for ff in f])


def test_sparse_membership_uses_only_declared_contributions():
    f = [np.linspace(499.8e6, 500.2e6, 101)] * 2
    model = KidsSweepModel(
        [500e6, 500.02e6],
        [10000, 12000],
        f,
        associations=np.array([[1, 1], [0, 1]], bool),
    )
    params = model.make_params()
    z = model.eval(params, f)
    result = fit_kids_model(
        f, z, model.fr_reference, model.qr_reference, associations=model.associations
    )
    assert result.success
    assert np.isnan(result.model.physical(result.params)["c"][1, 0])
    for a, b in zip(result.best_fit, z, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8)


def test_absolute_noise_scaling_and_fixed_input_phase_equivariance():
    f, z, fr, qr, _ = case()
    sigma = [np.full((len(ff), 2), 2e-4) for ff in f]
    a = fit_kids_model(f, z, fr, qr, sigma=sigma)
    b = fit_kids_model(f, z, fr, qr, sigma=[2 * s for s in sigma])
    np.testing.assert_allclose(b.covariance, 4 * a.covariance, rtol=2e-6, atol=1e-15)
    phase = np.exp(1j * np.array([0.7, -0.9]))
    rotated = fit_kids_model(
        f, [v * p for v, p in zip(z, phase, strict=True)], fr, qr, sigma=sigma
    )
    left, right = a.model.physical(a.params), rotated.model.physical(rotated.params)
    np.testing.assert_allclose(right["fr"], left["fr"], rtol=0, atol=0.01)
    np.testing.assert_allclose(right["c"], left["c"], rtol=1e-7)
    np.testing.assert_allclose(right["H0"], left["H0"] * phase, rtol=1e-7)
    np.testing.assert_allclose(
        rotated.covariance[:4, :4], a.covariance[:4, :4], rtol=1e-6, atol=1e-15
    )


def test_multiple_local_groups_do_not_choose_a_channel_calibration():
    source, plan = analysis_case()
    a = plan.isel(row=[0, 1]).copy(deep=True)
    b = a.copy(deep=True)
    b["group_id"] = ("row", ["alternative"] * 2)
    b["resonator_id"] = ("row", ["other_high", "other_low"])
    result = KidsModelAnalysis().make_fit(
        source.kids.multi_sweep, xr.concat([a, b], dim="row")
    )
    channel = result["chan_prop"].to_dataset().sel(chan=41)
    assert channel.group_id.item() == ""
    assert channel.fit_status.item() == "not_selected"
    assert (
        channel.recovery_reason.item() == "multiple_group_responses_require_selection"
    )
    # Both local fits survive even though they use the same acquired samples.
    assert result["joint"].sizes["group"] == 2
    validate_calibration(result)


def test_invalid_rational_seed_records_fallback_and_masks_keep_sample_positions():
    # A pole outside narrow declared frequency bounds triggers a visible fallback.
    f, z, fr, qr, _ = case()
    shifted = fr + np.array([2000.0, -2000.0])
    result = fit_kids_model(f, z, shifted, qr, fr_bounds=(shifted - 100, shifted + 100))
    assert result.initialization["used"] == "supplied"
    assert "outside" in result.initialization["fallback_reason"]
    source, plan = analysis_case()
    source.I.values[0, 15] = np.nan
    result = KidsModelAnalysis().make_fit(source.kids.multi_sweep, plan)
    samples = result["samples"].to_dataset()
    invalid = samples.isel(
        sample=np.flatnonzero(
            (samples.chan.values == 41) & (samples.sample_index.values == 15)
        )
    )
    assert invalid.sizes["sample"] == 1
    assert not invalid.fit_selected.item()
    assert not invalid.data_valid.item()
    assert np.isfinite(invalid.I_model.item())
