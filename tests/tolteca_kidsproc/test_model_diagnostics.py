"""Native constraints, repeat-fit comparisons and response coverage."""

import numpy as np
import pytest
import xarray as xr
from lmfit import Parameters

from tolteca_kids.model_fit_quality import (
    ModelFitBitMask,
    ModelFitQualityConfig,
    model_fit_quality,
)
from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.fitting import fit_kids_model
from tolteca_kidsproc.model_diagnostics import compare_kids_model_fits
from tolteca_kidsproc.models import KidsSweepModel, resonance_feature_frequency


@pytest.fixture
def constrained_fit():
    f = np.linspace(499.91e6, 500.09e6, 121)
    model = KidsSweepModel([500e6], [12000], [f], nonlinear=True)
    params = model.make_params()
    params["r0_a_nl"].value = 0.4
    params["c0_h1_re"].set(expr="0.01*c0_h0_re")
    z = model.eval(params, [f])[0]
    source = xr.Dataset(
        {
            "I": (("chan", "sweep"), z.real[None]),
            "Q": (("chan", "sweep"), z.imag[None]),
        },
        coords={"chan": [4], "frequency": (("chan", "sweep"), f[None])},
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
    mask = np.arange(len(f))[None] % 4 != 1
    base = fit_kids_model([f[mask[0]]], [z[mask[0]]], [500e6], [12000], nonlinear=True)
    p = base.params.copy()
    p["r0_xi"].set(value=0, vary=False)
    p["r0_a_nl"].value = 0.1
    p["c0_h1_re"].set(expr="0.01*c0_h0_re")
    analysis = KidsModelAnalysis(nonlinear=True)
    tree = analysis(source.kids.sweep, plan, mask=mask, initial_params={"g": p})
    return tree, source, plan, mask


def test_native_constraints_survive_storage_and_prediction(constrained_fit, tmp_path):
    tree, source, plan, mask = constrained_fit
    path = tmp_path / "fit.zarr"
    tree.to_zarr(path, zarr_format=2)
    with xr.open_datatree(path, engine="zarr") as reopened:
        view = reopened.kids.model_fit
        view.validate()
        assert float(view.fit.fr[0]) == 500e6
        assert float(view.fit.a_nl[0]) == pytest.approx(0.4, abs=1e-6)
        names = view.joint.optimizer_name.values.tolist()
        assert not bool(view.joint.optimizer_vary[names.index("r0_xi")])
        assert (
            view.joint.optimizer_expr[names.index("c0_h1_re")].item() == "0.01*c0_h0_re"
        )
        np.testing.assert_allclose(
            view.predict("g", 4, source.frequency[0]),
            source.I[0] + 1j * source.Q[0],
            atol=1e-8,
        )
    with pytest.raises(ValueError, match="unknown group"):
        KidsModelAnalysis()(
            source.kids.sweep, plan, mask=mask, initial_params={"unknown": Parameters()}
        )


def test_comparison_convergence_ranges_and_unchanged_identity(constrained_fit):
    tree, *_ = constrained_fit
    bad = tree.copy(deep=True)
    bad.kids.model_fit.joint.fit_status.values[:] = "optimizer_failed"
    good = tree.kids.model_fit
    comparison = compare_kids_model_fits(
        {"one": good, "two": good, "failed": bad.kids.model_fit}
    )
    assert comparison.near_best_count.item() == 2
    assert comparison.best_candidate.item() == "one"
    assert not comparison.near_best.sel(candidate="failed").any()
    np.testing.assert_array_equal(comparison.fr_min, comparison.fr_max)
    assert np.isfinite(comparison.excluded_rms).all()
    assert comparison.attrs["identity_action"] == "none"
    np.testing.assert_array_equal(good.fit.resonator_id, ["r"])
    failed = compare_kids_model_fits({"failed": bad.kids.model_fit})
    assert failed.near_best_count.item() == 0
    assert failed.best_candidate.item() == ""
    assert np.isnan(failed.fr_min).all()


@pytest.mark.parametrize("field", ["I", "fit_selected", "sigma_I"])
def test_comparison_rejects_different_data_selection_or_weights(constrained_fit, field):
    tree, *_ = constrained_fit
    other = tree.copy(deep=True)
    values = other.kids.model_fit.samples[field].values
    values[0] = not values[0] if field == "fit_selected" else 3.0
    with pytest.raises(ValueError, match="identical measurements"):
        compare_kids_model_fits(
            {"one": tree.kids.model_fit, "other": other.kids.model_fit}
        )


def test_nonlinear_feature_and_distinct_zero_drive_coverage(constrained_fit):
    tree, *_ = constrained_fit
    view = tree.kids.model_fit
    tree["tolteca_kidsproc.analysis.model_fit/fit"].attrs["nonlinear_branch"] = (
        "increasing"
    )
    view.fit.a_nl.values[:] = 2.0
    feature, kind = resonance_feature_frequency(
        500e6, 12000, 2.0, nonlinear_branch="increasing"
    )
    assert kind.item() == "upward_jump"
    # The jump obeys the turning-point equation and lies below zero-drive fr.
    y0 = 12000 * (feature / 500e6 - 1)
    roots = np.roots([4, -4 * y0, 1, -y0 - 2])
    folded = np.sort(roots.real)[0]
    assert 1 + 16 * folded / (1 + 4 * folded**2) ** 2 == pytest.approx(0, abs=2e-6)
    view.fit.f_min.values[:] = feature - 5000
    view.fit.f_max.values[:] = feature + 5000
    # Use the exact loaded Q used for the landmark in this policy fixture.
    view.fit.Qr.values[:] = 12000
    q = model_fit_quality(view, ModelFitQualityConfig())
    flag = q.group_flag.item()
    assert flag & ModelFitBitMask.zero_drive_outside_window
    assert not flag & ModelFitBitMask.resonance_outside_window
    assert not flag & ModelFitBitMask.nonlinear_limit
    names = view.joint.optimizer_name.values.tolist()
    index = names.index("r0_a_nl")
    view.joint.optimizer_value.values[index] = 2
    view.joint.optimizer_max.values[index] = 2
    assert (
        model_fit_quality(view, ModelFitQualityConfig()).group_flag.item()
        & ModelFitBitMask.nonlinear_limit
    )
    view.joint.optimizer_vary.values[index] = False
    assert (
        not model_fit_quality(view, ModelFitQualityConfig()).group_flag.item()
        & ModelFitBitMask.nonlinear_limit
    )
    np.testing.assert_allclose(
        resonance_feature_frequency(500e6, 12000, 0.5)[0], 500e6 * (1 - 0.5 / 12000)
    )
    np.testing.assert_allclose(
        resonance_feature_frequency(500e6, 12000, 2, nonlinear_branch="decreasing")[0],
        500e6 * (1 - 2 / 12000),
    )
