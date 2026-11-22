"""Quality policy, complete-comb power comparisons and artifact lifecycle."""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.table import QTable
from tollan.config import RuntimeContext
from tollan.pipeline import StepFailureKind
from tollan.plot.plotly import PlotlyViz

from tolteca_kids import ModelFit, ModelFitPlot, ModelFitPlotConfig
from tolteca_kids.core import Kids
from tolteca_kids.model_fit_quality import (
    ModelFitBitMask,
    ModelFitQualityConfig,
    model_fit_quality,
)
from tolteca_kids.model_fit_viz import ModelFitPlotArtifacts
from tolteca_kids.power_setup import (
    PowerSetupConfig,
    compare_drive_fit,
    power_setup_diagnostics,
)
from tolteca_kids.product import _arrays_store
from tolteca_kids.roach_tone import POWER_MODEL
from tolteca_kids.tone_match import ToneMatchConfig, match_tones
from tolteca_kids.tone_prep import TonePrepConfig
from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.models import KidsSweepModel


@pytest.fixture
def fitted():
    f = np.linspace(499.9e6, 500.12e6, 121)
    model = KidsSweepModel([500e6, 500.04e6], [12000, 15000], [f, f])
    z = np.array(model.eval(model.make_params(), [f, f]))
    source = xr.Dataset(
        {
            "I": (("chan", "sweep"), np.r_[z.real, z.real[:1]]),
            "Q": (("chan", "sweep"), np.r_[z.imag, z.imag[:1]]),
        },
        coords={
            "chan": [8, 2, 99],
            "sweep": np.arange(len(f)),
            "frequency": (("chan", "sweep"), np.tile(f, (3, 1))),
        },
    )
    source.frequency.attrs["units"] = "Hz"
    rows = [
        {
            "group_id": "pair",
            "chan": c,
            "resonator_id": str(j),
            "sample_start": 0,
            "sample_stop": len(f),
            "fr_initial": fr,
            "r_initial": 1 / (2 * qr),
        }
        for c in [8, 2]
        for j, (fr, qr) in enumerate(
            zip(model.fr_reference, model.qr_reference, strict=True)
        )
    ]
    plan = xr.Dataset({key: ("row", [row[key] for row in rows]) for key in rows[0]})
    return KidsModelAnalysis()(source.kids.sweep, plan)


def test_quality_preserves_channels_and_separates_selection_from_recovery(fitted):
    view = fitted.kids.model_fit
    q = model_fit_quality(view, ModelFitQualityConfig(condition_max=1.01))
    np.testing.assert_array_equal(q.chan, [8, 2, 99])
    assert q.channel_flag.sel(chan=99).item() == ModelFitBitMask.no_fit
    assert q.group_flag[0].item() & ModelFitBitMask.condition_high
    assert not q.group_flag[0].item() & ModelFitBitMask.not_converged
    assert q.attrs["acceptance"] == "not_assessed"
    assert view.chan_prop.recovery_mode[0].item() == "not_assessed"
    power = xr.Dataset(
        {
            "reference_status": ("chan", ["matched", "matched", "ambiguous"]),
            "power_delta_db": ("chan", [2.0, -2.0, np.nan]),
        },
        coords={"chan": [8, 2, 99]},
    )
    q = model_fit_quality(view, ModelFitQualityConfig(), power)
    assert int(q.channel_flag[0]) & ModelFitBitMask.power_high
    assert int(q.channel_flag[1]) & ModelFitBitMask.power_low
    assert int(q.channel_flag[2]) & ModelFitBitMask.power_reference_unmatched


def test_artifact_export_pair_choices_and_source_independence(fitted, tmp_path):
    view = fitted.kids.model_fit
    original = view.chan_prop.copy(deep=True)
    quality = model_fit_quality(view, ModelFitQualityConfig())
    spec = ModelFitPlotArtifacts()
    figures = spec(view, quality)
    group = figures[PlotlyViz(alias="group_0000")].data
    buttons = group.layout.updatemenus[0].buttons
    assert len(buttons) == 4
    assert {b.label for b in buttons} == {
        f"Channel {c} · resonator {r}" for c in [8, 2] for r in [0, 1]
    }
    for b in buttons:
        assert len(b.args[0]["visible"]) == len(group.data)
        assert sum(b.args[0]["visible"]) == 11
    xr.testing.assert_identical(original, view.chan_prop)
    fitted.close()
    receipts = spec.write(figures, path=tmp_path)
    assert all(receipt.path.exists() for receipt in receipts)
    assert {r.format for r in receipts} == {"html", "js"}
    browser = (tmp_path / "model_fit.html").read_text()
    assert 'id="flag"' in browser and 'id="search"' in browser and "99" in browser
    assert (
        "group_0000_correlation.html"
        in (tmp_path / "model_fit/group_0000.html").read_text()
    )


def test_plot_step_missing_input_and_pipeline_order():
    scope = xr.DataTree()
    ModelFitPlot()(scope)
    assert (
        ModelFitPlot.get_context(scope).failures[0].kind
        == StepFailureKind.missing_input
    )
    kids = Kids(RuntimeContext({"kids": {"model_fit": {}, "model_fit_plot": {}}}))
    assert kids.model_fit is not None and kids.model_fit_plot is not None
    steps = kids.sweep_pipeline.steps
    assert (
        steps.index(kids.model_fit)
        < steps.index(kids.model_fit_plot)
        < steps.index(kids.tone_prep)
    )
    cfg = ModelFitPlotConfig.model_validate(
        {"model_fit_alias": "test", "artifacts": {"groups": ["pair"]}}
    )
    assert ModelFitPlotConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert ModelFit().config.backend == "python"


def test_reference_power_sign_and_invalid_matches():
    f = np.array([500, 501.3, 503.8, 507.2]) * 1e6
    frac = np.array([0.1, 0.2, 0.3, 0.4])
    reference = xr.Dataset(
        {
            name: ("chan", values)
            for name, values in {
                "f_chan": f,
                "power_normalized": frac,
                "p_flag": [0, 0, 1, 0],
                "adrv_best": [12.0] * 4,
                "a_best": [0.2] * 4,
                "Qr_best": [10000.0] * 4,
            }.items()
        },
        coords={"chan": [7, 12, 2, 9]},
        attrs={"power_model": POWER_MODEL},
    )
    result = compare_drive_fit(f + 2000, frac, 10.0, reference, PowerSetupConfig())
    np.testing.assert_allclose(result["power_delta_db"][[0, 1, 3]], 2.0)
    assert np.isnan(result["power_delta_db"][2])
    assert result["reference_status"][2] == "invalid_power_selection"
    # Two current channels cannot silently inherit one reference detector.
    repeated = compare_drive_fit(
        np.insert(f, 1, f[0] + 1000), np.ones(5) / 5, 10, reference, PowerSetupConfig()
    )
    assert np.isnan(repeated["power_delta_db"][:2]).all()


def test_power_config_is_independent_and_full_comb_is_used(tmp_path):
    model = PowerSetupConfig.model_validate({"dac": {"lut_size": 1024}})
    prep = TonePrepConfig.model_validate(
        {
            "dac": {"lut_size": 2048},
            "drive_fit_reference": tmp_path / "proposed-reference",
        }
    )
    assert model.dac.lut_size == 1024 and prep.dac.lut_size == 2048
    assert model.drive_fit_reference is None
    f = np.array([500, 502, 505]) * 1e6
    output = power_setup_diagnostics(
        f,
        f - 501e6,
        [1, 0.5, 0.5],
        [0, 1, 2],
        [True] * 3,
        meta=SimpleNamespace(roach=0, atten_drive=10),
        config=model,
        channels=[8, 2, 99],
    )
    np.testing.assert_allclose(output.power_fraction, np.array([1, 0.25, 0.25]) / 1.5)
    assert "power_detector_dbm" not in output
    np.testing.assert_array_equal(output.chan, [8, 2, 99])


@pytest.mark.parametrize(
    "name", ["fit:Step with space.zarr", "fit%3AStep with space.zarr"]
)
def test_local_product_uri_is_decoded_exactly_once(tmp_path, name):
    # File URIs encode literal colons, spaces and percent signs. A reader must
    # decode exactly once to recover the original filename.
    path = tmp_path / name
    path.mkdir()
    item = SimpleNamespace(
        meta={"format": "zarr", "name": "Scientific data"}, source=object()
    )
    index = SimpleNamespace(
        product=SimpleNamespace(data_items=[item]),
        get_source_uri=lambda _: path.as_uri(),
    )
    assert _arrays_store(index).root == path


def test_power_reference_matches_tone_match_with_global_loading_shift():
    config = PowerSetupConfig()
    fields = ToneMatchConfig.model_fields
    assert config.match == fields["matcher"].get_default(call_default_factory=True)
    assert config.match_shift_max == fields["match_shift_max"].default
    assert config.match_shift_step == fields["match_shift_step"].default
    frequency = np.array([500, 501.3, 503.8, 507.2]) * 1e6
    shifted = frequency + 230e3  # More than four linewidths, but common to the comb.
    reference = xr.Dataset(
        {
            "f_chan": ("chan", frequency),
            "power_normalized": ("chan", np.full(4, 0.25)),
            "p_flag": ("chan", np.zeros(4)),
            "adrv_best": ("chan", np.full(4, 12.0)),
            "a_best": ("chan", np.full(4, 0.2)),
            "Qr_best": ("chan", np.full(4, 10000.0)),
        },
        coords={"chan": [10, 20, 30, 40]},
        attrs={"power_model": POWER_MODEL},
    )
    query = QTable(
        {"idx_chan": [0, 1, 2, 3], "f_chan": shifted * u.Hz, "amp_tone": np.ones(4)},
        meta={"uid_raw_obs": "query"},
    )
    ref = QTable(
        {
            "idx_chan": reference.chan.values,
            "f_chan": frequency * u.Hz,
            "amp_tone": np.ones(4),
        },
    )
    expected = match_tones(
        config.match,
        query,
        query["f_chan"],
        ref,
        ref["f_chan"],
        match_shift_max=config.match_shift_max,
        match_shift_step=config.match_shift_step,
    )
    actual = compare_drive_fit(shifted, np.full(4, 0.25), 12.0, reference, config)
    np.testing.assert_array_equal(
        actual["reference_chan"], expected.query["idx_chan_ref"]
    )
    assert isinstance(expected.matched.shift, u.Quantity)
    np.testing.assert_allclose(
        actual["reference_shift_hz"], expected.matched.shift.to_value(u.Hz)
    )
    np.testing.assert_allclose(
        actual["reference_distance_hz"], expected.query["dist_shifted"].to_value(u.Hz)
    )
    np.testing.assert_allclose(actual["reference_distance_unshifted_hz"], 230e3)
    assert np.all(actual["reference_status"] == "matched")
    assert np.max(abs(actual["reference_distance_hz"])) < 2e3


def test_selected_probe_does_not_inherit_unrelated_edge_fit_failure(fitted):
    """Failed alternative sweep fits remain visible without poisoning a route."""
    view = fitted.kids.model_fit
    fit = view.fit.copy(deep=True)
    fit["group_id"] = xr.full_like(fit.group_id, "edge", dtype=object)
    fit["fit_status"] = xr.full_like(fit.fit_status, "optimizer_failed", dtype=object)
    samples = view.samples.copy(deep=True)
    samples["group_id"] = xr.full_like(samples.group_id, "edge", dtype=object)
    joint = view.joint.copy(deep=True)
    joint["group_id"] = xr.full_like(joint.group_id, "edge", dtype=object)
    joint["fit_status"] = xr.full_like(
        joint.fit_status, "optimizer_failed", dtype=object
    )
    props = view.chan_prop.copy(deep=True)
    combined = SimpleNamespace(
        fit=xr.concat([view.fit, fit], dim="row", data_vars="minimal"),
        samples=xr.concat([view.samples, samples], dim="sample", data_vars="minimal"),
        joint=xr.concat([view.joint, joint], dim="group", data_vars="minimal"),
        chan_prop=props,
    )
    quality = model_fit_quality(combined, ModelFitQualityConfig())
    assert int(quality.group_flag[-1]) & ModelFitBitMask.not_converged
    assert not int(quality.channel_flag.sel(chan=8)) & ModelFitBitMask.not_converged
    assert (
        not int(quality.channel_flag.sel(chan=8)) & ModelFitBitMask.selection_required
    )
    props["group_id"].loc[{"chan": 8}] = ""
    props["recovery_reason"] = props.recovery_reason.astype(object)
    props["recovery_reason"].loc[{"chan": 8}] = (
        "multiple_group_responses_require_selection"
    )
    quality = model_fit_quality(combined, ModelFitQualityConfig())
    assert int(quality.channel_flag.sel(chan=8)) & ModelFitBitMask.not_converged
    assert int(quality.channel_flag.sel(chan=8)) & ModelFitBitMask.selection_required
