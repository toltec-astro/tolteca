"""Python step integration preserves measured channels and grouped calibration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from astropy.table import QTable
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext

from tolteca_kids.block_create import BlockCreate
from tolteca_kids.core import Kids
from tolteca_kids.model_calibration import ModelFitFlag, calibration_report
from tolteca_kids.model_fit import ModelFit, ModelFitConfig, ModelFitContext
from tolteca_kids.model_fit_artifacts import KidscppFitFiles, ModelFitTable
from tolteca_kids.model_fit_plot import ModelFitPlot, ModelFitPlotConfig
from tolteca_kids.model_joint import joint_models
from tolteca_kids.product import open_sweep_product, write_sweep_product
from tolteca_kids.sweep import ReducedSweepView
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kidsproc.analysis import KidsModelAnalysis


@pytest.mark.parametrize("nonlinear", [False, True])
def test_undersampled_group_is_retained_but_not_sent_to_optimizer(nonlinear):
    from tolteca_kids._model_fit import screen_plan

    frequency = np.array([500e6, 501e6])[:, None] + np.arange(20) * 1000
    plan = xr.Dataset(
        {
            "group_id": ("row", ["short", "measured"]),
            "resonator_id": ("row", ["r0", "r1"]),
            "chan": ("row", [8, 9]),
            "sample_start": ("row", [0, 0]),
            "sample_stop": ("row", [3, 20]),
            "fr_initial": ("row", [500e6, 501e6]),
            "r_initial": ("row", [1e-5, 1e-5]),
        }
    )
    saved = plan.copy(deep=True)
    result = screen_plan(
        plan,
        frequency,
        np.ones_like(frequency, complex),
        [8, 9],
        mask=np.ones_like(frequency, bool),
        analysis=KidsModelAnalysis(nonlinear=nonlinear),
    )
    np.testing.assert_array_equal(result.fit_eligible, [False, True])
    np.testing.assert_array_equal(
        result.fit_residual_dof, [-2 - nonlinear, 32 - nonlinear]
    )
    assert result.fit_plan_status.values[0] == "insufficient_residual_dof"
    xr.testing.assert_identical(plan, saved)


def test_undersampled_group_does_not_abort_neighboring_fit(target_sweep_path):
    with xr.open_dataset(target_sweep_path) as raw:
        source = raw.toltec_kids.select_channels([8, 9]).load()
    tree = BlockCreate()(ensure_datatree(source))
    SweepReduce()(tree)
    block = tree.toltec_kids.blocks[0]
    frequency = block.sweep.frequency.values
    signal = block.sweep.I.values + 1j * block.sweep.Q.values
    plan = xr.Dataset(
        {
            "group_id": ("row", ["short", "measured"]),
            "resonator_id": ("row", ["r0", "r1"]),
            "chan": ("row", [8, 9]),
            "sample_start": ("row", [0, 0]),
            "sample_stop": ("row", [3, signal.shape[1]]),
            "fr_initial": (
                "row",
                [frequency[0, 1], frequency[1, np.argmin(abs(signal[1]))]],
            ),
            "r_initial": ("row", [1 / 20000, 1 / 20000]),
        }
    )
    step = ModelFit(ModelFitConfig(analysis=KidsModelAnalysis(max_nfev=100)), plan=plan)
    step(block.tree)
    assert ModelFit.get_context(block.tree).completed
    view = ModelFit.view(block.tree)
    assert set(view.fit.group_id.values) == {"measured"}
    assert block.tree["model_fit/plan_quality"].fit_plan_status.values.tolist() == [
        "insufficient_residual_dof",
        "eligible",
    ]
    xr.testing.assert_identical(
        block.tree["model_fit/plan"].to_dataset(inherit=False), plan
    )
    report = calibration_report(view, block)
    assert report["flag"][0] & ModelFitFlag.fit_no_model
    assert report["flag"][0] & ModelFitFlag.fit_unusable
    assert np.isnan(report["fr"][0])
    assert np.isfinite(report["fr"][1])


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "kidscpp"},
        {"backend": "python", "executable": "/nonexistent/kidscpp"},
        {"backend": "python", "retain_processed": True},
        {"backend": "kidscpp", "executable": "/nonexistent/kidscpp", "analysis": {}},
    ],
)
def test_backend_selection_rejects_mixed_contracts(config):
    with pytest.raises(ValueError):
        ModelFitConfig(**config)


def test_analysis_configuration_survives_pipeline_serialization():
    config = ModelFitConfig(analysis=KidsModelAnalysis(qr_bounds=[500.0, 100000.0]))
    context = ModelFit.context_cls(step_cls=ModelFit, config=config)
    assert isinstance(context, ModelFitContext)
    reopened = ModelFit.load_context(ModelFit.dump_context(context))
    assert reopened.config == config
    assert ModelFitConfig.model_validate_json(config.model_dump_json()) == config
    kids = Kids(RuntimeContext({"kids": {"model_fit": config}}))
    steps = kids.sweep_pipeline.steps
    model_fit = kids.model_fit
    assert model_fit is not None
    assert steps.index(kids.kids_find) < steps.index(model_fit)
    assert steps.index(model_fit) < steps.index(kids.tone_prep)


def test_plan_uses_group_membership_and_coverage_not_nearest_probe():
    # Each of two channels sees both poles. Nonmonotonic acquisition order
    # produces two segments for one association without adding another pole.
    offsets = np.r_[np.linspace(-40000, 40000, 81), [200000.0, -20000.0, 0.0, 20000.0]]
    chans = [31, 7, 90]
    ds = xr.Dataset(
        {
            "I": (("chan", "sweep"), np.ones((3, len(offsets)))),
            "Q": (("chan", "sweep"), np.zeros((3, len(offsets)))),
            "f_chan": ("chan", [500e6, 500e6 + 10000, 501e6]),
        },
        coords={"chan": chans, "sweep": offsets},
    )
    finding = SimpleNamespace(
        detected=QTable(
            {
                "idx": [17, 9],
                "group": [42, 42],
                "f": np.array([500e6 - 10000, 500e6 + 10000]) * u.Hz,
                "Qr": [20000.0, 25000.0],
                # Matching evidence must not decide which channel sees a pole.
                "idx_chan": [31, 31],
            }
        )
    )
    mask = np.ones((3, len(offsets)), dtype=bool)
    plan = ModelFit.make_plan(ReducedSweepView(ds), finding, mask=mask)
    associations = set(zip(plan.chan.values, plan.resonator_id.values, strict=True))
    assert associations == {(31, "17"), (31, "9"), (7, "17"), (7, "9")}
    assert set(plan.group_id.values) == {"42"}
    assert plan.sizes["row"] > len(associations)
    assert all(plan.sample_stop > plan.sample_start)
    frequency = ReducedSweepView(ds).frequency
    assert frequency is not None
    for row in plan.to_dataframe().itertuples():
        f = frequency.sel(chan=row.chan).values
        selected = f[row.sample_start : row.sample_stop]
        qr = 20000 if row.resonator_id == "17" else 25000
        assert np.all(np.abs(selected - row.fr_initial) <= 2 * row.fr_initial / qr)
    mask[1] = False
    selected = ModelFit.make_plan(ReducedSweepView(ds), finding, mask=mask)
    assert set(selected.chan.values) == {31}
    empty = ModelFit.make_plan(ReducedSweepView(ds), finding, mask=np.zeros_like(mask))
    assert empty.sizes["row"] == 0
    assert empty.attrs["status"] == "no_usable_samples"


@pytest.mark.parametrize("family", ["affine", "delay"])
@pytest.mark.parametrize(
    "scope", ["acquisition", "collection", "block", "attached_block"]
)
def test_step_matches_numerical_analysis_and_published_result(  # noqa: PLR0915 -- Numerical parity and publication round trip.
    target_sweep_path,
    tmp_path,
    family,
    scope,
):
    with xr.open_dataset(target_sweep_path) as raw:
        accumlen = raw["Header.Toltec.AccumLen"].item()
        source = raw.toltec_kids.select_channels([8, 9, 10]).load()
    tree = BlockCreate()(ensure_datatree(source))
    if scope == "collection":
        tree = tree.toltec_kids.blocks.detach()
    elif scope == "block":
        tree = tree.toltec_kids.blocks[0].detach()
    elif scope == "attached_block":
        tree = tree["blocks/0"]
    SweepReduce()(tree)
    block = (
        tree.toltec_kids.block
        if scope in {"block", "attached_block"}
        else tree.toltec_kids.blocks[0]
    )
    sweep = block.sweep
    frequency = sweep.frequency.values
    signal = sweep.I.values + 1j * sweep.Q.values
    chan = sweep.I.coords[sweep.I.dims[0]].values[0]
    seed = frequency[0, np.argmin(abs(signal[0]))]
    plan = xr.Dataset(
        {
            "group_id": ("row", ["g"]),
            "chan": ("row", [chan]),
            "resonator_id": ("row", ["r"]),
            "sample_start": ("row", [0]),
            "sample_stop": ("row", [signal.shape[1]]),
            "fr_initial": ("row", [seed]),
            "r_initial": ("row", [1 / 20000]),
        }
    )
    before = plan.copy(deep=True)
    analysis = KidsModelAnalysis(gain_family=family, max_nfev=100)
    config = ModelFitConfig(backend="python", analysis=analysis, alias=family)
    step = ModelFit(config, plan=plan)
    # The fit owns a plan snapshot, not the caller's mutable table.
    plan.fr_initial.values[:] += 10
    step(block.tree)
    assert ModelFit.get_context(block.tree, alias=family).completed
    v = ModelFit.view(tree, block=0, alias=family)
    v.validate()
    saved_history = block.tree[step.result_path(alias=family)].attrs["sweep_history"]
    assert saved_history["status"] == "recorded"
    assert saved_history["direction"] == "increasing"
    with pytest.raises(KeyError):
        ModelFit.peak_links(tree, block=0, alias=family)
    probe = block.probe_axis
    m = probe.mapper
    counts = (
        m.get_arr(probe.data_source, m.schema.sample_end).values
        - m.get_arr(probe.data_source, m.schema.sample_start).values
    )
    sigma = (
        np.stack((sweep.unc_I.values, sweep.unc_Q.values), axis=-1)
        / np.sqrt(counts)[None, :, None]
    )
    expected = analysis(
        sweep.data_source.assign_coords(frequency=sweep.frequency).kids.sweep,
        before,
        sigma=sigma,
        probe_frequency=sweep.f_chan.values,
    ).kids.model_fit
    xr.testing.assert_identical(v.samples, expected.samples)
    xr.testing.assert_identical(v.fit, expected.fit)
    xr.testing.assert_identical(v.chan_prop, expected.chan_prop)
    assert list(v.chan_prop.chan.values) == [8, 9, 10]
    assert list(v.chan_prop.recovery_reason.values[1:]) == ["no_fit", "no_fit"]
    saved_plan = block.tree[f"model_fit_{family}/plan"].to_dataset(inherit=False)
    xr.testing.assert_identical(saved_plan, before)
    report = calibration_report(v, block)
    quality = block.tree[f"model_fit_{family}/quality"].dataset
    complete = joint_models(v, quality, include_single=True)
    assert report.meta["joint_models"]["groups"] == complete["groups"]
    assert len(complete["groups"]) == 1
    assert complete["groups"][0]["channels"] == [int(chan)]
    assert complete["groups"][0]["resonator_ids"] == ["r"]
    for flag in report["flag"][1:]:
        assert flag & ModelFitFlag.fit_no_model
        assert flag & ModelFitFlag.fit_unusable
    # A nonlinear singlet exports a fixed drive coefficient, never a zero-a inverse.
    nonlinear = block.tree[f"model_fit_{family}"].copy(deep=True)
    nonlinear.kids.model_fit.fit["a_nl"].values[:] = 0.1
    exported = calibration_report(nonlinear.kids.model_fit, block)
    assert exported.meta["model_schema"]["version"] == 2
    assert not exported["flag"][0] & ModelFitFlag.fit_unusable
    assert exported["b"][0] == pytest.approx(0.2 * exported["r"][0] ** 3)
    assert (
        joint_models(nonlinear.kids.model_fit, quality, include_single=True)["groups"]
        != []
    )
    # An extended directional fit remains available for diagnostics, but does
    # retains the fitted drive coefficient and its declared branch.
    nonlinear.kids.model_fit.fit["a_nl"].values[:] = 0.8
    nonlinear["tolteca_kidsproc.analysis.model_fit/fit"].attrs["nonlinear_branch"] = (
        "increasing"
    )
    unsupported = calibration_report(nonlinear.kids.model_fit, block)
    assert not unsupported["flag"][0] & ModelFitFlag.fit_unusable
    assert unsupported["b"][0] == pytest.approx(1.6 * unsupported["r"][0] ** 3)
    assert np.all(nonlinear.kids.model_fit.fit.a_nl.values == 0.8)
    original = v.samples.copy(deep=True)
    invalid = before.copy(deep=True)
    invalid.chan.values[:] = 99999
    with pytest.raises(ValueError, match="channel absent"):
        ModelFit(config, plan=invalid)(block.tree)
    xr.testing.assert_identical(
        ModelFit.view(block.tree, alias=family).samples, original
    )
    # Refit before publication: failed execution records are not successful fits.
    step(block.tree)
    ModelFitPlot(ModelFitPlotConfig(model_fit_alias=family))(block.tree)
    relative_scope = {"acquisition": "blocks/0", "collection": "0"}.get(scope, ".")
    index = write_sweep_product(tree, tmp_path)
    from tolteca_datamodels.data_prod import DataProdIndex
    from tolteca_datamodels.raw_obs_context import ArtifactBinding

    published = DataProdIndex.read(index)
    context = ModelFit.get_context(block.tree, alias=family)
    assert ModelFitTable in context.artifacts
    assert KidscppFitFiles not in context.artifacts
    [model_item] = [
        i for i in published.product.data_items if i.meta.get("name") == "model"
    ]
    model_path = index / model_item.source.path
    assert model_path.name == "kids_model.ecsv"
    assert model_path.read_bytes() == context.artifacts[ModelFitTable].data
    assert not (model_path.parent / "kidscpp").exists()
    binding = ArtifactBinding(
        product=published.product.reference,
        producer="tolteca_kids.model_fit:ModelFit",
        name=f"model_fit_{family}",
        interface=block.meta.interface,
        scope=relative_scope,
    )
    item = binding.resolve(published)
    assert item.source.path == (
        Path("array.zarr")
        / block.meta.interface
        / relative_scope
        / f"model_fit_{family}"
    )
    # Exact component evidence references the existing store, without a second
    # copy of fit arrays or the isolated-channel calibration projection.
    with xr.open_datatree(
        index / item.source.path, engine="zarr", consolidated=False
    ) as component:
        xr.testing.assert_identical(component.kids.model_fit.fit, v.fit)
    with open_sweep_product(index) as reopened:
        view = ModelFit.view(reopened, block=0, alias=family)
        view.validate()
        reopened_block = (
            reopened.toltec_kids.block
            if scope in {"block", "attached_block"}
            else reopened.toltec_kids.blocks[0]
        )
        assert reopened_block.sweep.data_source.attrs["iq_accumlen"] == accumlen
        restored_report = calibration_report(view, reopened_block)
        assert restored_report.meta == report.meta
        for name in report.colnames:
            np.testing.assert_array_equal(restored_report[name], report[name])
        pair = view.pair_model("g", chan, "r")
        model = view.predict("g", chan, frequency[0])
        np.testing.assert_allclose(
            pair.rotate(pair.derotate(model, frequency[0]), frequency[0]), model
        )
        xr.testing.assert_allclose(view.samples.compute(), original)
        assert view.covariance("g").ndim == 2
        assert "channel_flag" in reopened_block.tree[f"model_fit_{family}/quality"]
    assert list(tmp_path.rglob("model_fit.html"))
    assert np.isfinite(pair.eval(frequency[0])).all()
