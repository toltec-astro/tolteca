"""Recorded probe intervals, not sorted fit windows, select sweep history."""

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree

from tolteca_datamodels.toltec.kids import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids.block_create import BlockCreate
from tolteca_kids.model_fit import ModelFit, ModelFitConfig
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kidsproc.analysis import KidsModelAnalysis


@pytest.fixture
def recorded_block(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as ds:
        raw = ds.toltec_kids.select_channels([8]).load()
    tree = BlockCreate()(ensure_datatree(raw))
    SweepReduce()(tree)
    return tree.toltec_kids.blocks[0].detach(include_raw=False)


def test_history_survives_without_raw_samples(recorded_block, tmp_path):
    tree = recorded_block
    requested = KidsModelAnalysis(nonlinear=True, a_nl_max=10)
    resolved, history = ModelFit.bind_analysis(tree.toltec_kids.block, requested)
    assert resolved.nonlinear_branch == "increasing"
    assert requested.nonlinear_branch == "unique"
    assert history["direction"] == "increasing"
    assert history["f_lo_last_hz"] > history["f_lo_first_hz"]
    assert history["n_samples"] > history["n_steps"] > 1
    assert "raw" not in tree.children
    path = tmp_path / "sweep.zarr"
    tree.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    with xr.open_datatree(path, engine="zarr") as reopened:
        assert ModelFit.bind_analysis(reopened.toltec_kids.block, requested) == (
            resolved,
            history,
        )
    for analysis in (KidsModelAnalysis(), KidsModelAnalysis(nonlinear=True)):
        bound, _ = ModelFit.bind_analysis(tree.toltec_kids.block, analysis)
        assert bound is analysis


def test_explicit_branch_must_match_recorded_order(recorded_block):
    with pytest.raises(ValueError, match="contradicts recorded"):
        ModelFit.bind_analysis(
            recorded_block.toltec_kids.block,
            KidsModelAnalysis(
                nonlinear=True, nonlinear_branch="decreasing", a_nl_max=10
            ),
        )


@pytest.mark.parametrize("change", ["intervals", "lo", "reduced", "processed"])
def test_sorted_frequencies_cannot_replace_history(recorded_block, change):
    tree = recorded_block
    block = tree.toltec_kids.block
    probe = block.probe_axis
    mapper = probe.mapper
    if change == "intervals":
        key = mapper.get_name(mapper.schema.sample_start)
        tree["probe"][key].values[:] = tree["probe"][key].values[::-1]
    elif change == "lo":
        key = mapper.get_name(mapper.schema.f_lo)
        tree["probe"][key].values[1] = tree["probe"][key].values[0] - 1
    elif change == "reduced":
        lo = tree["sweep"]["f_lo"].copy(deep=True)
        lo.values[:] = lo.values[::-1]
        tree["sweep"]["f_lo"] = lo
    else:
        tree.attrs["data_kind"] = ToltecDataKind.reduced_targsweep.value
    with pytest.raises(ValueError, match="requires acquisition history"):
        ModelFit.bind_analysis(block, KidsModelAnalysis(nonlinear=True, a_nl_max=10))
    # Linear fitting still accepts nonmonotonic samples; provenance is explicit.
    requested = KidsModelAnalysis()
    resolved, history = ModelFit.bind_analysis(block, requested)
    assert resolved is requested
    assert history["status"] == "unavailable"


def test_decreasing_history_is_resolved_without_sorting(recorded_block):
    block = recorded_block.toltec_kids.block
    mapper = block.probe_axis.mapper
    key = mapper.get_name(mapper.schema.f_lo)
    lo = recorded_block["probe"][key].values
    lo[:] = lo[::-1]
    recorded_block["sweep"]["f_lo"].values[:] = lo
    resolved, history = ModelFit.bind_analysis(
        block, KidsModelAnalysis(nonlinear=True, a_nl_max=10)
    )
    assert resolved.nonlinear_branch == "decreasing"
    assert history["direction"] == "decreasing"
    assert np.diff(lo).max() < 0


def test_directional_step_publishes_native_history(recorded_block, tmp_path):
    block = recorded_block.toltec_kids.block
    sweep = block.sweep
    signal = sweep.I.values + 1j * sweep.Q.values
    plan = xr.Dataset(
        {
            "group_id": ("row", ["g"]),
            "chan": ("row", [8]),
            "resonator_id": ("row", ["r"]),
            "sample_start": ("row", [0]),
            "sample_stop": ("row", [signal.shape[-1]]),
            "fr_initial": (
                "row",
                [sweep.frequency.values[0, np.argmin(abs(signal[0]))]],
            ),
            "r_initial": ("row", [1 / 20000]),
        }
    )
    ModelFit(
        ModelFitConfig(
            analysis=KidsModelAnalysis(nonlinear=True, a_nl_max=10, max_nfev=10),
        ),
        plan=plan,
    )(recorded_block)
    result = recorded_block["model_fit"]
    assert result.kids.model_fit.fit_analysis.nonlinear_branch == "increasing"
    assert result.attrs["sweep_history"]["direction"] == "increasing"
    path = tmp_path / "model.zarr"
    result.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    with xr.open_datatree(path, engine="zarr") as native:
        view = native.kids.model_fit
        view.validate()
        samples = view.samples
        prediction = view.predict(
            samples.group_id.values, samples.chan.values, samples.frequency.values
        )
        np.testing.assert_allclose(
            prediction,
            samples.I_model.values + 1j * samples.Q_model.values,
            atol=1e-10,
            rtol=1e-10,
        )
        assert native.attrs["sweep_history"] == result.attrs["sweep_history"]
