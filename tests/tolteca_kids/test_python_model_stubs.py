"""A Python joint fit becomes an acquisition-owned model stub without refitting."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table
from tollan.accessor.xarray import ensure_datatree

from tolteca_datamodels.data_prod import DataProdIndex, DataProdStore
from tolteca_datamodels.raw_obs_context import open_acquisition, read_run_inputs
from tolteca_datamodels.raw_obs_context.stub import MODEL_FIELDS
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.block_create import BlockCreate
from tolteca_kids.model_fit import ModelFit, ModelFitConfig
from tolteca_kids.product import write_sweep_product
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kidsproc.analysis import KidsModelAnalysis


@pytest.mark.parametrize("family", ["affine", "delay"])
def test_python_joint_fit_to_published_stub(
    data_lmt_path, tmp_path, monkeypatch, family
):
    tune = data_lmt_path / (
        "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    source = data_lmt_path / (
        "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    import tolteca_kids.model_fit as model_fit_module

    def no_native_fit(*args, **kwargs):
        pytest.fail("Python model preparation invoked a native subprocess")

    monkeypatch.setattr(model_fit_module.subprocess, "run", no_native_fit)
    store = DataProdStore(tmp_path, namespace="tune-comparison")
    with open_toltec_kids(tune) as raw:
        tree = BlockCreate()(ensure_datatree(raw))
        SweepReduce()(tree)
        assert len(tree.toltec_kids.blocks) == 2
        for block in tree.toltec_kids.blocks:
            ModelFit(
                ModelFitConfig(
                    backend="python",
                    analysis=KidsModelAnalysis(gain_family=family, max_nfev=200),
                ),
                plan=_pair_plan(block.sweep),
            )(block.tree)
        model = DataProdIndex.read(write_sweep_product(tree, store))
    reports = {
        item.meta["scope"]: Table.read(
            model.get_source_uri(item.source), format="ascii.ecsv"
        )
        for item in model.product.data_items
        if item.meta.get("name") == "model"
    }
    assert set(reports) == {"blocks/0", "blocks/1"}
    for report in reports.values():
        [group] = report.meta["joint_models"]["groups"]
        assert group["channels"] == [494, 495]
        assert group["resonator_ids"] == ["a", "b"]
        assert group["gain_family"] == family

    # Stub preparation only needs headers. Reading detector samples or fitting
    # again here would violate the preparation contract.
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    getitem = NetCDF4ArrayWrapper.__getitem__

    def headers_only(array, key):
        assert array.variable_name not in {"Data.Toltec.Is", "Data.Toltec.Qs"}
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", headers_only)
    example = (
        Path(__file__).parents[2] / "examples/tune_model_comparison/prepare_stubs.py"
    )
    prepare = runpy.run_path(str(example))["prepare_stubs"]
    result = prepare(
        tune,
        source,
        model_index=model.base_path,
        output_path=tmp_path,
        namespace="tune-comparison",
    )
    index = DataProdIndex.read(result["index"])
    preparation = store.read_run(result["preparation_run_uid"])
    [original] = read_run_inputs(preparation)
    assert preparation.status == "completed"
    assert original.stubs == ()
    [context] = read_run_inputs(store.read_run(result["run_uid"]))
    assert context.model_dump(mode="json") == result["context"]
    assert index.product.meta.data_prod_type == "dp_raw_obs_proc"
    assert index.product.assocs[0].uid == "tcs-145609-0-2"
    assert index.provenance["raw_obs_context"] == original.model_dump(mode="json")
    assert index.run.uid == preparation.uid
    [application] = context.stubs
    application.resolve(index, raw_obs=context.raw_obs)
    path = application.artifact.resolve_path(index)
    with xr.open_dataset(path) as stub:
        assert "Data.Toltec.Is" not in stub
        assert "Data.Toltec.Qs" not in stub
        assert (
            json.loads(stub[MODEL_FIELDS[0]].attrs["joint_models_0"])
            == (reports["blocks/1"].meta["joint_models"])
        )
    with open_acquisition(source, stubs=[(path, "override")]) as effective:
        assert (
            json.loads(effective[MODEL_FIELDS[0]].attrs["model_schema_0"])
            == (reports["blocks/1"].meta["model_schema"])
        )
    assert result["data_item"]["stubs"] == [{"filepath": str(path), "mode": "override"}]


def _pair_plan(sweep):
    # Explicit local group for the measured close pair. Labels identify
    # fit components within this block, not detectors across Tune blocks.
    channels = [494, 495]
    centers = sweep.f_chan.sel(iqlen=channels).values
    return xr.Dataset(
        {
            "group_id": ("row", ["pair"] * 4),
            "chan": ("row", [494, 494, 495, 495]),
            "resonator_id": ("row", ["a", "b", "a", "b"]),
            "sample_start": ("row", [0] * 4),
            "sample_stop": ("row", [sweep.I.shape[1]] * 4),
            "fr_initial": ("row", np.tile(centers, 2)),
            "r_initial": ("row", [1 / 40000] * 4),
        }
    )
