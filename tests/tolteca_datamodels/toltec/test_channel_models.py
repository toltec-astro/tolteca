"""Original channel identities and recorded model tables follow block selections."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tests.kids_helpers import apply_step
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import BlockCreate, SweepReduce


def _tune(data_lmt_path):
    return next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_model_parameters_equal_v2(data_lmt_path, tmp_path, suffix):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for cross-environment parity")
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(pattern)
    reference = tmp_path / "models.npz"
    subprocess.run(  # noqa: S603 -- explicit local comparison interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_axis_reference.py")),
            str(path),
            str(reference),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with open_toltec_kids(path) as ds, np.load(reference) as expected:
        for block in BlockCreate()(ensure_datatree(ds)).toltec_kids.blocks:
            if suffix == "vnasweep":
                assert block.model_params is None
                assert block.model_params_table is None
                continue
            names = expected[f"{block.meta.block}_model_names"].tolist()
            actual = block.model_params_table
            assert actual.colnames == ["chan", *names]
            for name in names:
                np.testing.assert_array_equal(
                    actual[name], expected[f"{block.meta.block}_model_{name}"]
                )
                assert actual[name].unit is None
            np.testing.assert_array_equal(actual["chan"], block.chan_axis_table["chan"])
        selected = ds.toltec_kids.select_channels([10, 0, 5])
        for block in BlockCreate()(ensure_datatree(selected)).toltec_kids.blocks:
            np.testing.assert_array_equal(
                block.chan_axis_table["chan"],
                expected[f"{block.meta.block}_chan_id"][[10, 0, 5]],
            )
            if suffix != "vnasweep":
                for name in block.model_params.attrs["parameter_names"]:
                    np.testing.assert_array_equal(
                        block.model_params_table[name],
                        expected[f"{block.meta.block}_model_{name}"][[10, 0, 5]],
                    )


def test_repeated_channel_selection_and_reduction(data_lmt_path):
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        selected = ds.toltec_kids.select_channels(slice(0, 11, 5))
        selected = selected.toltec_kids.select_channels([2, 0])
        selected = selected.toltec_kids.select_channels(np.array([True, False]))
        assert "iqlen" not in ds.coords
        assert selected.iqlen.values.tolist() == [10]
        assert selected.toneFreqLen.values.tolist() == [10]
        np.testing.assert_array_equal(
            selected["Data.Toltec.Is"], ds["Data.Toltec.Is"].isel(iqlen=[10])
        )
        for block in BlockCreate()(ensure_datatree(selected)).toltec_kids.blocks:
            assert block.chan_axis_table["chan"].tolist() == [10]
            assert block.model_params_table["chan"].tolist() == [10]
            xr.testing.assert_equal(
                block.tree.toltec_kids.model_params, block.model_params
            )
            result = apply_step(SweepReduce(), block.tree).toltec_kids.sweep
            assert result.I.iqlen.values.tolist() == [10]
            np.testing.assert_array_equal(
                result.f_chan, block.chan_axis_table["f_chan"].value
            )


def test_named_mask_preserves_channel_ids(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as source:
        ds = source.toltec_kids.select_channels([10, 0, 5])
        mask = ds["Header.Toltec.ToneMask"].copy(data=np.array([[1, 0, 1]]))
        ds = ds.assign({mask.name: mask})
        selected = ds.toltec_kids.select_channels("tone_mask")
        assert BlockCreate()(ensure_datatree(selected)).toltec_kids.blocks[
            0
        ].model_params_table["chan"].tolist() == [
            10,
            5,
        ]
        with pytest.raises(ValueError, match="must be boolean"):
            ds.toltec_kids.select_channels("chan")


def test_model_projection_is_lazy(data_lmt_path, monkeypatch):
    original = NetCDF4ArrayWrapper.__getitem__
    forbidden = {
        "Data.Toltec.Is",
        "Data.Toltec.Qs",
        "Header.Toltec.AdcSnapData",
        "Header.Toltec.ModelParams",
    }

    def guarded(array, key):
        assert array.variable_name not in forbidden, array.variable_name
        return original(array, key)

    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
        selected = ds.toltec_kids.select_channels([10, 0, 5])
        block = BlockCreate()(ensure_datatree(selected)).toltec_kids.blocks[1]
        params = block.model_params
        assert not params.fr.variable._in_memory
        standalone = block.detach(include_raw=False)
        assert not standalone["model_params/fr"].variable._in_memory
        assert "raw" not in standalone.children
        forbidden.remove("Header.Toltec.ModelParams")
        assert standalone.toltec_kids.model_params_table["chan"].tolist() == [10, 0, 5]


@pytest.mark.parametrize("scope", ["acquisition", "collection", "block"])
@pytest.mark.parametrize("include_raw", [True, False])
def test_model_export_roundtrip(data_lmt_path, tmp_path, scope, include_raw):
    destination = tmp_path / "models.zarr"
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        selected = ds.toltec_kids.select_channels([10, 0, 5])
        tree = BlockCreate()(ensure_datatree(selected))
        reference = (
            BlockCreate()(ensure_datatree(selected))
            .toltec_kids.blocks[1]
            .model_params.load()
        )
        if scope == "collection":
            tree = tree["blocks"]
        elif scope == "block":
            tree = tree["blocks/1"]
        detached = tree.toltec_kids.detach(include_raw=include_raw)
        detached.toltec_kids.to_zarr(destination, include_raw=include_raw)
    with xr.open_datatree(destination, engine="zarr", chunks={}) as reopened:
        block = (
            reopened.toltec_kids.block
            if scope == "block"
            else reopened.toltec_kids.blocks[1]
        )
        assert ("model_params" in block.tree.children) is not include_raw
        assert not block.model_params.fr.variable._in_memory
        xr.testing.assert_equal(block.model_params, reference)
        assert block.model_params_table.colnames == [
            "chan",
            *reference.attrs["parameter_names"],
        ]
        assert block.model_params_table["chan"].tolist() == [10, 0, 5]
        assert block.chan_axis_table["chan"].tolist() == [10, 0, 5]
        for node in reopened.subtree:
            if node.children:
                assert not node.dataset.variables


@pytest.mark.parametrize(
    "invalid", ["missing_header", "missing_data", "duplicates", "channel_count"]
)
def test_invalid_model_headers(real_targsweep_file, invalid):
    with open_toltec_kids(real_targsweep_file) as source:
        ds = source.toltec_kids.select_channels([0, 5, 10])
        header, data = "Header.Toltec.ModelParamsHeader", "Header.Toltec.ModelParams"
        if invalid == "missing_header":
            ds = ds.drop_vars(header)
        elif invalid == "missing_data":
            ds = ds.drop_vars(data)
        elif invalid == "duplicates":
            ds[header] = xr.full_like(ds[header], b"fr")
        else:
            ds[data] = ds[data].rename(toneFreqLen="bad_chan").isel(bad_chan=[0])
        with pytest.raises(ValueError, match="ModelParams"):
            _ = BlockCreate()(ensure_datatree(ds)).toltec_kids.blocks[0].model_params
