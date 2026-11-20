"""Configured Python timestream reduction, publication and CLI reopening."""

import numpy as np
import pytest
import xarray as xr
from scipy.signal import welch
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from typer.testing import CliRunner

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import BlockCreate
from tolteca_kids.cli import app
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import (
    open_timestream_product,
    read_product_provenance,
    read_timestream_context,
)
from tolteca_kids.runner import run_data
from tolteca_kids.timestream_reduce import TimestreamReduce


@pytest.mark.parametrize("input_tree", [False, True])
def test_timestream_pipeline_borrowed_roundtrip(timestream_path, tmp_path, input_tree):
    config = RuntimeContext(
        {
            "kids": {
                "timestream_reduce": {
                    "psd": {"nperseg": 128, "stat_f_lims": ["10 Hz", "50 Hz"]}
                },
                "output": {"enabled": True, "path": str(tmp_path), "namespace": "time"},
            }
        }
    )
    with open_toltec_kids(timestream_path) as native:
        ds = native.isel(time=slice(0, 512)).toltec_kids.select_channels([10, 0, 5])
        data = BlockCreate()(ensure_datatree(ds)) if input_tree else ds
        result = run_data(data, timestream_path, config)
        assert TimestreamReduce.get_context(result["blocks/0"]).completed
        assert DataProdOutput.get_context(result).completed
        if input_tree:
            assert result is data
        actual = TimestreamReduce.get_context(result["blocks/0"]).data
        expected_signal = ds["Data.Toltec.Is"].values.T
        _, expected = welch(expected_signal, fs=122.0703125, nperseg=128)
        np.testing.assert_allclose(actual.I_psd, expected)
        assert actual.chan.values.tolist() == [10, 0, 5]
        # The caller's source is still usable and was not changed by the run.
        assert ds["Header.Toltec.ObsNum"].item() == 145609
        assert "kids_node_type" not in ds.attrs
    provenance = read_product_provenance(
        tmp_path / "time/145609-0-2/dp_raw_obs_proc"
    ).inputs["toltec0"]
    path = DataProdOutput.get_context(result).data
    with xr.open_datatree(path / "array.zarr", engine="zarr") as product:
        assert set(product.children) == {"toltec0"}
        assert not product.dataset.variables
        assert product["toltec0"].toltec_kids.meta.uid_raw_obs_file == (
            provenance.uid_raw_obs_file
        )
    assert "145609" in provenance.uid_raw_obs_file
    assert provenance.config.timestream_reduce.psd.nperseg == 128
    with open_timestream_product(
        tmp_path / "time/145609-0-2/dp_raw_obs_proc"
    ) as reopened:
        view = reopened.toltec_kids.timestream
        np.testing.assert_allclose(view.I_psd, expected)
        np.testing.assert_array_equal(view.I, expected_signal)
        assert view.chan.values.tolist() == [10, 0, 5]
        assert not view.I.variable._in_memory
        assert reopened.toltec_kids.blocks[0].model_params_table["chan"].tolist() == [
            10,
            0,
            5,
        ]
        for node in reopened.subtree:
            assert "input_row" not in node.attrs


def test_timestream_step_preserves_other_nodes(timestream_path):
    with open_toltec_kids(timestream_path) as ds:
        working = xr.DataTree(
            dataset=ds.isel(time=slice(0, 256)).toltec_kids.select_channels([5, 0])
        )
        working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"value": 7}))
        step = TimestreamReduce()
        step(working)
        view = step.get_context(working).data
        first = view.I_psd.data
        step(working)
        assert step.get_context(working).completed
        assert step.get_context(working).data.I_psd.data is first
        assert working["unrelated/value"].item() == 7


def test_disabled_psd_snapshot_survives_publication(timestream_path, tmp_path):
    with open_toltec_kids(timestream_path) as ds:
        data = ds.isel(time=slice(0, 128)).toltec_kids.select_channels([0])
        run_data(
            data,
            timestream_path,
            RuntimeContext(
                {
                    "kids": {
                        "timestream_reduce": {"psd": None},
                        "output": {
                            "enabled": True,
                            "path": str(tmp_path),
                            "namespace": "time",
                        },
                    }
                }
            ),
        )
    path = tmp_path / "time/145609-0-2/dp_raw_obs_proc"
    provenance = read_product_provenance(path)
    assert provenance.inputs["toltec0"].config.timestream_reduce.psd is None
    with open_timestream_product(path) as tree:
        assert read_timestream_context(tree).config.psd is None
        assert tree.toltec_kids.timestream.I_psd is None


def test_timestream_cli_produces_both_domains(timestream_path, tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "--kids.timestream_reduce.psd.nperseg",
            "256",
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(tmp_path),
            "--kids.output.namespace",
            "time",
            "--",
            str(timestream_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    with open_timestream_product(tmp_path / "time/145609-0-2/dp_raw_obs_proc") as tree:
        view = tree.toltec_kids.timestream
        assert view.I.shape == (632, 7802)
        assert view.I_psd.shape == (632, 129)
        assert view.meta.f_smp.to_value("Hz") == 122.0703125


def test_noop_then_timestream_export_and_alias(timestream_path, tmp_path):
    from tolteca_kids.product import write_timestream_product

    with open_toltec_kids(timestream_path) as source:
        data = source.isel(time=slice(0, 256)).toltec_kids.select_channels([0, 5])
        working = run_data(data, timestream_path, RuntimeContext(), dry_run=True)
        primary = TimestreamReduce()
        primary(working)
        alias = "short"
        TimestreamReduce(psd={"nperseg": 64}, alias=alias)(working)
        assert primary.get_context(working).data.I_psd.sizes["f_psd"] == 129
        assert (
            TimestreamReduce.get_context(working, alias=alias).data.I_psd.sizes["f_psd"]
            == 33
        )
        primary_context = primary.get_context(working)
        alias_context = TimestreamReduce.get_context(working, alias=alias)
        assert primary_context.completed
        assert alias_context.completed
        path = write_timestream_product(working, tmp_path)
        assert primary.get_context(working) is primary_context
        assert TimestreamReduce.get_context(working, alias=alias) is alias_context
    with open_timestream_product(path) as tree:
        assert (
            TimestreamReduce.view(
                tree,
                block=read_timestream_context(tree, alias=alias).block,
                alias=alias,
            ).I_psd.sizes["f_psd"]
            == 33
        )
        for context in (primary_context, alias_context):
            key = TimestreamReduce.get_context_key(alias=context.config.alias)
            assert isinstance(tree["tolteca_kids"].attrs[key], str)
            assert (
                read_timestream_context(tree, alias=context.config.alias).model_dump()
                == context.model_dump()
            )
        restored = read_timestream_context(tree, alias=alias)
        assert not TimestreamReduce.has_context(tree, alias=alias)
        assert restored.data is None
        assert restored.config.alias == alias
        assert restored.make_step().context_key == TimestreamReduce.get_context_key(
            alias=alias
        )
