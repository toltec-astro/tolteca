"""Acquisition publication with multiple interfaces and explicit runs."""

import pytest
import xarray as xr
from tollan.config import RuntimeContext

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.data_prod_store import DataProdStore
from tolteca_datamodels.toltec.file import guess_info_from_sources
from tolteca_kids.config import DataProdOutputConfig
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import (
    make_kids_product_index,
    open_sweep_product,
    read_product_interfaces,
    read_product_provenance,
    write_kids_product,
)
from tolteca_kids.runner import run_inputs
from tolteca_kids.sweep_reduce import SweepReduce


@pytest.mark.parametrize(
    ("kind", "scope"),
    [
        ("tune", "acquisition"),
        ("timestream", "acquisition"),
        ("timestream", "collection"),
        ("timestream", "block"),
    ],
)
def test_configured_chunks_preserve_dask_signals(data_lmt_path, tmp_path, kind, scope):
    from tollan.accessor.xarray import ensure_datatree

    from tolteca_kids.config import TimestreamReduceConfig
    from tolteca_kids.product import open_kids_product
    from tolteca_kids.timestream_reduce import TimestreamReduce

    name = (
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
        if kind == "tune"
        else "toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    source = data_lmt_path / "toltec/tcs/toltec0" / name
    with xr.open_dataset(source, chunks={}) as native:
        raw = native.toltec_kids.select_channels(list(range(33)))
        if kind == "timestream":
            mapper = raw.toltec_kids.mapper
            time_dim = mapper.get_name(mapper.schema.n_times)
            raw = raw.isel({time_dim: slice(0, 2300)})
        original_encoding = {
            name: dict(v.encoding) for name, v in raw.variables.items()
        }
        tree = ensure_datatree(raw)
        if scope != "acquisition":
            from tolteca_kids import BlockCreate

            BlockCreate()(tree)
            blocks = tree.toltec_kids.blocks
            tree = blocks.detach() if scope == "collection" else blocks[0].detach()
        step = (
            SweepReduce()
            if kind == "tune"
            else TimestreamReduce(TimestreamReduceConfig(psd=None))
        )
        step(tree)
        config = DataProdOutputConfig(n_chans_per_chunk=5, n_samples_per_chunk=127)
        path = write_kids_product(tree, tmp_path, output_config=config)
        assert {
            name: v.encoding for name, v in raw.variables.items()
        } == original_encoding
        assert read_product_provenance(path).output.n_chans_per_chunk == 5
        with open_kids_product(path) as saved:
            if kind == "tune":
                for block in tree.toltec_kids.blocks:
                    expected = block.sweep
                    actual = saved.toltec_kids.blocks[block.meta.block].sweep
                    for field in ("I", "Q", "unc_I", "unc_Q"):
                        before, after = getattr(expected, field), getattr(actual, field)
                        xr.testing.assert_identical(after.compute(), before.compute())
                        chan_dim = actual.f_chan.dims[0]
                        assert after.encoding["chunks"] == tuple(
                            5 if dim == chan_dim else size
                            for dim, size in after.sizes.items()
                        )
            else:
                actual = (
                    saved.toltec_kids.blocks[0].raw
                    if scope == "collection"
                    else saved.toltec_kids.data_source
                )
                mapper = raw.toltec_kids.mapper
                for field in (mapper.schema.I, mapper.schema.Q):
                    expected = mapper.get_arr(raw, field)
                    signal = mapper.get_arr(actual, field)
                    xr.testing.assert_identical(signal.compute(), expected.compute())
                    assert signal.encoding["chunks"] == (127, 5)


def _check_input_provenance_and_metadata_lookup(path, provenance, monkeypatch):
    assert not provenance.inputs["toltec7"].config.sweep_reduce.compute_uncertainty
    assert provenance.inputs["toltec11"].config.sweep_reduce.compute_uncertainty
    for selected_input in provenance.inputs.values():
        assert selected_input.config.input.select == "roach in [7, 11]"
        assert selected_input.config.timestream_reduce.psd is None
    from xarray.backends.zarr import ZarrArrayWrapper

    def unexpected(*args, **kwargs):
        pytest.fail("Interface discovery read an array chunk")

    with monkeypatch.context() as patch:
        patch.setattr(ZarrArrayWrapper, "__getitem__", unexpected)
        assert read_product_interfaces(path) == {
            "toltec7": "/toltec7",
            "toltec11": "/toltec11",
        }


def _check_republished_provenance(path, original):
    published = read_product_provenance(path)
    assert published.inputs == original.inputs
    assert published.output.run.name.key == "check-a"
    assert {"scipy", "astropy"} <= published.software.keys()


def _check_uniform_interface_layout(path, working, tmp_path, provenance):
    with pytest.raises(ValueError, match="Select an interface"):
        open_sweep_product(path)
    single_path = write_kids_product(working["interfaces/toltec7"], tmp_path / "single")
    assert read_product_interfaces(single_path) == {"toltec7": "/toltec7"}
    with (
        xr.open_datatree(path / "array.zarr", engine="zarr") as multiple,
        xr.open_datatree(single_path / "array.zarr", engine="zarr") as single,
    ):
        assert set(multiple.children) == {"toltec7", "toltec11"}
        assert set(single.children) == {"toltec7"}
        assert not single.dataset.variables and not multiple.dataset.variables
        xr.testing.assert_identical(single["toltec7"], multiple["toltec7"])
        for interface in provenance.inputs:
            assert multiple[interface].toltec_kids.meta.uid_raw_obs_file == (
                provenance.inputs[interface].uid_raw_obs_file
            )


def test_grouped_interfaces_and_run_publication(data_lmt_path, tmp_path, monkeypatch):
    sources = [
        next((data_lmt_path / f"toltec/tcs/toltec{nw}").glob("*156755*_tune.nc"))
        for nw in (7, 11)
    ]
    table = guess_info_from_sources(sources)
    runtime = RuntimeContext(
        [
            {
                "source": {
                    "kids": {
                        "input": {"select": "roach in [7, 11]"},
                        "timestream_reduce": {"psd": None},
                        "output": {"enabled": True, "path": tmp_path},
                    }
                }
            },
            {
                "enable_if": "roach == 7",
                "source": {
                    "kids": {
                        "sweep_reduce": {"compute_uncertainty": False},
                    }
                },
            },
        ]
    )
    with table.toltec_file.open() as opened:
        selected = opened.copy()
        for idx, raw in selected.io_obj.items():
            selected.at[idx, "io_obj"] = raw.toltec_kids.select_channels([0, 2])
        [(uid, working)] = list(run_inputs(selected, runtime))
        assert uid == "tcs-156755-0-1"
        path = DataProdOutput.get_context(working).data
        assert path.parent == tmp_path / "scratch/156755-0-1/dp_raw_obs_proc"
        index = DataProdIndex.read(path)
        assert index.uid == f"{uid}_proc/{index.run.uid}"
        assert len(index.assocs) == 1
        assert index.assocs[0].model_dump(
            exclude={"namespace_uid", "runs", "audit"}
        ) == {
            "uid": uid,
            "data_prod_assoc_type": "dpa_raw_obs_proc_raw_obs",
            "is_primary": True,
            "context": {},
        }
        assert {i.meta["format"] for i in index.data_items} == {
            "zarr",
            "data_prod_index",
        }
        assert len(list(path.glob("*.zarr"))) == 1
        assert not (path / "manifest.json").exists()
        assert DataProdIndex.read(path.parent).provenance is None
        provenance = read_product_provenance(path)
        assert set(provenance.inputs) == {"toltec7", "toltec11"}
        _check_input_provenance_and_metadata_lookup(path, provenance, monkeypatch)
        _check_uniform_interface_layout(path, working, tmp_path, provenance)
        for interface in provenance.inputs:
            with open_sweep_product(path, interface=interface) as saved:
                live = working[f"interfaces/{interface}"]
                xr.testing.assert_identical(
                    SweepReduce.view(saved, block=0).I,
                    SweepReduce.view(live, block=0).I,
                )
                assert saved.toltec_kids.meta.interface == interface
        original = (path / "index.yaml").read_bytes()
        initial_run = index.run.name
        next_path = write_kids_product(
            dict(working["interfaces"].children),
            tmp_path,
            output_config=DataProdOutputConfig(
                run={"name": {"generator": "counter", "key": "check-a"}}
            ),
        )
        assert (path / "index.yaml").read_bytes() == original
        assert next_path.name == "0000-check-a"
        _check_republished_provenance(next_path, provenance)
        assert (
            DataProdIndex.read(path.parent).tags["main"].current.uid
            == DataProdIndex.read(next_path).uid
        )
        assert DataProdIndex.read(path).run.name == initial_run
        published = (path.parent / "index.yaml").read_bytes()
        with pytest.raises(FileExistsError):
            write_kids_product(
                dict(working["interfaces"].children),
                tmp_path,
                output_config=DataProdOutputConfig(
                    run={
                        "name": {"generator": "counter", "value": 0, "key": "duplicate"}
                    }
                ),
            )
        assert (path.parent / "index.yaml").read_bytes() == published

        def fail(*args, **kwargs):
            msg = "test publication failure"
            raise OSError(msg)

        monkeypatch.setattr(xr.DataTree, "to_zarr", fail)
        with pytest.raises(OSError, match="publication failure"):
            write_kids_product(
                dict(working["interfaces"].children),
                tmp_path,
                output_config=DataProdOutputConfig(
                    run={"name": {"generator": "counter", "key": "failed"}}
                ),
            )
        inventory = DataProdIndex.read(path.parent)
        assert inventory.tags["main"].current.uid == DataProdIndex.read(next_path).uid
        assert (
            next(
                run
                for run in DataProdStore(tmp_path).read_namespace().runs.values()
                if run.name == "0001-failed"
            ).status
            == "failed"
        )
        assert DataProdStore(tmp_path).read_namespace().namespace.counter_next == 2
        assert not (path.parent / "0001-failed").exists()


def test_busy_tag_fails_before_block_processing(data_lmt_path, tmp_path, monkeypatch):
    from tolteca_datamodels.data_prod_store import DataProdStore
    from tolteca_kids.block_identify import BlockIdentify

    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc"))
    table = guess_info_from_sources([source])
    runtime = RuntimeContext({"kids": {"output": {"enabled": True, "path": tmp_path}}})

    def unexpected(*args, **kwargs):
        pytest.fail("A busy tag allowed block processing")

    monkeypatch.setattr(BlockIdentify, "run", unexpected)
    with table.toltec_file.open() as opened:
        uid = opened.uid_raw_obs.iloc[0]
        row = next(opened.itertuples(index=False))
        index = make_kids_product_index(
            row.master, int(row.obsnum), int(row.subobsnum), int(row.scannum)
        )
        with DataProdStore(tmp_path).publication(index, primary_uid=uid):
            [(result_uid, working)] = list(run_inputs(opened, runtime))
        assert result_uid == uid
        context = DataProdOutput.get_context(working)
        assert not context.completed and context.data is None
        assert context.failures[0].details["code"] == "tag_busy"
        assert not BlockIdentify.has_context(working)


@pytest.mark.parametrize("generator", ["uuid", "counter"])
def test_batch_uses_shared_ids_or_keys_and_portable_runs(
    data_lmt_path, tmp_path, generator
):
    import shutil

    sources = [
        next((data_lmt_path / "toltec/tcs/toltec0").glob(pattern))
        for pattern in ("*145606*_targsweep.nc", "*145605*_vnasweep.nc")
    ]
    table = guess_info_from_sources(sources)
    runtime = RuntimeContext(
        {
            "kids": {
                "output": {
                    "enabled": True,
                    "path": tmp_path,
                    "run": {
                        "name": None
                        if generator == "uuid"
                        else {"generator": generator}
                    },
                }
            }
        }
    )
    with table.toltec_file.open() as opened:
        selected = opened.copy()
        for idx, raw in selected.io_obj.items():
            selected.at[idx, "io_obj"] = raw.toltec_kids.select_channels([0])
        results = list(run_inputs(selected, runtime))
    paths = [DataProdOutput.get_context(tree).data for _, tree in results]
    from uuid import UUID

    runs = [DataProdIndex.read(path).run for path in paths]
    if generator == "uuid":
        assert len({run.name for run in runs}) == 1
        assert runs[0].uid.version == 4
    else:
        assert len({run.generator.key for run in runs}) == 1
        assert UUID(runs[0].generator.key).version == 4
        assert [run.generator.value for run in runs] == [0, 0]
    moved = tmp_path / "detached-run"
    shutil.copytree(paths[0], moved)
    shutil.rmtree(paths[0].parent)
    provenance = read_product_provenance(moved)
    assert provenance.runtime_info.hostname == runtime.runtime_info.hostname
    assert provenance.runtime_info.username == runtime.runtime_info.username
    with open_sweep_product(moved) as tree:
        assert SweepReduce.view(tree, block=0).I.size > 0


@pytest.mark.parametrize(
    ("options", "run_uid"),
    [
        (["--kids.output.run.name", "external-job-id"], "external-job-id"),
        (
            [
                "--kids.output.run.uid",
                "03f1f28e-c179-471f-a684-0767207f2975",
                "--kids.output.run.name",
                "external",
            ],
            "external",
        ),
        (
            [
                "--kids.output.run.name.generator",
                "counter",
                "--kids.output.run.name.value",
                "42",
                "--kids.output.run.name.key",
                "nightly",
                "--kids.output.run.name.format",
                "redu{value:04d}-{key}",
            ],
            "redu0042-nightly",
        ),
    ],
)
def test_cli_accepts_literal_and_counter_run_requests(
    target_sweep_path, tmp_path, options, run_uid
):
    from typer.testing import CliRunner

    from tolteca_kids.cli import app

    result = CliRunner().invoke(
        app,
        [
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(tmp_path),
            *options,
            "--",
            str(target_sweep_path),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    [path] = list(tmp_path.glob(f"scratch/*/dp_raw_obs_proc/{run_uid}/index.yaml"))
    index = DataProdIndex.read(path)
    assert index.run.name == run_uid
    assert index.uid.endswith(f"/{index.run.uid}")
    if "--kids.output.run.uid" in options:
        assert str(index.run.uid) == options[options.index("--kids.output.run.uid") + 1]
    assert DataProdIndex.read(path).run.name == run_uid
