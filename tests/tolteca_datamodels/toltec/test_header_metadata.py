"""Acquisition metadata and inherited block records have explicit read costs."""

import pytest
from pydantic import TypeAdapter
from tollan.accessor.xarray import ensure_datatree
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_kids import BlockCreate


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize("chunks", [None, {}])
def test_acquisition_metadata_reused_by_blocks(
    data_lmt_path, monkeypatch, suffix, chunks
):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    original = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def read(array, key):
        assert array.variable_name not in {
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
            "Header.Toltec.Xloc",
            "Header.Toltec.Yloc",
            "Header.Toltec.AdcSnapData",
        }
        reads.append(array.variable_name)
        return original(array, key)

    with open_toltec_kids(path, chunks=chunks) as source:
        data = ensure_datatree(source)
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
        acquisition = data.toltec_kids.meta
        assert type(acquisition) is ToltecRawObsMetadata
        assert acquisition is data.toltec_kids.meta
        assert not any(name.startswith("Data.") for name in reads)
        assert "Header.Toltec.ToneFreq" not in reads
        assert acquisition.n_sweepreps == (10 if suffix else None)
        assert not hasattr(acquisition, "n_blocks")
        assert not hasattr(acquisition, "n_sweeps")
        assert not hasattr(acquisition, "n_sweepsteps")
        assert acquisition.n_blocks_max == source.sizes["numSweeps"]
        reads.clear()
        identified = data.toltec_kids.block_index
        assert type(identified.meta) is ToltecRawObsBlocksMetadata
        for name in ToltecRawObsMetadata.model_fields:
            assert getattr(identified.meta, name) == getattr(acquisition, name), name
        tree = BlockCreate()(ensure_datatree(data))
        blocks = [item.meta for item in tree.toltec_kids.blocks]
        assert all(meta is tree.toltec_kids.blocks[meta.block].meta for meta in blocks)
        expected_blocks = 2 if suffix == "tune" else 1
        assert len(blocks) == expected_blocks
        assert sum(block.n_times for block in blocks) == acquisition.n_times
        for index, block in enumerate(blocks):
            assert type(block) is (
                ToltecSweepMetadata if suffix else ToltecTimestreamMetadata
            )
            assert isinstance(block, ToltecRawObsBlockMetadata)
            assert isinstance(block, ToltecRawObsBlocksMetadata)
            assert block.n_blocks == expected_blocks
            assert block.block == index
            assert not hasattr(block, "n_sweeps")
            for name in ToltecRawObsMetadata.model_fields:
                if name not in {"n_times", "t_int"}:
                    assert getattr(block, name) == getattr(acquisition, name), name
            assert block.t_int == (block.n_times / block.f_smp).to("s")
            adapter = TypeAdapter(type(block))
            restored = adapter.validate_json(adapter.dump_json(block, round_trip=True))
            assert restored.model_dump() == block.model_dump()
            assert restored.chan_axis_table is None
            assert block.chan_axis_table is not None
        assert set(reads) <= {
            "Header.Toltec.NumSweepSteps",
            "Header.Toltec.ToneFreq",
            "Header.Toltec.ToneMask",
            "Header.Toltec.ToneAmp",
            "Header.Toltec.TonePhase",
            "Data.Toltec.LoFreq",
        }
        assert reads.count("Data.Toltec.LoFreq") == (1 if suffix else 2)
        assert not source["Data.Toltec.Is"].variable._in_memory


@pytest.mark.parametrize(
    ("header", "error_type"),
    [("ObsType", ValueError), ("DriveAtten", ValueError), ("Filename", KeyError)],
)
def test_acquisition_metadata_requires_raw_fields(
    real_targsweep_file, header, error_type
):
    name = f"Header.Toltec.{header}"
    with (
        open_toltec_kids(real_targsweep_file) as data,
        pytest.raises(error_type, match=name),
    ):
        _ = data.drop_vars(name).toltec_kids.meta


def test_programmed_repetitions_are_required_in_sweep_acquisitions(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as data:
        changed = data.drop_vars("Header.Toltec.NumSamplesPerSweepStep")
        with pytest.raises(ValueError, match="NumSamplesPerSweepStep"):
            _ = changed.toltec_kids.meta


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_capacity_required_for_all_raw_acquisitions(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data, pytest.raises(KeyError, match="numSweeps"):
        _ = data.drop_dims("numSweeps").toltec_kids.meta


@pytest.mark.parametrize("layout", ["processed", "block"])
def test_acquisition_metadata_requires_unpartitioned_raw_input(
    real_targsweep_file, layout
):
    with open_toltec_kids(real_targsweep_file) as data:
        changed = (
            data.assign_attrs({"Header.Kids.kind": "processed_sweep"})
            if layout == "processed"
            else data.expand_dims(block=[0])
        )
        mapper = data.toltec_kids.mapper.from_data_source(changed)
        with pytest.raises(ValueError, match="requires (an unpartitioned|a raw KIDs)"):
            mapper.get_metadata(changed)
