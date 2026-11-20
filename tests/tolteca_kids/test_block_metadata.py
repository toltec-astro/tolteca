"""Metadata on blocks created by the KIDs workflow."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError
from tollan.accessor.xarray import ensure_datatree

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import BlockCreate


def test_processed_design_and_adc_can_be_absent(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as data:
        # Synthetic processed encoding, not a supported acquisition filename.
        changed = (
            data.drop_dims("loclen")
            .drop_vars("Header.Toltec.AdcSnapData")
            .assign_attrs({"Header.Kids.kind": "processed_sweep"})
        )
        meta = BlockCreate()(ensure_datatree(changed)).toltec_kids.blocks[0].meta
        assert meta.n_kids_design is None
        assert not hasattr(meta, "adc_snap")
        with pytest.raises(KeyError):
            _ = changed.toltec_kids.adc_snap


def test_processed_type_header_remains_optional(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as data:
        # Deliberately exercise a processed-header encoding, not a claim about
        # a real processed acquisition in the reference fixture inventory.
        processed = data.drop_vars("Header.Toltec.ObsType").assign_attrs(
            {"Header.Kids.kind": "processed_sweep"}
        )
        assert processed.toltec_kids.data_kind == ToltecDataKind.reduced_sweep
        assert (
            BlockCreate()(ensure_datatree(processed))
            .toltec_kids.blocks[0]
            .meta.obs_type_code
            is None
        )


def test_processed_metadata_can_omit_readout_settings(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as data:
        # Deliberately synthetic processed encoding, not an acquisition fixture.
        processed = data.drop_vars(
            [
                "Header.Toltec.DriveAtten",
                "Header.Toltec.SenseAtten",
                "Header.Toltec.TargSweepObsNum",
                "Header.Toltec.TargSweepSubObsNum",
                "Header.Toltec.TargSweepScanNum",
                "Header.Toltec.MaxNumTones",
                "Header.Toltec.RepeatLevel",
            ],
        ).assign_attrs({"Header.Kids.kind": "processed_sweep"})
        meta = BlockCreate()(ensure_datatree(processed)).toltec_kids.blocks[0].meta
        assert (
            meta.filepath_orig
            == BlockCreate()(ensure_datatree(data))
            .toltec_kids.blocks[0]
            .meta.filepath_orig
        )
        for field in (
            "atten_drive",
            "atten_sense",
            "cal_obsnum",
            "cal_subobsnum",
            "cal_scannum",
            "n_chans_max",
            "repeat",
        ):
            assert getattr(meta, field) is None


def test_processed_raw_metadata_still_requires_acquisition_path(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as data:
        # A deliberately incomplete processed encoding, not an acquisition fixture.
        incomplete = data.drop_vars("Header.Toltec.Filename").assign_attrs(
            {"Header.Kids.kind": "processed_sweep"}
        )
        with pytest.raises(ValidationError, match="filepath_orig"):
            _ = BlockCreate()(ensure_datatree(incomplete)).toltec_kids.blocks[0].meta


@pytest.mark.parametrize("processed", [False, True])
def test_timestream_lo_units(data_lmt_path, processed):
    root = data_lmt_path / "toltec/tcs/toltec0"
    (path,) = root.glob("*_09_27_57.nc")
    with open_toltec_kids(path) as data:
        name = "Header.Toltec.LoCenterFreq"
        frequency = data[name].item()
        # Equivalent unit representation of the recorded value.
        changed = data.assign({name: ((), frequency / 1e6, {"units": "MHz"})})
        if processed:
            changed = changed.assign_attrs({"Header.Kids.kind": "processed_timestream"})
        assert (
            BlockCreate()(ensure_datatree(changed))
            .toltec_kids.blocks[0]
            .meta.f_lo_center
            * u.Hz
            == frequency * u.Hz
        )


@pytest.mark.parametrize("kind", ["processed_sweep", "processed_timestream"])
def test_processed_lo_center_can_be_absent(target_sweep_path, kind):
    with open_toltec_kids(target_sweep_path) as data:
        # Deliberately synthetic processed encoding to check the omission policy.
        changed = data.drop_vars("Header.Toltec.LoCenterFreq").assign_attrs(
            {"Header.Kids.kind": kind}
        )
        assert (
            BlockCreate()(ensure_datatree(changed))
            .toltec_kids.blocks[0]
            .meta.f_lo_center
            is None
        )


def test_processed_sweep_does_not_require_raw_geometry(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as data:
        reps = int(data["Header.Toltec.NumSamplesPerSweepStep"].item())
        # Synthetic one-value-per-step layout, not a numerical reduction oracle.
        # Keep readout headers: numSweeps is also a dimension of the tone arrays.
        changed = (
            data.isel(time=slice(None, None, reps), numSweeps=0, drop=True)
            .rename({"time": "sweep"})
            .drop_vars("Header.Toltec.NumSamplesPerSweepStep")
            .assign_attrs({"Header.Kids.kind": "processed_sweep"})
        )
        meta = BlockCreate()(ensure_datatree(changed)).toltec_kids.blocks[0].meta
        assert meta.n_sweepreps is None
        assert meta.n_blocks_max == meta.n_blocks == 1
        assert meta.n_times == meta.n_sweepsteps == changed.sizes["sweep"]


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune"])
def test_mapper_and_accessor_block_metadata(data_lmt_path, suffix):
    """Identified collection metadata and created block records agree."""
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{suffix}.nc")
    with open_toltec_kids(path) as data:
        mapper = ToltecKidsIOMapper.from_data_source(data)
        identified = mapper.get_blocks(data)
        blocks = BlockCreate()(ensure_datatree(data)).toltec_kids.blocks
        assert identified.meta.n_blocks == len(blocks) == (2 if suffix == "tune" else 1)
        assert identified.meta.model_dump() == blocks.meta.model_dump()
        xr.testing.assert_identical(identified.index, blocks.data_source)
        for index, block in enumerate(blocks):
            meta = block.meta
            assert isinstance(meta, ToltecSweepMetadata)
            assert isinstance(meta, ToltecRawObsBlockMetadata)
            assert meta.block == index
            assert meta.n_times == block.raw.sizes["time"]
        assert not data["Data.Toltec.Is"].variable._in_memory


def test_invalid_block_raises(data_lmt_path):
    """An invalid selection fails after preparing a valid collection."""
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with (
        open_toltec_kids(path) as ds,
        pytest.raises(ValueError, match="Block .* out of range"),
    ):
        BlockCreate()(ensure_datatree(ds)).toltec_kids.blocks[5]


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_created_blocks_inherit_acquisition_metadata(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data:
        acquisition = data.toltec_kids.meta
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


@pytest.mark.parametrize("n_blocks", [0, 1, 2])
def test_created_blocks_follow_the_explicit_dimension(data_lmt_path, n_blocks):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with open_toltec_kids(path) as raw:
        indexed = raw.assign_coords(block=np.arange(n_blocks))
        blocks = BlockCreate()(ensure_datatree(indexed)).toltec_kids.blocks
        metadata = [block.meta for block in blocks]
        assert len(metadata) == n_blocks
        for i, meta in enumerate(metadata):
            assert meta.n_blocks == n_blocks
            assert meta.block == i
            assert meta.n_times == raw.sizes["time"]
            assert blocks[i].meta is meta
        with pytest.raises(ValueError, match="out of range"):
            _ = blocks[n_blocks]


def test_empty_tune_creates_no_blocks(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with open_toltec_kids(path) as raw:
        empty = raw.isel(time=slice(0, 0))
        blocks = BlockCreate()(ensure_datatree(empty)).toltec_kids.blocks
        assert list(blocks) == []
        assert blocks.meta.n_times == 0
        with pytest.raises(ValueError, match="out of range"):
            _ = blocks[0]
