"""Tests for TolTEC metadata models and accessor integration."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode
from tolteca_kids import BlockCreate

_OBS_HEADERS = {
    "Header.Toltec.Master": 0,
    "Header.Toltec.ObsNum": 12345,
    "Header.Toltec.SubObsNum": 0,
    "Header.Toltec.ScanNum": 0,
    "Header.Toltec.ObsStartTime": 1761902400,
    "Header.Toltec.ObsEndTime": 1761902460,
}


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(seed=42)


class TestMetadataModels:
    """Test metadata model structure and hierarchy."""

    def test_raw_obs_metadata_fields(self, kids_filepath_orig):
        """Test ToltecRawObsMetadata has required fields."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            n_blocks_max=1,
            n_times=1000,
            n_chans=256,
            f_smp=500 * u.Hz,
            filepath_orig=kids_filepath_orig,
            data_models=["toltec"],
            instru="toltec",
            interface="toltec0",
            interface_kind="roach",
            master="tcs",
            master_code=ToltecDBRawObsMasterCode.TCS,
            readout_code=0,
            roach=0,
            obsnum=12345,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_int=60.0 * u.s,
        )
        assert meta.data_models == ["toltec"]
        assert meta.master_code == ToltecDBRawObsMasterCode.TCS
        assert meta.roach == 0
        assert meta.obsnum == 12345
        assert meta.subobsnum == 1
        assert meta.scannum == 0

    def test_timestream_metadata_inherits_from_raw_obs(self, kids_filepath_orig):
        """Test ToltecTimestreamMetadata inherits from ToltecRawObsMetadata."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            filepath_orig=kids_filepath_orig,
            data_models=["toltec"],
            instru="toltec",
            interface="toltec0",
            interface_kind="roach",
            master="tcs",
            master_code=ToltecDBRawObsMasterCode.TCS,
            readout_code=0,
            roach=0,
            obsnum=10001,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_int=60.0 * u.s,
            n_times=1000,
            n_chans=256,
            f_smp=488.0 * u.Hz,
        )
        # Check inherited fields
        assert meta.data_models == ["toltec"]
        assert meta.roach == 0
        # Check timestream fields
        assert meta.n_times == 1000
        assert meta.n_chans == 256
        assert meta.f_smp == 488.0 * u.Hz

    def test_sweep_metadata_inherits_from_block(self, kids_filepath_orig):
        """Test ToltecSweepMetadata inherits from ToltecRawObsBlockMetadata."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecSweepMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            filepath_orig=kids_filepath_orig,
            data_models=["toltec"],
            instru="toltec",
            interface="toltec0",
            interface_kind="roach",
            master="tcs",
            master_code=ToltecDBRawObsMasterCode.TCS,
            readout_code=0,
            roach=0,
            obsnum=10002,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_int=60.0 * u.s,
            n_times=4910,
            n_chans=1000,
            f_smp=488.0 * u.Hz,
            n_sweepsteps=491,
            n_sweepreps=10,
            f_lo_center=4.5e9,
        )
        # Check inherited from raw obs
        assert meta.data_models == ["toltec"]
        assert meta.roach == 0
        # Check inherited from block
        assert meta.n_times == 4910
        assert meta.n_chans == 1000
        # Check sweep fields
        assert isinstance(meta, ToltecRawObsBlockMetadata)
        assert meta.n_sweepsteps == 491
        assert meta.n_sweepreps == 10
        assert meta.f_lo_center == 4.5e9

    def test_metadata_is_immutable(self, kids_filepath_orig):
        """Test metadata fields are frozen."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            n_blocks_max=1,
            n_times=1000,
            n_chans=256,
            f_smp=500 * u.Hz,
            filepath_orig=kids_filepath_orig,
            data_models=["toltec"],
            instru="toltec",
            interface="toltec0",
            interface_kind="roach",
            master="tcs",
            master_code=ToltecDBRawObsMasterCode.TCS,
            readout_code=0,
            roach=0,
            obsnum=10003,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_int=60.0 * u.s,
        )
        with pytest.raises(ValidationError, match="frozen"):
            setattr(meta, "roach", 1)  # noqa: B010 -- Test frozen validation.

    def test_derived_fields(self, kids_filepath_orig):
        """Test derived metadata fields."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            n_blocks_max=1,
            n_times=1000,
            n_chans=256,
            f_smp=500 * u.Hz,
            filepath_orig=kids_filepath_orig,
            data_models=["toltec"],
            instru="toltec",
            master="tcs",
            master_code=ToltecDBRawObsMasterCode.TCS,
            readout_code=0,
            roach=0,
            interface="toltec0",
            interface_kind="roach",
            nw=0,
            obsnum=10004,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_int=60.0 * u.s,
        )
        assert meta.interface == "toltec0"
        assert meta.nw == 0


class TestMetadataAccessor:
    """Test metadata accessor integration."""

    @pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune"])
    def test_mapper_and_accessor_block_metadata(self, data_lmt_path, suffix):
        """Identified collection metadata and created block records agree."""
        (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{suffix}.nc")
        with open_toltec_kids(path) as data:
            mapper = ToltecKidsIOMapper.from_data_source(data)
            identified = mapper.get_blocks(data)
            blocks = BlockCreate()(ensure_datatree(data)).toltec_kids.blocks
            assert (
                identified.meta.n_blocks
                == len(blocks)
                == (2 if suffix == "tune" else 1)
            )
            assert identified.meta.model_dump() == blocks.meta.model_dump()
            xr.testing.assert_identical(identified.index, blocks.data_source)
            for index, block in enumerate(blocks):
                meta = block.meta
                assert isinstance(meta, ToltecSweepMetadata)
                assert isinstance(meta, ToltecRawObsBlockMetadata)
                assert meta.block == index
                assert meta.n_times == block.raw.sizes["time"]
            assert not data["Data.Toltec.Is"].variable._in_memory

    def test_metadata_without_observation_type(self, rng):
        """A raw array shape cannot replace the recorded observation type."""
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["iqlen", "time"], rng.standard_normal((100, 1000))),
                "Data.Toltec.Qs": (["iqlen", "time"], rng.standard_normal((100, 1000))),
            },
            attrs={
                **_OBS_HEADERS,
                "Header.Toltec.RoachIndex": 0,
                "Header.Toltec.SampleFreq": 488.0,  # Add f_smp (required)
            },
        )

        with pytest.raises(ValueError, match="Header.Toltec.ObsType"):
            _ = ds.toltec_kids.meta


class TestMetadataEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_block_raises(self, data_lmt_path):
        """An invalid selection fails after preparing a valid collection."""
        (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
        with (
            open_toltec_kids(path) as ds,
            pytest.raises(ValueError, match="Block .* out of range"),
        ):
            BlockCreate()(ensure_datatree(ds)).toltec_kids.blocks[5]

    def test_metadata_caching(self, data_lmt_path):
        """Repeated metadata access returns the same cached instance."""
        (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
        with open_toltec_kids(path) as ds:
            meta1 = ds.toltec_kids.meta
            meta2 = ds.toltec_kids.meta
            assert meta1 is meta2
