"""Tests for TolTEC metadata dataclasses and accessor integration."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from tolteca_datamodels.toltec.kids.core import (
    ToltecKidsIOMapper as ToltecKidsFullMapper,
)
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimeStreamMetadata,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(seed=42)


class TestMetadataDataclasses:
    """Test metadata dataclass structure and hierarchy."""

    def test_raw_obs_metadata_fields(self):
        """Test ToltecRawObsMetadata has required fields."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            instru="toltec",
            interface="toltec0",
            master="tcs",
            roach=0,
            obsnum=12345,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_exp=60.0 * u.s,
        )
        assert meta.instru == "toltec"
        assert meta.master == "tcs"
        assert meta.roach == 0
        assert meta.obsnum == 12345
        assert meta.subobsnum == 1
        assert meta.scannum == 0

    def test_timestream_metadata_inherits_from_raw_obs(self):
        """Test ToltecTimeStreamMetadata inherits from ToltecRawObsMetadata."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecTimeStreamMetadata(
            instru="toltec",
            interface="toltec0",
            master="tcs",
            roach=0,
            obsnum=10001,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_exp=60.0 * u.s,
            n_times=1000,
            n_chans=256,
            f_smp=488.0 * u.Hz,
        )
        # Check inherited fields
        assert meta.instru == "toltec"
        assert meta.roach == 0
        # Check timestream fields
        assert meta.n_times == 1000
        assert meta.n_chans == 256
        assert meta.f_smp == 488.0 * u.Hz

    def test_sweep_metadata_inherits_from_timestream(self):
        """Test ToltecSweepMetadata inherits from ToltecTimeStreamMetadata."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecSweepMetadata(
            instru="toltec",
            interface="toltec0",
            master="tcs",
            roach=0,
            obsnum=10002,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_exp=60.0 * u.s,
            n_times=4910,
            n_chans=1000,
            f_smp=488.0 * u.Hz,
            n_sweeps=1,
            n_sweepsteps=491,
            n_sweepreps=10,
            f_lo_center=4.5e9,
        )
        # Check inherited from raw obs
        assert meta.instru == "toltec"
        assert meta.roach == 0
        # Check inherited from timestream
        assert meta.n_times == 4910
        assert meta.n_chans == 1000
        # Check sweep fields
        assert meta.n_sweeps == 1
        assert meta.n_sweepsteps == 491
        assert meta.n_sweepreps == 10
        assert meta.f_lo_center == 4.5e9

    def test_metadata_is_immutable(self):
        """Test metadata dataclasses are frozen."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            instru="toltec",
            interface="toltec0",
            master="tcs",
            roach=0,
            obsnum=10003,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_exp=60.0 * u.s,
        )
        with pytest.raises(
            (AttributeError, TypeError),
        ):  # pydantic frozen dataclass raises AttributeError or TypeError
            meta.roach = 1  # type: ignore[misc]

    def test_derived_fields(self):
        """Test derived metadata fields."""
        from astropy import units as u
        from astropy.time import Time

        meta = ToltecRawObsMetadata(
            instru="toltec",
            master="tcs",
            roach=0,
            interface="toltec0",
            nw=0,
            obsnum=10004,
            subobsnum=1,
            scannum=0,
            t0=Time("2025-01-01T00:00:00", format="isot", scale="utc"),
            t1=Time("2025-01-01T00:01:00", format="isot", scale="utc"),
            t_exp=60.0 * u.s,
        )
        assert meta.interface == "toltec0"
        assert meta.nw == 0


class TestMetadataAccessor:
    """Test metadata accessor integration."""

    def test_single_block_sweep_metadata(self, rng):
        """Test metadata property returns ToltecSweepMetadata for single-block sweep."""
        # Create mock sweep dataset
        n_chans = 1000
        n_sweeps = 491
        n_samples = n_sweeps * 10  # 10 reps per sweep

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (
                    ["iqlen", "time"],
                    rng.standard_normal((n_chans, n_samples)),
                ),
                "Data.Toltec.Qs": (
                    ["iqlen", "time"],
                    rng.standard_normal((n_chans, n_samples)),
                ),
                "Header.Toltec.ToneFreq": (
                    ["toneFreqLen"],
                    np.linspace(0, 250e6, n_chans),
                ),
                "Header.Toltec.ToneMask": (
                    ["toneFreqLen"],
                    np.ones(n_chans, dtype=bool),
                ),
            },
            coords={
                "sweeps": np.linspace(-245e6, 245e6, n_sweeps),
            },
            attrs={
                "Header.Toltec.RoachIndex": 0,
                "Header.Toltec.ObsNum": 12345,
                "Header.Toltec.SubObsNum": 1,
                "Header.Toltec.ScanNum": 0,
                "Header.Toltec.SampleFreq": 488.0,
                "Header.Toltec.LoCenterFreq": 4.5e9,
                "Header.Toltec.NumSweepSteps": n_sweeps,
                "Header.Toltec.NumSamplesPerSweepStep": 10,
                "Header.Toltec.Master": 1,
                "Header.Toltec.ObsType": 0,  # VNA sweep
            },
        )

        # Get metadata via accessor
        meta = ds.toltec_kids.meta

        # Check type
        assert isinstance(meta, ToltecSweepMetadata)

        # Check raw obs fields
        assert meta.roach == 0
        assert meta.obsnum == 12345
        assert meta.subobsnum == 1
        assert meta.scannum == 0
        assert meta.master == "tcs"

        # Check derived fields
        assert meta.interface == "toltec0"
        assert meta.nw == 0
        assert meta.array_name == "a1100"

        # Check timestream fields
        assert meta.n_times == n_samples
        assert meta.n_chans == n_chans
        assert meta.f_smp == 488.0 * u.Hz

        # Check sweep fields
        assert meta.n_sweeps == 1  # v2 convention: single block = 1 sweep
        assert meta.n_sweepsteps == n_sweeps
        assert meta.n_sweepreps == 10
        assert meta.f_lo_center == 4.5e9

    def test_multi_block_sweep_metadata(self, rng):
        """Test metadata property returns list for multi-block sweep."""
        # Create mock multi-block dataset
        n_chans = 632
        n_sweeps = 177
        n_blocks = 2

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((n_chans, n_blocks, n_sweeps)),
                ),
                "Data.Toltec.Qs": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((n_chans, n_blocks, n_sweeps)),
                ),
                "Data.Toltec.LoFreq": (
                    ["block", "sweeps"],
                    np.array(
                        [
                            np.linspace(4.3e9, 4.7e9, n_sweeps),  # Block 0
                            np.linspace(4.8e9, 5.2e9, n_sweeps),  # Block 1
                        ],
                    ),
                ),
                "Header.Toltec.ToneFreq": (
                    ["toneFreqLen"],
                    np.linspace(0, 250e6, n_chans),
                ),
                "Header.Toltec.ToneMask": (
                    ["block", "toneFreqLen"],
                    np.ones((n_blocks, n_chans), dtype=bool),
                ),
            },
            coords={
                "sweeps": np.linspace(-88e6, 88e6, n_sweeps),
            },
            attrs={
                "Header.Toltec.RoachIndex": 1,
                "Header.Toltec.ObsNum": 54321,
                "Header.Toltec.SubObsNum": 2,
                "Header.Toltec.ScanNum": 1,
                "Header.Toltec.SampleFreq": 488.0,
                "Header.Toltec.NumSweepSteps": n_sweeps,
                "Header.Toltec.Master": 1,
                "Header.Toltec.ObsType": 2,  # Tune
                "is_multi_block": True,
                "n_blocks": n_blocks,
            },
        )

        # Get metadata via accessor
        meta_list = ds.toltec_kids.meta

        # Check it's a list
        assert isinstance(meta_list, list)
        assert len(meta_list) == n_blocks

        # Check each block
        for _i, meta in enumerate(meta_list):
            assert isinstance(meta, ToltecSweepMetadata)
            # Raw obs fields (same for all blocks)
            assert meta.roach == 1
            assert meta.obsnum == 54321
            assert meta.array_name == "a1100"  # roach 1 maps to a1100 (range 0-6)
            # Block-specific
            assert meta.n_sweeps == 1
            assert meta.n_blocks == n_blocks
            # Each block has different f_lo_center
            assert meta.f_lo_center is not None

    def test_get_metadata_method(self, rng):
        """Test get_metadata() method on mapper directly."""
        n_chans = 500
        n_sweeps = 100

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (
                    ["iqlen", "sweep"],
                    rng.standard_normal((n_chans, n_sweeps)),
                ),
                "Data.Toltec.Qs": (
                    ["iqlen", "sweep"],
                    rng.standard_normal((n_chans, n_sweeps)),
                ),
            },
            coords={
                "sweep": np.linspace(-50e6, 50e6, n_sweeps),
            },
            attrs={
                "Header.Toltec.RoachIndex": 2,
                "Header.Toltec.SampleFreq": 488.0,
                "Header.Toltec.ObsNum": 99999,
                "Header.Toltec.ObsType": 0,  # VnaSweep
            },
        )

        # ToltecKidsFullMapper (core.ToltecKidsIOMapper) has get_metadata()
        mapper = ToltecKidsFullMapper.from_data_source(ds)
        meta = mapper.get_metadata(ds)

        assert isinstance(meta, ToltecSweepMetadata)
        assert meta.roach == 2
        assert meta.array_name == "a1100"  # roach 2 maps to a1100 (range 0-6)
        assert meta.obsnum == 99999

    def test_get_metadata_specific_block(self, rng):
        """Test get_metadata(block=N) returns single metadata for specific block."""
        n_chans = 100
        n_sweeps = 50
        n_blocks = 3

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((n_chans, n_blocks, n_sweeps)),
                ),
                "Data.Toltec.Qs": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((n_chans, n_blocks, n_sweeps)),
                ),
                "Data.Toltec.LoFreq": (
                    ["block", "sweeps"],
                    np.array(
                        [
                            np.linspace(4e9, 4.5e9, n_sweeps),
                            np.linspace(5e9, 5.5e9, n_sweeps),
                            np.linspace(6e9, 6.5e9, n_sweeps),
                        ],
                    ),
                ),
            },
            coords={
                "sweeps": np.linspace(-25e6, 25e6, n_sweeps),
            },
            attrs={
                "Header.Toltec.RoachIndex": 0,
                "Header.Toltec.SampleFreq": 488.0,
                "Header.Toltec.ObsType": 2,  # Tune sweep
                "Header.Toltec.NumSweepSteps": n_sweeps,
                "is_multi_block": True,
            },
        )

        # ToltecKidsFullMapper (core.ToltecKidsIOMapper) has get_metadata()
        mapper = ToltecKidsFullMapper.from_data_source(ds)
        meta_block_1 = mapper.get_metadata(ds, block=1)

        assert isinstance(meta_block_1, ToltecSweepMetadata)
        assert meta_block_1.n_blocks == n_blocks
        # f_lo_center should be from block 1 (around 5.25e9)
        assert 5e9 < meta_block_1.f_lo_center < 5.5e9

    def test_metadata_with_minimal_fields(self, rng):
        """Test metadata construction with minimal required fields."""
        # Minimal dataset
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["iqlen", "time"], rng.standard_normal((100, 1000))),
                "Data.Toltec.Qs": (["iqlen", "time"], rng.standard_normal((100, 1000))),
            },
            attrs={
                "Header.Toltec.RoachIndex": 0,
                "Header.Toltec.SampleFreq": 488.0,  # Add f_smp (required)
            },
        )

        meta = ds.toltec_kids.meta

        # Should still construct successfully
        assert isinstance(meta, (ToltecSweepMetadata, ToltecTimeStreamMetadata))
        assert meta.roach == 0


class TestMetadataEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_block_index_raises(self, rng):
        """Test get_metadata with invalid block index raises ValueError."""
        n_blocks = 2
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((100, n_blocks, 50)),
                ),
                "Data.Toltec.Qs": (
                    ["iqlen", "block", "sweeps"],
                    rng.standard_normal((100, n_blocks, 50)),
                ),
            },
            attrs={
                "is_multi_block": True,
                "Header.Toltec.SampleFreq": 488.0,
            },
        )

        mapper = ToltecKidsFullMapper.from_data_source(ds)
        with pytest.raises(ValueError, match="Block .* out of range"):
            mapper.get_metadata(ds, block=5)

    def test_metadata_caching(self, rng):
        """Test metadata is cached via functools.cached_property."""
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["iqlen", "sweeps"], rng.standard_normal((100, 50))),
                "Data.Toltec.Qs": (["iqlen", "sweeps"], rng.standard_normal((100, 50))),
            },
            attrs={
                "Header.Toltec.RoachIndex": 0,
                "Header.Toltec.SampleFreq": 488.0,
            },
        )

        # Access metadata twice
        meta1 = ds.toltec_kids.meta
        meta2 = ds.toltec_kids.meta

        # Should be the same object (cached)
        assert meta1 is meta2
