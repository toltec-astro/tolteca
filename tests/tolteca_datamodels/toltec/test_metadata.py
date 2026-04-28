"""Unit tests for TolTEC metadata models."""

from __future__ import annotations

from typing import Any

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic_core import ValidationError

from tolteca_datamodels.toltec import (
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimeStreamMetadata,
)


@pytest.fixture
def base_obs_fields() -> dict[str, Any]:
    """Common observation fields for all TolTEC metadata."""
    return {
        "obsnum": 12345,
        "subobsnum": 1,
        "scannum": 2,
        "interface": "toltec0",
        "t0": Time("2024-01-01T00:00:00", format="isot", scale="utc"),
        "t1": Time("2024-01-01T00:01:00", format="isot", scale="utc"),
        "t_exp": 60 * u.s,
    }


class TestToltecRawObsMetadata:
    """Test ToltecRawObsMetadata."""

    def test_creation_minimal(self, base_obs_fields: dict[str, Any]) -> None:
        """Test creation with minimal required fields."""
        meta = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        assert meta.obsnum == 12345
        assert meta.master == "ics"
        assert meta.instru == "toltec"
        assert meta.instru_component is None
        assert meta.roach is None
        assert meta.array_name is None

    def test_instru_field_accepts_multiple_values(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that instru accepts both 'toltec' and 'lmt'."""
        meta1 = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        assert meta1.instru == "toltec"

        meta2 = ToltecRawObsMetadata(**base_obs_fields, master="tcs", instru="lmt")
        assert meta2.instru == "lmt"

    def test_instru_component_options(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that instru_component accepts valid literals."""
        for component in ["roach", "hwpr", "hk", "tel"]:
            meta = ToltecRawObsMetadata(
                **base_obs_fields,
                master="ics",
                instru="toltec",
                instru_component=component,
            )
            assert meta.instru_component == component

    def test_roach_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test roach field accepts valid roach indices."""
        for roach_idx in [0, 5, 12]:
            meta = ToltecRawObsMetadata(
                **base_obs_fields,
                master="clip",
                instru="toltec",
                roach=roach_idx,
            )
            assert meta.roach == roach_idx

    def test_roach_defaults_to_none(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that roach defaults to None."""
        meta = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        assert meta.roach is None

    def test_array_name_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test array_name field accepts valid array names."""
        for array in ["a1100", "a1400", "a2000"]:
            meta = ToltecRawObsMetadata(
                **base_obs_fields,
                master="ics",
                instru="toltec",
                array_name=array,
            )
            assert meta.array_name == array

    def test_array_name_defaults_to_none(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that array_name defaults to None."""
        meta = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        assert meta.array_name is None

    def test_master_field_type_hint(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that master field accepts ToltecMasterNameT values."""
        for master in ["tcs", "ics", "clip"]:
            meta = ToltecRawObsMetadata(
                **base_obs_fields,
                master=master,
                instru="toltec",
            )
            assert meta.master == master

    def test_inheritance_from_lmt_raw_obs(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that ToltecRawObsMetadata inherits from LmtRawObsMetadata."""
        from tolteca_datamodels.lmt_base import LmtRawObsMetadata

        meta = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        assert isinstance(meta, LmtRawObsMetadata)
        # Should have all base fields
        assert hasattr(meta, "obsnum")
        assert hasattr(meta, "subobsnum")
        assert hasattr(meta, "scannum")
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t1")
        assert hasattr(meta, "t_exp")

    def test_frozen(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that metadata is immutable."""
        meta = ToltecRawObsMetadata(**base_obs_fields, master="ics", instru="toltec")
        with pytest.raises((AttributeError, TypeError)):
            meta.obsnum = 99999  # type: ignore[misc]
        with pytest.raises((AttributeError, TypeError)):
            meta.roach = 5  # type: ignore[misc]

    def test_keyword_only_enforcement(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that all fields are keyword-only."""
        with pytest.raises((TypeError, ValidationError)):
            ToltecRawObsMetadata(
                12345,  # obsnum  # type: ignore[arg-type]
                1,  # subobsnum
                2,  # scannum
                "toltec0",  # interface
                Time.now(),
                Time.now(),
                60 * u.s,
                "lmt",
                "toltec",
            )  # ty: ignore[missing-argument]


class TestToltecSweepMetadata:
    """Test ToltecSweepMetadata."""

    def test_creation(self, base_obs_fields: dict[str, Any]) -> None:
        """Test sweep metadata creation."""
        meta = ToltecSweepMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_sweeps=401,
            n_chans=512,
            n_times=1000,
            n_sweepsteps=100,
            f_smp=488.28125 * u.Hz,
        )
        assert meta.n_sweeps == 401
        assert meta.n_chans == 512
        assert meta.n_times == 1000
        assert meta.n_sweepsteps == 100
        assert meta.f_smp.value == 488.28125
        assert meta.f_smp.unit == u.Hz

    def test_frequency_quantity_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that f_smp accepts frequency quantities."""
        meta = ToltecSweepMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_sweeps=100,
            n_chans=256,
            n_times=500,
            n_sweepsteps=50,
            f_smp=1000 * u.Hz,
        )
        assert meta.f_smp.unit is not None
        assert meta.f_smp.unit.physical_type == "frequency"
        # Test unit conversion
        assert meta.f_smp.to(u.kHz).value == 1.0

    def test_inheritance_from_raw_obs(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that sweep metadata inherits from ToltecRawObsMetadata."""
        meta = ToltecSweepMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_sweeps=401,
            n_chans=512,
            n_times=1000,
            n_sweepsteps=100,
            f_smp=488.28125 * u.Hz,
        )
        assert isinstance(meta, ToltecRawObsMetadata)
        # Should have roach field
        assert meta.roach == 0
        # Should have base fields
        assert meta.obsnum == 12345

    def test_frozen(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that sweep metadata is immutable."""
        meta = ToltecSweepMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_sweeps=401,
            n_chans=512,
            n_times=1000,
            n_sweepsteps=100,
            f_smp=488.28125 * u.Hz,
        )
        with pytest.raises((AttributeError, TypeError)):
            meta.n_sweeps = 500  # type: ignore[misc]

    def test_missing_required_sweep_fields(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that missing sweep-specific fields raise validation error."""
        with pytest.raises(ValidationError):
            ToltecSweepMetadata(
                **base_obs_fields,
                master="lmt",
                instru="toltec",
                roach=0,
                # Missing: n_sweeps, n_chans, f_smp
            )


class TestToltecTimeStreamMetadata:
    """Test ToltecTimeStreamMetadata."""

    def test_creation(self, base_obs_fields: dict[str, Any]) -> None:
        """Test timestream metadata creation."""
        meta = ToltecTimeStreamMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        assert meta.n_times == 10000
        assert meta.n_chans == 512
        assert meta.f_smp.value == 488.28125
        assert meta.f_smp.unit == u.Hz

    def test_sample_rate_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that f_smp accepts sample rate frequencies."""
        meta = ToltecTimeStreamMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_times=5000,
            n_chans=256,
            f_smp=500 * u.Hz,
        )
        assert meta.f_smp.unit is not None
        assert meta.f_smp.unit.physical_type == "frequency"
        # Test different frequency units
        assert meta.f_smp.to(u.kHz).value == 0.5

    def test_inheritance_from_raw_obs(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that timestream metadata inherits from ToltecRawObsMetadata."""
        meta = ToltecTimeStreamMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        assert isinstance(meta, ToltecRawObsMetadata)
        # Should have roach field
        assert meta.roach == 0
        # Should have all base fields
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t_exp")

    def test_t_exp_from_base_class(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that t_exp is available from base class."""
        meta = ToltecTimeStreamMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        assert meta.t_exp == 60 * u.s
        assert meta.t_exp.unit is not None
        assert meta.t_exp.unit.physical_type == "time"

    def test_frozen(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that timestream metadata is immutable."""
        meta = ToltecTimeStreamMetadata(
            **base_obs_fields,
            master="clip",
            instru="toltec",
            roach=0,
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        with pytest.raises((AttributeError, TypeError)):
            meta.n_samples = 20000  # type: ignore[misc]

    def test_missing_required_timestream_fields(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that missing timestream-specific fields raise validation error."""
        with pytest.raises(ValidationError):
            ToltecTimeStreamMetadata(
                **base_obs_fields,
                master="lmt",
                instru="toltec",
                roach=0,
                # Missing: n_samples, n_chans, f_smp
            )
