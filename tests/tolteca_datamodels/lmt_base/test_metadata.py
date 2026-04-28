"""Unit tests for LMT base metadata models."""

from __future__ import annotations

import dataclasses

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic_core import ValidationError

from tolteca_datamodels.lmt_base import LmtRawObsMetadata


class TestLmtObsQuartetMixin:
    """Tests for LmtObsQuartetMixin through LmtRawObsMetadata."""

    def test_keyword_only_enforcement(self):
        """Test that all fields are keyword-only."""
        # Create with keyword arguments - should work
        meta = LmtRawObsMetadata(
            master="lmt",
            obsnum=100,
            subobsnum=2,
            scannum=5,
            instru="test",
            interface="test0",
            t0=Time("2024-01-01T00:00:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00"),  # pyright: ignore[reportArgumentType]
            t_exp=3600 * u.s,
        )
        assert meta.master == "lmt"
        assert meta.obsnum == 100

    def test_field_types(self):
        """Test that quartet fields have correct types."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=100,
            subobsnum=2,
            scannum=5,
            instru="test",
            interface="test0",
            t0=Time("2024-01-01T00:00:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00"),  # pyright: ignore[reportArgumentType]
            t_exp=3600 * u.s,
        )
        assert isinstance(meta.master, str)
        assert isinstance(meta.obsnum, int)
        assert isinstance(meta.subobsnum, int)
        assert isinstance(meta.scannum, int)

    def test_frozen_mixin(self):
        """Test that instances are immutable."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=100,
            subobsnum=2,
            scannum=5,
            instru="test",
            interface="test0",
            t0=Time("2024-01-01T00:00:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00"),  # pyright: ignore[reportArgumentType]
            t_exp=3600 * u.s,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.master = "new_value"  # type: ignore[attr-defined]


class TestLmtInstruMixin:
    """Tests for LmtInstruMixin through LmtRawObsMetadata."""

    def test_required_fields(self):
        """Test that instru and interface are required."""
        with pytest.raises(ValidationError):
            LmtRawObsMetadata(
                master="lmt",
                obsnum=1,
                subobsnum=0,
                scannum=0,
                # Missing instru and interface
                t0=Time.now(),
                t1=Time.now(),
                t_exp=60 * u.s,
            )  # type: ignore[arg-type]

    def test_optional_instru_component(self):
        """Test that instru_component is optional."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=Time("2024-01-01T00:00:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00"),  # pyright: ignore[reportArgumentType]
            t_exp=60 * u.s,
        )
        assert meta.instru_component is None

    def test_instru_component_can_be_set(self):
        """Test that instru_component can be provided."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            instru_component="subsystem1",
            t0=Time("2024-01-01T00:00:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00"),  # pyright: ignore[reportArgumentType]
            t_exp=60 * u.s,
        )
        assert meta.instru_component == "subsystem1"


class TestLmtRawObsMetadata:
    """Tests for LmtRawObsMetadata dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating metadata with all fields."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            instru="toltec",
            interface="toltec0",
            master="lmt",
            obsnum=12345,
            subobsnum=1,
            scannum=3,
            t0=Time("2024-01-15T10:30:00"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-15T10:35:00"),  # pyright: ignore[reportArgumentType]
            t_exp=300 * u.s,
        )

        assert meta.instru == "toltec"
        assert meta.interface == "toltec0"
        assert meta.master == "lmt"
        assert meta.obsnum == 12345
        assert meta.subobsnum == 1
        assert meta.scannum == 3
        assert isinstance(meta.t0, Time)
        assert isinstance(meta.t1, Time)
        assert meta.t_exp == 300 * u.s

    def test_time_field_types(self):
        """Test that time fields are properly typed."""
        t0 = Time("2024-01-01T00:00:00")
        t1 = Time("2024-01-01T01:00:00")

        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=t0,  # pyright: ignore[reportArgumentType]
            t1=t1,  # pyright: ignore[reportArgumentType]
            t_exp=60 * u.s,
        )

        assert isinstance(meta.t0, Time)
        assert isinstance(meta.t1, Time)
        assert meta.t0 == t0
        assert meta.t1 == t1

    def test_quantity_field_types(self):
        """Test that quantity fields are properly typed."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=Time.now(),  # pyright: ignore[reportArgumentType]
            t1=Time.now(),  # pyright: ignore[reportArgumentType]
            t_exp=120 * u.s,
        )

        assert isinstance(meta.t_exp, u.Quantity)
        assert meta.t_exp.unit is not None
        assert meta.t_exp.unit.physical_type == "time"

    def test_quantity_field_unit_conversion(self):
        """Test that quantity fields handle unit conversion."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=Time.now(),  # pyright: ignore[reportArgumentType]
            t1=Time.now(),  # pyright: ignore[reportArgumentType]
            t_exp=2 * u.min,
        )

        assert meta.t_exp == 2 * u.min
        assert meta.t_exp.to(u.s) == 120 * u.s

    def test_frozen_metadata(self):
        """Test that metadata instances are immutable."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=Time.now(),  # pyright: ignore[reportArgumentType]
            t1=Time.now(),  # pyright: ignore[reportArgumentType]
            t_exp=60 * u.s,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.obsnum = 999  # type: ignore[attr-defined]

    def test_multiple_inheritance(self):
        """Test that multiple inheritance works correctly."""
        meta: LmtRawObsMetadata = LmtRawObsMetadata(
            master="lmt",
            obsnum=1,
            subobsnum=0,
            scannum=0,
            instru="test",
            interface="test0",
            t0=Time.now(),  # pyright: ignore[reportArgumentType]
            t1=Time.now(),  # pyright: ignore[reportArgumentType]
            t_exp=60 * u.s,
        )

        # Check fields from LmtObsQuartetMixin
        assert hasattr(meta, "master")
        assert hasattr(meta, "obsnum")
        assert hasattr(meta, "subobsnum")
        assert hasattr(meta, "scannum")

        # Check fields from LmtInstruMixin
        assert hasattr(meta, "interface")
        assert hasattr(meta, "instru")
        assert hasattr(meta, "instru_component")

        # Check own fields
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t1")
        assert hasattr(meta, "t_exp")

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            LmtRawObsMetadata(
                master="lmt",
                obsnum=1,
                # Missing many required fields
            )  # type: ignore[arg-type]
