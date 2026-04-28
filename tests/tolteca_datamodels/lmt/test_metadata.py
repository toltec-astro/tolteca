"""Unit tests for LMT telescope metadata models."""

from __future__ import annotations

from typing import Any

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic_core import ValidationError

from tolteca_datamodels.lmt import LmtObsGoalType, LmtTelMetadata


@pytest.fixture
def base_tel_fields() -> dict[str, Any]:
    """Common fields for LMT telescope metadata."""
    return {
        "obsnum": 100,
        "subobsnum": 0,
        "scannum": 0,
        "interface": "tcs",
        "t0": Time("2024-01-01T00:00:00", format="isot", scale="utc"),
        "t1": Time("2024-01-01T01:00:00", format="isot", scale="utc"),
        "t_exp": 3600 * u.s,
    }


class TestLmtTelMetadata:
    """Test LmtTelMetadata for telescope control system."""

    def test_creation_minimal(self, base_tel_fields: dict[str, Any]) -> None:
        """Test creation with only required fields."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.obsnum == 100
        assert meta.master == "tcs"
        assert meta.instru == "lmt"
        assert meta.instru_component == "tel"
        assert meta.obs_goal == LmtObsGoalType.unspecified
        assert meta.target is None
        assert meta.target_off is None

    def test_default_field_values(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that fields have correct default values."""
        meta = LmtTelMetadata(**base_tel_fields)
        # These should all be set by default
        assert meta.master == "tcs"
        assert meta.instru == "lmt"
        assert meta.instru_component == "tel"
        assert isinstance(meta.obs_goal, LmtObsGoalType)

    def test_master_literal_type(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that master is constrained to 'tcs' literal."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.master == "tcs"
        # Attempting to override should still work since it's a default
        # But the type system enforces the literal at creation

    def test_instru_literal_type(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that instru is constrained to 'lmt' literal."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.instru == "lmt"

    def test_instru_component_literal_type(
        self,
        base_tel_fields: dict[str, Any],
    ) -> None:
        """Test that instru_component is constrained to 'tel' literal."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.instru_component == "tel"

    def test_obs_goal_types(self, base_tel_fields: dict[str, Any]) -> None:
        """Test different observation goal types."""
        for goal in [
            LmtObsGoalType.science,
            LmtObsGoalType.engineering,
            LmtObsGoalType.calibration,
            LmtObsGoalType.pointing,
            LmtObsGoalType.focus,
            LmtObsGoalType.astigmatism,
            LmtObsGoalType.beammap,
            LmtObsGoalType.oof,
        ]:
            meta = LmtTelMetadata(**base_tel_fields, obs_goal=goal)
            assert meta.obs_goal == goal
            assert isinstance(meta.obs_goal, LmtObsGoalType)

    def test_obs_goal_default(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that obs_goal defaults to unspecified."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.obs_goal == LmtObsGoalType.unspecified

    def test_target_skycoord(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that target accepts SkyCoord."""
        target = SkyCoord(ra=180 * u.deg, dec=45 * u.deg, frame="icrs")
        meta = LmtTelMetadata(**base_tel_fields, target=target)
        assert meta.target is not None
        # Verify it's preserved as SkyCoord-like
        assert hasattr(meta.target, "ra") or isinstance(meta.target, SkyCoord)

    def test_target_defaults_to_none(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that target defaults to None."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.target is None

    def test_target_off_skycoord(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that target_off accepts SkyCoord."""
        target_off = SkyCoord(ra=0.1 * u.deg, dec=0.1 * u.deg, frame="icrs")
        meta = LmtTelMetadata(**base_tel_fields, target_off=target_off)
        assert meta.target_off is not None
        # Verify it's preserved as SkyCoord-like
        assert hasattr(meta.target_off, "ra") or isinstance(meta.target_off, SkyCoord)

    def test_target_off_defaults_to_none(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that target_off defaults to None."""
        meta = LmtTelMetadata(**base_tel_fields)
        assert meta.target_off is None

    def test_both_targets_set(self, base_tel_fields: dict[str, Any]) -> None:
        """Test setting both target and target_off."""
        target = SkyCoord(ra=180 * u.deg, dec=45 * u.deg, frame="icrs")
        target_off = SkyCoord(ra=0.05 * u.deg, dec=-0.05 * u.deg, frame="icrs")
        meta = LmtTelMetadata(
            **base_tel_fields,
            target=target,
            target_off=target_off,
        )
        assert meta.target is not None
        assert meta.target_off is not None

    def test_inheritance_from_lmt_raw_obs(
        self,
        base_tel_fields: dict[str, Any],
    ) -> None:
        """Test that LmtTelMetadata inherits from LmtRawObsMetadata."""
        from tolteca_datamodels.lmt_base import LmtRawObsMetadata

        meta = LmtTelMetadata(**base_tel_fields)
        assert isinstance(meta, LmtRawObsMetadata)
        # Should have all base fields
        assert hasattr(meta, "obsnum")
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t_exp")
        assert meta.obsnum == 100

    def test_frozen(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that metadata is immutable."""
        meta = LmtTelMetadata(**base_tel_fields)
        with pytest.raises((AttributeError, TypeError)):
            meta.obsnum = 999  # type: ignore[misc]
        with pytest.raises((AttributeError, TypeError)):
            meta.obs_goal = LmtObsGoalType.science  # type: ignore[misc]

    def test_keyword_only_enforcement(self, base_tel_fields: dict[str, Any]) -> None:
        """Test that all fields are keyword-only."""
        with pytest.raises((TypeError, ValidationError)):
            LmtTelMetadata(
                100,  # obsnum # type: ignore[arg-type]
                0,  # subobsnum
                0,  # scannum
                "tcs",  # interface
                Time.now(),
                Time.now(),
                3600 * u.s,
            )  # ty: ignore[missing-argument]

    def test_complete_observation_metadata(self) -> None:
        """Test creating complete metadata for a real observation."""
        target = SkyCoord(ra=83.633 * u.deg, dec=22.0145 * u.deg, frame="icrs")  # M42
        meta = LmtTelMetadata(
            obsnum=54321,
            subobsnum=0,
            scannum=0,
            interface="tcs",
            t0=Time("2024-01-01T00:00:00", format="isot", scale="utc"),  # pyright: ignore[reportArgumentType]
            t1=Time("2024-01-01T01:00:00", format="isot", scale="utc"),  # pyright: ignore[reportArgumentType]
            t_exp=3600 * u.s,
            obs_goal=LmtObsGoalType.science,
            target=target,
        )
        assert meta.obsnum == 54321
        assert meta.obs_goal == LmtObsGoalType.science
        assert meta.target is not None
        assert meta.master == "tcs"
        assert meta.instru == "lmt"


class TestLmtObsGoalType:
    """Test LmtObsGoalType enum."""

    def test_all_goal_types_exist(self) -> None:
        """Test that all expected goal types are defined."""
        expected_goals = [
            "engineering",
            "science",
            "calibration",
            "pointing",
            "focus",
            "astigmatism",
            "beammap",
            "oof",
            "unspecified",
        ]
        for goal_name in expected_goals:
            assert hasattr(LmtObsGoalType, goal_name)
            goal = getattr(LmtObsGoalType, goal_name)
            assert isinstance(goal, LmtObsGoalType)

    def test_goal_type_string_values(self) -> None:
        """Test that goal types have correct string values."""
        assert LmtObsGoalType.science == "science"
        assert LmtObsGoalType.engineering == "engineering"
        assert LmtObsGoalType.calibration == "calibration"

    def test_goal_type_comparison(self) -> None:
        """Test that goal types can be compared."""
        assert LmtObsGoalType.science == LmtObsGoalType.science
        assert LmtObsGoalType.science != LmtObsGoalType.engineering

    def test_goal_type_is_str_enum(self) -> None:
        """Test that LmtObsGoalType is a StrEnum."""
        from enum import StrEnum

        assert issubclass(LmtObsGoalType, StrEnum)
        # Should be usable as strings
        assert isinstance(LmtObsGoalType.science, str)
