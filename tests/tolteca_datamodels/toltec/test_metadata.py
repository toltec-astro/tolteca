"""Unit tests for TolTEC metadata models."""

from __future__ import annotations

from typing import Any, assert_type

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic import ValidationError

from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode


@pytest.fixture
def base_obs_fields(kids_filepath_orig: str) -> dict[str, Any]:
    """Common observation fields for all TolTEC metadata."""
    return {
        "instru": "toltec",
        "filepath_orig": kids_filepath_orig,
        "obsnum": 12345,
        "subobsnum": 1,
        "scannum": 2,
        "interface": "toltec0",
        "interface_kind": "roach",
        "readout_code": 0,
        "roach": 0,
        "nw": 0,
        "array": 0,
        "array_name": "a1100",
        "t0": Time("2024-01-01T00:00:00", format="isot", scale="utc"),
        "t1": Time("2024-01-01T00:01:00", format="isot", scale="utc"),
        "t_int": 60 * u.s,
    }


class TestToltecRawObsMetadata:
    """Test ToltecRawObsMetadata."""

    def test_creation_minimal(self, base_obs_fields: dict[str, Any]) -> None:
        """Test creation with minimal required fields."""
        meta = ToltecRawObsMetadata(
            n_blocks_max=1,
            n_times=1000,
            n_chans=256,
            f_smp=500 * u.Hz,
            **base_obs_fields,
            master="ics",
            master_code=ToltecDBRawObsMasterCode.ICS,
            data_models=frozenset({"toltec"}),
        )
        assert_type(meta.data_models, frozenset[str])
        assert meta.obsnum == 12345
        assert meta.master_code == ToltecDBRawObsMasterCode.ICS
        assert meta.data_models == {"toltec"}
        assert meta.interface_kind == "roach"
        assert meta.roach == 0
        assert meta.array_name == "a1100"

    @pytest.mark.parametrize("instru", [None, "", "lmt", "other"])
    def test_instru_rejects_other_instruments(
        self,
        base_obs_fields: dict[str, Any],
        instru: Any,
    ) -> None:
        with pytest.raises(ValidationError, match="instru"):
            ToltecRawObsMetadata(
                n_blocks_max=1,
                n_times=1000,
                n_chans=256,
                f_smp=500 * u.Hz,
                **(base_obs_fields | {"instru": instru}),
                master="ics",
                master_code=ToltecDBRawObsMasterCode.ICS,
                data_models=frozenset({"toltec"}),
            )

    def test_roach_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test roach field accepts valid roach indices."""
        for roach_idx in [0, 5, 12]:
            values: dict[str, Any] = base_obs_fields | {"roach": roach_idx}
            meta = ToltecRawObsMetadata(
                n_blocks_max=1,
                n_times=1000,
                n_chans=256,
                f_smp=500 * u.Hz,
                **values,
                master="clip",
                master_code=ToltecDBRawObsMasterCode.CLIP,
                data_models=frozenset({"toltec"}),
            )
            assert meta.roach == roach_idx

    def test_array_name_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test array_name field accepts valid array names."""
        for array in ["a1100", "a1400", "a2000"]:
            values: dict[str, Any] = base_obs_fields | {"array_name": array}
            meta = ToltecRawObsMetadata(
                n_blocks_max=1,
                n_times=1000,
                n_chans=256,
                f_smp=500 * u.Hz,
                **values,
                master="ics",
                master_code=ToltecDBRawObsMasterCode.ICS,
                data_models=frozenset({"toltec"}),
            )
            assert meta.array_name == array

    @pytest.mark.parametrize("field", ["roach", "nw", "array", "array_name"])
    def test_readout_identity_requires_explicit_values(self, base_obs_fields, field):
        values = base_obs_fields | {
            "n_blocks_max": 1,
            "n_times": 1000,
            "n_chans": 256,
            "f_smp": 500 * u.Hz,
            "master": "ics",
            "master_code": ToltecDBRawObsMasterCode.ICS,
            "data_models": frozenset({"toltec"}),
        }
        meta = ToltecRawObsMetadata.model_validate(values | {field: None})
        assert getattr(meta, field) is None
        del values[field]
        with pytest.raises(ValidationError) as exc:
            ToltecRawObsMetadata.model_validate(values)
        [error] = exc.value.errors()
        assert (error["loc"], error["type"]) == ((field,), "missing")

    def test_master_field_type_hint(self, base_obs_fields: dict[str, Any]) -> None:
        """Master codes and labels retain their independent value types."""
        for master_code in ToltecDBRawObsMasterCode:
            meta = ToltecRawObsMetadata(
                n_blocks_max=1,
                n_times=1000,
                n_chans=256,
                f_smp=500 * u.Hz,
                **base_obs_fields,
                master=ToltecDBRawObsMasterCode.get_master(master_code),
                master_code=master_code,
                data_models=frozenset({"toltec"}),
            )
            assert meta.master_code == master_code
            assert meta.master == ToltecDBRawObsMasterCode.get_master(master_code)

    def test_inheritance_from_lmt_raw_obs(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that ToltecRawObsMetadata inherits from LmtRawObsMetadata."""
        from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata

        meta = ToltecRawObsMetadata(
            n_blocks_max=1,
            n_times=1000,
            n_chans=256,
            f_smp=500 * u.Hz,
            **base_obs_fields,
            master="ics",
            master_code=ToltecDBRawObsMasterCode.ICS,
            data_models=frozenset({"toltec"}),
        )
        assert isinstance(meta, LmtRawObsMetadata)
        # Should have all base fields
        assert hasattr(meta, "obsnum")
        assert hasattr(meta, "subobsnum")
        assert hasattr(meta, "scannum")
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t1")
        assert hasattr(meta, "t_int")


class TestToltecSweepMetadata:
    """Test ToltecSweepMetadata."""

    def test_creation(self, base_obs_fields: dict[str, Any]) -> None:
        """Test sweep metadata creation."""
        meta = ToltecSweepMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
            n_chans=512,
            n_times=1000,
            n_sweepsteps=100,
            f_smp=488.28125 * u.Hz,
        )
        assert meta.n_blocks == 1
        assert meta.n_chans == 512
        assert meta.n_times == 1000
        assert meta.n_sweepsteps == 100
        assert meta.f_smp.value == 488.28125
        assert meta.f_smp.unit == u.Hz

    def test_frequency_quantity_field(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that f_smp accepts frequency quantities."""
        meta = ToltecSweepMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
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
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
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
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
            n_chans=512,
            n_times=1000,
            n_sweepsteps=100,
            f_smp=488.28125 * u.Hz,
        )
        with pytest.raises(ValidationError, match="frozen"):
            setattr(meta, "n_blocks", 500)  # noqa: B010 -- Test frozen validation.

    def test_missing_required_sweep_fields(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that missing sweep-specific fields raise validation error."""
        with pytest.raises(ValidationError):
            ToltecSweepMetadata(
                **base_obs_fields,
                master="tcs",
                master_code=ToltecDBRawObsMasterCode.TCS,
                data_models=frozenset({"toltec"}),
                # Missing: n_blocks, block, n_chans, f_smp
            )


class TestToltecTimestreamMetadata:
    """Test ToltecTimestreamMetadata."""

    def test_creation(self, base_obs_fields: dict[str, Any]) -> None:
        """Test timestream metadata creation."""
        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
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
        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
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
        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        assert isinstance(meta, ToltecRawObsMetadata)
        # Should have roach field
        assert meta.roach == 0
        # Should have all base fields
        assert hasattr(meta, "t0")
        assert hasattr(meta, "t_int")

    def test_t_int_from_base_class(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that t_int is available from base class."""
        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        assert meta.t_int == 60 * u.s
        assert meta.t_int.unit is not None
        assert meta.t_int.unit.physical_type == "time"

    def test_frozen(self, base_obs_fields: dict[str, Any]) -> None:
        """Test that timestream metadata is immutable."""
        meta = ToltecTimestreamMetadata(
            n_blocks_max=1,
            n_blocks=1,
            block=0,
            **base_obs_fields,
            master="clip",
            master_code=ToltecDBRawObsMasterCode.CLIP,
            data_models=frozenset({"toltec"}),
            n_times=10000,
            n_chans=512,
            f_smp=488.28125 * u.Hz,
        )
        with pytest.raises(ValidationError, match="frozen"):
            setattr(meta, "n_times", 20000)  # noqa: B010 -- Test frozen validation.

    def test_missing_required_timestream_fields(
        self,
        base_obs_fields: dict[str, Any],
    ) -> None:
        """Test that missing timestream-specific fields raise validation error."""
        with pytest.raises(ValidationError):
            ToltecTimestreamMetadata(
                **base_obs_fields,
                master="tcs",
                master_code=ToltecDBRawObsMasterCode.TCS,
                data_models=frozenset({"toltec"}),
                # Missing: n_samples, n_chans, f_smp
            )
