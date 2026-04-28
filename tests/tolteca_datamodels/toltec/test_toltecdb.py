"""Unit tests for TolTEC database constants and mappings."""

from __future__ import annotations

import pytest

from tolteca_datamodels.toltec.toltecdb import (
    ToltecDBRawObsMaster,
    ToltecDBRawObsType,
)
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecMasterType


class TestToltecDBRawObsMaster:
    """Test ToltecDBRawObsMaster enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsMaster.TCS == 0
        assert ToltecDBRawObsMaster.ICS == 1
        assert ToltecDBRawObsMaster.CLIP == 2

    def test_get_master_type_tcs(self) -> None:
        """Test getting master type for TCS."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.TCS)
        assert master_type == ToltecMasterType.tcs

    def test_get_master_type_ics(self) -> None:
        """Test getting master type for ICS."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.ICS)
        assert master_type == ToltecMasterType.ics

    def test_get_master_type_clip(self) -> None:
        """Test getting master type for CLIP."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.CLIP)
        assert master_type == ToltecMasterType.clip

    def test_all_masters_mapped(self) -> None:
        """Test that all master enum values have master type mappings."""
        for master in ToltecDBRawObsMaster:
            master_type = ToltecDBRawObsMaster.get_master_type(master)
            assert master_type is not None
            assert isinstance(master_type, ToltecMasterType)


class TestToltecDBRawObsType:
    """Test ToltecDBRawObsType enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsType.Nominal == 0
        assert ToltecDBRawObsType.Timestream == 1
        assert ToltecDBRawObsType.VNA == 2
        assert ToltecDBRawObsType.TARG == 3
        assert ToltecDBRawObsType.TUNE == 4

    def test_get_data_kind_nominal(self) -> None:
        """Test getting data kind for nominal observation."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Nominal)
        assert data_kind == ToltecDataKind.RawTimeStream

    def test_get_data_kind_timestream(self) -> None:
        """Test getting data kind for timestream."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Timestream)
        assert data_kind == ToltecDataKind.RawTimeStream

    def test_get_data_kind_vna(self) -> None:
        """Test getting data kind for VNA sweep."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.VNA)
        assert data_kind == ToltecDataKind.VnaSweep

    def test_get_data_kind_targ(self) -> None:
        """Test getting data kind for target sweep."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TARG)
        assert data_kind == ToltecDataKind.TargetSweep

    def test_get_data_kind_tune(self) -> None:
        """Test getting data kind for tune."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TUNE)
        assert data_kind == ToltecDataKind.Tune

    def test_all_obs_types_mapped(self) -> None:
        """Test that all obs type enum values have data kind mappings."""
        for obs_type in ToltecDBRawObsType:
            data_kind = ToltecDBRawObsType.get_data_kind(obs_type)
            assert data_kind is not None
            assert isinstance(data_kind, ToltecDataKind)

    def test_obs_type_to_data_kind_consistency(self) -> None:
        """Test consistency between observation type and data kind."""
        # VNA and TARG should map to sweep types
        vna_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.VNA)
        targ_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TARG)
        assert vna_kind in (ToltecDataKind.VnaSweep, ToltecDataKind.RawSweep)
        assert targ_kind in (ToltecDataKind.TargetSweep, ToltecDataKind.RawSweep)

        # Nominal and Timestream should map to timestream types
        nominal_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Nominal)
        ts_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Timestream)
        assert nominal_kind in (ToltecDataKind.RawTimeStream,)
        assert ts_kind in (ToltecDataKind.RawTimeStream,)
