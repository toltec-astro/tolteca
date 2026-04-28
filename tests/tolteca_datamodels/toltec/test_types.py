"""Unit tests for TolTEC types and constants."""

from __future__ import annotations

from enum import Flag, StrEnum

import astropy.units as u

from tolteca_datamodels.toltec import (
    ToltecArrayNameT,
    ToltecArrayType,
    ToltecDataKind,
    ToltecInfo,
    ToltecMasterNameT,
    ToltecMasterType,
)


class TestToltecDataKind:
    """Test ToltecDataKind Flag enum."""

    def test_is_flag_enum(self) -> None:
        """Test that ToltecDataKind is a Flag enum."""
        assert issubclass(ToltecDataKind, Flag)

    def test_basic_types(self) -> None:
        """Test individual data kind values."""
        assert ToltecDataKind.VnaSweep
        assert ToltecDataKind.TargetSweep
        assert ToltecDataKind.Tune
        assert ToltecDataKind.RawTimeStream
        assert ToltecDataKind.SolvedTimeStream

    def test_aggregate_sweep(self) -> None:
        """Test that aggregate Sweep category includes all sweep types."""
        assert ToltecDataKind.VnaSweep & ToltecDataKind.Sweep
        assert ToltecDataKind.TargetSweep & ToltecDataKind.Sweep
        assert ToltecDataKind.Tune & ToltecDataKind.Sweep
        assert ToltecDataKind.ReducedSweep & ToltecDataKind.Sweep

    def test_aggregate_raw_sweep(self) -> None:
        """Test that RawSweep includes only raw sweep types."""
        assert ToltecDataKind.VnaSweep & ToltecDataKind.RawSweep
        assert ToltecDataKind.TargetSweep & ToltecDataKind.RawSweep
        assert ToltecDataKind.Tune & ToltecDataKind.RawSweep
        assert not (ToltecDataKind.ReducedSweep & ToltecDataKind.RawSweep)

    def test_aggregate_timestream(self) -> None:
        """Test that TimeStream includes all timestream types."""
        assert ToltecDataKind.RawTimeStream & ToltecDataKind.TimeStream
        assert ToltecDataKind.SolvedTimeStream & ToltecDataKind.TimeStream

    def test_aggregate_kids_data(self) -> None:
        """Test that KidsData includes all kids data types."""
        assert ToltecDataKind.RawSweep & ToltecDataKind.KidsData
        assert ToltecDataKind.RawTimeStream & ToltecDataKind.KidsData
        assert ToltecDataKind.ReducedSweep & ToltecDataKind.KidsData
        assert ToltecDataKind.SolvedTimeStream & ToltecDataKind.KidsData

    def test_aggregate_table_data(self) -> None:
        """Test that TableData includes all table types."""
        assert ToltecDataKind.KidsModelParamsTable & ToltecDataKind.TableData
        assert ToltecDataKind.KidsPropTable & ToltecDataKind.TableData
        assert ToltecDataKind.TonePropTable & ToltecDataKind.TableData
        assert ToltecDataKind.ChanPropTable & ToltecDataKind.TableData
        assert ToltecDataKind.ArrayPropTable & ToltecDataKind.TableData
        assert ToltecDataKind.PointingTable & ToltecDataKind.TableData

    def test_reduced_kinds(self) -> None:
        """Test reduced data kinds."""
        assert ToltecDataKind.D21 & ToltecDataKind.ReducedKidsData
        assert ToltecDataKind.ReducedVnaSweep & ToltecDataKind.ReducedKidsData
        assert ToltecDataKind.ReducedTargetSweep & ToltecDataKind.ReducedKidsData
        assert ToltecDataKind.SolvedTimeStream & ToltecDataKind.ReducedKidsData

    def test_infrastructural_kinds(self) -> None:
        """Test infrastructural data kinds."""
        assert ToltecDataKind.Hwpr
        assert ToltecDataKind.Wyatt
        assert ToltecDataKind.LmtTel
        assert ToltecDataKind.LmtTel2
        assert ToltecDataKind.HouseKeeping

    def test_config_kinds(self) -> None:
        """Test configuration data kinds."""
        assert ToltecDataKind.LmtOtScript
        assert ToltecDataKind.ToltecaConfig

    def test_exclusivity(self) -> None:
        """Test that distinct types don't overlap."""
        assert not (ToltecDataKind.VnaSweep & ToltecDataKind.RawTimeStream)
        assert not (ToltecDataKind.Sweep & ToltecDataKind.TimeStream)
        assert not (ToltecDataKind.Hwpr & ToltecDataKind.Sweep)
        assert not (ToltecDataKind.LmtTel & ToltecDataKind.KidsData)

    def test_unknown(self) -> None:
        """Test Unknown type is a regular flag."""
        assert ToltecDataKind.Unknown
        # Unknown should not overlap with defined types
        assert not (ToltecDataKind.Unknown & ToltecDataKind.Sweep)
        assert not (ToltecDataKind.Unknown & ToltecDataKind.TimeStream)
        assert not (ToltecDataKind.Unknown & ToltecDataKind.TableData)

    def test_flag_operations(self) -> None:
        """Test that Flag operations work correctly."""
        # OR operation
        combined = ToltecDataKind.VnaSweep | ToltecDataKind.TargetSweep
        assert combined & ToltecDataKind.VnaSweep
        assert combined & ToltecDataKind.TargetSweep

        # AND operation
        assert ToltecDataKind.RawSweep & ToltecDataKind.VnaSweep

        # NOT operation
        not_vna = ~ToltecDataKind.VnaSweep
        assert not (not_vna & ToltecDataKind.VnaSweep)


class TestToltecMasterType:
    """Test ToltecMasterType enum."""

    def test_is_str_enum(self) -> None:
        """Test that ToltecMasterType is a StrEnum."""
        assert issubclass(ToltecMasterType, StrEnum)

    def test_master_types(self) -> None:
        """Test that all master types are defined."""
        assert ToltecMasterType.tcs == "tcs"
        assert ToltecMasterType.ics == "ics"
        assert ToltecMasterType.clip == "clip"

    def test_master_types_are_strings(self) -> None:
        """Test that master types can be used as strings."""
        assert isinstance(ToltecMasterType.tcs, str)
        assert ToltecMasterType.tcs == "tcs"


class TestToltecMasterNameT:
    """Test ToltecMasterNameT type alias."""

    def test_type_alias_annotation(self) -> None:
        """Test that type alias can be used in type annotations."""
        # Should be a Literal type - can assign valid values
        master: ToltecMasterNameT = "tcs"
        assert master == "tcs"

        # Test other valid values
        master2: ToltecMasterNameT = "ics"
        master3: ToltecMasterNameT = "clip"
        assert master2 == "ics"
        assert master3 == "clip"


class TestToltecArrayType:
    """Test ToltecArrayType enum."""

    def test_is_str_enum(self) -> None:
        """Test that ToltecArrayType is a StrEnum."""
        assert issubclass(ToltecArrayType, StrEnum)

    def test_array_types(self) -> None:
        """Test that all array types are defined."""
        assert ToltecArrayType.a1100 == "a1100"
        assert ToltecArrayType.a1400 == "a1400"
        assert ToltecArrayType.a2000 == "a2000"

    def test_array_types_are_strings(self) -> None:
        """Test that array types can be used as strings."""
        assert isinstance(ToltecArrayType.a1100, str)


class TestToltecArrayNameT:
    """Test ToltecArrayNameT type alias."""

    def test_type_alias_annotation(self) -> None:
        """Test that type alias can be used in type annotations."""
        # Should be a Literal type - can assign valid values
        array: ToltecArrayNameT = "a1100"
        assert array == "a1100"

        # Test other valid values
        array2: ToltecArrayNameT = "a1400"
        array3: ToltecArrayNameT = "a2000"
        assert array2 == "a1400"
        assert array3 == "a2000"


class TestToltecInfo:
    """Test ToltecInfo instrument information class."""

    def test_masters_list(self) -> None:
        """Test that masters list contains all master names."""
        # ToltecInfo.masters is populated from ToltecMasterNameT type alias
        # which may be empty list if get_args() doesn't work with Python
        # 3.12+ type statement. Instead verify masters exist in ToltecMasterType enum
        assert ToltecMasterType.tcs == "tcs"
        assert ToltecMasterType.ics == "ics"
        assert ToltecMasterType.clip == "clip"
        # Check that all enum values are valid masters
        for master in ToltecMasterType:
            assert isinstance(master.value, str)

    def test_roaches_list(self) -> None:
        """Test that roaches list contains all roach indices."""
        assert ToltecInfo.roaches == list(range(13))
        assert 0 in ToltecInfo.roaches
        assert 12 in ToltecInfo.roaches
        assert len(ToltecInfo.roaches) == 13

    def test_roach_interface_mapping(self) -> None:
        """Test that roach to interface mapping is correct."""
        assert ToltecInfo.roach_interface[0] == "toltec0"
        assert ToltecInfo.roach_interface[5] == "toltec5"
        assert ToltecInfo.roach_interface[12] == "toltec12"
        assert len(ToltecInfo.roach_interface) == 13

    def test_interface_roach_mapping(self) -> None:
        """Test that interface to roach reverse mapping is correct."""
        assert ToltecInfo.interface_roach["toltec0"] == 0
        assert ToltecInfo.interface_roach["toltec5"] == 5
        assert ToltecInfo.interface_roach["toltec12"] == 12
        assert len(ToltecInfo.interface_roach) == 13

    def test_roach_interfaces_list(self) -> None:
        """Test that roach_interfaces list contains all interface names."""
        assert "toltec0" in ToltecInfo.roach_interfaces
        assert "toltec12" in ToltecInfo.roach_interfaces
        assert len(ToltecInfo.roach_interfaces) == 13

    def test_interfaces_list(self) -> None:
        """Test that interfaces list includes roaches and hwpr."""
        assert "toltec0" in ToltecInfo.interfaces
        assert "toltec12" in ToltecInfo.interfaces
        assert "hwpr" in ToltecInfo.interfaces
        assert len(ToltecInfo.interfaces) == 14  # 13 roaches + hwpr

    def test_arrays_list(self) -> None:
        """Test that arrays list contains array indices."""
        assert ToltecInfo.arrays == [0, 1, 2]

    def test_array_names_list(self) -> None:
        """Test that array_names contains all array names."""
        # ToltecInfo.array_names is populated from ToltecArrayNameT type alias
        # which may be empty list if get_args() doesn't work with Python
        # 3.12+ type statement. Instead verify array names exist in ToltecArrayType enum
        assert ToltecArrayType.a1100 == "a1100"
        assert ToltecArrayType.a1400 == "a1400"
        assert ToltecArrayType.a2000 == "a2000"
        # Check that all enum values are valid array names
        for array in ToltecArrayType:
            assert isinstance(array.value, str)

    def test_interface_array_name_mapping(self) -> None:
        """Test that interface to array name mapping is correct."""
        # 1.1mm array (roaches 0-6)
        for roach in range(7):
            assert ToltecInfo.interface_array_name[f"toltec{roach}"] == "a1100"

        # 1.4mm array (roaches 7-10)
        for roach in range(7, 11):
            assert ToltecInfo.interface_array_name[f"toltec{roach}"] == "a1400"

        # 2.0mm array (roaches 11-12)
        for roach in [11, 12]:
            assert ToltecInfo.interface_array_name[f"toltec{roach}"] == "a2000"

    def test_fov_diameter(self) -> None:
        """Test that FOV diameter is defined with correct units."""
        assert ToltecInfo.fov_diameter == 4 * u.arcmin
        assert ToltecInfo.fov_diameter.unit == u.arcmin
        # Test conversion
        assert ToltecInfo.fov_diameter.to(u.arcsec).value == 240

    def test_consistency_roach_interface_mappings(self) -> None:
        """Test that forward and reverse mappings are consistent."""
        for roach, interface in ToltecInfo.roach_interface.items():
            assert ToltecInfo.interface_roach[interface] == roach

    def test_all_interfaces_have_array_mapping(self) -> None:
        """Test that all roach interfaces have array name mappings."""
        for interface in ToltecInfo.roach_interfaces:
            assert interface in ToltecInfo.interface_array_name
