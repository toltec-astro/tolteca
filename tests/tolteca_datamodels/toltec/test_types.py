"""Unit tests for TolTEC types and constants."""

from __future__ import annotations

from enum import Flag

import astropy.units as u
import pytest
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt.types import LmtInfo, LmtInterfaceInfo
from tolteca_datamodels.lmt_base.metadata import InterfaceInfoMixin
from tolteca_datamodels.toltec import ToltecInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_instrument_literal_type():
    adapter = TypeAdapter(ToltecInfo.InstruT)
    assert adapter.validate_python("toltec") == "toltec"
    assert adapter.validate_json(adapter.dump_json("toltec")) == "toltec"
    with pytest.raises(ValidationError, match="literal_error"):
        adapter.validate_python("lmt")


def test_observation_type_data_kinds():
    assert ToltecInfo.obs_type_data_kind == {
        "nominal": ToltecDataKind.raw_timestream,
        "timestream": ToltecDataKind.raw_timestream,
        "vna": ToltecDataKind.vnasweep,
        "targ": ToltecDataKind.targsweep,
        "tune": ToltecDataKind.tune,
    }
    assert set(ToltecInfo.obs_type_data_kind) == set(
        TypeAdapter(ToltecInfo.ObsTypeT).json_schema()["enum"]
    )


def test_roach_type_matches_readout_indices():
    adapter = TypeAdapter(ToltecInfo.RoachT)
    assert ToltecInfo.roaches == list(range(13))
    for roach in ToltecInfo.roaches:
        assert adapter.validate_python(roach) == roach
        assert adapter.validate_python(str(roach)) == roach


@pytest.mark.parametrize("value", [-1, 13, "-1", "13", 0.5])
def test_roach_type_rejects_invalid_indices(value):
    with pytest.raises(ValidationError):
        TypeAdapter(ToltecInfo.RoachT).validate_python(value)


@pytest.mark.parametrize(
    ("annotation", "labels"),
    [
        (ToltecInfo.MasterT, ["tcs", "ics", "clip"]),
        (ToltecInfo.ObsTypeT, ["nominal", "timestream", "vna", "targ", "tune"]),
        (ToltecInfo.ArrayNameT, ["a1100", "a1400", "a2000"]),
        (LmtInfo.MasterT | ToltecInfo.MasterT, ["tcs", "ics", "clip"]),
    ],
)
def test_label_types_validate_strings_and_roundtrip(annotation, labels):
    adapter = TypeAdapter(annotation)
    for label in labels:
        value = adapter.validate_python(label, strict=True)
        assert type(value) is str
        assert adapter.dump_python(value) == label
        assert adapter.validate_json(adapter.dump_json(value), strict=True) == label
    for invalid in ("not-a-label", "", 0, None):
        with pytest.raises(ValidationError, match="literal_error"):
            adapter.validate_python(invalid, strict=True)


@pytest.mark.parametrize("array", [0, 1, 2])
def test_array_indices_are_integers(array):
    assert TypeAdapter(ToltecInfo.ArrayT).validate_python(array, strict=True) == array


@pytest.mark.parametrize("array", [-1, 3, "0", 0.5, None])
def test_invalid_array_indices(array):
    with pytest.raises(ValidationError):
        TypeAdapter(ToltecInfo.ArrayT).validate_python(array, strict=True)


def test_label_descriptions_cover_instrument_choices():
    assert set(ToltecInfo.master_description) == set(ToltecInfo.masters)
    assert ToltecInfo.master_description == {
        "tcs": "Telescope control system",
        "ics": "Instrument control system",
        "clip": "ROACH manager",
    }
    assert set(ToltecInfo.array_name_description) == set(ToltecInfo.array_names)
    assert ToltecInfo.array_name_description == {
        "a1100": "1.1 mm detector array",
        "a1400": "1.4 mm detector array",
        "a2000": "2.0 mm detector array",
    }


class TestToltecDataKind:
    """Test ToltecDataKind Flag enum."""

    def test_member_names_and_bit_values(self):
        assert {
            name: kind.value for name, kind in ToltecDataKind.__members__.items()
        } == {
            "vnasweep": 1,
            "targsweep": 2,
            "tune": 4,
            "raw_sweep": 7,
            "raw_timestream": 8,
            "raw_kids_data": 15,
            "d21": 16,
            "reduced_vnasweep": 32,
            "reduced_targsweep": 64,
            "reduced_sweep": 96,
            "solved_timestream": 128,
            "reduced_kids_data": 240,
            "sweep": 103,
            "timestream": 136,
            "kids_data": 255,
            "targ_freqs_dat": 256,
            "targ_amps_dat": 512,
            "kids_model_params_table": 1024,
            "kids_prop_table": 2048,
            "tone_prop_table": 4096,
            "chan_prop_table": 8192,
            "array_prop_table": 16384,
            "pointing_table": 32768,
            "kids_table_data": 15360,
            "table_data": 64512,
            "hwpr": 65536,
            "wyatt": 131072,
            "lmt_tel": 262144,
            "lmt_tel2": 524288,
            "housekeeping": 1048576,
            "lmt_ot_script": 2097152,
            "tolteca_config": 4194304,
            "unknown": 8388608,
        }

    def test_json_keeps_numeric_encoding(self):
        adapter = TypeAdapter(ToltecDataKind)
        for kind in ToltecDataKind.__members__.values():
            assert adapter.dump_python(kind, mode="json") == kind.value
            assert adapter.validate_json(adapter.dump_json(kind)) is kind
        combined = ToltecDataKind.tune | ToltecDataKind.lmt_tel
        assert adapter.dump_python(combined, mode="json") == 262148
        assert adapter.validate_json(adapter.dump_json(combined)) == combined

    def test_is_flag_enum(self) -> None:
        """Test that ToltecDataKind is a Flag enum."""
        assert issubclass(ToltecDataKind, Flag)

    def test_basic_types(self) -> None:
        """Test individual data kind values."""
        assert ToltecDataKind.vnasweep
        assert ToltecDataKind.targsweep
        assert ToltecDataKind.tune
        assert ToltecDataKind.raw_timestream
        assert ToltecDataKind.solved_timestream

    def test_aggregate_sweep(self) -> None:
        """Test that aggregate Sweep category includes all sweep types."""
        assert ToltecDataKind.vnasweep & ToltecDataKind.sweep
        assert ToltecDataKind.targsweep & ToltecDataKind.sweep
        assert ToltecDataKind.tune & ToltecDataKind.sweep
        assert ToltecDataKind.reduced_sweep & ToltecDataKind.sweep

    def test_aggregate_raw_sweep(self) -> None:
        """Test that RawSweep includes only raw sweep types."""
        assert ToltecDataKind.vnasweep & ToltecDataKind.raw_sweep
        assert ToltecDataKind.targsweep & ToltecDataKind.raw_sweep
        assert ToltecDataKind.tune & ToltecDataKind.raw_sweep
        assert not (ToltecDataKind.reduced_sweep & ToltecDataKind.raw_sweep)

    def test_aggregate_timestream(self) -> None:
        """Test that TimeStream includes all timestream types."""
        assert ToltecDataKind.raw_timestream & ToltecDataKind.timestream
        assert ToltecDataKind.solved_timestream & ToltecDataKind.timestream

    def test_aggregate_kids_data(self) -> None:
        """Test that KidsData includes all kids data types."""
        assert ToltecDataKind.raw_sweep & ToltecDataKind.kids_data
        assert ToltecDataKind.raw_timestream & ToltecDataKind.kids_data
        assert ToltecDataKind.reduced_sweep & ToltecDataKind.kids_data
        assert ToltecDataKind.solved_timestream & ToltecDataKind.kids_data

    def test_aggregate_table_data(self) -> None:
        """Test that TableData includes all table types."""
        assert ToltecDataKind.kids_model_params_table & ToltecDataKind.table_data
        assert ToltecDataKind.kids_prop_table & ToltecDataKind.table_data
        assert ToltecDataKind.tone_prop_table & ToltecDataKind.table_data
        assert ToltecDataKind.chan_prop_table & ToltecDataKind.table_data
        assert ToltecDataKind.array_prop_table & ToltecDataKind.table_data
        assert ToltecDataKind.pointing_table & ToltecDataKind.table_data

    def test_reduced_kinds(self) -> None:
        """Test reduced data kinds."""
        assert ToltecDataKind.d21 & ToltecDataKind.reduced_kids_data
        assert ToltecDataKind.reduced_vnasweep & ToltecDataKind.reduced_kids_data
        assert ToltecDataKind.reduced_targsweep & ToltecDataKind.reduced_kids_data
        assert ToltecDataKind.solved_timestream & ToltecDataKind.reduced_kids_data

    def test_infrastructural_kinds(self) -> None:
        """Test infrastructural data kinds."""
        assert ToltecDataKind.hwpr
        assert ToltecDataKind.wyatt
        assert ToltecDataKind.lmt_tel
        assert ToltecDataKind.lmt_tel2
        assert ToltecDataKind.housekeeping

    def test_config_kinds(self) -> None:
        """Test configuration data kinds."""
        assert ToltecDataKind.lmt_ot_script
        assert ToltecDataKind.tolteca_config

    def test_exclusivity(self) -> None:
        """Test that distinct types don't overlap."""
        assert not (ToltecDataKind.vnasweep & ToltecDataKind.raw_timestream)
        assert not (ToltecDataKind.sweep & ToltecDataKind.timestream)
        assert not (ToltecDataKind.hwpr & ToltecDataKind.sweep)
        assert not (ToltecDataKind.lmt_tel & ToltecDataKind.kids_data)

    def test_unknown(self) -> None:
        """Test Unknown type is a regular flag."""
        assert ToltecDataKind.unknown
        # Unknown should not overlap with defined types
        assert not (ToltecDataKind.unknown & ToltecDataKind.sweep)
        assert not (ToltecDataKind.unknown & ToltecDataKind.timestream)
        assert not (ToltecDataKind.unknown & ToltecDataKind.table_data)

    def test_flag_operations(self) -> None:
        """Test that Flag operations work correctly."""
        # OR operation
        combined = ToltecDataKind.vnasweep | ToltecDataKind.targsweep
        assert combined & ToltecDataKind.vnasweep
        assert combined & ToltecDataKind.targsweep

        # AND operation
        assert ToltecDataKind.raw_sweep & ToltecDataKind.vnasweep

        # NOT operation
        not_vna = ~ToltecDataKind.vnasweep
        assert not (not_vna & ToltecDataKind.vnasweep)


class TestToltecInfo:
    """Test ToltecInfo instrument information class."""

    def test_masters_list(self) -> None:
        """Test that masters list contains all master names."""
        assert ToltecInfo.masters == ["tcs", "ics", "clip"]

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
        """Interfaces include TolTEC streams and their telescope companion."""
        assert ToltecInfo.interfaces == [
            *(f"toltec{roach}" for roach in range(13)),
            "hwpr",
            "wyatt",
            "toltec_hk",
            "tel",
        ]
        assert ToltecInfo.interfaces == list(ToltecInfo.interface_interface_info)

    def test_interface_info_mapping(self) -> None:
        """All descriptors identify TolTEC, including the LMT-produced companion."""
        assert ToltecInfo.interface_interface_info == {
            **{
                f"toltec{roach}": InterfaceInfoMixin(
                    interface=f"toltec{roach}",
                    data_models=["toltec"],
                    instru="toltec",
                    interface_kind="roach",
                )
                for roach in range(13)
            },
            "hwpr": InterfaceInfoMixin(
                interface="hwpr",
                data_models=["toltec"],
                instru="toltec",
                interface_kind="hwpr",
            ),
            "wyatt": InterfaceInfoMixin(
                interface="wyatt",
                data_models=["toltec"],
                instru="toltec",
                interface_kind="wyatt",
            ),
            "toltec_hk": InterfaceInfoMixin(
                interface="toltec_hk",
                data_models=["toltec"],
                instru="toltec",
                interface_kind="toltec_hk",
            ),
            "tel": LmtInterfaceInfo(
                interface="tel",
                data_models=["lmt", "toltec"],
                instru="lmt",
                interface_kind="tel",
            ),
        }
        assert "toltec" not in ToltecInfo.interface_interface_info
        assert "hk" not in ToltecInfo.interface_interface_info

    def test_arrays_list(self) -> None:
        """Test that arrays list contains array indices."""
        assert ToltecInfo.arrays == [0, 1, 2]

    def test_array_names_list(self) -> None:
        """Test that array_names contains all array names."""
        assert ToltecInfo.array_names == ["a1100", "a1400", "a2000"]

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
