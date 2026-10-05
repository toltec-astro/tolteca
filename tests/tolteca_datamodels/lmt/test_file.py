"""Literal filename segments and their derived acquisition identity."""

from datetime import UTC, datetime
from typing import assert_type

import pytest
from pydantic import ValidationError

import tolteca_datamodels._filepath as file_module
from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.lmt_base.types import LabelT
from tolteca_datamodels.toltec.file import ToltecFilepathInfo, guess_info_from_sources
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize(
    ("model", "filename", "expected"),
    [
        (
            LmtFilepathInfo,
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ("tel_toltec", "tel", "lmt", "tcs", "tcs"),
        ),
        (
            ToltecFilepathInfo,
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
            ("toltec8", "toltec8", "toltec", "roach", None),
        ),
    ],
)
def test_identity_does_not_rewrite_filename_segments(
    model, filename, expected, lmtmc_records, toltecdb_records
):
    assert any(
        row["FileName"].endswith(f"/{filename}")
        for row in [*lmtmc_records, *toltecdb_records]
    )
    info = model.from_source(filename)
    assert info is not None
    assert (
        info.prefix,
        info.interface,
        info.instru,
        info.instru_component,
        info.master,
    ) == expected
    assert type(info.instru) is str
    assert type(info.instru_component) is str
    assert model.model_validate_json(info.model_dump_json(round_trip=True)) == info
    dumped = info.model_dump(round_trip=True)
    assert dumped["interface"] == info.interface
    assert dumped["instru"] == info.instru
    assert dumped["instru_component"] == info.instru_component
    assert not model.model_computed_fields


def test_telescope_filepath_rejects_instrument_identity():
    info = LmtFilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None
    with pytest.raises(ValidationError, match="instru"):
        LmtFilepathInfo.model_validate(info.model_dump() | {"instru": "toltec"})


def test_base_parser_uses_declared_format():
    info = FilepathInfo.from_source(
        "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    )
    assert type(info) is LmtFilepathInfo
    assert_type(info.instru, LmtInfo.InstruT)
    assert_type(info.instru_component, LabelT)
    assert info.prefix == "tel_toltec"
    assert info.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert info.master == "tcs"
    assert "roach" not in FilepathInfo.model_fields
    assert FilepathInfo.from_source("unknown.nc", raise_on_error=False) is None
    with pytest.raises(ValueError):
        FilepathInfo.from_source(
            "tel_toltec_2025-02-30_145609_00_0002.nc", raise_on_error=False
        )


def test_lmt_filepath_component_is_a_required_label(lmtmc_records):
    info = LmtFilepathInfo.from_source(lmtmc_records[0]["FileName"])
    assert info is not None
    field = LmtFilepathInfo.model_fields["instru_component"]
    assert field.annotation is LabelT
    assert field.is_required()
    with pytest.raises(ValidationError, match="instru_component"):
        LmtFilepathInfo.model_validate(info.model_dump() | {"instru_component": None})


def test_processing_suffix_belongs_to_toltec_filename_model():
    assert "proc_suffix" not in FilepathInfo.model_fields
    assert "proc_suffix" not in LmtFilepathInfo.model_fields
    assert "proc_suffix" in ToltecFilepathInfo.__annotations__


@pytest.mark.parametrize("directory", ["tel_toltec", "tel"])
def test_misplaced_telescope_copy_does_not_change_master(directory):
    # Deliberately misplaced copy, not a telescope acquisition directory.
    info = FilepathInfo.from_source(
        f"/data_lmt/toltec/ics/{directory}/tel_toltec_2025-10-31_145609_00_0002.nc"
    )
    assert isinstance(info, LmtFilepathInfo)
    assert info.interface == "tel"
    assert info.master == "tcs"


@pytest.mark.parametrize(
    "filename",
    [
        "toltec_145609_000_0002_2025_10_31_09_27_57.nc",
        "hk_145609_000_0002_2025_10_31_09_27_57.nc",
        "tel_test_2025-10-31_145609_00.nc",
        "toltec8_2025-10-31_145609_00_0002.nc",
        "tel_toltec_145609_000_0002_2025_10_31_09_27_57.nc",
        "toltec8_145609_000_0002.nc",
        "toltec8_145609_000_0002_tune.nc",
        "toltecX_145609_000_0002_2025_10_31_09_27_57.nc",
        "toltec8_145609_000_0002_2025_10_31_09_27_57.",
    ],
)
def test_malformed_acquisition_filenames_are_rejected(filename):
    # Deliberately invalid variants of recorded names, not acquisition examples.
    with pytest.raises(ValueError, match="Unrecognized acquisition filename"):
        FilepathInfo.from_source(filename)
    assert FilepathInfo.from_source(filename, raise_on_error=False) is None


@pytest.mark.parametrize(
    "prefix", ["tel_toltec", "tel_toltec2", "tel_dummy_instru", "tel_unregistered"]
)
def test_telescope_prefix_does_not_select_instrument(prefix):
    # Dummy and unregistered labels exercise instrument-independent parsing.
    info = LmtFilepathInfo.from_source(f"{prefix}_2025-10-31_145609_00_0002.nc")
    assert type(info) is LmtFilepathInfo
    assert info.prefix == prefix
    assert (info.interface, info.instru, info.instru_component) == ("tel", "lmt", "tcs")
    assert (info.obsnum, info.subobsnum, info.scannum) == (145609, 0, 2)


def test_telescope_parser_does_not_accept_readout_files():
    name = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    with pytest.raises(
        ValueError, match="Unrecognized acquisition filename for LmtFilepathInfo"
    ):
        LmtFilepathInfo.from_source(name)
    assert LmtFilepathInfo.from_source(name, raise_on_error=False) is None


def test_telescope_prefix_and_interface_selection(monkeypatch):
    catalog = guess_info_from_sources(
        [
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            "tel_toltec_2025-10-31_145605_00_0001.nc",
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        ]
    )
    assert catalog["interface"].tolist() == ["tel", "tel", "toltec8"]
    assert catalog["data_kind"].tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.vnasweep.value,
    ]
    assert catalog["file_prefix"].iloc[1] == "tel_toltec"

    def unexpected(*args, **kwargs):
        pytest.fail("Selection recomputed per-row model identity")

    monkeypatch.setattr(file_module.FilepathInfo, "from_source", unexpected)
    for expression, indices in [
        ('interface == "tel"', [0, 1]),
        ('file_prefix == "tel_toltec" & obsnum == 145605', [1]),
        ('file_prefix == "tel_toltec"', [0, 1]),
        (
            'file_instru == "lmt" & file_instru_component == "tcs"',
            [0, 1],
        ),
        ('file_instru == "toltec" & file_instru_component == "roach"', [2]),
    ]:
        assert catalog.toltec_file.query(expression).index.tolist() == indices
