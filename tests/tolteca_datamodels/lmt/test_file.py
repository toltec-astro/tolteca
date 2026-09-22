"""Literal filename segments and their derived acquisition identity."""

from datetime import UTC, datetime

import pytest

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import (
    FileIdentifier,
    FilepathInfo,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.file import RoachFilepathInfo


@pytest.mark.parametrize(
    ("model", "filename", "expected"),
    [
        (
            FilepathInfo,
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ("tel_toltec", "tel", "lmt", "tcs", "tcs"),
        ),
        (
            RoachFilepathInfo,
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
    info = model.from_path(filename)
    assert info is not None
    assert (
        info.prefix,
        info.interface,
        info.instru,
        info.instru_component,
        info.master_str,
    ) == expected
    assert type(info.instru) is str
    assert type(info.instru_component) is str
    assert model.model_validate_json(info.model_dump_json(round_trip=True)) == info
    dumped = info.model_dump(round_trip=True)
    assert dumped["interface"] == info.interface
    assert dumped["instru"] == info.instru
    assert dumped["instru_component"] == info.instru_component
    assert not model.model_computed_fields


def test_copy_of_complete_record_preserves_identity():
    original = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert original is not None
    _ = original.model_dump()
    replacement = RoachFilepathInfo.from_path(
        "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    assert replacement is not None
    copied = original.model_copy(update=replacement.model_dump(round_trip=True))
    assert copied.interface == "toltec8"
    assert copied.instru == "toltec"
    assert copied.instru_component == "roach"
    assert original.interface == "tel"


def test_base_parser_uses_declared_format():
    info = FilepathInfo.from_path(
        "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    )
    assert info is not None
    assert info.prefix == "tel_toltec"
    assert info.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert info.master_str == "tcs"
    assert type(info) is FilepathInfo
    assert "roach" not in FilepathInfo.model_fields
    assert FilepathInfo.from_path("unknown.nc", raise_on_error=False) is None
    with pytest.raises(ValueError):
        FilepathInfo.from_path(
            "tel_toltec_2025-02-30_145609_00_0002.nc", raise_on_error=False
        )


def test_processing_suffix_is_a_shared_record_field():
    raw = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert raw is not None
    assert raw.proc_suffix is None
    assert "proc_suffix" in FilepathInfo.model_fields
    assert "proc_suffix" not in RoachFilepathInfo.__annotations__
    # Test the shared record's capability, not a telescope product-name grammar.
    fields = raw.model_dump(round_trip=True)
    fields["proc_suffix"] = "example_processing_tag"
    processed = FilepathInfo.model_validate(fields)
    assert processed.proc_suffix == "example_processing_tag"
    assert (
        FilepathInfo.model_validate_json(processed.model_dump_json(round_trip=True))
        == processed
    )


@pytest.mark.parametrize("directory", ["tel_toltec", "tel"])
def test_misplaced_telescope_copy_does_not_change_master(directory):
    # Deliberately misplaced copy, not a telescope acquisition directory.
    info = FilepathInfo.from_path(
        f"/data_lmt/toltec/ics/{directory}/tel_toltec_2025-10-31_145609_00_0002.nc"
    )
    assert info is not None
    assert info.interface == "tel"
    assert info.master_str == "tcs"


@pytest.mark.parametrize(
    "filename",
    [
        "toltec_145609_000_0002_2025_10_31_09_27_57.nc",
        "hk_145609_000_0002_2025_10_31_09_27_57.nc",
        "tel_test_2025-10-31_145609_00_0002.nc",
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
        FilepathInfo.from_path(filename)
    assert FilepathInfo.from_path(filename, raise_on_error=False) is None


def test_common_record_declarations_do_not_accept_readout_files():
    name = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    assert all(
        pattern.match(name) is None
        for pattern, parser in FileIdentifier.filename_pattern_parser.items()
        if parser.filepath_info_cls is FilepathInfo
    )


def test_telescope_prefix_and_interface_selection(monkeypatch):
    catalog = guess_info_from_sources(
        [
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            "tel_toltec_2025-10-31_145605_00_0001.nc",
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        ]
    )
    assert catalog["interface"].tolist() == ["tel", "tel", "toltec8"]
    assert catalog["data_kind"].tolist() == ["LmtTel", "LmtTel", "VnaSweep"]
    assert catalog.toltec_file.to_info_list()[1].file_prefix == "tel_toltec"

    def unexpected(*args, **kwargs):
        pytest.fail("Selection recomputed per-row model identity")

    monkeypatch.setattr(file_module.FileIdentifier, "parse_filepath", unexpected)
    for expression, indices in [
        ('interface == "tel"', [0, 1]),
        ('file_prefix == "tel_toltec" & obsnum == 145605', [1]),
        ('filepath_info.prefix == "tel_toltec"', [0, 1]),
        (
            'filepath_info.instru == "lmt" & filepath_info.instru_component == "tcs"',
            [0, 1],
        ),
        ('file_instru == "toltec" & file_instru_component == "roach"', [2]),
    ]:
        assert catalog.toltec_file.query(expression).index.tolist() == indices
