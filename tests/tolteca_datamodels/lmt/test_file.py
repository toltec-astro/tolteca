"""Literal filename segments and their derived acquisition identity."""

from datetime import UTC, datetime

import pytest

from tolteca_datamodels.file import (
    FilepathInfo,
    filename_patterns,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.file import FilepathInfo as ToltecFilepathInfo


@pytest.mark.parametrize(
    ("model", "filename", "expected"),
    [
        (
            FilepathInfo,
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ("tel_toltec", "tel", "lmt", "tcs"),
        ),
        (
            ToltecFilepathInfo,
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
            ("toltec8", "toltec8", "toltec", "roach"),
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
    assert (info.prefix, info.interface, info.instru, info.instru_component) == expected
    assert info.master_str is None
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
    replacement = ToltecFilepathInfo.from_path(
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
    assert info.master_str is None
    assert type(info) is FilepathInfo
    assert "roach" not in FilepathInfo.model_fields
    assert FilepathInfo.from_path("unknown.nc", raise_on_error=False) is None
    with pytest.raises(ValueError):
        FilepathInfo.from_path(
            "tel_toltec_2025-02-30_145609_00_0002.nc", raise_on_error=False
        )


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
        for pattern, model in filename_patterns.items()
        if model is FilepathInfo
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

    monkeypatch.setattr(FilepathInfo, "_derive_fields", classmethod(unexpected))
    monkeypatch.setattr(ToltecFilepathInfo, "_derive_fields", classmethod(unexpected))
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
