"""Typed filename metadata and vectorized catalog selection."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
            (
                "toltec8",
                145605,
                1,
                datetime(2025, 10, 31, 9, 20, 6, tzinfo=UTC),
                "vnasweep",
            ),
        ),
        (
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ("tel_toltec", 145609, 2, datetime(2025, 10, 31, tzinfo=UTC), None),
        ),
        (
            "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc",
            (
                "toltec8",
                145609,
                1,
                datetime(2025, 10, 31, 9, 26, 59, tzinfo=UTC),
                "tune",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "prefix", ["/recorded", "host:/recorded", "https://example.org"]
)
def test_filename_info(name, expected, prefix, monkeypatch):
    file_prefix, obsnum, scannum, ut, suffix = expected

    def unexpected(*args, **kwargs):
        pytest.fail("Filename parsing accessed the filesystem")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", unexpected)
        info = FilepathInfo.from_source(f"{prefix}/{name}")
    assert isinstance(info, (LmtFilepathInfo, ToltecFilepathInfo))
    assert (info.prefix, info.obsnum, info.subobsnum, info.scannum) == (
        file_prefix,
        obsnum,
        0,
        scannum,
    )
    assert (info.ut, info.suffix, info.ext) == (ut, suffix, "nc")
    assert type(info).model_validate_json(info.model_dump_json(round_trip=True)) == info
    with pytest.raises(ValidationError, match="frozen"):
        info.obsnum = 1  # ty: ignore[invalid-assignment]


def test_unknown_filename_and_invalid_date():
    assert FilepathInfo.from_source("renamed.nc", raise_on_error=False) is None
    with pytest.raises(ValueError, match="Unrecognized"):
        FilepathInfo.from_source("renamed.nc")
    with pytest.raises(ValueError, match="does not match format"):
        FilepathInfo.from_source(
            "tel_toltec_2025-99-31_145609_00_0002.nc", raise_on_error=False
        )


def test_projection_does_not_reparse_paths(monkeypatch):
    location = (
        "host:/data_lmt/toltec/tcs/toltec8/"
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    )
    parsed = FilepathInfo.from_source(location)
    assert isinstance(parsed, ToltecFilepathInfo)
    assert parsed.suffix == "vnasweep"

    def unexpected(*args, **kwargs):
        pytest.fail("Filename projection reparsed a filepath")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    catalog = ToltecFileAccessor.from_models([parsed], columns=parsed._columns)
    assert catalog.loc[0, "file_suffix"] == "vnasweep"
    assert "filepath_orig" not in catalog
    assert "filepath_info_orig" not in catalog


@pytest.fixture
def sources():
    return [
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "unknown.nc",
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc",
    ]


def test_catalog_flat_columns(sources):
    catalog = guess_info_from_sources(sources)
    assert catalog.filepath_info.notna().tolist() == [True, True, False, True]
    assert catalog["source"].tolist() == sources
    assert all(isinstance(value, int) for value in catalog["data_kind_toltec"].tolist())
    assert str(catalog["file_ut"].dtype) == "datetime64[ns, UTC]"
    assert catalog["file_suffix"].isna().tolist() == [False, True, True, False]


@pytest.mark.parametrize(
    ("expression", "indices"),
    [
        ('file_suffix in ["vnasweep", "tune"]', [0, 3]),
        ('file_suffix != "tune"', [0, 1, 2]),
        ('obsnum >= 145606 & file_suffix == "tune"', [3]),
        ('obsnum < 145606 | file_prefix == "tel_toltec"', [0, 1]),
        ('file_ut >= "2025-10-31T00:00:00Z"', [0, 1, 3]),
        ("file_ut.isna()", [2]),
        ("file_interface.isna()", [2]),
    ],
)
def test_flat_selection(sources, expression, indices):
    catalog = guess_info_from_sources(sources)
    selected = catalog.toltec_file.query(expression)
    assert selected.index.tolist() == indices
    assert selected["source"].tolist() == [sources[i] for i in indices]


@pytest.mark.parametrize("size", [0, 2])
def test_empty_and_all_unknown_filenames(size):
    catalog = guess_info_from_sources(["unknown.nc"] * size)
    assert catalog.filepath_info.dtype == object
    assert catalog.filepath_info.tolist() == [None] * size
    assert catalog.toltec_file.query('file_suffix == "tune"').empty
    assert len(catalog.toltec_file.query("file_ut.isna()")) == size
    assert len(list(catalog.itertuples(index=False))) == size


def test_unknown_field_is_not_silently_ignored(sources):
    catalog = guess_info_from_sources(sources)
    with pytest.raises(AttributeError):
        catalog.toltec_file.query('filepath_info.not_a_field == "tune"')
    with pytest.raises(pd.errors.UndefinedVariableError):
        catalog.toltec_file.query("missing > 0")
    with pytest.raises(pd.errors.UndefinedVariableError):
        catalog.toltec_file.query("file_not_a_field > 0")


def test_cached_source_keeps_master_from_identification_location(tmp_path):
    original = (
        "/data_lmt/toltec/simu/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    record = get_info_from_source(tmp_path / "downloaded.nc", source_loc=original)
    assert record is not None
    catalog = guess_info_from_sources(
        [{"source": tmp_path / "downloaded.nc", "source_loc": original}]
    )
    assert catalog.toltec_file.query(
        "master == 'simu' & file_master == 'simu'"
    ).index.tolist() == [0]
    assert catalog.toltec_file.source.tolist() == [record.source]
    assert catalog.toltec_file.source_loc.tolist() == [record.source_loc]


def test_database_paths_identify_cached_files(
    toltecdb_records, lmtmc_records, tmp_path
):
    rows = [*toltecdb_records[:20], *lmtmc_records[:20]]
    sources = [tmp_path / f"cache-{index}.nc" for index in range(len(rows))]
    locations = [f"host:{row['FileName']}" for row in rows]
    catalog = guess_info_from_sources(
        {"source": source, "source_loc": location}
        for source, location in zip(sources, locations, strict=True)
    )
    assert not catalog["file_prefix"].isna().any()
    assert catalog.toltec_file.source.tolist() == sources
    assert [location.as_rsync_arg() for location in catalog.source_loc] == locations
    assert len(catalog.toltec_file.query('file_interface == "tel"')) > 0
