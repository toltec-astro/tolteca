"""Filename identification preserves local and remote source locations."""

from pathlib import Path

import pandas as pd
import pytest
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec.file import (
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize("interface", ["toltec08", "toltec13"])
def test_unregistered_roach_interface_is_rejected(toltecdb_records, interface):
    # Deliberately replace a recorded prefix with an unregistered spelling/index.
    path = Path(toltecdb_records[0]["FileName"])
    _, remainder = path.name.split("_", 1)
    invalid_path = path.with_name(f"{interface}_{remainder}")
    with pytest.raises(KeyError, match=interface):
        FilepathInfo.from_source(invalid_path)


def test_unknown_telescope_suffix_does_not_change_kind():
    # A deliberately renamed copy exercises suffix-independent identification.
    info = get_info_from_source("tel_toltec_2025-10-31_145609_00_0002_custom.nc")
    assert info is not None
    assert info.file_suffix == "custom"
    assert info.data_kind_toltec == ToltecDataKind.lmt_tel.value


@pytest.mark.parametrize(
    "prefix", ["host:/data", "file://host/data", "https://example.org/data"]
)
def test_remote_filename_and_catalog(prefix, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote identification attempted local file access")

    source = fileloc(f"{prefix}/tel_toltec_2025-10-31_145609_00_0002.nc")
    info = get_info_from_source(source)
    assert info is not None
    assert info.source is source
    assert info.obsnum == 145609
    assert info.interface == "tel"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", unexpected)
        catalog = guess_info_from_sources([source])
    assert catalog.loc[0, "source"] is source
    restored = next(iter(catalog.itertuples(index=False)))
    assert restored.source is source
    assert restored.obsnum == info.obsnum


def test_url_query_is_not_part_of_filename():
    source = "https://example.org/tel_toltec_2025-10-31_145609_00_0002.nc?q=a%26b#part"
    info = get_info_from_source(source)
    assert info is not None
    assert info.file_ext == "nc"
    assert info.source == source


def test_unknown_remote_file_keeps_its_location():
    source = "host:/data/unrecognized.nc"
    info = next(iter(guess_info_from_sources([source]).itertuples(index=False)))
    assert pd.isna(info.obsnum)
    assert info.source is source


def test_local_source_keeps_original_path_and_parent_context(tmp_path):
    source = fileloc(
        "tel_toltec_2025-10-31_145609_00_0002.nc", local_parent_path=tmp_path
    )
    info = get_info_from_source(source)
    assert info is not None
    assert info.source is source
    assert fileloc(info.source).path == tmp_path / source.path_orig
    assert info._asdict()["source"] is source


@pytest.mark.parametrize(
    "prefix", ["/data_lmt", "host:/data_lmt", "https://example.org"]
)
def test_explicit_source_location_uses_the_same_parser(prefix):
    master = "tcs"
    original = (
        f"{prefix}/toltec/{master}/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    parsed = FilepathInfo.from_source(original)
    current = get_info_from_source(original)
    source = fileloc("downloaded.nc")
    fallback = get_info_from_source(source, source_loc=original)
    assert isinstance(parsed, ToltecFilepathInfo)
    assert current is not None and fallback is not None
    assert parsed.master == master
    assert "master_code" not in parsed.model_dump()
    assert parsed.roach == 8
    assert fallback.source is source
    assert fallback.source_loc == fileloc(original)
    for name in ("file_prefix", "obsnum", "master", "file_ut", "data_kind_toltec"):
        assert getattr(current, name) == getattr(fallback, name)
    assert fallback.master == parsed.master
    assert parsed.ut is not None
    assert fallback.file_ut == parsed.ut
    assert fallback.file_suffix == parsed.suffix
    assert fallback.file_ext == parsed.ext


@pytest.mark.parametrize(
    "directory",
    ["/data_lmt/toltec", "/data_lmt/toltec/toltec8", "/code/toltec/project/python"],
)
def test_layout_without_master_does_not_use_filename_or_interface(directory):
    parsed = FilepathInfo.from_source(
        f"{directory}/toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    assert isinstance(parsed, ToltecFilepathInfo)
    assert parsed.master is None


def test_explicit_telescope_source_location():
    original = "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    info = get_info_from_source("downloaded.nc", source_loc=original)
    assert info is not None
    assert info.interface == "tel"
    assert info.obsnum == 145609
    assert info.scannum == 2
    assert pd.isna(info.file_suffix)
    assert info.master == "tcs"


@pytest.mark.parametrize(
    "source", ["downloaded.nc", "tel_toltec_2025-10-31_145609_00_0002.nc"]
)
def test_unknown_explicit_location_does_not_fall_back_to_source(source):
    info = get_info_from_source(source, source_loc="/recorded/unknown.nc")
    assert info.source == source
    assert pd.isna(info.file_prefix)
    assert pd.isna(info.obsnum)
    assert info.data_kind_toltec == ToltecDataKind.unknown.value


@pytest.mark.parametrize("suffix", ["custom", "custom-tag"])
def test_unknown_kids_suffix_is_not_classified(suffix):
    # Deliberately change a recorded VNA filename to an unsupported suffix.
    record = get_info_from_source(
        f"toltec8_145605_000_0001_2025_10_31_09_20_06_{suffix}.nc"
    )
    assert record is not None
    assert record.data_kind_toltec == ToltecDataKind.unknown.value
    assert record.file_suffix == suffix


@pytest.mark.parametrize(
    "stem",
    [
        "tel_toltec_2025-10-31_145609_00_0002",
        "tel_toltec2_2025-10-31_145609_00_0002",
        "wyatt_2025-10-31_145609_00_0002",
        "hwpr_145609_000_0002_2025_10_31_01_02_03",
        "toltec_hk_145609_000_0002_2025_10_31_01_02_03",
    ],
)
def test_auxiliary_kind_requires_netcdf_extension(stem):
    record = get_info_from_source(f"{stem}.txt")
    assert record is not None
    assert record.data_kind_toltec == ToltecDataKind.unknown.value


def test_telescope_filename_suffix():
    # User-renamed copy of an actual file; the suffix is not an acquired field.
    record = get_info_from_source("tel_toltec_2025-10-31_145609_00_0002_custom-tag.nc")
    assert record is not None
    assert record.file_suffix == "custom-tag"
    assert record.data_kind_toltec == ToltecDataKind.lmt_tel.value
