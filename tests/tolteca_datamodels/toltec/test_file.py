"""Filename identification preserves local and remote source locations."""

from dataclasses import replace
from pathlib import Path

import pytest
from tollan.utils.fileloc import fileloc

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import (
    FileKindKey,
    FilepathInfo,
    SourceInfoModel,
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.file import FilepathInfo as ToltecFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_file_identity_and_kind_are_computed_once(monkeypatch):
    lookups = []

    class KindMap(dict):
        def get(self, key, default=None):
            lookups.append(key)
            return super().get(key, default)

    specs = tuple(
        replace(spec, kind_map=KindMap(spec.kind_map))
        for spec in file_module._get_file_specs()
    )
    monkeypatch.setattr(file_module, "_get_file_specs", lambda: specs)
    info = FilepathInfo.from_path(
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    )
    assert info is not None
    assert lookups == [
        FileKindKey(
            instru="toltec", instru_component="roach", suffix="vnasweep", ext="nc"
        )
    ]
    lookups.clear()

    def unexpected(*args, **kwargs):
        pytest.fail("Stored filename identity was recomputed")

    monkeypatch.setattr(FilepathInfo, "_derive_fields", classmethod(unexpected))
    for _ in range(2):
        assert info.interface == "toltec8"
        assert info.instru == "toltec"
        assert info.instru_component == "roach"
        assert info.data_kind == ToltecDataKind.VnaSweep
        assert type(info).model_validate(info.model_dump()) == info
        assert type(info).model_validate_json(info.model_dump_json()) == info
    assert lookups == []
    assert not FilepathInfo.model_computed_fields


def test_unknown_telescope_suffix_does_not_change_kind():
    # A deliberately renamed copy exercises suffix-independent identification.
    info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002_custom.nc")
    assert info is not None
    assert info.suffix == "custom"
    assert info.data_kind == ToltecDataKind.LmtTel


@pytest.mark.parametrize(
    "prefix", ["host:/data", "file://host/data", "https://example.org/data"]
)
def test_remote_filename_and_catalog(prefix, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote identification attempted local file access")

    monkeypatch.setattr(Path, "exists", unexpected)
    source = fileloc(f"{prefix}/tel_toltec_2025-10-31_145609_00_0002.nc")
    info = guess_info_from_source(source)
    assert info is not None
    assert info.source is source
    assert info.obsnum == 145609
    assert info.interface == "tel"
    assert (
        SourceInfoModel.model_validate_json(info.model_dump_json(round_trip=True))
        == info
    )
    catalog = guess_info_from_sources([source])
    assert catalog.loc[0, "source"] == source.url
    restored = catalog.toltec_file.to_info_list()[0]
    assert restored.source.url == source.url
    assert restored.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    ) == info.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )
    with pytest.raises(ValueError, match="no local filepath"):
        _ = info.filepath


def test_url_query_is_not_part_of_filename():
    source = "https://example.org/tel_toltec_2025-10-31_145609_00_0002.nc?q=a%26b#part"
    info = guess_info_from_source(source)
    assert info is not None
    assert info.file_ext == "nc"
    assert info.source.url == source


def test_unknown_remote_file_keeps_its_location():
    source = "host:/data/unrecognized.nc"
    info = guess_info_from_sources([source]).toltec_file.to_info_list()[0]
    assert info.obsnum is None
    assert info.source.url == fileloc(source).url


def test_local_source_keeps_original_path_and_parent_context(tmp_path):
    source = fileloc(
        "tel_toltec_2025-10-31_145609_00_0002.nc", local_parent_path=tmp_path
    )
    info = guess_info_from_source(source)
    assert info is not None
    assert info.source is source
    assert info.filepath == tmp_path / source.path_orig
    assert info.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )["source"] == str(info.filepath)
    assert (
        SourceInfoModel.model_validate_json(info.model_dump_json(round_trip=True))
        == info
    )


@pytest.mark.parametrize(
    "prefix", ["/data_lmt", "host:/data_lmt", "https://example.org"]
)
def test_current_and_original_paths_use_the_same_parser(prefix):
    master_str = "tcs"
    original = (
        f"{prefix}/toltec/{master_str}/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    parsed = FilepathInfo.from_path(original)
    current = guess_info_from_source(original)
    source = fileloc("downloaded.nc")
    fallback = guess_info_from_source(source, filepath_orig=original)
    assert parsed is not None and current is not None and fallback is not None
    assert parsed.master_str == master_str
    assert "master" not in parsed.model_dump()
    assert isinstance(parsed, ToltecFilepathInfo)
    assert parsed.roach == 8
    assert fallback.source is source
    assert fallback.filepath_orig == original
    assert fallback.filepath_info_orig == parsed
    assert current.model_dump(
        round_trip=True, exclude={"source", "filepath_orig"}
    ) == fallback.model_dump(round_trip=True, exclude={"source", "filepath_orig"})
    assert fallback.master == parsed.master_str
    assert parsed.ut is not None
    assert fallback.file_ut == parsed.ut
    assert fallback.file_suffix == parsed.suffix
    assert fallback.file_ext == parsed.ext


@pytest.mark.parametrize(
    "directory",
    ["/data_lmt/toltec", "/data_lmt/toltec/toltec8", "/code/toltec/project/python"],
)
def test_layout_without_master_does_not_use_filename_or_interface(directory):
    parsed = FilepathInfo.from_path(
        f"{directory}/toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    assert parsed is not None
    assert parsed.master_str is None


def test_original_telescope_filename_fallback():
    original = "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    info = guess_info_from_source("downloaded.nc", filepath_orig=original)
    assert info is not None
    assert info.interface == "tel"
    assert info.obsnum == 145609
    assert info.scannum == 2
    assert info.file_suffix is None
    assert info.master is None


def test_conflicting_current_filename_retains_its_fields():
    original = (
        "/data_lmt/toltec/tcs/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    # Deliberately supply two different recorded observations to exercise
    # source/original-path precedence, not a valid acquisition association.
    current = (
        "/copied/toltec/ics/toltec8/"
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    )
    info = guess_info_from_source(current, filepath_orig=original)
    assert info is not None
    assert info.source.path.name == Path(current).name
    assert info.roach == 8
    assert (info.obsnum, info.subobsnum, info.scannum) == (145605, 0, 1)
    assert info.file_ut is not None
    assert info.file_ut.day == 31
    assert info.file_suffix == "vnasweep"
    assert info.master == "tcs"


def test_unknown_original_filename_does_not_invent_identification():
    assert (
        guess_info_from_source(
            "downloaded.nc", filepath_orig="/recorded/unknown.nc", raise_on_error=False
        )
        is None
    )


@pytest.mark.parametrize("suffix", ["custom", "custom-tag"])
def test_unknown_kids_suffix_is_not_classified(suffix):
    # Deliberately change a recorded VNA filename to an unsupported suffix.
    record = guess_info_from_source(
        f"toltec8_145605_000_0001_2025_10_31_09_20_06_{suffix}.nc"
    )
    assert record is not None
    assert record.data_kind == ToltecDataKind.Unknown
    assert record.file_suffix == suffix


@pytest.mark.parametrize("ext", ["nc.gz", "nc-custom"])
def test_unknown_extension_retains_identifiers(ext):
    # Unsupported extensions exercise discovery fallback, not new file formats.
    record = guess_info_from_source(
        f"toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.{ext}"
    )
    assert record is not None
    assert record.data_kind == ToltecDataKind.Unknown
    assert record.file_ext == ext
    assert (record.obsnum, record.roach) == (145605, 8)


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
    record = guess_info_from_source(f"{stem}.txt")
    assert record is not None
    assert record.data_kind == ToltecDataKind.Unknown


def test_telescope_filename_suffix():
    # User-renamed copy of an actual file; the suffix is not an acquired field.
    record = guess_info_from_source(
        "tel_toltec_2025-10-31_145609_00_0002_custom-tag.nc"
    )
    assert record is not None
    assert record.file_suffix == "custom-tag"
    assert record.data_kind == ToltecDataKind.LmtTel
