"""Filename identification preserves local and remote source locations."""

from pathlib import Path

import pytest
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.toltec.file import (
    FilepathInfo,
    SourceInfoModel,
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


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
    assert info.interface == "tel_toltec"
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


@pytest.mark.parametrize("master_str", ["tcs", "ics", "clip", "simu", "custom-master"])
@pytest.mark.parametrize(
    "prefix", ["/data_lmt", "host:/data_lmt", "https://example.org"]
)
def test_current_and_original_paths_use_the_same_parser(master_str, prefix):
    original = (
        f"{prefix}/toltec/{master_str}/toltec8/"
        "toltec8_145609_000_0002_2025_10_31_01_02_03_tune.nc"
    )
    parsed = FilepathInfo.from_path(original)
    current = guess_info_from_source(original)
    source = fileloc("downloaded.nc")
    fallback = guess_info_from_source(source, filepath_orig=original)
    assert parsed is not None and current is not None and fallback is not None
    assert parsed.master_str == master_str
    assert "master" not in parsed.model_dump()
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
        f"{directory}/toltec8_145609_000_0002_2025_10_31_01_02_03_tune.nc"
    )
    assert parsed is not None
    assert parsed.master_str is None


def test_original_telescope_filename_fallback():
    original = "/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    info = guess_info_from_source("downloaded.nc", filepath_orig=original)
    assert info is not None
    assert info.interface == "tel_toltec"
    assert info.obsnum == 145609
    assert info.scannum == 2
    assert info.file_suffix is None
    assert info.master is None


@pytest.mark.parametrize("suffix", ["", "_tune_processed"])
def test_recognized_current_filename_retains_its_fields(suffix):
    original = (
        "/data_lmt/toltec/tcs/toltec8/"
        "toltec8_145609_000_0002_2025_10_31_01_02_03_tune.nc"
    )
    current = (
        "/copied/toltec/ics/toltec9/"
        f"toltec9_000001_002_0003_2025_10_30_00_00_00{suffix}.nc"
    )
    info = guess_info_from_source(current, filepath_orig=original)
    assert info is not None
    assert info.source.path.name == Path(current).name
    assert info.roach == 9
    assert (info.obsnum, info.subobsnum, info.scannum) == (1, 2, 3)
    assert info.file_ut is not None
    assert info.file_ut.day == 30
    assert info.file_suffix == (suffix.removeprefix("_") or None)
    assert info.master == "tcs"


def test_unknown_original_filename_does_not_invent_identification():
    assert (
        guess_info_from_source(
            "downloaded.nc", filepath_orig="/recorded/unknown.nc", raise_on_error=False
        )
        is None
    )


@pytest.mark.parametrize(
    ("ending", "kind"),
    [
        ("vnasweep.nc", ToltecDataKind.VnaSweep),
        ("vnasweep_processed.nc", ToltecDataKind.ReducedVnaSweep),
        ("targsweep_processed.nc", ToltecDataKind.ReducedTargetSweep),
        ("tune_processed.nc", ToltecDataKind.ReducedTargetSweep),
        ("timestream_processed.nc", ToltecDataKind.SolvedTimeStream),
        ("custom.nc", ToltecDataKind.Unknown),
        ("custom-tag.nc", ToltecDataKind.Unknown),
        ("vnasweep.txt", ToltecDataKind.KidsModelParamsTable),
        ("tune_kids_find.ecsv", ToltecDataKind.KidsPropTable),
        ("tune_targamps.dat", ToltecDataKind.TargAmpsDat),
    ],
)
def test_kind_rules_keep_processed_and_unknown_suffixes_distinct(ending, kind):
    record = guess_info_from_source(
        f"toltec8_145609_000_0002_2025_10_31_01_02_03_{ending}"
    )
    assert record is not None
    assert record.data_kind == kind
    assert record.file_suffix == ending.rsplit(".", 1)[0]


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
    record = guess_info_from_source(
        "tel_toltec_2025-10-31_145609_00_0002_custom-tag.nc"
    )
    assert record is not None
    assert record.file_suffix == "custom-tag"
    assert record.data_kind == ToltecDataKind.LmtTel


@pytest.mark.parametrize("roach", range(13))
@pytest.mark.parametrize(
    ("suffix", "ext", "kind"),
    [
        ("", "nc", ToltecDataKind.RawTimeStream),
        ("timestream", "nc", ToltecDataKind.RawTimeStream),
        ("timestream_processed", "nc", ToltecDataKind.SolvedTimeStream),
        ("vnasweep", "nc", ToltecDataKind.VnaSweep),
        ("targsweep", "nc", ToltecDataKind.TargetSweep),
        ("targetsweep", "nc", ToltecDataKind.TargetSweep),
        ("tune", "nc", ToltecDataKind.Tune),
        ("vnasweep_processed", "nc", ToltecDataKind.ReducedVnaSweep),
        ("targsweep_processed", "nc", ToltecDataKind.ReducedTargetSweep),
        ("tune_processed", "nc", ToltecDataKind.ReducedTargetSweep),
        ("sweep", "nc", ToltecDataKind.ReducedSweep),
        *[
            (sweep, "txt", ToltecDataKind.KidsModelParamsTable)
            for sweep in ("vnasweep", "targsweep", "tune")
        ],
        *[
            (f"{sweep}_targamps", "dat", ToltecDataKind.TargAmpsDat)
            for sweep in ("vnasweep", "targsweep", "tune")
        ],
        *[
            (f"{sweep}_{product}", "ecsv", ToltecDataKind.KidsPropTable)
            for sweep in ("vnasweep", "targsweep", "tune")
            for product in ("kids_find", "kids_fit", "chan_prop", "adrv")
        ],
    ],
)
def test_all_kids_filename_kinds(roach, suffix, ext, kind):
    ending = f"_{suffix}" if suffix else ""
    parsed = FilepathInfo.from_path(
        f"toltec{roach}_145609_000_0002_2025_10_31_01_02_03{ending}.{ext}"
    )
    assert parsed is not None
    assert parsed.guess_data_kind() == kind


@pytest.mark.parametrize(
    ("interface", "kind"),
    [
        ("tel_toltec", ToltecDataKind.LmtTel),
        ("tel_toltec2", ToltecDataKind.LmtTel2),
        ("tel_test", ToltecDataKind.LmtTel),
        ("hwpr", ToltecDataKind.Hwpr),
        ("hk", ToltecDataKind.HouseKeeping),
        ("toltec_hk", ToltecDataKind.HouseKeeping),
        ("wyatt", ToltecDataKind.Wyatt),
    ],
)
@pytest.mark.parametrize("suffix", [None, "custom", "tune", "timestream_processed"])
def test_auxiliary_kinds_do_not_depend_on_suffix(interface, kind, suffix):
    parsed = FilepathInfo(
        interface=interface,
        roach=None,
        master_str=None,
        obsnum=145609,
        subobsnum=0,
        scannum=2,
        ut=None,
        suffix=suffix,
        ext="nc",
    )
    assert parsed.guess_data_kind() == kind
