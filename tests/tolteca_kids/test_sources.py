"""Shared discovery, catalog selection, and kind-based dispatch tests."""

import json
from datetime import UTC, datetime

import pytest
import xarray as xr
from tollan.config import RuntimeContext
from typer.testing import CliRunner

from tolteca_datamodels.toltec.file import (
    SourceInfoModel,
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.filestore import build_input_catalog, resolve_sources
from tolteca_datamodels.toltec.kids import SweepReducer
from tolteca_datamodels.toltec.kids.source import identify_source
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids.cli import app
from tolteca_kids.core import Kids
from tolteca_kids.runner import run_source


@pytest.mark.parametrize(
    ("filename", "kind"),
    [
        ("toltec0_000001_000_0000_vnasweep.nc", ToltecDataKind.VnaSweep),
        ("toltec0_000001_000_0000_targsweep.nc", ToltecDataKind.TargetSweep),
        ("toltec0_000001_000_0000_tune.nc", ToltecDataKind.Tune),
        ("toltec0_000001_000_0000.nc", ToltecDataKind.RawTimeStream),
        ("toltec0_000001_000_0000_timestream.nc", ToltecDataKind.RawTimeStream),
        (
            "toltec0_000001_000_0000_timestream_processed.nc",
            ToltecDataKind.SolvedTimeStream,
        ),
        (
            "toltec0_000001_000_0000_tune_processed.nc",
            ToltecDataKind.ReducedTargetSweep,
        ),
        (
            "toltec0_000001_000_0000_vnasweep_processed.nc",
            ToltecDataKind.ReducedVnaSweep,
        ),
        ("toltec0_000001_000_0000_tune_kids_find.ecsv", ToltecDataKind.KidsPropTable),
        ("toltec0_000001_000_0000_tune.txt", ToltecDataKind.KidsModelParamsTable),
        ("toltec0_000001_000_0000_tune_targamps.dat", ToltecDataKind.TargAmpsDat),
        ("hwpr_000001_000_0000_timestream.nc", ToltecDataKind.Hwpr),
        ("tel_toltec_2026-09-01_000001_00_0000.nc", ToltecDataKind.LmtTel),
        ("tel_toltec2_2026-09-01_000001_00_0000.nc", ToltecDataKind.LmtTel2),
        (
            "toltec_hk_000001_000_0000_2026_09_01_01_02_03.nc",
            ToltecDataKind.HouseKeeping,
        ),
        ("wyatt_2026-09-01_000001_00_0000.nc", ToltecDataKind.Wyatt),
    ],
)
def test_filename_kinds_and_catalog_roundtrip(filename, kind):
    info = guess_info_from_source(filename)
    assert info is not None
    assert info.data_kind == kind
    assert info.obsnum == 1
    assert info.master is None
    restored = guess_info_from_sources([filename]).toltec_file.to_info_list()[0]
    assert restored.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    ) == info.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )
    assert restored.source.url == info.source.url


def test_unknown_stays_partial_and_master_comes_from_layout(tmp_path):
    assert guess_info_from_source("renamed.nc", raise_on_error=False) is None
    catalog = guess_info_from_sources(
        [
            tmp_path / "renamed.nc",
            tmp_path / "toltec/tcs/toltec0/toltec0_000001_000_0000_tune.nc",
        ]
    )
    unknown, known = catalog.toltec_file.to_info_list()
    assert unknown.obsnum is None
    assert unknown.data_kind == ToltecDataKind.Unknown
    assert known.master == "tcs"
    assert catalog.toltec_file.to_info_list("roach == 0 & obsnum == 1") == [known]


@pytest.mark.parametrize(
    "spec", ["1", "1-0", "1-0-0", "1/0", "1/toltec0", "1-0-0-0", "tcs-1/0"]
)
def test_observation_selectors(tmp_path, spec):
    path = tmp_path / "toltec/tcs/toltec0/toltec0_000001_000_0000.nc"
    path.parent.mkdir(parents=True)
    path.touch()
    catalog = resolve_sources([spec], data_root=tmp_path)
    assert catalog.toltec_file.to_info_list()[0].filepath == path


def test_paths_globs_deduplicate_and_unmatched_raise(tmp_path):
    path = tmp_path / "unknown.nc"
    path.touch()
    catalog = resolve_sources([path, str(tmp_path / "*.nc")])
    assert len(catalog) == 1
    with pytest.raises(FileNotFoundError):
        resolve_sources([str(tmp_path / "*.fits")])
    assert resolve_sources([]).toltec_file.to_info_list() == []


def test_header_enrichment_corrects_guesses():
    data = xr.Dataset({"Header.Toltec.ObsType": 1, "Header.Toltec.RoachIndex": 8})
    info = identify_source(data, SourceInfoModel(source="renamed.nc"))
    assert info.data_kind == ToltecDataKind.RawTimeStream
    assert info.obsnum is None
    assert info.roach == 8
    assert info.interface == "toltec8"
    assert info.file_suffix is None
    guessed = SourceInfoModel(
        source="renamed.nc", roach=0, interface="toltec0", data_kind=ToltecDataKind.Tune
    )
    corrected = identify_source(data, guessed)
    assert corrected.roach == 8
    assert corrected.interface == "toltec8"
    assert corrected.data_kind == ToltecDataKind.RawTimeStream
    assert guessed.roach == 0


def test_original_filename_enrichment_creates_a_fresh_record():
    original = "/recorded/toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
    record = SourceInfoModel(source="renamed.nc")
    assert record.filepath_info_orig is None
    data = xr.Dataset({"Header.Toltec.ObsType": 1, "Header.Toltec.Filename": original})
    enriched = identify_source(data, record)
    assert enriched.source == record.source
    assert enriched.filepath_orig == original
    assert enriched.filepath_info_orig is not None
    assert enriched.filepath_info_orig.suffix == "timestream"
    assert enriched.file_suffix == "timestream"
    assert enriched.file_ext == "nc"
    assert enriched.file_ut == datetime(2025, 10, 31, 1, 2, 3, tzinfo=UTC)
    assert (enriched.obsnum, enriched.subobsnum, enriched.scannum) == (145609, 0, 2)
    assert enriched.roach == 8
    assert enriched.interface == "toltec8"
    assert record.filepath_orig is None
    assert record.filepath_info_orig is None


@pytest.mark.parametrize(
    ("field", "header", "value", "expected"),
    [
        ("roach", "RoachIndex", 9, 9),
        ("obsnum", "ObsNum", 123, 123),
        ("subobsnum", "SubObsNum", 3, 3),
        ("scannum", "ScanNum", 4, 4),
        ("master", "Master", 1, "ics"),
    ],
)
def test_headers_win_with_original_consistency_check_opt_in(
    field, header, value, expected
):
    original = (
        "/data_lmt/toltec/tcs/toltec8/"
        "toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
    )
    data = xr.Dataset(
        {
            "Header.Toltec.Filename": original,
            "Header.Toltec.ObsType": 1,
            f"Header.Toltec.{header}": value,
        }
    )
    record = SourceInfoModel(source="downloaded.nc")
    corrected = identify_source(data, record)
    assert getattr(corrected, field) == expected
    assert corrected.filepath_info_orig is not None
    assert corrected.filepath_info_orig.master_str == "tcs"
    with pytest.raises(ValueError, match=f"Conflicting {field}") as exc_info:
        identify_source(data, record, check_filepath_orig=True)
    assert f"Header.Toltec.{header}" in str(exc_info.value)
    assert original in str(exc_info.value)


def test_original_path_master_overrides_current_directory_guess():
    record = guess_info_from_source(
        "/copied/toltec/ics/toltec8/"
        "toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
    )
    assert record is not None
    assert record.master == "ics"
    data = xr.Dataset(
        {
            "Header.Toltec.Filename": (
                "/data_lmt/toltec/tcs/toltec8/"
                "toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
            ),
            "Header.Toltec.ObsType": 1,
            "Header.Toltec.Master": 0,
        }
    )
    for data_to_read in [data, data.drop_vars("Header.Toltec.Master")]:
        corrected = identify_source(data_to_read, record, check_filepath_orig=True)
        assert corrected.master == "tcs"
        assert corrected.source == record.source


def test_unregistered_master_is_preserved_without_a_header_code():
    original = (
        "/data_lmt/toltec/simu/toltec8/"
        "toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
    )
    data = xr.Dataset({"Header.Toltec.Filename": original, "Header.Toltec.ObsType": 1})
    record = SourceInfoModel(source="downloaded.nc")
    corrected = identify_source(data, record, check_filepath_orig=True)
    assert corrected.master == "simu"
    assert corrected.filepath_info is not None
    assert corrected.filepath_info_orig is not None
    assert corrected.filepath_info.master_str == "simu"
    assert corrected.filepath_info_orig.master_str == "simu"
    with_master = data.assign({"Header.Toltec.Master": 0})
    assert identify_source(with_master, record).master == "tcs"
    with pytest.raises(ValueError, match="Conflicting master"):
        identify_source(with_master, record, check_filepath_orig=True)


def test_processed_kind_and_suffix_are_not_replaced_by_raw_original():
    record = guess_info_from_source(
        "toltec8_145609_000_0002_2025_10_31_01_02_03_vnasweep_processed.nc"
    )
    assert record is not None
    data = xr.Dataset(
        {
            "Header.Toltec.Filename": (
                "/recorded/toltec8_145609_000_0002_2025_10_31_01_02_03_vnasweep.nc"
            ),
            "Header.Kids.kind": "processed_sweep",
        }
    )
    corrected = identify_source(data, record, check_filepath_orig=True)
    assert corrected.data_kind == ToltecDataKind.ReducedVnaSweep
    assert corrected.file_suffix == "vnasweep_processed"
    assert corrected.filepath_info_orig is not None
    assert corrected.filepath_info_orig.suffix == "vnasweep"


@pytest.mark.parametrize("check", [False, True])
def test_reference_headers_match_original_filename(data_lmt_path, monkeypatch, check):
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    reads = []
    getitem = NetCDF4ArrayWrapper._getitem

    def record_read(array, key):
        reads.append(array.variable_name)
        assert array.variable_name.startswith("Header.")
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", record_read)
    for path in sorted((data_lmt_path / "toltec").rglob("*.nc")):
        with xr.open_dataset(path) as data:
            info = identify_source(
                data, SourceInfoModel(source="downloaded.nc"), check_filepath_orig=check
            )
        assert info.filepath_info_orig is not None
        assert info.obsnum == info.filepath_info_orig.obsnum
        assert info.master == info.filepath_info_orig.master_str == "tcs"
        assert info.file_suffix == info.filepath_info_orig.suffix
    assert reads


@pytest.mark.parametrize(
    "suffix_field",
    [
        "file_suffix",
        "file_suffix_orig",
        "filepath_info.suffix",
        "filepath_info_orig.suffix",
    ],
)
def test_cli_selects_original_filename_in_renamed_file(tmp_path, suffix_field):
    path = tmp_path / "downloaded.nc"
    original = "/recorded/toltec8_145609_000_0002_2025_10_31_01_02_03_timestream.nc"
    xr.Dataset(
        {
            "Header.Toltec.ObsType": 1,
            "Header.Toltec.RoachIndex": 8,
            "Header.Toltec.Filename": original,
        }
    ).to_netcdf(path, engine="scipy")
    result = CliRunner().invoke(
        app,
        [
            "--show_inputs",
            "--select",
            f"roach == 8 & {suffix_field} == 'timestream'",
            "--",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.exception
    [row] = json.loads(result.stdout)
    assert row["source"] == str(path)
    assert row["filepath_orig"] == original
    assert row["filepath_info_orig"]["suffix"] == "timestream"
    assert row["filepath_info_orig"]["ut"] == "2025-10-31T01:02:03.000Z"


@pytest.mark.parametrize(
    "kind",
    [
        ToltecDataKind.RawTimeStream,
        ToltecDataKind.SolvedTimeStream,
        ToltecDataKind.ReducedSweep,
        ToltecDataKind.Unknown,
    ],
)
def test_unsupported_kind_never_enters_sweep_pipeline(kind, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported input invoked reduction")

    monkeypatch.setattr(SweepReducer, "__call__", unexpected)
    with pytest.raises(NotImplementedError, match=kind.name):
        Kids(RuntimeContext()).pipeline_for(kind)


def test_cli_selection_and_timestream_rejection(tmp_path, monkeypatch):
    path = tmp_path / "toltec8_000001_000_0000_timestream.nc"
    xr.Dataset({"Header.Toltec.ObsType": 1}).to_netcdf(path, engine="scipy")

    def unexpected(*args, **kwargs):
        pytest.fail("metadata-only command invoked reduction")

    monkeypatch.setattr(SweepReducer, "__call__", unexpected)
    result = CliRunner().invoke(
        app,
        [
            str(path),
            "--show_inputs",
            "--select",
            "roach == 8 & data_kind == 'RawTimeStream'",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout)[0]["data_kind"] == "RawTimeStream"
    info = build_input_catalog([path]).toltec_file.to_info_list()[0]
    with pytest.raises(NotImplementedError, match="RawTimeStream"):
        run_source(info, RuntimeContext())
