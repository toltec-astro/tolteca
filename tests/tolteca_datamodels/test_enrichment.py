"""Filename guesses and authoritative metadata are explicit ordered passes."""

from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from tolteca_datamodels.file import FileAccessor, FilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.model_table import (
    ModelColumnMapper,
    ModelTableAccessor,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimeStreamMetadata,
)
from tolteca_datamodels.toltec.source import read_source
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_metadata_declarations_are_compatible():
    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            QuantityColumnRule(physical_type="frequency", unit="Hz"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
        include_computed=True,
        default=None,
    )
    declared = {}
    for model in (
        LmtTelMetadata,
        ToltecRawObsMetadata,
        ToltecTimeStreamMetadata,
        ToltecRawObsBlockMetadata,
        ToltecSweepMetadata,
    ):
        for name, column in mapper(model).items():
            if name in declared:
                previous = declared[name]
                assert (previous.attr, previous.dtype) == (column.attr, column.dtype)
                if isinstance(previous.converter, partial):
                    assert isinstance(column.converter, partial)
                    assert previous.converter.func is column.converter.func
                    assert previous.converter.args == column.converter.args
                    assert previous.converter.keywords == column.converter.keywords
                else:
                    assert previous.converter is column.converter
            declared[name] = column
    assert set(declared) == set(ToltecFileAccessor._metadata_columns)


def test_metadata_mapping_reads_duck_objects_without_model_validation():
    columns = ToltecFileAccessor._metadata_columns
    objects = [SimpleNamespace(obsnum=3, uid_raw_obs="example"), object()]
    table = ModelTableAccessor.from_models(objects, columns=columns)
    assert table["obsnum"].iloc[0] == 3
    assert table["uid_raw_obs"].iloc[0] == "example"
    assert table.iloc[1].isna().all()


def test_remote_rows_have_usable_identifiers_without_io(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote table construction attempted file access")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(ToltecFileAccessor, "_source_opener", xr.open_dataset)
    table = guess_info_from_sources(
        [
            (
                "https://example.invalid/toltec/tcs/toltec8/"
                "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
            ),
            "https://example.invalid/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        ]
    )
    first, second = table.itertuples(index=False)
    assert first.uid_raw_obs_file == "tcs-145605-0-1-toltec8"
    assert first.instru == "toltec"
    assert second.obsnum == 145605 and pd.isna(second.uid_raw_obs)


def test_headers_override_guesses_and_own_derived_identifiers(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    table = guess_info_from_sources([path])
    guessed = table.iloc[0]
    with xr.open_dataset(path) as raw:
        changed = raw.assign({"Header.Toltec.ObsNum": int(guessed.obsnum) + 1})
        with table.toltec_file.open(io_objs=[changed]) as opened:
            row = next(opened.itertuples(index=False))
            assert row.obsnum == guessed.obsnum + 1
            assert row.file_obsnum == guessed.obsnum
            assert row.uid_raw_obs_file == changed.toltec_kids.meta.uid_raw_obs_file
            assert row.uid_raw_obs_file != guessed.uid_raw_obs_file
    assert table.iloc[0].obsnum == guessed.obsnum


def test_guessed_columns_are_not_reactive():
    table = guess_info_from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    uid = table.loc[0, "uid_raw_obs"]
    table.loc[0, "obsnum"] = 3
    assert table.loc[0, "uid_raw_obs"] == uid


def test_generic_file_sources_need_no_record_or_filename_schema(tmp_path):
    table = FileAccessor.from_sources(
        [{"source": tmp_path / "cache", "source_loc": "archive:/file.nc"}]
    )
    assert list(table) == [
        "source",
        "source_loc",
        "source_loc_is_local",
        "source_loc_host",
        "source_loc_scheme",
    ]
    assert FileAccessor(table).source.tolist() == [tmp_path / "cache"]
    assert FilepathInfo.from_source("unrecognized.nc", raise_on_error=False) is None


def test_unknown_explicit_location_never_falls_back_to_cache_filename():
    table = guess_info_from_sources(
        [
            {
                "source": "tel_toltec_2025-10-31_145609_00_0002.nc",
                "source_loc": "archive:/unknown.nc",
            }
        ],
    )
    assert pd.isna(table.iloc[0].obsnum)
    assert pd.isna(table.iloc[0].file_prefix)


def test_partial_projection_nulls_override_guesses_without_touching_other_rows():
    table = guess_info_from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"] * 2)
    result = ModelTableAccessor(table).update_from_models(
        {1: SimpleNamespace(obsnum=3, uid_raw_obs=None, uid_raw_obs_file=None)},
        columns=ToltecFileAccessor._metadata_columns,
    )
    assert pd.notna(result["uid_raw_obs"].iloc[0])
    assert pd.isna(result["uid_raw_obs"].iloc[1])
    assert result["obsnum"].tolist() == [145609, 3]
    pd.testing.assert_series_equal(result["file_obsnum"], table["file_obsnum"])


def test_minimal_source_table_can_be_enriched(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    table = FileAccessor.from_sources([path]).assign(user_tag="selected")
    read = table.toltec_file.read_metadata()
    assert read["user_tag"].tolist() == ["selected"]
    assert read["obsnum"].tolist() == [145609]
    assert read["data_kind"].tolist() == [ToltecDataKind.tune.value]
    assert not any(name.startswith("file_") for name in read)


def test_rejected_payload_keeps_guesses_and_records_open_failure():
    table = guess_info_from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    table["user_tag"] = "selected"
    with table.toltec_file.open(io_objs=[xr.Dataset()]) as opened:
        pd.testing.assert_frame_equal(
            opened.drop(columns=["diagnostics", "io_obj"]), table
        )
        assert opened.io_obj.iloc[0] is None
        [diagnostic] = opened.diagnostics.iloc[0]
        assert diagnostic.stage == "open"
        assert diagnostic.exception_type == "ValueError"


def test_processed_payload_supplies_kind_not_raw_metadata(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    with xr.open_dataset(path) as raw:
        processed = raw.assign_attrs({"Header.Kids.kind": "processed_sweep"})
        metadata, kind = read_source(processed)
    assert metadata is None
    assert kind == ToltecDataKind.reduced_sweep


def test_fresh_header_read_does_not_reuse_cached_view_metadata(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc"))
    with xr.open_dataset(path) as raw:
        cached = raw.toltec_kids.meta
        first, kind = read_source(raw)
        raw["Header.Toltec.DriveAtten"] = 1.25
        second, kind_second = read_source(raw)
        assert raw.toltec_kids.read_metadata().atten_drive == 1.25
        assert raw.toltec_kids.meta is cached
    assert isinstance(first, ToltecRawObsMetadata)
    assert isinstance(second, ToltecRawObsMetadata)
    assert first.atten_drive == cached.atten_drive == 7.25
    assert second.atten_drive == 1.25
    assert kind_second == kind == ToltecDataKind.targsweep


def test_telescope_source_read_sees_headers_changed_after_cached_metadata(
    data_lmt_path,
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as raw:
        cached = raw.lmt_tel.meta
        raw["Header.Dcs.ObsNum"] = cached.obsnum + 1
        metadata, kind = read_source(raw)
        assert metadata.obsnum == cached.obsnum + 1
        assert kind == ToltecDataKind.lmt_tel
        assert raw.lmt_tel.meta is cached


@pytest.mark.parametrize(
    "original",
    [
        "unrecognized.nc",
        "tel_toltec_2025-10-31_145609_00_0002.unsupported",
    ],
)
def test_unknown_original_telescope_name_does_not_fall_back_to_guess(
    data_lmt_path, original
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    table = guess_info_from_sources([path])
    with xr.open_dataset(path) as raw:
        changed = raw.assign({"Header.File.Name": original})
        metadata, kind = read_source(changed)
        assert isinstance(metadata, LmtTelMetadata)
        assert kind == ToltecDataKind.unknown
        with table.toltec_file.open(io_objs=[changed]) as opened:
            assert opened["filepath_orig"].iloc[0] == original
            assert opened["file_prefix"].iloc[0] == "tel_toltec"
            assert opened["data_kind"].iloc[0] == kind.value
            assert opened["obsnum"].iloc[0] == metadata.obsnum


def test_misnamed_file_keeps_whole_filename_provenance(data_lmt_path):
    tel = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    kids = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    # Deliberately incorrect identification tests correction, not an acquisition name.
    table = guess_info_from_sources([{"source": kids, "source_loc": tel}])
    read = table.toltec_file.read_metadata()
    assert read["instru"].iloc[0] == "toltec"
    assert read["instru_component"].iloc[0] == "roach"
    assert read["interface"].iloc[0] == "toltec0"
    assert read["data_kind"].iloc[0] == ToltecDataKind.tune.value
    provenance = [name for name in table if str(name).startswith("file_")]
    pd.testing.assert_frame_equal(read[provenance], table[provenance])


def test_original_header_filename_does_not_fill_missing_identifiers(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    table = guess_info_from_sources([path])
    with xr.open_dataset(path) as raw:
        missing = raw.drop_vars("Header.Toltec.ObsNum")
        with (
            pytest.raises(ValueError, match="Header.Toltec.ObsNum"),
            table.toltec_file.open(io_objs=[missing], on_error="raise"),
        ):
            pytest.fail("Missing required header was filled from a filename")


def test_read_metadata_overwrites_prefilled_identifiers(data_lmt_path, tmp_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    cached = tmp_path / "download.nc"
    cached.symlink_to(path)
    table = guess_info_from_sources([cached])
    table["obsnum"] = 1
    table["master"] = "ics"
    read = table.toltec_file.read_metadata()
    assert read["obsnum"].iloc[0] == 145609
    assert read["master"].iloc[0] == "tcs"
    assert pd.isna(read["file_obsnum"].iloc[0])
