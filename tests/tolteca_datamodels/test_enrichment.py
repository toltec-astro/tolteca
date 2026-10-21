"""Filename guesses and authoritative metadata are explicit ordered passes."""

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr
from tollan.accessor.pandas.model_table import (
    ModelColumnMapper,
    ModelTableAccessor,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.hwpr import ToltecHwprMetadata
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_metadata_declarations_are_compatible():
    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule.model_validate({"physical_type": "time", "unit": "s"}),
            QuantityColumnRule.model_validate(
                {"physical_type": "frequency", "unit": "Hz"}
            ),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule.model_validate({"attrs": ["ra", "dec"], "unit": "deg"}),
            ScalarColumnRule(),
        ],
        include_computed=True,
        default=None,
    )
    declared = {}
    for model in (
        LmtTelMetadata,
        ToltecHwprMetadata,
        ToltecRawObsMetadata,
        ToltecTimestreamMetadata,
        ToltecRawObsBlockMetadata,
        ToltecSweepMetadata,
    ):
        for name, column in mapper(model).items():
            if name in declared:
                previous = declared[name]
                assert (previous.dtype, previous.default) == (
                    column.dtype,
                    column.default,
                )
            declared[name] = column
    assert set(declared) | {"data_models"} == set(ToltecFileAccessor._metadata_columns)
    assert ToltecFileAccessor._metadata_columns["data_models"].dtype is object


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
    assert first.uid_obs == "tcs-145605"
    assert first.uid_raw_obs_file == "tcs-145605-0-1-toltec8"
    assert first.instru == "toltec"
    assert second.obsnum == 145605 and pd.isna(second.master)
    assert second.uid_obs == "145605"
    assert second.uid_raw_obs == "145605-0-1"
    assert second.uid_raw_obs_file == "145605-0-1-toltec8"


def test_headers_override_guesses_and_own_derived_identifiers(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    table = guess_info_from_sources([path])
    guessed = table.iloc[0]
    with xr.open_dataset(path) as raw:
        changed = raw.assign({"Header.Toltec.ObsNum": int(guessed.obsnum) + 1})
        with table.toltec_file.open(io_objs=[changed]) as opened:
            opened.toltec_file.update_from_source_metadata(inplace=True)
            row = next(opened.itertuples(index=False))
            assert row.obsnum == guessed.obsnum + 1
            assert row.file_obsnum == guessed.obsnum
            assert row.uid_raw_obs_file == changed.toltec_kids.meta.uid_raw_obs_file
            assert row.uid_raw_obs_file != guessed.uid_raw_obs_file
    assert table.iloc[0].obsnum == guessed.obsnum


def test_generic_unidentified_sources_preserve_locations_and_nullable_guesses(tmp_path):
    table = FileAccessor.from_sources(
        [{"source": tmp_path / "cache", "source_loc": "archive:/file.nc"}]
    )
    assert list(table)[:5] == [
        "source",
        "source_loc",
        "source_loc_is_local",
        "source_loc_host",
        "source_loc_scheme",
    ]
    assert FileAccessor(table).source.tolist() == [tmp_path / "cache"]
    assert table.file_prefix.isna().all()
    assert table.data_kind_toltec.eq(ToltecDataKind.unknown.value).all()
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
    ] == ["filepath"]
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


def test_generic_source_table_can_be_enriched(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    table = FileAccessor.from_sources([path]).assign(user_tag="selected")
    read = table.toltec_file.update_from_source_metadata()
    assert read["user_tag"].tolist() == ["selected"]
    assert read["obsnum"].tolist() == [145609]
    assert read["data_kind_toltec"].tolist() == [ToltecDataKind.tune.value]
    filename_columns = [
        name for name in table if isinstance(name, str) and name.startswith("file_")
    ]
    pd.testing.assert_frame_equal(read[filename_columns], table[filename_columns])


def test_rejected_payload_keeps_guesses_and_records_identification_failure():
    table = guess_info_from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    table["user_tag"] = "selected"
    with table.toltec_file.open(io_objs=[xr.Dataset()]) as opened:
        pd.testing.assert_frame_equal(
            opened[table.columns].drop(columns="source_trace"),
            table.drop(columns="source_trace"),
        )
        assert isinstance(opened.io_obj.iloc[0], xr.Dataset)
        assert opened.accessor_obj.iloc[0] is None
        [diagnostic] = opened.source_trace.map(source_trace_issues).iloc[0]
        assert diagnostic.stage == "identify"
        assert diagnostic.exception_type == "ValueError"


def test_processed_payload_remains_openable_pending_metadata_support(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as raw:
        processed = raw.assign_attrs({"Header.Kids.kind": "processed_sweep"})
        assert processed.toltec_kids.meta is None
        table = ToltecFileAccessor.from_sources(
            [{"source": processed, "source_loc": path}], mode="read"
        )
        with table.toltec_file.open(on_error="raise") as opened:
            assert opened.io_obj.iloc[0] is processed
            reader = opened.accessor_obj.iloc[0]
            assert reader.data_kind is ToltecDataKind.reduced_sweep
            assert reader.meta is None
            assert not processed["Data.Toltec.Is"].variable._in_memory
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.reduced_sweep.value]
    assert "atten_drive" not in table
    assert table.obsnum.tolist() == [145609]
    assert table.source_trace.map(source_trace_issues).iloc[0] == []
    assert table.metadata_read.tolist() == [False]
    [metadata_event] = [
        event for event in table.source_trace.iloc[0] if event.stage == "metadata"
    ]
    assert metadata_event.outcome == "skipped"
    assert "no metadata" in metadata_event.message


@pytest.mark.parametrize(
    "original",
    [
        "unrecognized.nc",
        "tel_toltec_2025-10-31_145609_00_0002.unsupported",
    ],
)
def test_receiver_identifies_telescope_independently_of_original_filename(
    data_lmt_path, original
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    table = guess_info_from_sources([path])
    with xr.open_dataset(path) as raw:
        changed = raw.assign({"Header.File.Name": original})
        metadata = changed.lmt_tel.meta
        assert isinstance(metadata, LmtTelMetadata)
        with table.toltec_file.open(io_objs=[changed]) as opened:
            opened.toltec_file.update_from_source_metadata(inplace=True)
            assert opened["filepath_orig"].iloc[0] == original
            assert opened["file_prefix"].iloc[0] == "tel_toltec"
            assert opened["data_kind_toltec"].iloc[0] == ToltecDataKind.lmt_tel.value
            assert opened["obsnum"].iloc[0] == metadata.obsnum


def test_original_header_filename_does_not_fill_missing_identifiers(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    table = guess_info_from_sources([path])
    with xr.open_dataset(path) as raw:
        missing = raw.drop_vars("Header.Toltec.ObsNum")
        with (
            table.toltec_file.open(io_objs=[missing], on_error="raise") as opened,
            pytest.raises(ValueError, match="Header.Toltec.ObsNum"),
        ):
            opened.toltec_file.update_from_source_metadata(
                inplace=True, on_error="raise"
            )


def test_read_metadata_identifies_a_cached_unrecognized_filename(
    data_lmt_path, tmp_path
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    cached = tmp_path / "download.nc"
    cached.symlink_to(path)
    table = guess_info_from_sources([cached])
    assert pd.isna(table.obsnum.iloc[0])
    assert pd.isna(table.master.iloc[0])
    read = table.toltec_file.update_from_source_metadata()
    assert read["obsnum"].iloc[0] == 145609
    assert read["master"].iloc[0] == "tcs"
    assert pd.isna(read["file_obsnum"].iloc[0])
