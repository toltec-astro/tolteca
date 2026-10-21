"""Queries and row-wise access to projected header metadata."""

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import ModelTableAccessor
from tollan.config.sources.base import ConfigSourceBase
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.file import ToltecFileAccessor, read_info_from_sources
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata


@pytest.fixture
def catalog(data_lmt_path):
    return read_info_from_sources(
        [
            *sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*1456*.nc")),
            data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        ]
    )


def test_header_columns_support_queries(catalog):
    assert len(catalog.toltec_file.query("atten_drive == 7.25")) == 4
    assert len(catalog.toltec_file.query("obs_pgm.notna()")) == 1
    assert len(catalog.toltec_file.query("obs_pgm == 'lissajous'")) == 1
    assert catalog.toltec_file.query("obs_pgm == 'Lissajous'").empty


def test_header_columns_supply_config_context(catalog):
    for record in catalog.itertuples(index=False):
        context = record._asdict()
        assert context["master"] == record.master
        assert context["master_code"] == record.master_code
        condition_master = ConfigSourceBase(
            order=0, enable_if="master == 'tcs' & master_code == 0"
        )
        assert condition_master.is_enabled_for(context)
        assert "metadata" not in context
        condition = ConfigSourceBase(order=0, enable_if="atten_drive == 7.25")
        assert condition.is_enabled_for(context) == (
            pd.notna(record.roach) and record.roach == 0
        )
        assert context["atten_drive"] is record.atten_drive


def test_master_label_and_header_code_have_distinct_table_columns(catalog):
    assert not {"master_str", "file_master_str"}.intersection(catalog)
    assert catalog["master"].tolist() == ["tcs"] * len(catalog)
    assert catalog["master_code"].tolist() == [0] * len(catalog)
    assert catalog["file_master"].dtype == pd.StringDtype()
    assert all(info.master == "tcs" for info in catalog.filepath_info)
    selected = catalog.toltec_file.query("file_master == master & master_code == 0")
    assert len(selected) == len(catalog)
    assert "obs_type_code" in catalog
    assert "obs_type" in catalog
    assert "array" in catalog
    assert catalog.toltec_file.query("obs_type == 'tune'")[
        "obs_type_code"
    ].tolist() == [4]


def test_filename_only_and_empty_catalogs_do_not_load_metadata(catalog):
    for paths in ([], catalog["source"].tolist()):
        frame = ToltecFileAccessor.from_sources(paths)
        assert "metadata" not in frame
        assert "atten_drive" not in frame
        assert all(
            not hasattr(item, "metadata")
            for item in list(frame.itertuples(index=False))
        )


def test_projection_uses_actual_typed_reader_values(catalog):
    for record in catalog.itertuples(index=False):
        opener = open_lmt_tel if record.interface == "tel" else open_toltec_kids
        with opener(fileloc(record.source).path) as data:
            metadata = (
                data.lmt_tel.meta
                if record.interface == "tel"
                else data.toltec_kids.meta
            )
            assert record.t0_mjd == metadata.t0.utc.mjd
            assert record.t1_mjd == metadata.t1.utc.mjd
            assert record.master == metadata.master
            assert record.uid_obs == metadata.uid_obs
            assert record.uid_raw_obs == metadata.uid_raw_obs
            assert record.uid_raw_obs_file == metadata.uid_raw_obs_file
            if record.interface == "tel":
                assert isinstance(metadata, LmtTelMetadata)
                assert metadata.target is not None
                assert record.target_ra_deg == metadata.target.ra.deg
                assert record.target_dec_deg == metadata.target.dec.deg
                assert record.t_int_s == metadata.t_int.to_value("s")
            else:
                assert isinstance(metadata, ToltecRawObsMetadata)
                assert record.f_smp_Hz == metadata.f_smp.to_value("Hz")
            assert record.filepath_info.obsnum == record.file_obsnum
            assert "metadata" not in catalog


def test_header_read_qualifies_masterless_filename_uids(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    guessed = ToltecFileAccessor.from_sources(
        [{"source": path, "source_loc": path.name}]
    )
    assert guessed.master.isna().all()
    assert guessed.uid_obs.tolist() == ["145609"]
    assert guessed.uid_raw_obs.tolist() == ["145609-0-1"]
    read = guessed.toltec_file.update_from_source_metadata()
    assert read.uid_obs.tolist() == ["tcs-145609"]
    assert read.uid_raw_obs.tolist() == ["tcs-145609-0-1"]
    assert read.uid_raw_obs_file.tolist() == ["tcs-145609-0-1-toltec0"]
    assert guessed.uid_obs.tolist() == ["145609"]


def test_projection_and_context_do_not_parse_original_paths(catalog, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Header projection parsed an original filename")

    records = list(catalog.itertuples(index=False))
    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    for record in records:
        opener = open_lmt_tel if record.interface == "tel" else open_toltec_kids
        with opener(fileloc(record.source).path) as data:
            metadata = (
                data.lmt_tel.meta
                if record.interface == "tel"
                else data.toltec_kids.meta
            )
            projected = ModelTableAccessor.from_models(
                [metadata], columns=ToltecFileAccessor._metadata_columns
            )
        assert projected is not None
        assert "filepath_info_orig" not in projected
        context = record._asdict()
        assert "filepath_info_orig" not in context
        assert "file_suffix" in context


def test_selected_rows_keep_header_context_aligned(catalog):
    selected = catalog.iloc[[4, 0, 2]]
    for record, position in zip(
        selected.itertuples(index=False), [4, 0, 2], strict=True
    ):
        assert record.source == catalog.iloc[position]["source"]
        assert record.obsnum == catalog.iloc[position]["obsnum"]
        assert record.scannum == catalog.iloc[position]["scannum"]
        assert record.t0_mjd == catalog.iloc[position]["t0_mjd"]


def test_extra_columns_are_accessible_on_rows(catalog):
    before = catalog.copy()
    catalog["returncode"] = 0
    catalog["data_obj"] = [object() for _ in range(len(catalog))]
    pd.testing.assert_frame_equal(catalog[before.columns], before)
    for position, row in enumerate(catalog.itertuples(index=False)):
        assert row.returncode == 0
        assert row.data_obj is catalog.iloc[position]["data_obj"]
