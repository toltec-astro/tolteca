"""Typed acquisition metadata in source records and Arrow catalogs."""

import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from tollan.config.sources.base import ConfigSourceBase

from tolteca_datamodels.file import (
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
)
from tolteca_datamodels.lmt import LmtTelMetadata, open_lmt_tel
from tolteca_datamodels.toltec import open_toltec


@pytest.fixture
def records(data_lmt_path):
    result = []
    for path in sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*.nc")):
        with open_toltec(path) as data:
            meta = data.toltec_kids.meta_header
            info = guess_info_from_source(path)
            assert info is not None
            fields = info.model_dump(round_trip=True)
            fields["metadata"] = meta
            result.append(SourceInfoModel.model_validate(fields))
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with open_lmt_tel(path) as data:
        info = guess_info_from_source(path)
        assert info is not None
        fields = info.model_dump(round_trip=True)
        fields["metadata"] = data.lmt_tel.meta
        result.append(SourceInfoModel.model_validate(fields))
    return result


def test_metadata_roundtrips_json_and_mixed_parquet(records, tmp_path):
    adapter = TypeAdapter(list[SourceInfoModel])
    payload = adapter.dump_json(records, round_trip=True)
    restored = adapter.validate_json(payload)
    assert [type(item.metadata) for item in restored] == [
        type(item.metadata) for item in records
    ]
    assert adapter.dump_json(restored, round_trip=True) == payload
    for source, restored_item in zip(records, restored, strict=True):
        assert restored_item.metadata is not None
        assert source.metadata.t0 == restored_item.metadata.t0
        assert source.metadata.t1 == restored_item.metadata.t1
        assert source.metadata.t_int == restored_item.metadata.t_int
        if source.interface == "tel":
            assert isinstance(restored_item.metadata, LmtTelMetadata)
            assert source.metadata.target == restored_item.metadata.target
            assert source.metadata.target_ref == restored_item.metadata.target_ref
        flat = source.model_dump(context={"flat": True}, round_trip=True)
        assert "atten_drive" not in flat
        assert type(SourceInfoModel.model_validate(flat).metadata) is type(
            source.metadata
        )
    frame = ToltecFileAccessor.from_info_list(records)
    assert isinstance(frame["metadata"].dtype, pd.ArrowDtype)
    assert "atten_drive" not in frame
    path = tmp_path / "catalog.parquet"
    frame.to_parquet(path)
    loaded = pd.read_parquet(path, dtype_backend="pyarrow")
    restored = loaded.toltec_file.to_info_list()
    assert adapter.dump_json(restored, round_trip=True) == payload


def test_metadata_aliases_are_vectorized_and_match_config_context(records, monkeypatch):
    frame = ToltecFileAccessor.from_info_list(records)

    def unexpected(*args, **kwargs):
        pytest.fail("A catalog query serialized or restored Python metadata")

    monkeypatch.setattr(SourceInfoModel, "model_validate", unexpected)
    monkeypatch.setattr(SourceInfoModel, "model_dump", unexpected)
    selected = frame.toltec_file.query("atten_drive == 7.25 & roach == 0")
    assert len(selected) == 4
    assert len(frame.toltec_file.query("metadata.atten_drive == 7.25")) == 4
    assert len(frame.toltec_file.query("obs_pgm == metadata.obs_pgm")) == 1
    # File-level master labels win the unqualified name over raw numeric codes.
    assert len(frame.toltec_file.query("master == 'tcs' & metadata.master == 0")) == 5


def test_flat_metadata_aliases_are_available_for_conditions(records):
    for record in records:
        context = record.model_dump(
            mode="json",
            context={"flat": True},
            exclude={"filepath_info", "filepath_info_orig"},
        )
        assert context["master"] == record.master
        condition = ConfigSourceBase(order=0, enable_if="atten_drive == 7.25")
        assert condition.is_enabled_for(context) == (record.roach == 0)


def test_metadata_requires_matching_known_kind(records):
    fields = records[0].model_dump(round_trip=True)
    for kind in ("Unknown", "not_a_kind"):
        with pytest.raises(ValidationError):
            SourceInfoModel.model_validate({**fields, "data_kind": kind})
    with pytest.raises(ValidationError, match="requires ToltecTimeStreamMetadata"):
        SourceInfoModel.model_validate(
            {**fields, "data_kind": "RawTimeStream", "metadata": records[0].metadata}
        )


def test_filename_only_and_empty_catalogs_keep_metadata_absent(records):
    record = guess_info_from_source(records[0].source)
    assert record is not None
    for rows in ([], [record]):
        frame = ToltecFileAccessor.from_info_list(rows)
        assert all(item.metadata is None for item in frame.toltec_file.to_info_list())
