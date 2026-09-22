"""Source-record serialization for models, catalogs, and config conditions."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.file import (
    FilepathInfo,
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
)
from tolteca_datamodels.toltec.file import FilepathInfo as ToltecFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture
def record(tmp_path, toltecdb_records):
    row = next(
        row
        for row in toltecdb_records
        if row["ObsNum"] == 145605 and row["RoachIndex"] == 8
    )
    source = fileloc(
        Path(row["FileName"]).name,
        local_parent_path=tmp_path,
    )
    return guess_info_from_source(source)


def test_normal_serialization_retains_location_and_typed_aliases(record):
    dumped = record.model_dump()
    assert dumped["source"] == record.source.model_dump()
    assert dumped["data_kind"] is ToltecDataKind.VnaSweep
    assert "data_kind" not in dumped["filepath_info"]
    assert dumped["file_ut"] == datetime(2025, 10, 31, 9, 20, 6, tzinfo=UTC)
    assert dumped["file_suffix"] == "vnasweep"
    assert dumped["file_ext"] == "nc"
    dumped_json = record.model_dump(mode="json")
    assert dumped_json["source"] == record.source.model_dump(mode="json")
    assert dumped_json["data_kind"] == ToltecDataKind.VnaSweep.value
    assert "data_kind" not in dumped_json["filepath_info"]
    assert dumped_json["file_ut"] == "2025-10-31T09:20:06Z"
    restored = SourceInfoModel.model_validate_json(
        record.model_dump_json(round_trip=True)
    )
    assert restored == record
    assert restored.source.path_orig == record.source.path_orig
    with pytest.raises(ValidationError, match="frozen"):
        record.obsnum = 1


def test_flat_config_dump_needs_no_original_path_parsing(record, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Scalar projection parsed the original path")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    flat = record.model_dump(
        mode="json",
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )
    assert flat == {
        "source": str(record.filepath),
        "filepath_orig": None,
        "interface": "toltec8",
        "roach": 8,
        "master": None,
        "obsnum": 145605,
        "subobsnum": 0,
        "scannum": 1,
        "data_kind": "VnaSweep",
        "metadata": None,
        "file_prefix": "toltec8",
        "file_ut": "2025-10-31T09:20:06Z",
        "file_suffix": "vnasweep",
        "file_ext": "nc",
    }


def test_aliases_follow_replaced_path_info_without_cached_values(record):
    assert record.file_suffix == "vnasweep"
    replacement = guess_info_from_source(
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    assert replacement is not None
    updated = record.model_copy(
        update={
            "source": replacement.source,
            "filepath_info": replacement.filepath_info,
            "obsnum": replacement.obsnum,
            "data_kind": replacement.data_kind,
        }
    )
    assert updated.file_suffix == "tune"
    assert updated.model_dump()["file_suffix"] == "tune"
    assert record.file_suffix == "vnasweep"
    schema = SourceInfoModel.model_json_schema(mode="serialization")
    for name in ("file_prefix", "file_ut", "file_suffix", "file_ext"):
        assert schema["properties"][name]["readOnly"] is True
        assert name not in record.__dict__
        assert name not in record.model_dump(round_trip=True)


@pytest.mark.parametrize(
    "kind",
    [
        *ToltecDataKind.__members__.values(),
        ToltecDataKind.VnaSweep | ToltecDataKind.LmtTel,
        ToltecDataKind(0),
    ],
)
def test_catalog_kind_serialization_and_validation(kind, record):
    # Exercise the enum codec, not classification of these artificial combinations.
    record = SourceInfoModel(
        source="https://example.org/unknown.nc",
        data_kind=kind,
        filepath_info=record.filepath_info,
    )
    assert (
        SourceInfoModel.model_validate(record.model_dump(round_trip=True)).data_kind
        == kind
    )
    assert (
        SourceInfoModel.model_validate_json(
            record.model_dump_json(round_trip=True)
        ).data_kind
        == kind
    )
    serialized = record.model_dump(context={"flat": True}, round_trip=True)
    assert serialized["source"] == record.source.url
    assert SourceInfoModel.model_validate(serialized).data_kind == kind
    catalog = ToltecFileAccessor.from_info_list([record])
    assert catalog.toltec_file.to_info_list()[0].data_kind == kind
    dtype = catalog["filepath_info"].dtype
    assert isinstance(dtype, pd.ArrowDtype)
    assert "data_kind" not in dtype.pyarrow_dtype.names


def test_invalid_kind_name_raises_validation_error():
    with pytest.raises(ValidationError, match="Unknown data kind"):
        SourceInfoModel(source="unknown.nc", data_kind="not_a_kind")


def test_catalog_uses_one_native_dump_without_alias_columns(record, monkeypatch):
    dump = SourceInfoModel.model_dump
    calls = []

    def traced(self, **kwargs):
        calls.append(kwargs)
        return dump(self, **kwargs)

    monkeypatch.setattr(SourceInfoModel, "model_dump", traced)
    catalog = ToltecFileAccessor.from_info_list(iter([record]))
    assert len(calls) == 1
    assert calls[0].get("mode", "python") == "python"
    assert not {"file_ut", "file_suffix", "file_ext"} & set(catalog.columns)
    assert catalog["filepath_info"].struct.field("ut")[0] == record.file_ut
    assert catalog.toltec_file.to_info_list()[0].filepath_info == record.filepath_info


def test_type_adapter_roundtrip_omits_all_computed_fields(record):
    adapter = TypeAdapter(list[SourceInfoModel])
    payload = adapter.dump_json([record], round_trip=True)
    assert adapter.validate_json(payload) == [record]


def test_mixed_record_types_survive_json_and_parquet(tmp_path, record, monkeypatch):
    telescope = guess_info_from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert telescope is not None
    records = [record, telescope]
    adapter = TypeAdapter(list[SourceInfoModel])
    restored = adapter.validate_json(adapter.dump_json(records, round_trip=True))
    assert restored == records
    assert type(restored[0].filepath_info) is ToltecFilepathInfo
    assert type(restored[1].filepath_info) is FilepathInfo
    assert restored[0].filepath_info.roach == 8
    assert "roach" not in restored[1].filepath_info.model_dump()

    def unexpected(*args, **kwargs):
        pytest.fail("Restoring a stored record repeated filename deduction")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    catalog = ToltecFileAccessor.from_info_list(records)
    output = tmp_path / "catalog.parquet"
    catalog.to_parquet(output, index=False)
    catalog = pd.read_parquet(output, dtype_backend="pyarrow")
    restored = catalog.toltec_file.to_info_list()
    for before, after in zip(records, restored, strict=True):
        assert after.filepath_info == before.filepath_info
        # Catalogs store resolved locations, not FileLoc's original spelling
        # and parent-path context.
        assert after.model_dump(context={"flat": True}) == before.model_dump(
            context={"flat": True}
        )
    assert catalog.toltec_file.query("file_roach == 8").index.tolist() == [0]
    assert catalog.toltec_file.query("file_roach.isna()").index.tolist() == [1]
