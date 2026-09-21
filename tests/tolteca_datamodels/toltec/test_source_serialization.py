"""Source-record serialization for models, catalogs, and config conditions."""

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.toltec.file import (
    FilepathInfo,
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture
def record(tmp_path):
    source = fileloc(
        "toltec8_145609_000_0002_2025_10_31_01_02_03_vnasweep.nc",
        local_parent_path=tmp_path,
    )
    return guess_info_from_source(source)


def test_normal_serialization_retains_location_and_typed_aliases(record):
    dumped = record.model_dump()
    assert dumped["source"] == record.source.model_dump()
    assert dumped["data_kind"] is ToltecDataKind.VnaSweep
    assert dumped["file_ut"] == datetime(2025, 10, 31, 1, 2, 3, tzinfo=UTC)
    assert dumped["file_suffix"] == "vnasweep"
    assert dumped["file_ext"] == "nc"
    dumped_json = record.model_dump(mode="json")
    assert dumped_json["source"] == record.source.model_dump(mode="json")
    assert dumped_json["data_kind"] == ToltecDataKind.VnaSweep.value
    assert dumped_json["file_ut"] == "2025-10-31T01:02:03Z"
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
        "obsnum": 145609,
        "subobsnum": 0,
        "scannum": 2,
        "data_kind": "VnaSweep",
        "file_ut": "2025-10-31T01:02:03Z",
        "file_suffix": "vnasweep",
        "file_ext": "nc",
    }


def test_aliases_follow_replaced_path_info_without_cached_values(record):
    assert record.file_suffix == "vnasweep"
    updated_path = record.filepath_info.model_copy(update={"suffix": "tune"})
    updated = record.model_copy(update={"filepath_info": updated_path})
    assert updated.file_suffix == "tune"
    assert updated.model_dump()["file_suffix"] == "tune"
    assert record.file_suffix == "vnasweep"
    schema = SourceInfoModel.model_json_schema(mode="serialization")
    for name in ("file_ut", "file_suffix", "file_ext"):
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
def test_catalog_kind_serialization_and_validation(kind):
    record = SourceInfoModel(source="https://example.org/unknown.nc", data_kind=kind)
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
