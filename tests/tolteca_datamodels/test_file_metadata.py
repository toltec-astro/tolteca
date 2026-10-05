"""Typed, cached acquisition filepath metadata."""

from typing import assert_type

import pytest
from pydantic import ValidationError

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt_base._filepath import FilepathOrigMetadataMixin
from tolteca_datamodels.toltec.file import ToltecFilepathInfo


@pytest.mark.parametrize("data", [{}, {"filepath_orig": None}])
def test_original_path_is_required_and_nonnullable(data):
    with pytest.raises(ValidationError, match="filepath_orig"):
        FilepathOrigMetadataMixin[FilepathInfo].model_validate(data)


def test_unrecognized_original_path():
    meta = FilepathOrigMetadataMixin[FilepathInfo](filepath_orig="unrecognized.nc")
    assert meta.filepath_orig == "unrecognized.nc"
    assert meta.filepath_info_orig is None
    assert "filepath_info_orig" in meta.__dict__


def test_original_path_uses_registered_parser_once(monkeypatch):
    path = "tel_toltec_2025-10-31_145609_00_0002.nc"
    calls = []
    parse = file_module.FilepathInfo.from_source

    def trace(source, **kwargs):
        calls.append(source)
        return parse(source, **kwargs)

    monkeypatch.setattr(file_module.FilepathInfo, "from_source", trace)
    meta = FilepathOrigMetadataMixin[FilepathInfo](filepath_orig=path)
    assert calls == []
    assert "filepath_info_orig" not in meta.__dict__
    payload = meta.model_dump_json(round_trip=True)
    restored = type(meta).model_validate_json(payload)
    assert calls == []
    assert "filepath_info_orig" not in restored.__dict__

    parsed = meta.filepath_info_orig
    assert isinstance(parsed, LmtFilepathInfo)
    assert parsed.interface == "tel"
    assert parsed.obsnum == 145609
    assert meta.filepath_info_orig is parsed
    assert meta.model_dump(mode="json")["filepath_info_orig"] == parsed.model_dump(
        mode="json"
    )
    meta.model_dump_json()
    assert calls == [path]

    other = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"filepath_orig": "unrecognized.nc"}
    )
    assert "filepath_info_orig" not in other.__dict__
    assert other.filepath_info_orig is None
    assert calls == [path, "unrecognized.nc"]
    assert meta.filepath_info_orig is parsed


def test_invalid_filename_date_is_not_treated_as_unrecognized():
    meta = FilepathOrigMetadataMixin[LmtFilepathInfo](
        filepath_orig="tel_toltec_2025-13-31_145609_00_0002.nc"
    )
    with pytest.raises(ValueError):
        _ = meta.filepath_info_orig


def test_generic_intermediate_and_concrete_subclass():
    class Metadata[InfoT: FilepathInfo](FilepathOrigMetadataMixin[InfoT]):
        pass

    class Concrete(Metadata[LmtFilepathInfo]):
        pass

    meta = Concrete(filepath_orig="tel_toltec_2025-10-31_145609_00_0002.nc")
    parsed = meta.filepath_info_orig
    assert_type(parsed, LmtFilepathInfo | None)
    assert type(parsed) is LmtFilepathInfo
    assert Concrete.model_computed_fields["filepath_info_orig"].return_type == (
        LmtFilepathInfo | None
    )


def test_specializations_do_not_share_runtime_binding():
    telescope_path = "tel_toltec_2025-10-31_145609_00_0002.nc"
    toltec_path = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    telescope = FilepathOrigMetadataMixin[LmtFilepathInfo](filepath_orig=telescope_path)
    toltec = FilepathOrigMetadataMixin[ToltecFilepathInfo](filepath_orig=toltec_path)
    assert type(telescope.filepath_info_orig) is LmtFilepathInfo
    assert type(toltec.filepath_info_orig) is ToltecFilepathInfo
    assert (
        FilepathOrigMetadataMixin[LmtFilepathInfo](
            filepath_orig=toltec_path
        ).filepath_info_orig
        is None
    )
    assert (
        FilepathOrigMetadataMixin[ToltecFilepathInfo](
            filepath_orig=telescope_path
        ).filepath_info_orig
        is None
    )


def test_computed_path_information_is_readonly_in_schema():
    cls = FilepathOrigMetadataMixin[LmtFilepathInfo]
    schema_input = cls.model_json_schema()
    assert "filepath_orig" in schema_input["required"]
    assert schema_input["properties"]["filepath_orig"]["type"] == "string"
    assert "filepath_info_orig" not in schema_input["properties"]
    schema = cls.model_json_schema(mode="serialization")
    assert schema["properties"]["filepath_info_orig"]["readOnly"]
    assert schema["properties"]["filepath_info_orig"]["anyOf"] == [
        {"$ref": "#/$defs/LmtFilepathInfo"},
        {"type": "null"},
    ]
