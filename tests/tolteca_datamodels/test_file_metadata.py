"""Polymorphic, cached acquisition filepath metadata."""

from typing import assert_type

import pytest
from pydantic import ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt_base.filepath import FilepathOrigMetadataMixin


@pytest.mark.parametrize("data", [{}, {"filepath_orig": None}])
def test_original_path_is_required_and_nonnullable(data):
    with pytest.raises(ValidationError, match="filepath_orig"):
        FilepathOrigMetadataMixin.model_validate(data)


def test_unrecognized_original_path():
    meta = FilepathOrigMetadataMixin(filepath_orig="unrecognized.nc")
    assert meta.filepath_orig == "unrecognized.nc"
    assert meta.filepath_info_orig is None
    assert "filepath_info_orig" in meta.__dict__


def test_original_path_uses_registered_parser_once(monkeypatch):
    path = "tel_toltec_2025-10-31_145609_00_0002.nc"
    calls = []
    parse = FilepathInfo.from_source

    def trace(source, **kwargs):
        calls.append(source)
        return parse(source, **kwargs)

    monkeypatch.setattr(FilepathInfo, "from_source", trace)
    meta = FilepathOrigMetadataMixin(filepath_orig=path)
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
    meta = FilepathOrigMetadataMixin(
        filepath_orig="tel_toltec_2025-13-31_145609_00_0002.nc"
    )
    with pytest.raises(ValueError):
        _ = meta.filepath_info_orig


@pytest.mark.parametrize(
    "path",
    [
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
        "chan01-tcs-1-0-2-20251031T120000.nc",
    ],
)
def test_metadata_uses_the_same_public_factory(path):
    from tolteca_dummy_instru import (
        filepath,  # noqa: F401 -- Enable only filename parsing.
    )

    class Metadata(FilepathOrigMetadataMixin):
        pass

    meta = Metadata(filepath_orig=path)
    expected = FilepathInfo.from_source(path)
    assert expected is not None
    assert_type(meta.filepath_info_orig, FilepathInfo | None)
    assert type(meta.filepath_info_orig) is type(expected)
    assert meta.filepath_info_orig == expected
    assert meta.model_dump()["filepath_info_orig"] == expected.model_dump()


def test_computed_path_information_is_readonly_in_schema():
    cls = FilepathOrigMetadataMixin
    schema_input = cls.model_json_schema()
    assert "filepath_orig" in schema_input["required"]
    assert schema_input["properties"]["filepath_orig"]["type"] == "string"
    assert "filepath_info_orig" not in schema_input["properties"]
    schema = cls.model_json_schema(mode="serialization")
    assert schema["properties"]["filepath_info_orig"]["readOnly"]
    assert schema["properties"]["filepath_info_orig"]["anyOf"] == [
        {"$ref": "#/$defs/FilepathInfo"},
        {"type": "null"},
    ]
