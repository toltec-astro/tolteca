"""Shared time and acquisition-path properties on frozen metadata."""

from dataclasses import replace

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

import tolteca_datamodels.file as file_module
from tolteca_datamodels.lmt_base import FilepathInfo, LmtRawObsMetadata
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMaster


@pytest.fixture
def meta():
    return LmtRawObsMetadata(
        master="tcs",
        obsnum=145609,
        subobsnum=0,
        scannum=2,
        instru="lmt",
        interface="tel",
        t0=Time("2025-10-31T09:27:56"),
        t1=Time("2025-10-31T09:28:56"),
        t_int=0 * u.s,
    )


def test_elapsed_is_cached_and_independent_of_integration_time(meta):
    assert "t_elapsed" not in meta.__dict__
    elapsed = meta.t_elapsed
    assert elapsed == (meta.t1 - meta.t0).to(u.s)
    assert elapsed.unit == u.s
    assert meta.t_elapsed is elapsed
    assert meta.t_int == 0 * u.s

    other = replace(meta, t1=meta.t1 + 1 * u.s)
    assert "t_elapsed" not in other.__dict__
    assert other.t_elapsed == (other.t1 - other.t0).to(u.s)
    assert other.t_elapsed != elapsed
    assert meta.t_elapsed is elapsed


def test_computed_fields_are_readonly_and_not_roundtrip_inputs(meta):
    adapter = TypeAdapter(LmtRawObsMetadata)
    payload = adapter.dump_json(meta, round_trip=True)
    assert "t_elapsed" not in meta.__dict__
    assert "filepath_info_orig" not in meta.__dict__
    assert b'"t_int"' in payload
    assert b'"t_exp"' not in payload
    restored = adapter.validate_json(payload)
    assert restored.t_elapsed == meta.t_elapsed
    assert restored.filepath_info_orig is None
    schema = adapter.json_schema(mode="serialization")["properties"]
    for name in ("t_elapsed", "filepath_info_orig"):
        assert schema[name]["readOnly"]
        assert name not in adapter.json_schema()["properties"]
    # Integration time remains a required input under its public name.
    with pytest.raises(ValidationError, match="t_int"):
        adapter.validate_json(payload.replace(b'"t_int"', b'"t_exp"'))


@pytest.mark.parametrize("path", [None, "unrecognized.nc"])
def test_absent_or_unrecognized_original_path(meta, path):
    meta = replace(meta, filepath_orig=path)
    assert meta.filepath_orig == path
    assert meta.filepath_info_orig is None
    assert "filepath_info_orig" in meta.__dict__


def test_original_path_uses_registered_parser_once(meta, monkeypatch):
    path = "tel_toltec_2025-10-31_145609_00_0002.nc"
    meta = replace(meta, filepath_orig=path)
    calls = []
    parse = file_module.FileIdentifier.parse_filepath

    def trace(source, **kwargs):
        calls.append(source)
        return parse(source, **kwargs)

    monkeypatch.setattr(file_module.FileIdentifier, "parse_filepath", trace)
    assert calls == []
    parsed = meta.filepath_info_orig
    assert isinstance(parsed, FilepathInfo)
    assert parsed.interface == "tel"
    assert parsed.obsnum == meta.obsnum
    assert meta.filepath_info_orig is parsed
    adapter = TypeAdapter(LmtRawObsMetadata)
    payload = adapter.dump_python(meta, mode="json")
    assert payload["filepath_info_orig"] == parsed.model_dump(mode="json")
    adapter.dump_json(meta)
    assert calls == [path]

    other = replace(meta, filepath_orig="unrecognized.nc")
    assert "filepath_info_orig" not in other.__dict__
    assert other.filepath_info_orig is None
    assert calls == [path, "unrecognized.nc"]
    assert meta.filepath_info_orig is parsed


def test_invalid_filename_date_is_not_treated_as_unrecognized(meta):
    meta = replace(meta, filepath_orig="tel_toltec_2025-13-31_145609_00_0002.nc")
    with pytest.raises(ValueError):
        _ = meta.filepath_info_orig


def test_toltec_original_path_retains_subclass_fields(meta):
    fields = TypeAdapter(LmtRawObsMetadata).dump_python(meta, round_trip=True)
    fields.update(
        master=ToltecDBRawObsMaster.TCS,
        instru="toltec",
        interface="toltec8",
        roach=8,
        filepath_orig=(
            "/data_lmt/toltec/tcs/toltec8/"
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
        ),
    )
    meta = ToltecRawObsMetadata(**fields)
    parsed = meta.filepath_info_orig
    assert parsed is not None
    expected = parsed.model_dump(mode="json")
    assert expected["roach"] == 8
    for cls in (LmtRawObsMetadata, ToltecRawObsMetadata):
        # The common computed field preserves the domain-specific record.
        payload = TypeAdapter(cls).dump_python(meta, mode="json")
        assert payload["filepath_info_orig"] == expected
    adapter = TypeAdapter(ToltecRawObsMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.filepath_info_orig == parsed
    assert restored.t_elapsed == meta.t_elapsed
    assert replace(meta, filepath_orig=None).filepath_info_orig is None
