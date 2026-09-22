"""KIDs acquisition paths are metadata, not current source locations."""

import dataclasses

import pytest
from pydantic import TypeAdapter

from tolteca_datamodels.file import FilepathInfo, SourceInfoModel
from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.kids.source import identify_source


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_recorded_acquisition_path(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec(path) as data:
        accessor = data.toltec_kids
        mapper = accessor.mapper
        meta = accessor.meta
        expected = f"/data_lmt/toltec/tcs/toltec0/{path.name}"
        assert meta.filepath_orig == expected
        assert mapper.get_name(mapper.schema.filepath_orig) == "Header.Toltec.Filename"
        parsed = FilepathInfo.from_path(meta.filepath_orig)
        assert parsed is not None
        assert (parsed.obsnum, parsed.subobsnum, parsed.scannum) == (
            meta.obsnum,
            meta.subobsnum,
            meta.scannum,
        )
        # Deliberately renamed local copy: header discovery retains its location.
        record = identify_source(data, SourceInfoModel(source="downloaded.nc"))
        assert record.filepath_orig == meta.filepath_orig
        assert record.source.path.name == "downloaded.nc"
        assert not data["Data.Toltec.Is"].variable._in_memory

    adapter = TypeAdapter(type(meta))
    dumped = adapter.dump_python(meta, round_trip=True)
    assert dumped["filepath_orig"] == expected
    assert "filename_orig" not in dumped
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.filepath_orig == expected


def test_path_field_does_not_parse_or_require_a_filename(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc")
    with open_toltec(path) as data:
        # Unrecognized text is kept verbatim, not silently replaced by a guess.
        meta = dataclasses.replace(data.toltec_kids.meta, filepath_orig="unknown.nc")
        assert meta.filepath_orig == "unknown.nc"
        missing = data.drop_vars("Header.Toltec.Filename")
        assert missing.toltec_kids.meta.filepath_orig is None
