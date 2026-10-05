"""Payload identification separate from metadata reads and resource ownership."""

from io import StringIO, TextIOBase
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt import LmtTelAccessor, identify_lmt_tel
from tolteca_datamodels.lmt.tel import LmtTelIOMapper
from tolteca_datamodels.model_table import ModelColumn
from tolteca_datamodels.source import identify_source


def test_reader_order_and_deferred_metadata():
    payload = object()
    calls = []

    class Reader:
        def read_metadata(self):
            calls.append("metadata")
            return SimpleNamespace(obsnum=1)

    reader = Reader()

    def miss(data):
        assert data is payload
        calls.append("miss")

    def match(data):
        assert data is payload
        calls.append("match")
        return reader

    def unexpected(data):
        pytest.fail("Identification continued after a match")

    identified = identify_source(payload, readers=(miss, match, unexpected))
    assert identified is reader
    assert calls == ["miss", "match"]
    assert identified.read_metadata().obsnum == 1
    assert calls == ["miss", "match", "metadata"]
    assert identify_source(payload, readers=(miss,)) is None
    assert identify_source(payload, readers=()) is None


@pytest.mark.parametrize("error_type", [ValueError, KeyError, RuntimeError])
def test_identification_exceptions_are_not_misses(error_type):
    def failed(data):
        msg = "Invalid payload"
        raise error_type(msg)

    def unexpected(data):
        pytest.fail("Identification continued after an exception")

    with pytest.raises(error_type, match="Invalid payload"):
        identify_source(object(), readers=(failed, unexpected))


@pytest.mark.parametrize("container", ["dataset", "tree", "node"])
def test_telescope_identification_does_not_read_or_close(
    data_lmt_path, monkeypatch, container
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"

    def unexpected_read(*args, **kwargs):
        pytest.fail("Identification read a netCDF variable")

    with xr.open_dataset(path) as dataset:
        source = dataset
        if container == "tree":
            source = xr.DataTree(dataset)
        elif container == "node":
            source = xr.DataTree.from_dict({"telescope": dataset})["telescope"]
        with monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "__getitem__", unexpected_read)
            patch.setattr(xr.Dataset, "close", unexpected_read)
            patch.setattr(xr.DataTree, "close", unexpected_read)
            reader = identify_source(source, readers=(identify_lmt_tel,))
            assert isinstance(reader, LmtTelAccessor)
            assert reader is source.lmt_tel
            assert identify_lmt_tel(source, raise_on_error=True) is reader
            assert "meta" not in reader.__dict__
        assert reader.meta.obsnum == 145609
        assert reader.az.isel(time=0).item() is not None


@pytest.mark.parametrize("strict", [False, True])
def test_telescope_identity_does_not_require_valid_metadata(data_lmt_path, strict):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as dataset:
        corrupt = dataset.drop_vars("Header.Source.Ra")
        reader = identify_lmt_tel(corrupt, raise_on_error=strict)
        assert reader is not None
        with pytest.raises(KeyError, match="Header.Source.Ra"):
            _ = reader.meta
    assert identify_lmt_tel(path) is None
    assert identify_lmt_tel(xr.Dataset()) is None


@pytest.mark.parametrize("source", [object(), "tel.nc", xr.DataArray([1])])
def test_strict_telescope_identification_rejects_unsupported_types(source):
    assert identify_lmt_tel(source) is None
    with pytest.raises(TypeError, match="Dataset or DataTree"):
        identify_lmt_tel(source, raise_on_error=True)


@pytest.mark.parametrize("source", [xr.Dataset(), xr.DataTree()])
def test_strict_telescope_identification_rejects_missing_timestamps(source):
    assert identify_lmt_tel(source) is None
    with pytest.raises(ValueError, match="missing tel_time variable"):
        identify_lmt_tel(source, raise_on_error=True)


def test_telescope_identification_checks_structure_once(data_lmt_path, monkeypatch):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    checks = []
    has_arr = LmtTelIOMapper.has_arr

    def track_check(self, data_source, field):
        checks.append(field)
        return has_arr(self, data_source, field)

    monkeypatch.setattr(LmtTelIOMapper, "has_arr", track_check)
    with xr.open_dataset(path) as dataset:
        reader = identify_lmt_tel(dataset)
        assert reader is not None
        assert identify_lmt_tel(dataset, raise_on_error=True) is reader
        assert dataset.lmt_tel is reader
        assert checks == [reader.mapper.schema.tel_time]


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("error_type", [KeyError, RuntimeError])
def test_telescope_identification_propagates_unexpected_failures(
    monkeypatch, error_type, strict
):
    error = error_type("Unexpected validation failure")

    def fail(self):
        raise error

    monkeypatch.setattr(LmtTelAccessor, "_validate", fail)
    with pytest.raises(error_type) as exc:
        identify_lmt_tel(xr.Dataset(), raise_on_error=strict)
    assert exc.value is error


def test_telescope_value_error_is_a_nonmatch_unless_strict(monkeypatch):
    error = ValueError("Incompatible telescope payload")

    def fail(self):
        raise error

    monkeypatch.setattr(LmtTelAccessor, "_validate", fail)
    assert identify_lmt_tel(xr.Dataset()) is None
    with pytest.raises(ValueError) as exc:
        identify_lmt_tel(xr.Dataset(), raise_on_error=True)
    assert exc.value is error


def test_shared_metadata_loop_supports_non_xarray_readers(tmp_path):
    class TextReader:
        def __init__(self, data):
            self.data = data

        def read_metadata(self):
            return SimpleNamespace(title=self.data.readline().strip())

    def identify_text(data):
        return TextReader(data) if isinstance(data, TextIOBase) else None

    class TextFiles(FileAccessor):
        _io_obj_types = (TextIOBase,)
        _source_readers = (identify_text,)
        _metadata_file_extensions = (".txt",)
        _metadata_columns: ClassVar = {"title": ModelColumn("title", dtype="string")}

        _source_opener = Path.open

    path = tmp_path / "observation.txt"
    path.write_text("Owned text\n")
    with StringIO("Borrowed text\n") as borrowed:
        table = TextFiles.from_sources(
            [path, {"source": borrowed, "source_loc": "archive:/borrowed.txt"}],
            mode="read",
        )
        assert table.title.tolist() == ["Owned text", "Borrowed text"]
        assert not borrowed.closed
        assert "io_obj" not in table
