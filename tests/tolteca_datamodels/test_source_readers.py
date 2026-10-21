"""Payload identification separate from metadata reads and resource ownership."""

from io import StringIO, TextIOBase
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
import xarray as xr
from tollan.accessor.pandas.model_table import ModelColumn
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.core.file import accessor as file_module
from tolteca_datamodels.core.file._resolution import _SourceReaderResolver
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
)
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt import LmtTelAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelIOMapper
from tolteca_datamodels.toltec.types import ToltecDataModel


class _TextDataModel(DataModelInfo):
    name = "text"


@pytest.mark.parametrize("container", ["dataset", "tree", "node"])
def test_known_telescope_binding_does_not_read_or_close(
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
            identified = _SourceReaderResolver(
                handlers={
                    "lmt": LmtFileAccessor._reader_rules,
                }
            )(source, choices={"lmt": LmtTelAccessor})
            assert identified is not None
            reader, _ = identified
            assert isinstance(reader, LmtTelAccessor)
            assert isinstance(source.lmt_tel, type(reader))
            assert "meta" not in reader.__dict__
        assert reader.meta.obsnum == 145609
        assert reader.az.isel(time=0).item() is not None


def test_telescope_binding_does_not_require_valid_metadata(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as dataset:
        corrupt = dataset.drop_vars("Header.Source.Ra")
        reader = corrupt.lmt_tel
        assert reader is not None
        with pytest.raises(KeyError, match="Header.Source.Ra"):
            _ = reader.meta


def test_named_telescope_accessor_checks_structure_once(data_lmt_path, monkeypatch):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    checks = []
    has_arr = LmtTelIOMapper.has_arr

    def track_check(self, data_source, field):
        checks.append(field)
        return has_arr(self, data_source, field)

    monkeypatch.setattr(LmtTelIOMapper, "has_arr", track_check)
    with xr.open_dataset(path) as dataset:
        reader = dataset.lmt_tel
        assert reader is not None
        assert dataset.lmt_tel is reader
        assert checks == [reader.mapper.schema.tel_time]


def test_shared_metadata_loop_supports_non_xarray_readers(tmp_path, monkeypatch):
    class TextReader:
        def __init__(self, data):
            self.data = data

        @property
        def meta(self):
            return SimpleNamespace(title=self.data.readline().strip())

    def prepare_record(data):
        return {} if isinstance(data, TextIOBase) else None

    class TextFiles(FileAccessor[_TextDataModel]):
        _source_opener = SourceOpener(rules=(({"ext": "txt"}, Path.open),))
        _reader_rules = SourceReaderRules(
            rules=(({}, TextReader),), prepare_record=prepare_record
        )
        _metadata_columns: ClassVar = {"title": ModelColumn("title", dtype="string")}

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


@pytest.mark.parametrize("operation", ["read", "open"])
def test_table_rules_use_current_values_and_batch_matching(
    tmp_path, monkeypatch, operation
):
    """Temporary text payloads exercise format dispatch, not instrument readers."""
    opened = []
    batches = []
    match = file_module.FileAccessor._get_opener_choices

    def batch(accessor):
        batches.append(len(accessor._obj))
        return match(accessor)

    monkeypatch.setattr(file_module.FileAccessor, "_get_opener_choices", batch)

    def opener(path):
        resource = path.open()
        opened.append(resource)
        return resource

    class TextReader:
        def __init__(self, data):
            self.data = data

        @property
        def meta(self):
            return SimpleNamespace(title=self.data.readline().strip())

    class TextFiles(FileAccessor[ToltecDataModel]):
        _source_opener = SourceOpener(
            rules=(({"ext": "txt", "suffix": "current"}, opener),)
        )
        _reader_rules = SourceReaderRules(
            rules=(({}, TextReader),),
            prepare_record=lambda data: {} if isinstance(data, TextIOBase) else None,
        )
        _metadata_columns: ClassVar = {"title": ModelColumn("title", dtype="string")}

        @staticmethod
        def _guess_derived_columns(table):
            return pd.DataFrame({"suffix": "current"}, index=table.index)

    cached = tmp_path / "cached"
    cached.write_text("model parameters")
    location = "archive:/toltec8_145605_000_0001_2025_10_31_09_20_06_tune.txt"
    table = TextFiles.from_sources([{"source": cached, "source_loc": location}])
    # A handler's derived values override filename snapshots for rule selection.
    assert table.suffix.tolist() == ["current"]
    assert table.file_suffix.tolist() == ["tune"]
    if operation == "read":
        result = TextFiles(table).update_from_source_metadata()
    else:
        with TextFiles(table).open(on_error="raise") as result:
            TextFiles(result).update_from_source_metadata(inplace=True)
            assert not opened[0].closed
    assert result.title.tolist() == ["model parameters"]
    assert batches == [1]
    assert len(opened) == 1 and opened[0].closed
