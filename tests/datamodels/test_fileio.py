import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from tollan.utils.fileloc import FileLoc

from tolteca_datamodels.io.core import (
    FileIO,
    FileIODataModelBase,
    identify_io_cls,
    io_cls_registry,
)


class DummyFileIOData(FileIODataModelBase):
    """Dummy file IO data."""

    @classmethod
    def validate_source(cls, source, **kwargs):  # noqa: ARG003
        source = FileLoc(source)
        if source.path.suffix not in [".yaml", ".yml"]:
            raise ValueError("invalid file format.")
        return source

    def _set_open_state(self, **kwargs):
        with self.source.path.open(**kwargs) as fo:
            self.io_obj = yaml.safe_load(fo)


@dataclass(kw_only=True)
class DummyFileIOMetadata:
    a: str = ...
    b: int = ...


class DummyFileIO(FileIO[DummyFileIOData, DummyFileIOMetadata]):
    """Dummy file IO."""

    def _update_meta_from_io_data(self):
        if self.is_open():
            self.meta.__dict__.update(self.io_obj)


def test_identify():
    assert DummyFileIO.identify("data.yaml")


def test_registry():
    assert DummyFileIO in io_cls_registry
    assert identify_io_cls("data.yaml", type=DummyFileIO) is DummyFileIO
    assert identify_io_cls("data.yaml", type=DummyFileIO | str) is DummyFileIO


def test_dummy_file_io():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        f = tmp / "data.yaml"
        with f.open("w") as fo:
            fo.write(
                """
---
a: "value"
b: 1
""",
            )
        fio = DummyFileIO(f, arg=True)
        assert fio.io_data.source.path.samefile(f)
        assert fio.filepath.samefile(f)
        assert not fio.is_open()
        assert fio.meta.a == ...
        assert fio.meta.b == ...
        with fio.open():
            assert fio.is_open()
            assert fio.meta.a == "value"
            assert fio.meta.b == 1
        assert not fio.is_open()
        assert fio.meta.a == "value"
        assert fio.meta.b == 1
