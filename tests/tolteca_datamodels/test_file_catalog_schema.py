"""Filename parsers independent of table projection declarations."""

import re
from typing import ClassVar

import pandas as pd

import tolteca_datamodels._filepath as file_module
from tolteca_datamodels.file import FileAccessor, FilepathInfo
from tolteca_datamodels.model_table import ModelColumn
from tolteca_datamodels.toltec.file import ToltecFileAccessor


def test_mixed_models_only_project_declared_attributes():
    models = [
        FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc"),
        FilepathInfo.from_source(
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
        ),
    ]
    table = ToltecFileAccessor.from_models(
        models, columns=ToltecFileAccessor._filepath_info_columns
    )
    assert table["file_prefix"].tolist() == ["tel_toltec", "toltec8"]
    assert pd.isna(table["roach"].iloc[0])
    assert table["roach"].iloc[1] == 8
    assert not {"filepath_info", "_record", "metadata"}.intersection(table)
    selected = ToltecFileAccessor.from_models(
        models,
        columns={
            "prefix": ModelColumn("prefix", dtype="string"),
        },
    )
    assert list(selected) == ["prefix"]
    assert selected["prefix"].tolist() == table["file_prefix"].tolist()


def test_parser_registration_needs_no_storage_schema(monkeypatch):
    monkeypatch.setattr(file_module, "_filename_pattern_parser", {})

    class Filename(FilepathInfo):
        stem: str
        extension: str

        @staticmethod
        def _parse_extension(data, path):
            return {"extension": path.suffix.removeprefix(".")}

        _filename_patterns: ClassVar = {
            re.compile(r"^(?P<stem>tel_toltec_2025-10-31_145609_00_0002)\.nc$"): (
                _parse_extension,
            )
        }

    info = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert isinstance(info, Filename)
    assert info.extension == "nc"
    table = FileAccessor.from_models([info], columns={"ext": ModelColumn("extension")})
    assert table["ext"].tolist() == ["nc"]
