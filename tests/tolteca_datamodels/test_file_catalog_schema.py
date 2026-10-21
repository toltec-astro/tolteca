"""Filename parsers independent of table projection declarations."""

import re

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import ModelColumn

import tolteca_datamodels.core.filepath as file_module
from tolteca_datamodels.core.filepath import register_filepath_parser
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec.file import ToltecFileAccessor


def test_mixed_models_only_project_declared_attributes():
    models = [
        FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc"),
        FilepathInfo.from_source(
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
        ),
    ]
    table = ToltecFileAccessor.from_models(
        models,
        columns={
            "file_prefix": ModelColumn("prefix", dtype="string"),
            "roach": ModelColumn("roach", dtype="Int64", default=None),
        },
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
    monkeypatch.setattr(
        file_module, "_filepath_parsers", [*file_module._filepath_parsers]
    )

    class Filename(FilepathInfo):
        stem: str
        extension: str

        @staticmethod
        def _parse_extension(data, path):
            return {"extension": path.suffix.removeprefix(".")}

    register_filepath_parser(
        pattern=re.compile(r"^(?P<stem>example)\.nc$"),
        parsers=(Filename._parse_extension,),
        filepath_info_cls=Filename,
    )

    info = FilepathInfo.from_source("example.nc")
    assert isinstance(info, Filename)
    assert info.extension == "nc"
    table = FileAccessor.from_models([info], columns={"ext": ModelColumn("extension")})
    assert table["ext"].tolist() == ["nc"]

    def unexpected(*args, **kwargs):
        pytest.fail("A parsed filename without memberships triggered source reading")

    monkeypatch.setattr(ToltecFileAccessor, "_acquire_source", unexpected)
    table = ToltecFileAccessor.from_sources(["example.nc"], mode="read")
    assert isinstance(table.filepath_info.iloc[0], Filename)
    assert table.data_models.tolist() == [[]]
    summary = table.toltec_file.source_trace_summary()
    scope = summary.loc[summary.stage == "scope", "outcome"].tolist()
    assert scope == ["out_of_scope"]
    updated = table.toltec_file.update_from_source_metadata(on_error="raise")
    assert updated.filepath_info.iloc[0] is table.filepath_info.iloc[0]
    with updated.toltec_file.open(on_error="raise") as opened:
        assert opened.io_obj.iloc[0] is None
        assert opened.accessor_obj.iloc[0] is None
        assert opened.filepath_info.iloc[0] is table.filepath_info.iloc[0]
