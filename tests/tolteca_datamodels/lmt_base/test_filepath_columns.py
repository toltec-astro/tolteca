"""Shared acquisition filepath projections and instrument-specific aliases."""

import pytest

from tolteca_datamodels.lmt._filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt_base._filepath import (
    FilepathInfoMixin,
    make_filepath_info_column_mapper,
)
from tolteca_datamodels.toltec._filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.filepath import DummyInstruFilepathInfo


@pytest.mark.parametrize(
    ("model_cls", "accessor_cls", "extra_obs_fields"),
    [
        (LmtFilepathInfo, LmtFileAccessor, ()),
        (ToltecFilepathInfo, ToltecFileAccessor, ("roach",)),
        (DummyInstruFilepathInfo, DummyInstruFileAccessor, ("channel",)),
    ],
)
def test_filepath_columns_and_observation_aliases(
    model_cls, accessor_cls, extra_obs_fields
):
    columns = make_filepath_info_column_mapper(*extra_obs_fields)(model_cls)
    obs_fields = {
        "interface",
        "instru",
        "instru_component",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        *extra_obs_fields,
    }
    expected = [
        alias
        for name in model_cls.model_fields
        for alias in (
            [f"file_{name}", name] if name in obs_fields else [f"file_{name}"]
        )
    ]
    assert list(columns) == expected
    assert {
        name: accessor_cls._filepath_info_columns[name] for name in columns
    } == columns
    assert all(column.default is None for column in columns.values())
    assert columns["file_ut"].dtype == "datetime64[ns, UTC]"
    for name in obs_fields:
        assert columns[name] == columns[f"file_{name}"]
        assert columns[name].attr == name
    assert columns["obsnum"].dtype == "Int64"
    assert columns["master"].dtype == "string"


def test_extra_aliases_do_not_modify_the_shared_mapping():
    extended = make_filepath_info_column_mapper("prefix", "obsnum")
    assert extended(FilepathInfoMixin)["prefix"].attr == "prefix"
    assert "prefix" not in make_filepath_info_column_mapper()(FilepathInfoMixin)
    # Extra names are literal; regex metacharacters must not select other fields.
    columns = make_filepath_info_column_mapper("prefix|suffix")(FilepathInfoMixin)
    assert "prefix" not in columns and "suffix" not in columns
