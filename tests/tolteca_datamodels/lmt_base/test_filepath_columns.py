"""Shared acquisition filepath projections and instrument-specific aliases."""

import pytest

from tolteca_datamodels.lmt._filepath import LmtFilepathInfo
from tolteca_datamodels.toltec._filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)
from tolteca_dummy_instru.filepath import (
    DummyInstruFilepathInfo,
    DummyInstruLmtFilepathInfo,
)


@pytest.mark.parametrize(
    ("model_cls", "specific_aliases"),
    [
        (LmtFilepathInfo, ()),
        (ToltecFilepathInfo, ("roach",)),
        (DummyInstruFilepathInfo, ("channel",)),
    ],
)
def test_filepath_columns_and_observation_aliases(model_cls, specific_aliases):
    columns = model_cls._columns
    obs_fields = {
        "interface",
        "instru",
        "producer",
        "interface_kind",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        *specific_aliases,
    }
    expected = [
        alias
        for name in model_cls.model_fields
        for alias in (
            [f"file_{name}", name] if name in obs_fields else [f"file_{name}"]
        )
    ]
    assert list(columns) == expected
    assert all(column.default is None for column in columns.values())
    assert columns["file_ut"].dtype == "datetime64[ns, UTC]"
    for name in obs_fields:
        assert columns[name] == columns[f"file_{name}"]
        assert columns[name].attr == name
    assert columns["obsnum"].dtype == "Int64"
    assert columns["master"].dtype == "string"


def test_concrete_columns_are_built_from_the_completed_model():
    class Extended(DummyInstruFilepathInfo):
        readout_code: int
        payload: dict[str, int]

    columns = Extended._columns
    assert columns["readout_code"].attr == "readout_code"
    assert columns["readout_code"].dtype == "Int64"
    assert columns["readout_code"] == columns["file_readout_code"]
    assert columns["channel"] == columns["file_channel"]
    assert "payload" not in columns and "file_payload" not in columns
    assert "readout_code" not in DummyInstruFilepathInfo._columns


@pytest.mark.parametrize("model", [ToltecLmtFilepathInfo, DummyInstruLmtFilepathInfo])
def test_thin_telescope_subclasses_keep_telescope_columns(model):
    assert model._columns == LmtFilepathInfo._columns
