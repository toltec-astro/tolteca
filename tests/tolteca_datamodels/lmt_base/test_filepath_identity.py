"""Shared observation identity in parsed acquisition filepaths."""

import pytest
from pydantic import ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt_base.metadata import (
    InterfaceInfoMixin,
    LmtObsQuartetMixin,
)
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_dummy_instru.filepath import DummyInstruFilepathInfo


@pytest.fixture(
    params=[
        (
            LmtFilepathInfo,
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            "tcs",
            None,
        ),
        (
            ToltecFilepathInfo,
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
            None,
            None,
        ),
        (
            ToltecFilepathInfo,
            (
                "/data_lmt/toltec/tcs/toltec8/"
                "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
            ),
            "tcs",
            "tune",
        ),
        (
            DummyInstruFilepathInfo,
            "chan00-tcs-1-0-2-20251031T120000.nc",
            "tcs",
            None,
        ),
    ],
)
def filepath_case(request):
    model, source, master, suffix = request.param
    result = FilepathInfo.from_source(source)
    assert isinstance(result, model)
    return result, master, suffix


def test_parser_supplies_nullable_values(filepath_case):
    info, master, suffix = filepath_case
    assert isinstance(info, InterfaceInfoMixin)
    assert isinstance(info, LmtObsQuartetMixin)
    assert {"master", "suffix"} <= info.model_fields_set
    assert info.master == master
    assert info.suffix == suffix
    assert "file_quartet" not in info._columns
    assert "quartet" not in info._columns


@pytest.mark.parametrize("name", ["master", "suffix"])
def test_missing_parser_values_are_not_defaulted(filepath_case, name):
    info, _, _ = filepath_case
    data = info.model_dump(round_trip=True, exclude={name})
    with pytest.raises(ValidationError) as caught:
        type(info).model_validate(data)
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        ((name,), "missing")
    ]
