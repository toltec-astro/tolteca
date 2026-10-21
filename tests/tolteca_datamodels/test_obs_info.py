"""Partial observation identifiers and their file-table projections."""

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt_base.filepath import FilepathInfoMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata
from tolteca_datamodels.lmt_base.types import SeqNumT
from tolteca_datamodels.toltec.file import (
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)


def test_shared_filepath_values_belong_to_observation_mixin():
    assert not FilepathInfo.model_fields
    assert {"prefix", "suffix", "ext", "obsnum", "ut"} <= (
        FilepathInfoMixin.model_fields.keys()
    )
    assert all(field.is_required() for field in FilepathInfoMixin.model_fields.values())


@pytest.fixture(params=[ToltecLmtFilepathInfo, ToltecFilepathInfo])
def concrete_filepath_info(request, lmtmc_records, toltecdb_records):
    rows = lmtmc_records if request.param is ToltecLmtFilepathInfo else toltecdb_records
    info = FilepathInfo.from_source(rows[0]["FileName"])
    assert type(info) is request.param
    return info


@pytest.mark.parametrize(
    "name",
    [
        "prefix",
        "interface",
        "interface_kind",
        "obsnum",
        "subobsnum",
        "scannum",
        "ut",
        "ext",
    ],
)
def test_concrete_filepath_requires_guaranteed_values(concrete_filepath_info, name):
    model = type(concrete_filepath_info)
    data = concrete_filepath_info.model_dump(round_trip=True)
    del data[name]
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data)
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data | {name: None})


def test_telescope_master_is_required(lmtmc_records):
    parsed = FilepathInfo.from_source(lmtmc_records[0]["FileName"])
    assert isinstance(parsed, LmtFilepathInfo)
    assert parsed.master == "tcs"
    assert LmtFilepathInfo.model_fields["master"].is_required()
    data = parsed.model_dump(round_trip=True)
    del data["master"]
    with pytest.raises(ValidationError, match="master"):
        LmtFilepathInfo.model_validate(data)
    with pytest.raises(ValidationError, match="master"):
        LmtFilepathInfo.model_validate(data | {"master": None})


def test_toltec_readout_requires_instrument(toltecdb_records):
    info = FilepathInfo.from_source(toltecdb_records[0]["FileName"])
    assert info is not None
    data = info.model_dump(round_trip=True, exclude={"instru"})
    for affiliation in ({}, {"instru": None}, {"instru": "other"}):
        with pytest.raises(ValidationError, match="instru"):
            ToltecFilepathInfo.model_validate(data | affiliation)


@pytest.mark.parametrize("name", ["obsnum", "subobsnum", "scannum"])
def test_concrete_observation_numbers_remain_nonnegative(concrete_filepath_info, name):
    model = type(concrete_filepath_info)
    data = concrete_filepath_info.model_dump(round_trip=True)
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data | {name: -1})
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data | {name: "0"})
    assert getattr(model.model_validate(data | {name: 0}), name) == 0


def test_concrete_filepaths_keep_timezone_validation(concrete_filepath_info):
    data = concrete_filepath_info.model_dump(round_trip=True)
    data["ut"] = data["ut"].replace(tzinfo=None)
    with pytest.raises(ValidationError, match="ut"):
        type(concrete_filepath_info).model_validate(data)


def test_strict_raw_metadata_does_not_inherit_partial_identifiers():
    for name in ("obsnum", "subobsnum", "scannum"):
        field = LmtRawObsMetadata.model_fields[name]
        assert field.is_required()
        assert field.annotation is SeqNumT


def test_partial_models_do_not_turn_unknown_filenames_into_matches():
    assert FilepathInfo.from_source("unrecognized.nc", raise_on_error=False) is None
    with pytest.raises(ValueError, match="Unrecognized"):
        FilepathInfo.from_source("unrecognized.nc")
    with pytest.raises(ValueError):
        FilepathInfo.from_source(
            "tel_toltec_2025-13-31_145609_00_0002.nc", raise_on_error=False
        )


def test_catalog_projects_all_common_identifiers(data_lmt_path):
    paths = [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")),
    ]
    catalog = guess_info_from_sources(paths)
    assert "master_code" not in catalog
    assert catalog["master"].tolist() == ["tcs", "tcs"]
    assert catalog["data_models"].tolist() == [["lmt", "toltec"], ["toltec"]]
    assert catalog["instru"].tolist() == ["lmt", "toltec"]
    assert catalog["interface_kind"].tolist() == ["tel", "roach"]
    mask = catalog.data_models.map(
        lambda names: isinstance(names, list) and "toltec" in names
    )
    assert len(catalog.loc[mask].toltec_file.query("master == 'tcs'")) == 2
    for name in (
        "interface",
        "data_models",
        "interface_kind",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
    ):
        pd.testing.assert_series_equal(
            catalog[name], catalog[f"file_{name}"], check_names=False
        )


def test_concrete_filepath_nullable_values_and_roundtrips(concrete_filepath_info):
    info = concrete_filepath_info
    model = type(info)
    assert model.model_validate_json(info.model_dump_json(round_trip=True)) == info
    data = info.model_dump(round_trip=True)
    assert model.model_validate(data) == info
    assert model.model_validate(data | {"suffix": None}).suffix is None
