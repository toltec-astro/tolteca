"""Partial observation identifiers and their file-table projections."""

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt_base._filepath import FilepathInfoMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata
from tolteca_datamodels.lmt_base.types import SeqNumT
from tolteca_datamodels.toltec.file import (
    ToltecFilepathInfo,
    guess_info_from_sources,
)


def test_shared_filepath_values_belong_to_observation_mixin():
    assert not FilepathInfo.model_fields
    assert {"prefix", "suffix", "ext", "obsnum", "ut"} <= (
        FilepathInfoMixin.model_fields.keys()
    )
    assert all(
        field.is_required()
        for name, field in FilepathInfoMixin.model_fields.items()
        if name not in {"suffix", "master"}
    )


@pytest.fixture(params=[LmtFilepathInfo, ToltecFilepathInfo])
def concrete_filepath_info(request, lmtmc_records, toltecdb_records):
    rows = lmtmc_records if request.param is LmtFilepathInfo else toltecdb_records
    info = FilepathInfo.from_source(rows[0]["FileName"])
    assert type(info) is request.param
    return info


@pytest.mark.parametrize(
    "name",
    [
        "prefix",
        "interface",
        "instru",
        "instru_component",
        "obsnum",
        "subobsnum",
        "scannum",
        "ut",
        "ext",
    ],
)
def test_concrete_filepath_requires_guaranteed_values(concrete_filepath_info, name):
    model = type(concrete_filepath_info)
    assert model.model_fields[name].is_required()
    data = concrete_filepath_info.model_dump()
    del data[name]
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data)
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data | {name: None})


def test_telescope_master_is_required(lmtmc_records):
    parsed = LmtFilepathInfo.from_source(lmtmc_records[0]["FileName"])
    assert parsed is not None and parsed.master == "tcs"
    assert LmtFilepathInfo.model_fields["master"].is_required()
    data = parsed.model_dump()
    del data["master"]
    with pytest.raises(ValidationError, match="master"):
        LmtFilepathInfo.model_validate(data)
    with pytest.raises(ValidationError, match="master"):
        LmtFilepathInfo.model_validate(data | {"master": None})


def test_concrete_filepath_rejects_wrong_instrument(concrete_filepath_info):
    data = concrete_filepath_info.model_dump() | {"instru": "other"}
    with pytest.raises(ValidationError, match="instru"):
        type(concrete_filepath_info).model_validate(data)


@pytest.mark.parametrize("name", ["obsnum", "subobsnum", "scannum"])
def test_concrete_observation_numbers_remain_nonnegative(concrete_filepath_info, name):
    model = type(concrete_filepath_info)
    data = concrete_filepath_info.model_dump()
    with pytest.raises(ValidationError, match=name):
        model.model_validate(data | {name: -1})
    assert getattr(model.model_validate(data | {name: "0"}), name) == 0


def test_concrete_filepaths_keep_timezone_validation(concrete_filepath_info):
    data = concrete_filepath_info.model_dump()
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
        next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc")),
    ]
    catalog = guess_info_from_sources(paths)
    assert "master_code" not in catalog
    assert catalog["master"].tolist() == ["tcs", "tcs"]
    assert catalog["instru"].tolist() == ["lmt", "toltec"]
    assert catalog["instru_component"].tolist() == ["tcs", "roach"]
    assert len(catalog.toltec_file.query("instru == 'toltec' & master == 'tcs'")) == 1
    for name in (
        "interface",
        "instru",
        "instru_component",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
    ):
        pd.testing.assert_series_equal(
            catalog[name], catalog[f"file_{name}"], check_names=False
        )


def test_concrete_filepath_optional_values_and_roundtrips(concrete_filepath_info):
    info = concrete_filepath_info
    model = type(info)
    assert model.model_validate_json(info.model_dump_json()) == info
    assert model.model_validate(info.model_dump()) == info
    assert model.model_validate(info.model_dump() | {"suffix": None}).suffix is None
