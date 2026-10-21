"""Data-model memberships, producing instrument, and interface classes."""

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt_base.metadata import InterfaceInfoMixin, LmtRawObsMetadata
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)
from tolteca_datamodels.toltec.hwpr import ToltecHwprMetadata
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_dummy_instru.filepath import DummyInstruLmtFilepathInfo
from tolteca_dummy_instru.metadata import DummyInstruMetadata


@pytest.mark.parametrize(
    "model",
    [
        LmtRawObsMetadata,
        LmtTelMetadata,
        ToltecRawObsMetadata,
        ToltecRawObsBlocksMetadata,
        ToltecRawObsBlockMetadata,
        ToltecSweepMetadata,
        ToltecTimestreamMetadata,
        ToltecHwprMetadata,
        DummyInstruMetadata,
    ],
)
def test_metadata_requires_explicit_interface_identity(model):
    names = {"data_models", "instru", "interface_kind", "interface"}
    with pytest.raises(ValidationError) as exc:
        model.model_validate({})
    missing = {
        error["loc"][0] for error in exc.value.errors() if error["type"] == "missing"
    }
    assert names <= missing
    schema = model.model_json_schema()
    assert names <= set(schema["required"])
    assert "producer" not in schema["properties"]
    for name in names:
        assert "default" not in schema["properties"][name]


@pytest.mark.parametrize(
    "model",
    [
        InterfaceInfoMixin,
        LmtTelMetadata,
        ToltecRawObsMetadata,
        DummyInstruMetadata,
        ToltecFilepathInfo,
        ToltecLmtFilepathInfo,
        DummyInstruLmtFilepathInfo,
    ],
)
def test_interface_identity_fields_are_grouped(model):
    names = list(model.model_fields)
    start = names.index("instru")
    assert names[start : start + 4] == [
        "instru",
        "data_models",
        "interface_kind",
        "interface",
    ]


@pytest.mark.parametrize(
    ("prefix", "model", "data_models"),
    [
        ("tel_toltec", ToltecLmtFilepathInfo, ["lmt", "toltec"]),
        ("tel_toltec2", ToltecLmtFilepathInfo, ["lmt", "toltec"]),
        ("tel_dummy_instru", DummyInstruLmtFilepathInfo, ["lmt", "dummy_instru"]),
    ],
)
def test_telescope_layout_selects_specialized_result(prefix, model, data_models):
    name = f"{prefix}_2025-10-31_145609_00_0002.nc"
    info = FilepathInfo.from_source(name)
    assert isinstance(info, LmtFilepathInfo)
    assert type(info) is model
    assert (info.data_models, info.instru, info.interface_kind, info.interface) == (
        data_models,
        "lmt",
        "tel",
        "tel",
    )
    assert info.master == "tcs"
    with pytest.raises(ValidationError, match="instru"):
        model.model_validate(info.model_dump(round_trip=True) | {"instru": "unrelated"})


@pytest.mark.parametrize("accessor", [LmtFileAccessor, ToltecFileAccessor])
def test_telescope_identity_survives_header_enrichment(data_lmt_path, accessor):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    guessed = accessor.from_sources([path])
    actual = accessor(guessed).update_from_source_metadata()
    names = ["data_models", "instru", "interface_kind", "interface", "master"]
    expected = [["lmt", "toltec"], "lmt", "tel", "tel", "tcs"]
    assert guessed[names].iloc[0].tolist() == expected
    assert actual[names].iloc[0].tolist() == expected
    for name in names:
        pd.testing.assert_series_equal(
            actual[name], actual[f"file_{name}"], check_names=False
        )
    assert actual.uid_raw_obs_file.tolist() == guessed.uid_raw_obs_file.tolist()
    assert actual.uid_raw_obs_file.iloc[0] == "tcs-145609-0-2-tel"
    assert (
        accessor(actual).query('instru == "lmt" and interface_kind == "tel"').shape[0]
        == 1
    )
    assert "instru_component" not in actual
    assert "file_instru_component" not in actual
    assert "producer_component" not in actual


@pytest.mark.parametrize("interface", ["toltec0", "toltec7", "hwpr", "toltec_hk"])
def test_interface_class_is_declared_separately_from_instance(interface):
    name = f"{interface}_145609_000_0002_2025_10_31_09_27_57.nc"
    info = FilepathInfo.from_source(name)
    assert isinstance(info, ToltecFilepathInfo)
    assert (info.data_models, info.instru, info.interface) == (
        ["toltec"],
        "toltec",
        interface,
    )
    assert info.interface_kind == (
        "roach" if interface in {"toltec0", "toltec7"} else interface
    )
