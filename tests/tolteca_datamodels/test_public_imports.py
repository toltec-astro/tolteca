"""Package entry points and module-level building blocks."""

import importlib
import importlib.util

import pytest


@pytest.mark.parametrize(
    "package_name",
    ["tolteca_datamodels.lmt", "tolteca_datamodels.toltec", "tolteca_dummy_instru"],
)
def test_instrument_obs_spec_contexts_are_private(package_name):
    from tolteca_datamodels.core import obs_spec

    package = importlib.import_module(package_name)
    context_module = importlib.import_module(f"{package_name}._obs_spec")
    assert "ObsSpecContext" in obs_spec.__all__
    assert isinstance(context_module._obs_spec_context, obs_spec.ObsSpecContext)
    assert not hasattr(context_module, "obs_spec_context")
    assert not hasattr(package, "obs_spec_context")
    assert importlib.util.find_spec(f"{package_name}.obs_spec") is None


@pytest.mark.parametrize(
    ("package_name", "exports", "building_blocks"),
    [
        (
            "tolteca_datamodels.lmt",
            {
                ".types": {"LmtInfo"},
                ".tel": {"LmtTelAccessor", "open_lmt_tel"},
            },
            {
                ".types": {
                    "LmtInterfaceInfo",
                    "LmtObsGoal",
                    "LmtObsPgm",
                    "LmtRawObsMasterCode",
                },
                ".metadata": {"LmtTelMetadata"},
                ".filepath": {"LmtFilepathInfo"},
                ".input_discovery": {"LmtInputDiscovery"},
                ".tel": {"LmtTelIOMapper", "LmtTelIOSchema"},
            },
        ),
        (
            "tolteca_datamodels.toltec",
            {
                ".hwpr": {"open_toltec_hwpr"},
                ".types": {"ToltecInfo"},
                ".kids.core": {"ToltecKidsAccessor", "open_toltec_kids"},
            },
            {
                ".types": {"ToltecDataKind"},
                ".filepath": {"ToltecFilepathInfo", "ToltecLmtFilepathInfo"},
                ".input_discovery": {"ToltecInputDiscovery"},
                ".metadata": {
                    "ToltecRawObsBlocksMetadata",
                    "ToltecRawObsBlockMetadata",
                    "ToltecRawObsMetadata",
                    "ToltecSweepMetadata",
                    "ToltecTimestreamMetadata",
                },
                ".axes": {"ToltecAxisMapper", "ToltecAxisSchema", "ToltecAxisView"},
                ".kids": {
                    "ToltecBlockIndex",
                    "ToltecKidsIOMapper",
                    "ToltecKidsIOSchema",
                },
            },
        ),
        (
            "tolteca_datamodels.toltec.kids",
            {
                ".blocks": {"ToltecBlockIndex"},
                ".core": {
                    "ToltecKidsAccessor",
                    "ToltecKidsIOMapper",
                    "ToltecKidsIOSchema",
                    "open_toltec_kids",
                },
            },
            {},
        ),
        (
            "tolteca_datamodels.lmt_base",
            {".filepath": {"FilepathInfoMixin", "FilepathOrigMetadataMixin"}},
            {
                ".metadata": {
                    "InterfaceInfoMixin",
                    "LmtObsQuartetMixin",
                    "LmtRawObsMetadata",
                },
                ".input_discovery": {
                    "LmtBaseFileStoreInputDiscovery",
                    "SourcesInputDiscovery",
                },
            },
        ),
    ],
)
def test_package_exports_and_building_block_ownership(
    package_name, exports, building_blocks
):
    package = importlib.import_module(package_name)
    assert set(package.__all__) == {
        name for names in exports.values() for name in names
    }
    for suffix, names in exports.items():
        owner = importlib.import_module(package_name + suffix)
        for name in names:
            assert getattr(package, name) is getattr(owner, name)
    for suffix, names in building_blocks.items():
        owner = importlib.import_module(package_name + suffix)
        for name in names:
            assert name in owner.__all__
            assert getattr(owner, name) is not None
            # Removing only __all__ entries would leave accidental re-exports.
            assert not hasattr(package, name)


@pytest.mark.parametrize(
    "package_name",
    ["tolteca_datamodels.lmt", "tolteca_datamodels.toltec", "tolteca_dummy_instru"],
)
def test_data_model_definition_has_an_explicit_module(package_name):
    from tolteca_datamodels.core.data_model import DataModelSpec

    package = importlib.import_module(package_name)
    module = importlib.import_module(f"{package_name}.data_model")
    assert "data_model" not in package.__all__
    assert module.__all__ == ["data_model"]
    assert isinstance(module.data_model, DataModelSpec)
    assert (
        module.data_model.name
        == module.data_model.file_accessor_cls.data_model_cls.name
    )
    assert importlib.util.find_spec(f"{package_name}.core") is None


@pytest.mark.parametrize(
    "package_name",
    [
        "tolteca_datamodels.lmt_base",
        "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec",
        "tolteca_dummy_instru",
    ],
)
def test_discovery_settings_have_one_public_module(package_name):
    module = importlib.import_module(f"{package_name}.input_discovery")
    for name in module.__all__:
        assert getattr(module, name).__module__ == module.__name__
    assert importlib.util.find_spec(f"{package_name}._input_discovery") is None
