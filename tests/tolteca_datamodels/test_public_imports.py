"""Package entry points and module-level building blocks."""

import importlib

import pytest


@pytest.mark.parametrize(
    ("package_name", "exports", "building_blocks"),
    [
        (
            "tolteca_datamodels.lmt",
            {
                "._filepath": {"LmtFilepathInfo"},
                ".types": {"LmtInfo"},
                ".tel": {"LmtTelAccessor", "identify_lmt_tel", "open_lmt_tel"},
            },
            {
                ".types": {
                    "LmtMaster",
                    "LmtObsGoal",
                    "LmtObsPgm",
                    "LmtRawObsMasterCode",
                },
                ".metadata": {"LmtTelMetadata"},
                ".tel": {"LmtTelIOMapper", "LmtTelIOSchema"},
            },
        ),
        (
            "tolteca_datamodels.toltec",
            {
                "": {"open_toltec", "reduce_raw_sweep"},
                "._filepath": {"ToltecFilepathInfo"},
                ".types": {"ToltecInfo"},
                ".kids.core": {"ToltecKidsAccessor"},
            },
            {
                ".types": {
                    "ToltecArray",
                    "ToltecArrayNameT",
                    "ToltecDataKind",
                    "ToltecMaster",
                    "ToltecMasterNameT",
                },
                ".metadata": {
                    "ToltecBlocksMetadata",
                    "ToltecRawObsBlockMetadata",
                    "ToltecRawObsMetadata",
                    "ToltecSweepMetadata",
                    "ToltecTimeStreamMetadata",
                },
                ".axes": {"ToltecAxisMapper", "ToltecAxisSchema", "ToltecAxisView"},
                ".kids": {
                    "ReducedSweepView",
                    "SweepReducer",
                    "ToltecBlockView",
                    "ToltecBlocksView",
                },
                ".kids.core": {"ToltecKidsIOMapper", "ToltecKidsIOSchema"},
                ".kids.blocks": {"ToltecBlocksMapper", "ToltecBlocksSchema"},
            },
        ),
        (
            "tolteca_datamodels.toltec.kids",
            {
                ".blocks": {"ToltecBlockView", "ToltecBlocksView"},
                ".core": {"ToltecKidsAccessor"},
                ".sweep": {"ReducedSweepView", "SweepReducer"},
                ".timestream": {"TimestreamReducer", "TimestreamView"},
                ".zarr": {"open_zarr_dataset"},
            },
            {
                ".blocks": {"ToltecBlocksMapper", "ToltecBlocksSchema"},
                ".core": {"ToltecKidsIOMapper", "ToltecKidsIOSchema"},
                ".sweep": {"ToltecSweepMapper", "ToltecSweepSchema"},
                ".timestream": {"ToltecTimestreamMapper", "ToltecTimestreamSchema"},
            },
        ),
        (
            "tolteca_datamodels.lmt_base",
            {"._filepath": {"FilepathInfoMixin", "FilepathOrigMetadataMixin"}},
            {
                ".metadata": {
                    "LmtInstruMixin",
                    "LmtObsQuartetMixin",
                    "LmtRawObsMetadata",
                }
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
