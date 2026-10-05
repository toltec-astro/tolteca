"""Typed instrument wrappers over the shared source workflow."""

from typing import TYPE_CHECKING

from tolteca_datamodels import toltec
from tolteca_datamodels.toltec.kids import core
from tolteca_dummy_instru import data

if TYPE_CHECKING:
    from io import StringIO
    from pathlib import Path
    from typing import assert_type

    import xarray as xr
    from tollan.utils.fileloc import FileLoc

    from tolteca_datamodels.source import open_source
    from tolteca_datamodels.toltec import ToltecKidsDataset, open_toltec
    from tolteca_datamodels.toltec.kids.core import (
        ToltecKidsAccessor,
        ToltecKidsDataTree,
        identify_toltec_kids,
    )
    from tolteca_dummy_instru.data import (
        DummyInstruAccessor,
        DummyInstruDataset,
        identify_dummy_instru,
        open_dummy_instru,
    )

    def _check_backend_types(source: str | Path | FileLoc):
        assert_type(open_source(source, opener=xr.open_dataset), xr.Dataset)
        assert_type(open_source(source, opener=lambda _: StringIO()), StringIO)

    def _check_kids_types(
        source: str | Path | FileLoc, tree: ToltecKidsDataTree, *, strict: bool
    ):
        dataset = open_toltec(source=source)
        assert_type(dataset, ToltecKidsDataset)
        assert_type(dataset.toltec_kids, ToltecKidsAccessor)
        assert_type(tree.toltec_kids, ToltecKidsAccessor)
        assert_type(identify_toltec_kids(dataset), ToltecKidsAccessor | None)
        assert_type(
            identify_toltec_kids(dataset, raise_on_error=True), ToltecKidsAccessor
        )
        assert_type(
            identify_toltec_kids(dataset, raise_on_error=strict),
            ToltecKidsAccessor | None,
        )
        with dataset as opened:
            assert_type(opened, ToltecKidsDataset)

    def _check_dummy_types(source: str | Path | FileLoc, *, strict: bool):
        dataset = open_dummy_instru(source=source)
        assert_type(dataset, DummyInstruDataset)
        assert_type(dataset.dummy_instru, DummyInstruAccessor)
        assert_type(identify_dummy_instru(dataset), DummyInstruAccessor | None)
        assert_type(
            identify_dummy_instru(dataset, raise_on_error=True), DummyInstruAccessor
        )
        assert_type(
            identify_dummy_instru(dataset, raise_on_error=strict),
            DummyInstruAccessor | None,
        )
        with dataset as opened:
            assert_type(opened, DummyInstruDataset)


def test_container_types_do_not_create_runtime_wrappers():
    for module in (toltec, core):
        for name in ("ToltecKidsDataset", "ToltecKidsDataTree"):
            assert not hasattr(module, name)
            assert name not in module.__all__
    assert not hasattr(data, "DummyInstruDataset")
    assert "DummyInstruDataset" not in data.__all__
