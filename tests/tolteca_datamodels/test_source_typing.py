"""Typed instrument wrappers over the shared source workflow."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import assert_type

    from tollan.utils.fileloc import FileLoc

    from tolteca_datamodels.toltec import ToltecKidsDataset, open_toltec_kids
    from tolteca_datamodels.toltec.hwpr import (
        ToltecHwprAccessor,
        ToltecHwprDataset,
        open_toltec_hwpr,
    )
    from tolteca_datamodels.toltec.kids.core import (
        ToltecKidsAccessor,
        ToltecKidsDataTree,
    )
    from tolteca_dummy_instru.data import (
        DummyInstruAccessor,
        DummyInstruDataset,
        open_dummy_instru,
    )

    def _check_kids_types(source: str | Path | FileLoc, tree: ToltecKidsDataTree):
        dataset = open_toltec_kids(source=source)
        assert_type(dataset, ToltecKidsDataset)
        assert_type(dataset.toltec_kids, ToltecKidsAccessor)
        assert_type(tree.toltec_kids, ToltecKidsAccessor)
        with dataset as opened:
            assert_type(opened, ToltecKidsDataset)

    def _check_dummy_types(source: str | Path | FileLoc):
        dataset = open_dummy_instru(source=source)
        assert_type(dataset, DummyInstruDataset)
        assert_type(dataset.dummy_instru, DummyInstruAccessor)
        with dataset as opened:
            assert_type(opened, DummyInstruDataset)

    def _check_hwpr_types(source: str | Path | FileLoc):
        dataset = open_toltec_hwpr(source)
        assert_type(dataset, ToltecHwprDataset)
        assert_type(dataset.toltec_hwpr, ToltecHwprAccessor)
