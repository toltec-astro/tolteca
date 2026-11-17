"""Static telescope accessor types without runtime dataset wrappers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import assert_type

    from tollan.utils.fileloc import FileLoc

    from tolteca_datamodels.lmt import (
        LmtTelAccessor,
        LmtTelDataset,
        LmtTelDataTree,
        open_lmt_tel,
    )
    from tolteca_datamodels.lmt.metadata import LmtTelMetadata

    def _check_telescope_types(
        source: str | Path | FileLoc, tree: LmtTelDataTree
    ) -> None:
        dataset = open_lmt_tel(source=source)
        assert_type(dataset, LmtTelDataset)
        assert_type(dataset.lmt_tel, LmtTelAccessor)
        assert_type(dataset.lmt_tel.meta, LmtTelMetadata)
        with dataset as opened:
            assert_type(opened, LmtTelDataset)
            assert_type(opened.lmt_tel, LmtTelAccessor)
            assert_type(opened.lmt_tel.meta, LmtTelMetadata)
        with tree as node:
            assert_type(node, LmtTelDataTree)
            assert_type(node.lmt_tel, LmtTelAccessor)
            assert_type(node.lmt_tel.meta, LmtTelMetadata)
