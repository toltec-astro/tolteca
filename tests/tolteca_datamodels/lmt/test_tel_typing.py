"""Static telescope accessor types without runtime dataset wrappers."""

from typing import TYPE_CHECKING

from tolteca_datamodels import lmt
from tolteca_datamodels.lmt import tel

if TYPE_CHECKING:
    from pathlib import Path
    from typing import assert_type

    from tollan.utils.fileloc import FileLoc

    from tolteca_datamodels.lmt import (
        LmtTelAccessor,
        LmtTelDataset,
        LmtTelDataTree,
        identify_lmt_tel,
        open_lmt_tel,
    )
    from tolteca_datamodels.lmt.metadata import LmtTelMetadata

    def _check_telescope_types(
        source: str | Path | FileLoc, tree: LmtTelDataTree, *, strict: bool
    ) -> None:
        dataset = open_lmt_tel(source=source)
        assert_type(dataset, LmtTelDataset)
        assert_type(dataset.lmt_tel, LmtTelAccessor)
        assert_type(dataset.lmt_tel.meta, LmtTelMetadata)
        assert_type(identify_lmt_tel(dataset), LmtTelAccessor | None)
        assert_type(
            identify_lmt_tel(dataset, raise_on_error=False), LmtTelAccessor | None
        )
        assert_type(identify_lmt_tel(dataset, raise_on_error=True), LmtTelAccessor)
        assert_type(
            identify_lmt_tel(dataset, raise_on_error=strict), LmtTelAccessor | None
        )
        assert_type(dataset.lmt_tel.read_metadata(), LmtTelMetadata)
        with dataset as opened:
            assert_type(opened, LmtTelDataset)
            assert_type(opened.lmt_tel, LmtTelAccessor)
            assert_type(opened.lmt_tel.meta, LmtTelMetadata)
        with tree as node:
            assert_type(node, LmtTelDataTree)
            assert_type(node.lmt_tel, LmtTelAccessor)
            assert_type(node.lmt_tel.meta, LmtTelMetadata)


def test_telescope_container_types_are_type_checking_only():
    for module in (lmt, tel):
        for name in ("LmtTelDataset", "LmtTelDataTree"):
            assert not hasattr(module, name)
            assert name not in module.__all__
