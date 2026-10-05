"""Metadata for synthetic counter observations."""

from typing import Literal

from tolteca_datamodels.lmt_base import FilepathOrigMetadataMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata

from .filepath import DummyInstruFilepathInfo
from .types import ChannelT, DummyInstruInfo

__all__ = ["DummyInstruMetadata"]


class DummyInstruMetadata(
    LmtRawObsMetadata, FilepathOrigMetadataMixin[DummyInstruFilepathInfo]
):
    """Observation metadata for a synthetic counter channel."""

    instru: DummyInstruInfo.InstruT = DummyInstruInfo.instru
    instru_component: Literal["counter"] = "counter"
    channel: ChannelT
