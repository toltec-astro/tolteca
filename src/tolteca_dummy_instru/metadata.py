"""Metadata for synthetic counter observations."""

from typing import Literal

from tolteca_datamodels.lmt_base import FilepathOrigMetadataMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata

from . import filepath as _filepath  # noqa: F401 -- register dummy filename parsing.
from .types import ChannelT, DummyInstruInfo

__all__ = ["DummyInstruInterfaceKindT", "DummyInstruMetadata"]

type DummyInstruInterfaceKindT = Literal["counter"]


class DummyInstruMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin):
    """Observation metadata for a synthetic counter channel."""

    instru: DummyInstruInfo.InstruT
    interface_kind: DummyInstruInterfaceKindT
    channel: ChannelT
