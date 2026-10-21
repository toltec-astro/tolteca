"""Identity and channel declarations for the dummy instrument."""

from enum import Flag, auto
from typing import Annotated, ClassVar, Literal

from pydantic import Field
from tollan.utils.typing import get_literal_value

from tolteca_datamodels.core.types import DataModelInfo

__all__ = ["ChannelT", "DummyInstruDataKind", "DummyInstruDataModel", "DummyInstruInfo"]

type ChannelT = Annotated[int, Field(ge=0, le=3)]


class DummyInstruDataModel(DataModelInfo):
    """Dummy-instrument data-model information, including telescope companions."""

    name = "dummy_instru"


class DummyInstruDataKind(Flag):
    """Data kinds for dummy-instrument processing."""

    counts = auto()
    """Synthetic counter observations."""

    telescope = auto()
    """Companion telescope observations."""

    unknown = auto()
    """Unidentified or unsupported data."""


class DummyInstruInfo:
    """Four synthetic counter channels, identified by zero-padded labels."""

    type InstruT = Literal["dummy_instru"]

    instru: ClassVar[InstruT] = get_literal_value(InstruT)
    interfaces: ClassVar = tuple(f"chan{number:02d}" for number in range(4))
    interface_channel: ClassVar = dict(zip(interfaces, range(4), strict=True))
