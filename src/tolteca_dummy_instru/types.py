"""Identity and channel declarations for the dummy instrument."""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

__all__ = ["ChannelT", "DummyInstruInfo"]

type ChannelT = Annotated[int, Field(ge=0, le=3)]


class DummyInstruInfo:
    """Four synthetic counter channels, identified by zero-padded labels."""

    type InstruT = Literal["dummy_instru"]

    instru: ClassVar[InstruT] = "dummy_instru"
    interfaces: ClassVar = tuple(f"chan{number:02d}" for number in range(4))
    interface_channel: ClassVar = dict(zip(interfaces, range(4), strict=True))
