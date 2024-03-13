import enum

from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column

from ..toltec.types import ToltecDataKind


class Instrument(enum.Enum):
    """The instrument."""

    lmt = "lmt"
    """The Large Millimeter Telescope."""

    toltec = "toltec"
    """The TolTEC Camera."""


class Interface(enum.Enum):
    """The instrument interface."""

    lmttcs = "lmttcs"


class FileInfoMixin(MappedAsDataclass):
    """A mixin orm model for file info."""

    instrument: Mapped[ToltecDataKind] = (mapped_column(default=None),)
    datamodels_toltec_data_kind: Mapped[ToltecDataKind] = mapped_column(default=None)


class LmtFileInfoMixin(MappedAsDataclass):
    """A mixin orm model for LMT file info."""

    instrument: Mapped[ToltecDataKind] = mapped_column(default=None)
    datamodels_toltec_data_kind: Mapped[ToltecDataKind] = mapped_column(default=None)


class ToltecFileInfoMixin(MappedAsDataclass):
    """A mixin orm model for TolTEC file info."""

    instrument: Mapped[ToltecDataKind] = mapped_column(default=None)
    datamodels_toltec_data_kind: Mapped[ToltecDataKind] = mapped_column(default=None)
