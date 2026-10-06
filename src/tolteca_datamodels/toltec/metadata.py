"""TolTEC observation metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr
from tollan.config import FrozenBaseModel
from tollan.config.types import FrequencyQuantityField

from ..lmt_base._filepath import FilepathOrigMetadataMixin
from ..lmt_base.metadata import LmtRawObsMetadata
from ..lmt_base.types import SeqNumT
from ._filepath import ToltecFilepathInfo
from .types import RoachT, ToltecInfo

if TYPE_CHECKING:
    from astropy.table import QTable

__all__ = [
    "ToltecBlocksMetadata",
    "ToltecRawObsBlockMetadata",
    "ToltecRawObsMetadata",
    "ToltecSweepMetadata",
    "ToltecTimeStreamMetadata",
]


class ToltecRawObsMetadata(
    LmtRawObsMetadata, FilepathOrigMetadataMixin[ToltecFilepathInfo]
):
    """TolTEC acquisition metadata and readout settings."""

    # Acquisition identity and mode.
    instru: ToltecInfo.InstruT = Field(
        default="toltec", description="TolTEC instrument"
    )
    master: ToltecInfo.MasterT = Field(description="Acquisition master label")
    master_code: ToltecInfo.MasterCodeT = Field(description="Recorded master code")
    obs_type: ToltecInfo.ObsTypeT | None = Field(
        default=None, description="Acquisition type, independent of block data kind"
    )
    obs_type_code: ToltecInfo.ObsTypeCodeT | None = Field(
        default=None, description="Recorded acquisition type code"
    )
    repeat: int | None = Field(
        default=None, description="Recorded acquisition repeat level"
    )

    # Readout identity.
    roach: RoachT | None = Field(default=None, description="ROACH board index")
    nw: RoachT | None = Field(default=None, description="Network index (ROACH index)")
    array: ToltecInfo.ArrayT | None = Field(
        default=None, description="Detector array index"
    )
    array_name: ToltecInfo.ArrayNameT | None = Field(
        default=None, description="Array name"
    )

    # Acquisition size and programmed readout settings.
    n_blocks_max: int = Field(ge=0, description="Allocated acquisition block capacity")
    n_times: int = Field(ge=0, description="Number of time samples in this record")
    n_chans: int = Field(description="Number of readout channels")
    n_chans_max: int | None = Field(default=None, description="Maximum number of tones")
    n_kids_design: int | None = Field(
        default=None, description="Number of designed KIDs"
    )
    n_sweepreps: int | None = Field(
        default=None, description="Programmed samples per sweep step, when recorded"
    )
    f_smp: FrequencyQuantityField = Field(description="Sample rate")
    f_lo_center: float | None = Field(
        default=None, description="Center LO frequency in Hz"
    )
    atten_drive: float | None = Field(
        default=None, description="Drive attenuation in dB"
    )
    atten_sense: float | None = Field(
        default=None, description="Sense attenuation in dB"
    )

    # Associated calibration acquisition.
    cal_roach: RoachT | None = Field(
        default=None, description="Calibration ROACH index"
    )
    cal_obsnum: SeqNumT | None = Field(
        default=None, description="Calibration observation number"
    )
    cal_subobsnum: SeqNumT | None = Field(
        default=None, description="Calibration sub-observation number"
    )
    cal_scannum: SeqNumT | None = Field(
        default=None, description="Calibration scan number"
    )


class ToltecBlocksMetadata(FrozenBaseModel):
    """Small, eager summary of a discovered block collection."""

    n_blocks: int = Field(ge=0, description="Number of discovered blocks")
    n_chans: int = Field(ge=0, description="Fixed readout channel count")


class ToltecRawObsBlockMetadata(ToltecRawObsMetadata):
    """One block's complete scalar metadata and optional table projections.

    Scalar records are self-contained. Table access is attached by a block view
    and is excluded from JSON; scientific arrays live in DataTree leaves.
    """

    n_blocks: int = Field(ge=1, description="Number of discovered acquisition blocks")
    block: int = Field(ge=0, description="Zero-based acquisition block index")
    _block_view: Any = PrivateAttr(default=None)

    @property
    def chan_axis_table(self) -> QTable | None:
        """Channel rows for this block, when attached to a data view."""
        return (
            self._block_view.chan_axis_table if self._block_view is not None else None
        )

    @property
    def probe_axis_table(self) -> QTable | None:
        """Probe dwells for this block, when attached to a data view."""
        return (
            self._block_view.probe_axis_table if self._block_view is not None else None
        )


class ToltecTimeStreamMetadata(ToltecRawObsBlockMetadata):
    """One TolTEC timestream block."""


class ToltecSweepMetadata(ToltecRawObsBlockMetadata):
    """One TolTEC sweep block and its programmed sweep settings."""

    n_sweepsteps: int = Field(description="Number of frequency steps per sweep")
