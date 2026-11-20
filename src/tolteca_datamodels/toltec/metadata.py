"""TolTEC observation metadata."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import xarray as xr
from pydantic import Field, PrivateAttr
from tollan.accessor.xarray import dataset_to_qtable
from tollan.config.types import FrequencyQuantityField

from ..core.types import DataModelsT  # noqa: TC001 -- resolved by Pydantic
from ..lmt_base.filepath import FilepathOrigMetadataMixin
from ..lmt_base.metadata import LmtRawObsMetadata
from ..lmt_base.types import SeqNumT
from .axes import ToltecAxisView
from .toltec_db import ToltecDBRawObsMasterCode
from .types import ToltecInfo

if TYPE_CHECKING:
    from astropy.table import QTable

__all__ = [
    "ToltecProcessedKidsMetadata",
    "ToltecRawObsBlockMetadata",
    "ToltecRawObsBlocksMetadata",
    "ToltecRawObsMetadata",
    "ToltecSweepMetadata",
    "ToltecTimestreamMetadata",
]


class ToltecProcessedKidsMetadata(FilepathOrigMetadataMixin):
    """The acquisition named by a native processed KIDs file.

    `filepath_orig` is the source recorded by Kidscpp. Inspect
    `filepath_info_orig` for identity encoded in that name, or open the referenced
    acquisition explicitly to read its full headers.
    """

    instru: ToltecInfo.InstruT
    data_models: DataModelsT


class ToltecRawObsMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin):
    """TolTEC acquisition metadata and readout settings."""

    # Acquisition identity and mode.
    instru: ToltecInfo.InstruT = Field(description="TolTEC instrument")
    master: ToltecInfo.MasterT = Field(description="Acquisition master label")
    master_code: ToltecDBRawObsMasterCode = Field(description="Recorded master code")
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
    readout_code: int = Field(description="Recorded readout code")
    roach: ToltecInfo.RoachT | None = Field(description="ROACH board index")
    nw: ToltecInfo.RoachT | None = Field(description="Network index (ROACH index)")
    array: ToltecInfo.ArrayT | None = Field(description="Detector array index")
    array_name: ToltecInfo.ArrayNameT | None = Field(description="Array name")

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
    dac_normalization: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Applied gain of the DAC Fourier sum before integer quantization",
    )
    atten_sense: float | None = Field(
        default=None, description="Sense attenuation in dB"
    )

    # Associated calibration acquisition.
    # The acquisition header currently omits the calibration master. DPDB can
    # resolve it from the referenced ToltecDB record; the file alone cannot.
    # TODO(upstream acquisition): map the header field once the writer adds it.
    cal_master: ToltecInfo.MasterT | None = Field(
        default=None, description="Calibration acquisition master, when known"
    )
    cal_roach: ToltecInfo.RoachT | None = Field(
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


class ToltecRawObsBlocksMetadata(ToltecRawObsMetadata):
    """Acquisition metadata, block index and shared axes."""

    n_blocks: int = Field(ge=0, description="Number of discovered blocks")

    _block_index: xr.Dataset | None = PrivateAttr(default=None)
    _chan_axis: ToltecAxisView | None = PrivateAttr(default=None)
    _probe_axis: ToltecAxisView | None = PrivateAttr(default=None)

    @property
    def block_index(self) -> xr.Dataset | None:
        """Block kinds and selection bounds, when attached to acquisition data."""
        return self._block_index

    @cached_property
    def block_index_table(self) -> QTable | None:
        """Block-index QTable, materialized on first access when data is attached."""
        if self.block_index is None:
            return None
        return dataset_to_qtable(
            self.block_index,
            dim="block",
            columns={str(name): str(name) for name in self.block_index.variables},
        )

    @property
    def chan_axis_table(self) -> QTable | None:
        """Channel rows across blocks, when attached to acquisition data."""
        return self._chan_axis.axis_table if self._chan_axis is not None else None

    @property
    def probe_axis_table(self) -> QTable | None:
        """Probe dwells across blocks, when attached to acquisition data."""
        return self._probe_axis.axis_table if self._probe_axis is not None else None


class ToltecRawObsBlockMetadata(ToltecRawObsBlocksMetadata):
    """Acquisition identity and readout settings for one block."""

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


class ToltecTimestreamMetadata(ToltecRawObsBlockMetadata):
    """One TolTEC timestream block."""


class ToltecSweepMetadata(ToltecRawObsBlockMetadata):
    """One TolTEC sweep block and its programmed sweep settings."""

    n_sweepsteps: int = Field(description="Number of frequency steps per sweep")
