"""TolTEC metadata models.

Pydantic dataclasses with Field descriptions for self-documenting metadata.
All metadata classes are frozen (immutable) DTOs.

"""

from __future__ import annotations

from functools import cached_property
from typing import Literal

from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass
from tollan.config.types import FrequencyQuantityField

from ..core import metadata_dataclass_config
from ..lmt_base import LmtRawObsMetadata
from .toltecdb import ToltecDBRawObsMaster
from .types import ToltecArrayNameT, ToltecMasterType

__all__ = [
    "ToltecRawObsMetadata",
    "ToltecSweepMetadata",
    "ToltecTimeStreamMetadata",
]


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class ToltecRawObsMetadata(LmtRawObsMetadata):
    """TolTEC acquisition metadata and readout settings.

    ``master`` retains the recorded controller code. ``master_str`` supplies
    its readable name, used by the quartet and observation identifiers.
    """

    master: ToltecDBRawObsMaster = Field(description="Recorded acquisition master code")

    @computed_field
    @cached_property
    def master_str(self) -> ToltecMasterType:
        """Return the acquisition master's name."""
        return ToltecDBRawObsMaster.get_master_type(self.master)

    # Override parent fields with defaults where appropriate
    instru: Literal["toltec", "lmt"] | str = Field(
        default="toltec",
        description="TolTEC instrument",
    )

    # TolTEC-specific fields
    array_name: ToltecArrayNameT | str | None = Field(
        default=None,
        description="Array name",
    )

    roach: int | None = Field(
        default=None,
        description="ROACH board index",
    )

    # Derived field for network index (alias for roach in v2)
    nw: int | None = Field(
        default=None,
        description="Network index (alias for roach)",
    )

    # Observation type (from Header.Toltec.ObsType)
    obs_type: int | None = Field(
        default=None,
        description="Observation type code",
    )

    # Instrument settings
    atten_drive: float | None = Field(
        default=None,
        description="Drive attenuation in dB",
    )
    atten_sense: float | None = Field(
        default=None,
        description="Sense attenuation in dB",
    )

    # Calibration/association references
    cal_roach: int | None = Field(
        default=None,
        description="Associated calibration ROACH index",
    )
    cal_obsnum: int | None = Field(
        default=None,
        description="Associated calibration observation number",
    )
    cal_subobsnum: int | None = Field(
        default=None,
        description="Associated calibration sub-observation number",
    )
    cal_scannum: int | None = Field(
        default=None,
        description="Associated calibration scan number",
    )

    # ADC snap data
    adc_snap: str | None = Field(
        default=None,
        description="ADC snapshot data reference",
    )

    # Design/configuration parameters
    n_kids_design: int | None = Field(
        default=None,
        description="Number of designed KIDs (loclen)",
    )
    n_chans_max: int | None = Field(
        default=None,
        description="Maximum number of tones",
    )

    # File information
    filepath_orig: str | None = Field(
        default=None,
        description="Original acquisition file path",
    )

    repeat: int | None = Field(
        default=None,
        description="Recorded acquisition repeat level",
    )


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class ToltecTimeStreamMetadata(ToltecRawObsMetadata):
    """TolTEC time stream metadata with sample count and timing.

    Metadata for time-series KIDs data. All TolTEC data has time dimension,
    including sweeps which are time-series with frequency sweeping.
    """

    n_times: int = Field(description="Number of time samples")
    n_chans: int = Field(description="Number of readout channels")
    f_smp: FrequencyQuantityField = Field(description="Sample rate")
    f_lo_center: float | None = Field(
        default=None,
        description="Center LO frequency in Hz",
    )


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class ToltecSweepMetadata(ToltecTimeStreamMetadata):
    """TolTEC frequency sweep metadata (VNA, target, tune).

    Extends timestream metadata with sweep-specific fields. Sweeps are
    time-series data with frequency sweeping, so they inherit from timestream.
    Matches v2 structure where sweeps have all timestream fields plus sweep info.
    """

    n_sweeps: int = Field(description="Number of sweeps (blocks for multi-block data)")
    n_sweepsteps: int = Field(description="Number of frequency steps per sweep")
    n_sweepreps: int | None = Field(
        default=None,
        description="Number of repetitions per sweep step (for raw sweeps)",
    )

    # Multi-block fields (from v2)
    n_blocks: int | None = Field(
        default=None,
        description="Number of blocks (for multi-block data)",
    )
    n_blocks_max: int | None = Field(
        default=None,
        description="Maximum number of blocks",
    )
