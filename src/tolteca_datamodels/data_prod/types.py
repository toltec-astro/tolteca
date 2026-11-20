"""Shared classifications for TolTECA data products."""

import re
from enum import Flag, auto
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, Field
from tollan.config.types import PathSegment

NamedGroupUid = Annotated[UUID, Field(strict=False)]
"""Portable collection UUID, independent of display name and directory."""


def _named_group_path(value: str) -> str:
    if re.fullmatch(r"(?:[A-Za-z][A-Za-z0-9]*-)?[0-9]+(?:-[0-9]+-[0-9]+)?", value):
        msg = "Named-group path is reserved for an observation or acquisition"
        raise ValueError(msg)
    return value


NamedGroupPath = Annotated[PathSegment, AfterValidator(_named_group_path)]
"""One readable directory component, excluding observation/acquisition labels."""

PRIMARY_PRODUCT_TYPES = frozenset({"dp_obs", "dp_raw_obs", "dp_named_group"})
"""Logical products that are their own primary, regardless of membership."""

LOGICAL_PRODUCT_TYPES = PRIMARY_PRODUCT_TYPES | {"dp_cal_group", "dp_version_group"}
"""Global products independent of a producing namespace or execution."""

ASSOC_TARGET_TYPES: dict[str, str | None] = {
    "dpa_obs_raw_obs": "dp_raw_obs",
    "dpa_raw_obs_cal_obs": "dp_raw_obs",
    "dpa_raw_obs_proc_raw_obs": "dp_raw_obs",
    "dpa_raw_obs_proc_data_prod": None,
    "dpa_cal_group_raw_obs": "dp_raw_obs",
    "dpa_drivefit_raw_obs": "dp_raw_obs",
    "dpa_focus_group_raw_obs": "dp_raw_obs",
    "dpa_astig_group_raw_obs": "dp_raw_obs",
    "dpa_oof_group_raw_obs": "dp_raw_obs",
    "dpa_named_group_data_prod": None,
    "dpa_version_group_data_prod": None,
    "dpa_version_group_obs": "dp_obs",
    "dpa_version_group_raw_obs": "dp_raw_obs",
    "dpa_version_group_named_group": "dp_named_group",
    "dpa_quicklook_obs": "dp_obs",
    "dpa_quicklook_raw_obs": "dp_raw_obs",
    "dpa_quicklook_data_prod": None,
    "dpa_reduction_obs": "dp_obs",
    "dpa_reduction_raw_obs": "dp_raw_obs",
    "dpa_reduction_named_group": "dp_named_group",
    "dpa_reduction_data_prod": None,
    "dpa_apt_obs": "dp_obs",
    "dpa_apt_raw_obs": "dp_raw_obs",
    "dpa_apt_named_group": "dp_named_group",
    "dpa_apt_ref_apt": "dp_apt",
    "dpa_apt_data_prod": None,
    "dpa_tone_match_raw_obs": "dp_raw_obs",
    "dpa_tone_match_data_prod": None,
    "dpa_skydip_named_group": "dp_named_group",
    "dpa_skydip_data_prod": None,
    "dpa_atten_grid_obs": "dp_obs",
    "dpa_atten_grid_raw_obs": "dp_raw_obs",
}
"""Registered association types and the product type each one references.

None means the association can reference a product of any type.
"""

VERSION_GROUP_PRIMARY_ASSOC_TYPES = {
    "dp_obs": "dpa_version_group_obs",
    "dp_raw_obs": "dpa_version_group_raw_obs",
    "dp_named_group": "dpa_version_group_named_group",
}
"""Association that places a version group at its primary, by primary type."""


class DataKind(Flag):
    """TolTEC raw observation data acquisition kinds.

    Bitwise flags used to differentiate observation types within the
    ``dp_raw_obs`` product type without creating separate table rows.
    Calibration is a *relationship* (association), not a type.

    Reference: tolteca_datamodels/toltec/types.py
    """

    # Sweep types (KIDs calibration modes)
    VnaSweep = auto()
    """Full-range VNA sweep (bootstrapping)."""

    TargetSweep = auto()
    """Target sweep on a list of frequencies (refinement)."""

    Tune = auto()
    """Pack of back-to-back target sweeps (fine adjustment)."""

    RawSweep = VnaSweep | TargetSweep | Tune
    """All sweep kinds combined."""

    # Science observation mode
    RawTimeStream = auto()
    """Continuous capture at probe tones (science mode)."""

    RawKidsData = RawSweep | RawTimeStream
    """All raw KIDs data kinds."""

    # Reduced KIDs data
    D21 = auto()
    """D21 processed sweep data."""

    ReducedVnaSweep = auto()
    ReducedTargetSweep = auto()
    ReducedSweep = ReducedVnaSweep | ReducedTargetSweep
    SolvedTimeStream = auto()
    ReducedKidsData = D21 | ReducedSweep | SolvedTimeStream

    # Telescope / housekeeping
    LmtTel = auto()
    """LMT telescope interface data (tel_toltec)."""

    LmtTel2 = auto()
    """Supplementary LMT telescope data at original sample rate."""

    Unknown = auto()
    """Unknown data kind."""

    # Append persisted bits: existing values must not move.
    Hwpr = auto()
    """Half-wave plate readout data."""
