"""Recorded calibration acquisitions with explicit sweep-block bindings."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, FiniteFloat, model_validator

from ..core.types import FrozenDataModel
from .artifact import ArtifactBinding
from .model import KidsModelBinding
from .types import RawObsRef

__all__ = [
    "ArrayFlux",
    "ArrayPropTableCalItem",
    "CalItem",
    "PhotometryCalItem",
    "PointingCalItem",
    "PointingOffset",
    "SweepCalDataItem",
    "SweepCalItem",
]


class SweepCalDataItem(FrozenDataModel):
    """One explicitly selected interface and block of a calibration sweep.

    Parameters
    ----------
    interface : str
        Nonempty interface name in the calibration acquisition, for example
        ``toltec0``. Its name identifies the target data item's interface too.
    block : int
        Zero-based, nonnegative block index supplied by block identification.
        Negative or symbolic selections such as "last" are not resolved bindings.
    model : KidsModelBinding or None, optional
        Exact fitted realization of this sweep. None leaves model selection
        unresolved; it does not request fitting or select a header model.

    Notes
    -----
    The selection denotes the complete sweep block with its recorded channels.
    Construction does not verify that the referenced interface/block exists or
    contains sweep data. It does not establish a detector correspondence or
    verify fitted model payloads.
    """

    model_config = ConfigDict(extra="forbid")

    interface: str = Field(min_length=1)
    block: int = Field(ge=0)
    model: KidsModelBinding | None = None


class SweepCalItem(FrozenDataModel):
    """A recorded calibration acquisition with exact interface/block bindings.

    Parameters
    ----------
    type : {'sweep'}
        Explicit calibration-role discriminator.
    raw_obs : RawObsRef
        Acquisition containing the recorded calibration block. This is an
        acquisition relationship, not a reduction setting or candidate choice.
    data_items : tuple of SweepCalDataItem
        Nonempty selections with distinct interfaces. A context-wide binding can
        select several networks from the same acquisition. An item-local binding
        selects only that item's interface.

    Raises
    ------
    ValueError
        An interface occurs more than once in the calibration selection.

    Notes
    -----
    Use RawObsContext.from_product to derive this reference from the acquisition's
    portable calibration lineage. A directly constructed or restored binding
    can be checked with RawObsContext.validate_lineage for exact
    source/target block identity.
    Different reductions of the recorded sweep must retain its raw block identity.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["sweep"]
    raw_obs: RawObsRef
    data_items: tuple[SweepCalDataItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_interfaces(self) -> Self:
        interfaces = [item.interface for item in self.data_items]
        if len(set(interfaces)) != len(interfaces):
            msg = "Sweep calibration requires distinct interfaces"
            raise ValueError(msg)
        return self


class ArrayPropTableCalItem(FrozenDataModel):
    """An exact array-property table, common or scoped to a detector item.

    Parameters
    ----------
    type : {'array_prop_table'}
        Calibration role.
    artifact : ArtifactBinding
        Pinned table artifact. The consumer validates detector correspondence.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["array_prop_table"]
    artifact: ArtifactBinding


class PointingOffset(FrozenDataModel):
    """Resolved pointing correction samples for one telescope axis.

    Parameters
    ----------
    axis : {'az', 'alt'}
        Telescope axis.
    values_arcsec : tuple of float
        Finite correction samples in arcseconds, using the consumer's documented
        sign convention. A single sample is constant.
    """

    model_config = ConfigDict(extra="forbid")
    axis: Literal["az", "alt"]
    values_arcsec: tuple[FiniteFloat, ...] = Field(min_length=1)


class PointingCalItem(FrozenDataModel):
    """Resolved telescope pointing corrections.

    Parameters
    ----------
    type : {'pointing'}
        Calibration role.
    offsets : tuple of PointingOffset
        Corrections in azimuth and altitude; both axes must occur once.
    mjd : tuple of float, optional
        Increasing MJD sample times. Empty denotes constant corrections.
    artifacts : tuple of ArtifactBinding, optional
        Exact PPT sources in sample order. Empty for explicitly supplied
        corrections; otherwise one binding per MJD sample. Each PPT scope is
        its measurement acquisition UID.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["pointing"]
    offsets: tuple[PointingOffset, PointingOffset]
    mjd: tuple[FiniteFloat, ...] = ()
    artifacts: tuple[ArtifactBinding, ...] = ()

    @model_validator(mode="after")
    def _samples(self):
        if {x.axis for x in self.offsets} != {"az", "alt"}:
            msg = "Pointing requires az and alt corrections"
            raise ValueError(msg)
        count = len(self.mjd) if self.mjd else 1
        if any(len(x.values_arcsec) != count for x in self.offsets):
            msg = "Pointing corrections and times must have equal lengths"
            raise ValueError(msg)
        if any(b <= a for a, b in zip(self.mjd, self.mjd[1:], strict=False)):
            msg = "Pointing times must increase"
            raise ValueError(msg)
        if self.artifacts and len(self.artifacts) != len(self.mjd):
            msg = "Pointing artifacts and sample times must have equal lengths"
            raise ValueError(msg)
        return self


class ArrayFlux(FrozenDataModel):
    """Calibrator flux density and uncertainty for one TolTEC array.

    Parameters
    ----------
    array_name : {'a1100', 'a1400', 'a2000'}
        TolTEC array.
    value_mJy : float
        Positive finite flux density in milliJanskys.
    uncertainty_mJy : float
        Nonnegative finite flux uncertainty in milliJanskys.
    """

    model_config = ConfigDict(extra="forbid")
    array_name: Literal["a1100", "a1400", "a2000"]
    value_mJy: FiniteFloat = Field(gt=0)  # noqa: N815 -- physical unit symbol
    uncertainty_mJy: FiniteFloat = Field(ge=0)  # noqa: N815 -- physical unit symbol


class PhotometryCalItem(FrozenDataModel):
    """Resolved calibrator flux densities, with one entry per selected array.

    Parameters
    ----------
    type : {'photometry'}
        Calibration role.
    fluxes : tuple of ArrayFlux
        Nonempty flux collection with distinct array names.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["photometry"]
    fluxes: tuple[ArrayFlux, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _arrays(self):
        if len({f.array_name for f in self.fluxes}) != len(self.fluxes):
            msg = "Photometry requires distinct arrays"
            raise ValueError(msg)
        return self


CalItem = Annotated[
    SweepCalItem | ArrayPropTableCalItem | PointingCalItem | PhotometryCalItem,
    Field(discriminator="type"),
]
