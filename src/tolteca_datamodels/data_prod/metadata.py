"""Metadata for products, file sources and association evidence.

Source metadata includes complete instrument metadata from opened files and
partial acquisition-database records for files that are unavailable locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, get_args
from uuid import UUID

from adaptix import Retort, dumper, loader
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    model_serializer,
)

from ..core.data_prod import DataProdRef
from ..lmt.metadata import LmtTelMetadata
from ..lmt_base.types import SeqNumT
from ..toltec.hwpr import ToltecHwprMetadata
from ..toltec.metadata import ToltecRawObsMetadata
from .calibration import RawObsFileCalObsFileMeta
from .types import DataKind

# ====================  Shared Mixin Classes  ====================


class _MetadataModel(BaseModel):
    """Editable metadata whose field assignments retain validation."""

    model_config = ConfigDict(
        strict=True, validate_default=True, extra="forbid", validate_assignment=True
    )


class ObsIdMixin(_MetadataModel):
    """Core observation identification fields: obsnum quartet + master.

    Shared between interface-level and data product-level metadata.
    """

    obsnum: SeqNumT = 0
    subobsnum: SeqNumT = 0
    scannum: SeqNumT = 0
    master: str = ""


class RoachMetaMixin(_MetadataModel):
    """Roach interface metadata fields for toltecN roach boards."""

    nw_id: int | None = None
    roach: int | None = None
    interface: str | None = None
    hostname: str | None = None


class TelMetaMixin(_MetadataModel):
    """LMT telescope metadata fields shared between interface and product levels.

    Unavailable telescope values remain None.
    """

    source_name: str | None = None
    obs_goal: str | None = None
    project_id: str | None = None
    obs_pgm: str | None = None
    integration_time: float | None = None
    az_deg: float | None = None
    el_deg: float | None = None
    user_az_offset_arcsec: float | None = None
    user_el_offset_arcsec: float | None = None
    paddle_az_offset_arcsec: float | None = None
    paddle_el_offset_arcsec: float | None = None
    m1_zernike: list[float] | None = None
    m2_offset_mm: tuple[float, float, float] | None = None
    tau: float | None = None
    crane_in_beam: bool | None = None


# ====================  Base Classes  ====================


class DataProdMetaBase(_MetadataModel):
    """Base metadata for all data products.

    Construct metadata from typed values, or use ``model_validate`` for a
    dictionary. Unknown fields and incompatible values raise validation errors.
    Field assignments are validated too. Use ``model_validate_json`` to restore
    persisted JSON values.

    Attributes
    ----------
    name : str
        Human-readable product name.
    data_prod_type : str
        TolTEC product type string (e.g. "dp_raw_obs").
    description : str | None
        Optional human-readable description.
    obs_datetime : datetime | None
        Observation datetime or latest member datetime for groups.
    """

    name: str = ""
    data_prod_type: str = ""
    description: str | None = None
    obs_datetime: datetime | None = None


# ====================  DataProd Metadata Classes  ====================


class VersionedProductMeta(_MetadataModel):
    """Metadata of a realized product that belongs to a version group."""

    version_group: DataProdRef | None = Field(
        default=None,
        description=(
            "Version group of this realization. Materialization sets it; a "
            "product description before materialization has none."
        ),
    )


class ObsMeta(DataProdMetaBase):
    """Metadata for all raw observations sharing a master and observation number.

    Attributes
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``.
    obsnum : int
        Observation number, shared by the associated raw observations.
    obs_datetime : datetime.datetime or None
        Earliest known start among the associated raw observations. Adding an
        acquisition or enriching its start can move this value earlier.
    data_kind : int
        Union of acquisition-mode flags from the associated raw observations.
        Zero denotes no known kinds.
    tag : {"obs"}
        Discriminator for union type deserialization.
    """

    master: str
    obsnum: SeqNumT
    tag: Literal["obs"] = "obs"
    data_kind: int = 0


class RawObsMeta(DataProdMetaBase, ObsIdMixin, TelMetaMixin):
    """Metadata for dp_raw_obs — aggregates from all interface sources.

    One product describes a complete acquisition quartet across its interfaces.
    Supply master, obsnum, subobsnum and scannum explicitly; zero is a recorded
    value, not a replacement for an unknown component.

    Attributes
    ----------
    tag : Literal["raw_obs"]
        Discriminator for union type deserialization.
    data_kind : int
        Data acquisition mode flags (from DataKind enum).
    obs_datetime : datetime.datetime or None
        Earliest known interface start. Additional interface metadata can move
        this acquisition-level value earlier; file timestamps remain unchanged.
    """

    master: str
    obsnum: SeqNumT
    subobsnum: SeqNumT
    scannum: SeqNumT
    tag: Literal["raw_obs"] = "raw_obs"
    data_kind: int = 0


class RawObsProcMeta(DataProdMetaBase, ObsIdMixin, VersionedProductMeta):
    """Transforms and diagnostics for one raw acquisition."""

    tag: Literal["raw_obs_proc"] = "raw_obs_proc"


class ScienceMeta(DataProdMetaBase, VersionedProductMeta):
    """Science maps, optional catalogs and diagnostics for selected observations.

    Primary and contributor associations define the observation or collection
    scope. Exact acquisition and calibration selections belong to saved contexts.
    """

    tag: Literal["science"] = "science"


class PointingMeta(DataProdMetaBase, VersionedProductMeta):
    """Measured pointing solution with its supporting maps and diagnostics."""

    tag: Literal["pointing"] = "pointing"


class BeammapMeta(DataProdMetaBase, VersionedProductMeta):
    """Detector beam characterization, fitted properties and supporting maps."""

    tag: Literal["beammap"] = "beammap"


class AptMeta(DataProdMetaBase, VersionedProductMeta):
    """Detector rosters, array properties and acquired-channel projections."""

    tag: Literal["apt"]


class CalGroupMeta(DataProdMetaBase, ObsIdMixin):
    """VNA acquisition anchoring an intrinsic calibration group.

    The quartet identifies the opening VNA. Membership and current group
    resolution are available through ``get_calibration_group``.
    """

    tag: Literal["cal_group"] = "cal_group"


class DrivefitMeta(DataProdMetaBase, ObsIdMixin, VersionedProductMeta):
    """Metadata for dp_drivefit.

    Attributes
    ----------
    tag : Literal["drivefit"]
        Discriminator for union type deserialization.
    n_items : int
        Number of raw observations used.
    fit_method : str | None
        Fitting method/algorithm.
    convergence_status : str | None
        Fit convergence status.
    chi_squared : float | None
        Chi-squared goodness of fit.
    """

    tag: Literal["drivefit"] = "drivefit"
    n_items: int = 0
    fit_method: str | None = None
    convergence_status: str | None = None
    chi_squared: float | None = None


class QuicklookMeta(DataProdMetaBase, VersionedProductMeta):
    """Integrated feedback over explicitly selected scientific results."""

    tag: Literal["quicklook"]
    workflow: str
    n_inputs: int


class ToneMatchMeta(DataProdMetaBase, VersionedProductMeta):
    """Tone correspondences among selected block/result datasets."""

    tag: Literal["tone_match"]
    n_inputs: int


class SkydipMeta(DataProdMetaBase, VersionedProductMeta):
    """Curated loading-response diagnostics from exact tone comparisons."""

    tag: Literal["skydip"]
    n_inputs: int


class AttenGridMeta(DataProdMetaBase, VersionedProductMeta):
    """Readout attenuation measurements and candidate initialization presets."""

    tag: Literal["atten_grid"]
    n_inputs: int
    condition: str
    window: Literal["closed", "open", "unknown"]


class FocusGroupMeta(DataProdMetaBase, ObsIdMixin, VersionedProductMeta):
    """Metadata for dp_focus_group.

    Attributes
    ----------
    tag : Literal["focus_group"]
        Discriminator for union type deserialization.
    n_items : int
        Number of raw observations in group.
    focus_positions : list[float] | None
        Focus positions sampled.
    best_focus : float | None
        Optimal focus position.
    focus_metric : str | None
        Metric used (FWHM, Strehl, etc.).
    """

    tag: Literal["focus_group"] = "focus_group"
    n_items: int = 0
    focus_positions: list[float] | None = None
    best_focus: float | None = None
    focus_metric: str | None = None


class AstigGroupMeta(DataProdMetaBase, ObsIdMixin, VersionedProductMeta):
    """Metadata for dp_astig_group.

    Attributes
    ----------
    tag : Literal["astig_group"]
        Discriminator for union type deserialization.
    n_items : int
        Number of raw observations in group.
    astig_positions : list[tuple[float, float]] | None
        Astigmatism correction positions sampled (x, y).
    best_astig : tuple[float, float] | None
        Optimal astigmatism correction (x, y).
    astig_metric : str | None
        Metric used (FWHM, Strehl, etc.).
    """

    tag: Literal["astig_group"] = "astig_group"
    n_items: int = 0
    astig_positions: list[tuple[float, float]] | None = None
    best_astig: tuple[float, float] | None = None
    astig_metric: str | None = None


class OofGroupMeta(DataProdMetaBase, ObsIdMixin, VersionedProductMeta):
    """Metadata for dp_oof_group (Out-of-Focus holography).

    Attributes
    ----------
    tag : Literal["oof_group"]
        Discriminator for union type deserialization.
    n_items : int
        Number of raw observations in group.
    oof_positions : list[float] | None
        Out-of-focus positions sampled.
    surface_rms : float | None
        Measured surface RMS from holography.
    oof_metric : str | None
        Metric used for analysis.
    """

    tag: Literal["oof_group"] = "oof_group"
    n_items: int = 0
    oof_positions: list[float] | None = None
    surface_rms: float | None = None
    oof_metric: str | None = None


class NamedGroupMeta(DataProdMetaBase):
    """Logical collection with stable identity and refreshable membership.

    Attributes
    ----------
    tag : Literal["named_group"]
        Discriminator for union type deserialization.
    group_name : str
        User-defined group name.
    n_items : int
        Number of products in group.
    tags : list[str] | None
        Optional tags for categorization.
    owner : str | None
        Group owner/creator.
    notes : str | None
        Free-form notes.
    context : dict
        Producer-owned project, selection or import context. Membership is held
        by associations, independently of this descriptive metadata.
    """

    tag: Literal["named_group"] = "named_group"
    group_name: str = ""
    n_items: int = 0
    tags: list[str] | None = None
    owner: str | None = None
    notes: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class VersionGroupMeta(DataProdMetaBase):
    """Versions of one logical product sharing a declared product type.

    Parameters
    ----------
    content_data_prod_type : str
        Registered product type required for every version in the group.
    """

    content_data_prod_type: str
    tag: Literal["version_group"] = "version_group"


# Union type for DataProd.meta (polymorphic, Literal discriminators)
AnyDataProdMeta = (
    VersionGroupMeta
    | ObsMeta
    | RawObsMeta
    | RawObsProcMeta
    | ScienceMeta
    | PointingMeta
    | BeammapMeta
    | AptMeta
    | CalGroupMeta
    | DrivefitMeta
    | ToneMatchMeta
    | SkydipMeta
    | AttenGridMeta
    | QuicklookMeta
    | FocusGroupMeta
    | AstigGroupMeta
    | OofGroupMeta
    | NamedGroupMeta
)


# ====================  Interface-Level Metadata  ====================


@dataclass
class InterfaceFileMeta:
    """Simple interface file metadata (nw_id and roach only).

    Lightweight struct for use in dagster asset metadata creation.
    Product items can instead carry the fuller AnyInterfaceMeta records.

    Attributes
    ----------
    nw_id : int | None
        Network/roach ID.
    roach : int | None
        Roach board index (0-12).
    """

    nw_id: int | None = None
    roach: int | None = None


class RoachInterfaceMeta(ObsIdMixin, RoachMetaMixin):
    """Partial ROACH metadata supplied by acquisition-database records.

    Available files supply the complete ``ToltecRawObsMetadata`` instead.

    Attributes
    ----------
    type : Literal["roach"]
        Type discriminator for adaptix union handling.
    data_kind : int | None
        DataKind flag value for this interface file.
    cal_master, cal_obsnum, cal_subobsnum, cal_scannum : optional
        Calibration acquisition referenced by this interface. Unavailable
        components remain None; the acquisition master is not inferred from
        the source acquisition's master.
    """

    obs_type: str | None = None
    obs_type_code: int | None = None
    repeat: int | None = None
    type: Literal["roach"] = "roach"
    data_kind: int | None = None
    readout_code: int | None = None
    cal_master: str | None = None
    cal_obsnum: int | None = None
    cal_subobsnum: int | None = None
    cal_scannum: int | None = None


class HwprInterfaceMeta(ObsIdMixin):
    """Half-wave plate source metadata.

    Attributes
    ----------
    interface : str
        Half-wave plate readout interface.
    readout_code : int or None
        Recorded readout identifier, when headers were read. Not a ROACH index.
    data_kind : int
        Persisted data-kind flag.
    """

    type: Literal["hwpr"] = "hwpr"
    interface: Literal["hwpr"] = "hwpr"
    readout_code: int | None = None
    data_kind: int = DataKind.Hwpr.value


class TelInterfaceMeta(ObsIdMixin, TelMetaMixin):
    """LMT telescope source metadata.

    Stored in DataProdItem.meta for tel interface files.

    Attributes
    ----------
    type : str
        Telescope source discriminator, always ``tel``.
    interface : str
        Telescope writer prefix: ``tel_toltec`` or ``tel_toltec2``.
    data_kind : int
        DataKind flag value for this source.
    receiver : str
        Receiver name.
    instrument : str
        Instrument name.
    main_time : float
        Main beam time (seconds).
    ref_time : float
        Reference time (seconds).
    valid : bool
        Valid data flag.
    """

    type: Literal["tel"] = "tel"
    data_kind: int = DataKind.LmtTel.value
    interface: Literal["tel_toltec", "tel_toltec2"] = "tel_toltec"
    receiver: str = "Toltec"
    instrument: str = "tel"
    main_time: float = 0.0
    ref_time: float = 0.0
    valid: bool = True


# Metadata carried by product items, independently of their storage references.
AnyInterfaceMeta = TelInterfaceMeta | RoachInterfaceMeta | HwprInterfaceMeta


class _ItemMetaModel(_MetadataModel):
    """Item metadata whose unset optional fields are omitted when serialized."""

    @model_serializer(mode="wrap")
    def _omit_none(self, handler):
        return {k: v for k, v in handler(self).items() if v is not None}


class ArtifactMeta(_ItemMetaModel):
    """File written into a product by a producing run.

    Attributes
    ----------
    type : {"artifact"}
        Item kind.
    format : str
        File format, such as ``ecsv``, ``zarr`` or ``png``.
    name : str
        Name that selects the file within its product.
    producer : str
        Writer of the file: the qualified key of a pipeline step
        (``module:Class`` or ``module:Class:alias``), or the name of an external
        program, such as ``citlali``.
    run_uid : uuid.UUID
        Run that wrote the file.
    interface, scope, kind, array : str or None
        Selection keys: the readout interface, the scope of the content, a
        producer-defined category and the detector array. None when the file
        has no such selection.
    raw_obs_uid : str or None
        Acquisition for an acquisition-specific artifact, such as a matched APT.
        None leaves applicability unscoped.
    source_file, fit_file : str or None
        Source data and saved fit record used by a figure, relative to its
        product directory. None when the artifact declares no such reference.
    context : dict or None
        Producer-specific values that are not selection keys.
    """

    type: Literal["artifact"]
    format: str
    name: str
    producer: str
    run_uid: UUID
    interface: str | None = None
    scope: str | None = None
    kind: str | None = None
    array: str | None = None
    raw_obs_uid: str | None = None
    source_file: str | None = None
    fit_file: str | None = None
    context: dict[str, Any] | None = None


class IndexItemMeta(_ItemMetaModel):
    """A product's own index file in a namespace.

    Attributes
    ----------
    type : {"data_prod_index"}
        Item kind.
    namespace_uid : uuid.UUID
        Namespace whose store wrote the index file.
    run_uid : uuid.UUID or None
        Run of a realization. None for logical descriptors and version groups.
    """

    type: Literal["data_prod_index"]
    namespace_uid: UUID
    run_uid: UUID | None = None


AnyDataProdItemMeta = (
    ToltecRawObsMetadata
    | ToltecHwprMetadata
    | LmtTelMetadata
    | AnyInterfaceMeta
    | ArtifactMeta
    | IndexItemMeta
)


# Database JSON and portable documents use the models' field contracts.
_validate_product_metadata_json = TypeAdapter(
    Annotated[AnyDataProdMeta, Field(discriminator="tag")]
).validate_json
_validate_interface_metadata_json = TypeAdapter(
    Annotated[AnyInterfaceMeta, Field(discriminator="type")]
).validate_json
_validate_item_metadata_json = TypeAdapter(
    Annotated[
        AnyInterfaceMeta | ArtifactMeta | IndexItemMeta, Field(discriminator="type")
    ]
).validate_json

_retort = Retort(
    recipe=[
        loader(
            AnyDataProdMeta,
            lambda value: _validate_product_metadata_json(json.dumps(value)),
        ),
        loader(
            AnyInterfaceMeta,
            lambda value: _validate_interface_metadata_json(json.dumps(value)),
        ),
        *[
            provider
            for model in (
                *get_args(AnyDataProdMeta),
                *get_args(AnyInterfaceMeta),
                ArtifactMeta,
                IndexItemMeta,
                ToltecRawObsMetadata,
                ToltecHwprMetadata,
                LmtTelMetadata,
                RawObsFileCalObsFileMeta,
            )
            for provider in (
                loader(
                    model,
                    lambda value, model=model: model.model_validate_json(
                        json.dumps(value)
                    ),
                ),
                dumper(
                    model, lambda value: value.model_dump(mode="json", round_trip=True)
                ),
            )
        ],
    ]
)


def _retain_calibration_master(
    meta: AnyDataProdItemMeta, previous: AnyDataProdItemMeta
) -> AnyDataProdItemMeta:
    """Retain a known calibration master for an unchanged interface reference."""
    # Acquisition-defect workaround: file headers omit the calibration master.
    # Reopening files or importing a partial record must not erase the value
    # resolved from ToltecDB. Never reuse it for a different calibration triple.
    # TODO(upstream acquisition): revisit when headers carry this field.
    kinds = (ToltecRawObsMetadata, RoachInterfaceMeta)
    if (
        isinstance(meta, kinds)
        and isinstance(previous, kinds)
        and meta.cal_master is None
        and previous.cal_master is not None
        and all(
            getattr(meta, name) == getattr(previous, name)
            for name in ("cal_obsnum", "cal_subobsnum", "cal_scannum")
        )
    ):
        return meta.model_copy(update={"cal_master": previous.cal_master})
    return meta


def _load_source_metadata(value):
    """Restore file headers through their model's field contracts."""
    if "interface_kind" in value:
        model = {
            "roach": ToltecRawObsMetadata,
            "hwpr": ToltecHwprMetadata,
            "tel": LmtTelMetadata,
        }[value["interface_kind"]]
        return model.model_validate_json(json.dumps(value))
    return _validate_item_metadata_json(json.dumps(value))


_retort = _retort.extend(recipe=[loader(AnyDataProdItemMeta, _load_source_metadata)])


# ====================  Process Context  ====================


@dataclass
class ProcessContext:
    """Process metadata for DataProdAssoc.context field.

    Attributes
    ----------
    module : str or None
        Processing module name.
    version : str or None
        Processing version.
    config : dict or None
        Processing configuration.
    """

    tag: Literal["process"] = "process"
    module: str | None = None
    version: str | None = None
    config: dict[str, Any] | None = None


@dataclass(kw_only=True)
class RawObsCalObsContext:
    """Interfaces whose recorded calibration reference establishes this edge.

    Attributes
    ----------
    interfaces : list of str
        Source interface names, sorted and without duplicates.
    tag : {"raw_obs_cal_obs"}
        Discriminator for association context deserialization.
    """

    interfaces: list[str]
    tag: Literal["raw_obs_cal_obs"] = "raw_obs_cal_obs"


@dataclass(kw_only=True)
class ReductionObsContext:
    """Roles of one observation in an exact reduction result.

    Attributes
    ----------
    roles : list of str
        Contribution roles, such as science, pointing or photometry. An
        observation can supply both measurements and selected calibrations.
    """

    roles: Annotated[list[str], Field(min_length=1)]
    tag: Literal["reduction_obs"] = "reduction_obs"


@dataclass(kw_only=True)
class AptSourceContext:
    """Roles of an exact input product in a detector roster.

    Attributes
    ----------
    sources : list of str
        Aliases used by the roster's observation and association evidence.
    design_ref : bool
        Whether this product supplies the roster's design identity reference.
    """

    sources: list[str]
    design_ref: bool = False
    tag: Literal["apt_source"] = "apt_source"


AnyAssocContext = (
    RawObsCalObsContext | ProcessContext | ReductionObsContext | AptSourceContext
)


__all__ = [
    "AnyAssocContext",
    "AnyDataProdItemMeta",
    "AnyDataProdMeta",
    "AnyInterfaceMeta",
    "AptMeta",
    "AptSourceContext",
    "ArtifactMeta",
    "AstigGroupMeta",
    "AttenGridMeta",
    "BeammapMeta",
    "CalGroupMeta",
    "DataProdMetaBase",
    "DrivefitMeta",
    "FocusGroupMeta",
    "HwprInterfaceMeta",
    "IndexItemMeta",
    "InterfaceFileMeta",
    "MetadataType",
    "NamedGroupMeta",
    "ObsIdMixin",
    "ObsMeta",
    "OofGroupMeta",
    "PointingMeta",
    "ProcessContext",
    "ProductMetadata",
    "QuicklookMeta",
    "RawObsCalObsContext",
    "RawObsMeta",
    "RawObsProcMeta",
    "ReductionObsContext",
    "RoachInterfaceMeta",
    "RoachMetaMixin",
    "ScienceMeta",
    "SkydipMeta",
    "TelInterfaceMeta",
    "TelMetaMixin",
    "ToneMatchMeta",
    "VersionGroupMeta",
    "VersionedProductMeta",
    "_retort",
]

# Union type for all metadata (includes interface metadata and dict fallback)
MetadataType = AnyDataProdMeta | AnyDataProdItemMeta | dict[str, Any]


def _product_type(value):
    return (
        value.get("data_prod_type")
        if isinstance(value, dict)
        else getattr(value, "data_prod_type", None)
    )


ProductMetadata = Annotated[
    Annotated[VersionGroupMeta, Tag("dp_version_group")]
    | Annotated[ObsMeta, Tag("dp_obs")]
    | Annotated[RawObsMeta, Tag("dp_raw_obs")]
    | Annotated[RawObsProcMeta, Tag("dp_raw_obs_proc")]
    | Annotated[ScienceMeta, Tag("dp_science")]
    | Annotated[PointingMeta, Tag("dp_pointing")]
    | Annotated[BeammapMeta, Tag("dp_beammap")]
    | Annotated[AptMeta, Tag("dp_apt")]
    | Annotated[CalGroupMeta, Tag("dp_cal_group")]
    | Annotated[DrivefitMeta, Tag("dp_drivefit")]
    | Annotated[ToneMatchMeta, Tag("dp_tone_match")]
    | Annotated[SkydipMeta, Tag("dp_skydip")]
    | Annotated[AttenGridMeta, Tag("dp_atten_grid")]
    | Annotated[QuicklookMeta, Tag("dp_quicklook")]
    | Annotated[FocusGroupMeta, Tag("dp_focus_group")]
    | Annotated[AstigGroupMeta, Tag("dp_astig_group")]
    | Annotated[OofGroupMeta, Tag("dp_oof_group")]
    | Annotated[NamedGroupMeta, Tag("dp_named_group")],
    Discriminator(_product_type),
]
"""Typed metadata selected by the product's ``data_prod_type``."""
