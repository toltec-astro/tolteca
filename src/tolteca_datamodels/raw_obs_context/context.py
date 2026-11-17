"""Selected raw blocks and their observation-wide or local calibrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping  # noqa: TC003 -- public annotations
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from ..core.types import FrozenDataModel
from ..data_prod import DataProd, DataProdRef
from ..data_prod.lineage import (
    collect_cal_refs,
    resolve_block_relation,
)
from .artifact import ArtifactBinding
from .calibration import CalItem, SweepCalDataItem, SweepCalItem
from .stub import StubApplication
from .types import RawObsRef

__all__ = ["RawObsContext", "RawObsDataItem"]


class RawObsDataItem(FrozenDataModel):
    """An acquired detector block or ancillary interface in a raw observation.

    Parameters
    ----------
    interface : str
        Nonempty logical interface name, for example ``toltec0`` or ``tel``.
    block : int or None, optional
        Acquired detector block index. None selects an ancillary whole interface,
        such as the telescope record; it cannot bind a detector sweep.
    representation : ArtifactBinding or None, optional
        Exact prepared representation of this item. Telescope variable corrections
        use context.stubs so the raw telescope remains the base.
        None retains the raw representation.
    cal_items : tuple of CalItem, optional
        Item-local calibrations, empty by default. At most one binding per role.
        Sweep identities are derived from recorded block history.

    Raises
    ------
    ValueError
        A local binding selects another interface or lacks a source block.
    """

    model_config = ConfigDict(extra="forbid")
    interface: str = Field(min_length=1)
    block: int | None = Field(default=None, ge=0)
    representation: ArtifactBinding | None = None
    cal_items: tuple[CalItem, ...] = ()

    @model_validator(mode="after")
    def _validate_cal_items(self) -> Self:
        sweeps = [cal for cal in self.cal_items if isinstance(cal, SweepCalItem)]
        if len(sweeps) > 1:
            msg = f"Conflicting sweep calibrations for {self.interface!r}"
            raise ValueError(msg)
        if len({cal.type for cal in self.cal_items}) != len(self.cal_items):
            msg_0 = "Conflicting item-local calibration roles"
            raise ValueError(msg_0)
        if sweeps and self.block is None:
            msg = "Sweep calibration requires an acquired source block"
            raise ValueError(msg)
        for cal in sweeps:
            if tuple(item.interface for item in cal.data_items) != (self.interface,):
                msg = "Item-local sweep calibration must select only its interface"
                raise ValueError(msg)
        return self


class RawObsContext(FrozenDataModel):
    """One raw acquisition with selected blocks and their calibration sweeps.

    Parameters
    ----------
    raw_obs : RawObsRef
        Acquisition being interpreted. Its product owns membership and locations.
    stubs : tuple of StubApplication, optional
        Explicit ordered acquisition-file patches. Empty means use no stubs,
        regardless of which files exist. Each selects an interface in data_items.
    data_items : tuple of RawObsDataItem
        Nonempty selection with distinct (interface, block) pairs. Multiple
        blocks on one detector interface can have different calibrations.
    cal_items : tuple of CalItem, optional
        Common calibrations, empty by default. Each common sweep applies to
        selected blocks on its interface. Other roles apply to the observation.
        Common/local bindings may not overlap. Telescope companions do not
        inherit detector sweep calibrations.

    Notes
    -----
    from_product derives exact sweep identities from shared block history.
    Direct construction and JSON loading check structure only; validate_lineage
    checks recorded source/target block relationships before consumption.
    Neither opens payloads or certifies tone/power compatibility, fitted-model
    normalization or availability. Inspection-only contexts may omit bindings.
    Serialize with model_dump_json and restore with model_validate_json.
    """

    model_config = ConfigDict(extra="forbid")
    raw_obs: RawObsRef
    data_items: tuple[RawObsDataItem, ...] = Field(min_length=1)
    cal_items: tuple[CalItem, ...] = ()
    stubs: tuple[StubApplication, ...] = ()

    @model_validator(mode="after")
    def _validate_stubs(self) -> Self:
        interfaces = {item.interface for item in self.data_items}
        if any(stub.interface not in interfaces for stub in self.stubs):
            msg = "Stub interface is not selected in this raw observation"
            raise ValueError(msg)
        return self

    @classmethod
    def from_product(
        cls,
        product: DataProd,
        *,
        data_items: tuple[RawObsDataItem, ...],
        cal_products: Mapping[str, DataProd] | None = None,
        metadata: Iterable[object] = (),
    ) -> Self:
        """Prepare exact sweep bindings from recorded block histories.

        Parameters
        ----------
        product : DataProd
            Source raw acquisition including portable calibration/block evidence.
        data_items : tuple of RawObsDataItem
            Ordered source selections without preassigned cal_items. Detector
            selections require a source block; ancillary interfaces use None.
        cal_products : mapping of str to DataProd or None, optional
            Referenced raw products keyed by UID, with their interface histories.
            None supplies no additional acquisitions. Missing evidence fails;
            no nearest/latest acquisition or fitted realization is selected.
        metadata : iterable of object, optional
            Additional source metadata accepted by the shared lineage collector.

        Returns
        -------
        RawObsContext
            Selected timestreams with derived item-local sweeps. Setup lineage
            belongs to sweep history and does not populate cal_items.

        Raises
        ------
        ValueError
            Selection or history is missing, conflicting, ambiguous or cyclic,
            or the caller supplied calibration choices instead of source blocks.

        Notes
        -----
        This operation performs no I/O. Supply recorded headers and inspected
        block-index metadata, or their portable history. Fit success is not
        required to establish identity. Payload and applicability checks remain
        separate. Sweep selections do not require loading setup ancestors; use
        resolve_block_relation to inspect that history independently.
        """
        metadata = tuple(metadata)
        references = collect_cal_refs(product, metadata=metadata)
        items = []
        for item in data_items:
            if item.cal_items:
                msg = "from_product derives cal_items; supply source selections only"
                raise ValueError(msg)
            cal = _binding(product, item, references, cal_products, metadata)
            items.append(
                item.model_copy(update={"cal_items": () if cal is None else (cal,)})
            )
        return cls(raw_obs=product.reference, data_items=tuple(items))

    def with_model(
        self, calibration: SweepCalItem, *, interface: str, block: int
    ) -> Self:
        """Attach a fitted realization to one selected timestream block.

        Parameters
        ----------
        calibration : SweepCalItem
            One network's recorded sweep binding with an exact model attached.
        interface : str
            Destination timestream network.
        block : int
            Destination acquired block index. The calibration binding supplies
            its own block index, which may have the same numeric value.

        Returns
        -------
        RawObsContext
            New context with item-local bindings. Common sweeps are expanded
            without changing their meaning; the original context is unchanged.

        Raises
        ------
        ValueError
            The destination is absent, has no sweep calibration, or the supplied
            binding changes its recorded acquisition/interface/block.

        Notes
        -----
        No payloads are opened. Model preparation checks the native report;
        prepare_run_inputs checks all exact indexes before saving run inputs.
        """
        if len(calibration.data_items) != 1 or calibration.data_items[0].model is None:
            msg = "Supply one sweep network with an exact fitted model"
            raise ValueError(msg)
        common = {
            item.interface: SweepCalItem(
                type="sweep", raw_obs=cal.raw_obs, data_items=(item,)
            )
            for cal in self.cal_items
            if isinstance(cal, SweepCalItem)
            for item in cal.data_items
        }
        items, found = [], False
        for item in self.data_items:
            other = tuple(
                cal for cal in item.cal_items if not isinstance(cal, SweepCalItem)
            )
            bound = tuple(
                cal for cal in item.cal_items if isinstance(cal, SweepCalItem)
            ) or ((common[item.interface],) if item.interface in common else ())
            if (item.interface, item.block) == (interface, block):
                found = True
                target = calibration.data_items[0]
                if (
                    not bound
                    or bound[0].raw_obs != calibration.raw_obs
                    or (target.interface, target.block)
                    != (
                        bound[0].data_items[0].interface,
                        bound[0].data_items[0].block,
                    )
                ):
                    msg = "Model binding changes the recorded sweep calibration"
                    raise ValueError(msg)
                bound = (calibration,)
            items.append(item.model_copy(update={"cal_items": bound + other}))
        if not found:
            msg = "Model destination is not selected in this raw-observation context"
            raise ValueError(msg)
        return type(self)(
            raw_obs=self.raw_obs,
            stubs=self.stubs,
            data_items=tuple(items),
            cal_items=tuple(
                cal for cal in self.cal_items if not isinstance(cal, SweepCalItem)
            ),
        )

    def validate_lineage(
        self,
        product: DataProd,
        *,
        cal_products: Mapping[str, DataProd] | None = None,
        metadata: Iterable[object] = (),
    ) -> None:
        """Check common and local bindings against exact recorded block history.

        Parameters
        ----------
        product : DataProd
            Owning raw acquisition with portable history evidence.
        cal_products : mapping of str to DataProd or None, optional
            Referenced raw acquisitions, as in from_product.
        metadata : iterable of object, optional
            Additional source metadata, as in from_product.

        Raises
        ------
        ValueError
            Source identity differs, history cannot resolve, a required binding
            is omitted, or its acquisition/interface/block differs from history.

        Notes
        -----
        No I/O or mutation occurs. A common binding must agree with every source
        block it covers; grouping does not override local history.
        """
        if product.reference != self.raw_obs:
            msg = "Lineage record does not match the context's raw acquisition"
            raise ValueError(msg)
        metadata = tuple(metadata)
        references = collect_cal_refs(product, metadata=metadata)
        common = {
            target.interface: (cal.raw_obs, target.block)
            for cal in self.cal_items
            if isinstance(cal, SweepCalItem)
            for target in cal.data_items
        }
        for item in self.data_items:
            expected = _binding(product, item, references, cal_products, metadata)
            actual = common.get(item.interface)
            sweeps = [cal for cal in item.cal_items if isinstance(cal, SweepCalItem)]
            if sweeps:
                [cal] = sweeps
                actual = cal.raw_obs, cal.data_items[0].block
            target = (
                None
                if expected is None
                else (expected.raw_obs, expected.data_items[0].block)
            )
            if actual != target:
                msg = (
                    f"Sweep binding differs from recorded block calibration for "
                    f"{item.interface!r}/{item.block}"
                )
                raise ValueError(msg)

    @model_validator(mode="after")
    def _validate_cal_items(self) -> Self:  # noqa: C901 -- scope and sweep checks
        keys = [(item.interface, item.block) for item in self.data_items]
        if len(set(keys)) != len(keys):
            msg = "Raw-observation context requires distinct source selections"
            raise ValueError(msg)
        whole = {item.interface for item in self.data_items if item.block is None}
        blocks = {item.interface for item in self.data_items if item.block is not None}
        if whole & blocks:
            msg = "Whole-interface and block selections cannot overlap"
            raise ValueError(msg)
        bound = {
            item.interface
            for item in self.data_items
            if any(isinstance(cal, SweepCalItem) for cal in item.cal_items)
        }
        roles = [
            cal.type for cal in self.cal_items if not isinstance(cal, SweepCalItem)
        ]
        if len(set(roles)) != len(roles):
            msg_0 = "Conflicting common calibration roles"
            raise ValueError(msg_0)
        for item in self.data_items:
            if set(roles) & {cal.type for cal in item.cal_items}:
                msg_0 = "Common and item-local calibration roles overlap"
                raise ValueError(msg_0)
        for cal in self.cal_items:
            if not isinstance(cal, SweepCalItem):
                continue
            for item in cal.data_items:
                if item.interface not in blocks:
                    msg = (
                        "Common sweep targets an unselected detector "
                        f"interface {item.interface!r}"
                    )
                    raise ValueError(msg)
                if item.interface in bound:
                    msg = f"Conflicting sweep calibrations for {item.interface!r}"
                    raise ValueError(msg)
                bound.add(item.interface)
        return self


def _binding(product, item, references, products, metadata):
    if item.block is None:
        if item.interface in references:
            msg = (
                f"Recorded detector interface {item.interface!r} "
                "requires a source block"
            )
            raise ValueError(msg)
        return None
    entry = references.get(item.interface)
    history = entry.meta.block_history if entry is not None else None
    if (
        history is not None
        and 0 <= item.block < len(history.blocks)
        and history.blocks[item.block].kind == "sweep"
    ):
        # Setup ancestry is queryable independently; it is not a sweep cal item.
        return None
    resolved = resolve_block_relation(
        product, item.interface, item.block, products=products, metadata=metadata
    )
    if resolved is None or resolved.kind != "calibration":
        return None
    return SweepCalItem(
        type="sweep",
        raw_obs=DataProdRef(uid=resolved.target_uid),
        data_items=(
            SweepCalDataItem(interface=item.interface, block=resolved.target_block),
        ),
    )
