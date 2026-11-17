"""Per-interface calibration references for raw-observation products."""

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from ..core.data_prod import DataProdAudit
from ..core.types import FrozenDataModel
from ..lmt_base.funcs import make_uid_raw_obs

__all__ = [
    "DataProdCalRef",
    "RawObsBlockHistory",
    "RawObsBlockRelation",
    "RawObsFileCalObsFileMeta",
    "RawObsHistoryBlock",
]


class RawObsHistoryBlock(FrozenDataModel):
    """One acquired block in the recorded setup sequence.

    Parameters
    ----------
    kind : {'sweep', 'timestream'}
        Recorded block role. The position in RawObsBlockHistory.blocks is the
        zero-based acquired block index, not a preallocated header slot.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["sweep", "timestream"]


class RawObsBlockHistory(FrozenDataModel):
    """Ordered acquired blocks and the state at acquisition entry.

    Parameters
    ----------
    initial : {'recorded', 'root', 'unknown'}
        Start from the interface's recorded calibration acquisition, an
        established root, or unknown evidence. Root is not inferred from a
        missing header. A referenced acquisition contributes its final active
        sweep, even if timestream blocks follow it.
    blocks : tuple of RawObsHistoryBlock
        Nonempty acquired prefix in block-index order. Current Tune history follows
        the measured sweep order. Later snapshots may append acquired blocks.

    Notes
    -----
    This is portable acquisition evidence, not a policy selecting a fitted
    realization. Construction does not inspect files or certify detector state.
    """

    model_config = ConfigDict(extra="forbid")
    initial: Literal["recorded", "root", "unknown"]
    blocks: tuple[RawObsHistoryBlock, ...] = Field(min_length=1)

    def sweep_before(self, block: int | None = None) -> int | str:
        """Return the active local sweep index or unresolved boundary marker.

        Parameters
        ----------
        block : int or None, optional
            Acquired block whose incoming state is wanted. None (default)
            returns the final state after all recorded blocks.

        Returns
        -------
        int or str
            Earlier local sweep index, or 'recorded', 'root', or 'unknown'.
            Each sweep replaces the preceding sweep; timestreams preserve it.

        Raises
        ------
        ValueError
            The requested block is outside the acquired prefix.
        """
        if block is not None and not 0 <= block < len(self.blocks):
            msg = f"Block {block} is not acquired in this history"
            raise ValueError(msg)
        state: int | str = self.initial
        for index, entry in enumerate(self.blocks[:block]):
            if entry.kind == "sweep":
                state = index
        return state


class RawObsBlockRelation(FrozenDataModel):
    """A block's recorded relationship to an exact sweep on the same interface.

    Parameters
    ----------
    kind : {'calibration', 'setup_lineage'}
        A timestream has a calibration relation requiring matching detector
        settings. A sweep has setup lineage, which may include tone/power changes.
    target_uid : str
        Raw acquisition containing the preceding sweep.
    target_block : int
        Zero-based acquired sweep index within that acquisition.

    Notes
    -----
    Derived by resolve_block_relation from headers and acquired order. This
    records identity, not fit validity or verified physical applicability.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["calibration", "setup_lineage"]
    target_uid: str = Field(min_length=1)
    target_block: int = Field(ge=0)


class RawObsFileCalObsFileMeta(FrozenDataModel):
    """Known acquisition anchor and optional block history for one interface.

    Missing components remain None until acquisition metadata supplies them.
    A VNA uses its own quartet, regardless of older target-sweep header values.
    block_history preserves acquired block order when supplied;
    absent history does not imply a single-block acquisition.
    """

    model_config = ConfigDict(extra="forbid")
    cal_master: Annotated[str, Field(pattern="^[a-z][a-z0-9]*$")] | None = None
    cal_obsnum: Annotated[int, Field(ge=0)] | None = None
    cal_subobsnum: Annotated[int, Field(ge=0)] | None = None
    cal_scannum: Annotated[int, Field(ge=0)] | None = None
    block_history: RawObsBlockHistory | None = None

    @property
    def uid(self) -> str | None:
        """UID from the complete quartet, or None while any component is unknown."""
        values = (
            self.cal_master,
            self.cal_obsnum,
            self.cal_subobsnum,
            self.cal_scannum,
        )
        return (
            None
            if any(value is None for value in values)
            else make_uid_raw_obs(*values)
        )


class DataProdCalRef(FrozenDataModel):
    """Recorded calibration evidence for one interface of a raw acquisition.

    Parameters
    ----------
    interface : str
        Readout interface, such as ``toltec0``, within the owning DataProd.
    meta : RawObsFileCalObsFileMeta
        Recorded calibration acquisition and inspected block history. Missing
        identity components or block history remain explicitly unknown.
    target_uid : str or None, optional
        Resolved calibration acquisition UID; None until resolved, even when
        meta.uid is known. A supplied value must agree with the complete quartet
        in meta. Construction checks that agreement without looking up a target.
    audit : DataProdAudit or None, optional
        Attribution of the recorded evidence, when available.

    Notes
    -----
    Stored in DataProd.cal_refs, with at most one reference per interface.
    Use cal_ref_from_metadata
    or collect_cal_refs to derive these records from file metadata.
    RawObsContext uses them to determine sweep bindings; they do not select a
    fitted realization or an alternative calibration observation.
    """

    model_config = ConfigDict(extra="forbid")
    interface: str = Field(min_length=1)
    meta: RawObsFileCalObsFileMeta
    target_uid: str | None = Field(
        default=None, description="Resolved raw product; None while unresolved."
    )
    audit: DataProdAudit | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        if self.target_uid is not None and self.target_uid != self.meta.uid:
            msg = "Calibration target UID must match the complete reference quartet"
            raise ValueError(msg)
        return self
