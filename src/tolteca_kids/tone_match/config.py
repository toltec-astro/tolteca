"""Inputs, frequency matching and publication settings for tone lineage."""

from pydantic import Field
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsAnyPath

from tolteca_kids.config import DataProdOutputConfig
from tolteca_kids.match1d import FrequencyMatchConfig

from .viz import ToneLineageViz, ToneMatchViz


class ToneMatchInput(FrozenBaseModel):
    """One saved sweep block and its role in per-interface matching."""

    source: AbsAnyPath = Field(
        description="Published processed-product directory or index."
    )
    interface: str = Field(description="Readout interface in the saved product.")
    block: int = Field(default=0, ge=0, description="Saved sweep block number.")
    reference: bool = Field(
        default=False, description="Use as this interface's reference."
    )


class ToneMatchConfig(FrequencyMatchConfig):
    """Match saved tones independently for each interface and publish the results."""

    uid: str | None = Field(
        default=None,
        description=(
            "Logical comparison UID. Defaults to <primary acquisition>_tone_match."
        ),
    )
    inputs: list[ToneMatchInput] = Field(
        min_length=2,
        description=(
            "Pinned sweep selections. Distinct block/result selections; "
            "at most one explicit reference per "
            "interface; otherwise the earliest acquisition is used."
        ),
    )
    plot: ToneMatchViz | None = Field(
        default_factory=ToneMatchViz,
        description="Pair diagnostics and export settings; None disables plots.",
    )
    lineage_plot: ToneLineageViz | None = Field(
        default_factory=ToneLineageViz,
        description="Lineage diagnostics and export settings; None disables plots.",
    )
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)
