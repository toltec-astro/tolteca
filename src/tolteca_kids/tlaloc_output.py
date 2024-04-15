from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from tollan.config.types import AbsDirectoryPath, AbsFilePath
from tollan.utils.log import timeit

from tolteca_kidsproc.kidsdata.sweep import MultiSweep

from .kids_find import KidsFind
from .output import OutputConfig, OutputMixin
from .pipeline import Step, StepContext


class _TonePhasesBase(BaseModel):
    use_cached: bool = Field(
        default=True,
        description="Use cached tone phases file.",
    )
    cache_path: None | AbsDirectoryPath = Field(
        default=None,
        description="Cache the generated phase file.",
    )


class TonePhasesRandom(_TonePhasesBase):
    """Random tone phases."""

    method: Literal["random"] = Field(
        description="Use random phases",
    )
    seed: None | int = Field(
        default=None,
        description="Random seed.",
    )
    optimize_waveform: bool = Field(
        default=False,
        description="Optimize by generating an array of waveforms.",
    )
    n_waveforms: int = Field(
        default=None,
        description="Number of waveforms to generate for optimization.",
    )


class TonePhasesFromFile(_TonePhasesBase):
    """Tone phases from file."""

    method: Literal["file"] = Field(
        description="Use phases from file.",
    )
    path: str | AbsFilePath = Field(
        description="The fileapth.",
    )


TonePhasesType = Annotated[
    TonePhasesRandom | TonePhasesFromFile,
    Field(discriminator="method"),
]


class TlalocOutputConfig(OutputConfig):
    """The tlaloc output config."""

    path: AbsDirectoryPath = Field(
        default=".",
        description="Tlaloc etc folder.",
    )
    n_tones_max: int = Field(
        default=1000,
        description="Maximum number of tones allowed.",
    )
    n_placeholders: int = Field(default=20, description="Number of placeholder tones.")
    tone_phases: TonePhasesType = Field(
        default={
            "method": "random",
        },
        description="The tone phases config.",
    )


@dataclass(kw_only=True)
class TlalocOutputData:
    """The data class for tlaloc output."""


class TlalocOutputContext(StepContext["TlalocOutput", TlalocOutputConfig]):
    """The context class for tlaloc output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: TlalocOutputData = Field(default_factory=TlalocOutputData)


class TlalocOutput(OutputMixin, Step[TlalocOutputConfig, TlalocOutputContext]):
    """TlalocOutput.

    This step write files to tlaloc folder.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):
        """Run kids find plot."""
        # ctx_sc = SweepCheck.get_context(data)
        ctx_kf = KidsFind.get_context(data)
        if not ctx_kf.completed:
            raise ValueError("kids find step has not run yet.")
        swp = data
        # ctd_sc = ctx_sc.data
        ctd_kf = ctx_kf.data
        cfg_kf = ctx_kf.config
        # cfg1 = context.config
        ctd = context.data
        return True
