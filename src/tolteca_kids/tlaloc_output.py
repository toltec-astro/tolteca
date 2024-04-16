from dataclasses import dataclass
from typing import Annotated, Literal

import astropy.units as u
import numpy as np
from astropy.table import QTable
from pydantic import BaseModel, ConfigDict, Field
from tollan.config.types import AbsDirectoryPath, AbsFilePath, FrequencyQuantityField
from tollan.utils.log import logger, timeit
from typing_extensions import assert_never

from tolteca_kidsproc.kidsdata.sweep import MultiSweep

from .kids_find import KidsFind
from .output import OutputConfig, OutputMixin
from .pipeline import Step, StepContext
from .roach_tone import RoachToneProps, TlalocEtcDataStore


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

    def __call__(self, n_chans):
        if self.optimize_waveform:
            # TODO: integrate in the tone power optimizations.
            raise NotImplementedError
        return RoachToneProps.make_random_phases(n_chans, seed=self.seed)


class TonePhasesFromFile(_TonePhasesBase):
    """Tone phases from file."""

    method: Literal["file"] = Field(
        description="Use phases from file.",
    )
    path: str | AbsFilePath = Field(
        description="The fileapth.",
    )

    def __call__(self, n_chans):
        tbl = QTable.read(self.path, format="ascii.no_header", names=["phase_tone"])
        phases = tbl["phase_tone"]
        if len(phases) < n_chans:
            raise ValueError("two few phases in file.")
        return phases


TonePhasesType = Annotated[
    TonePhasesRandom | TonePhasesFromFile,
    Field(discriminator="method"),
]


class PlaceHolderTones(BaseModel):
    """Helper to generate placeholder tones."""

    n: int = Field(
        default=0,
        description="number of placehold tones.",
        ge=0,
    )
    loc: Literal["inner", "outer"] = "outer"
    sep_f_lo_min: FrequencyQuantityField = Field(
        default=5 << u.MHz,
        description="Minimum separation from the LO frequency.",
    )
    sep_f_lo_max: FrequencyQuantityField = Field(
        default=500 << u.MHz,
        description="Maximum separation from the LO frequency.",
    )
    sep_f_det: FrequencyQuantityField = Field(
        default=100 << u.kHz,
        description="separation from the closest detector.",
    )
    f_step: FrequencyQuantityField = Field(
        default=50 << u.kHz,
        description="The step size of place holder tones.",
    )

    def __call__(self, f_det, f_lo):
        """Return a list of place holder tone frequencies."""
        if self.n == 0:
            return None
        f_det_neg = f_det[f_det < f_lo]
        f_det_pos = f_det[f_det >= f_lo]

        f_det_neg_min = np.min(f_det_neg)
        f_det_neg_max = np.max(f_det_neg)

        f_det_pos_min = np.min(f_det_pos)
        f_det_pos_max = np.max(f_det_pos)

        f_neg_min = f_lo - self.sep_f_lo_max
        f_neg_max = f_lo - self.sep_f_lo_min

        f_pos_min = f_lo + self.sep_f_lo_min
        f_pos_max = f_lo + self.sep_f_lo_max

        n_neg = self.n // 2
        n_pos = self.n - n_neg
        f_unit = u.Hz

        if self.loc == "inner":
            neg0 = f_det_neg_max + self.sep_f_det
            neg1 = min(neg0 + (n_neg - 1) * self.f_step, f_neg_max)
            pos1 = f_det_pos_min - self.sep_f_det
            pos0 = max(pos1 - (n_pos - 1) * self.f_step, f_pos_min)
        elif self.loc == "outer":
            neg1 = f_det_neg_min - self.sep_f_det
            neg0 = max(neg1 - (n_neg - 1) * self.f_step, f_neg_min)
            pos0 = f_det_pos_max + self.sep_f_det
            pos1 = min(pos0 + (n_pos - 1) * self.f_step, f_pos_max)
        else:
            assert_never()
        return (
            np.r_[
                np.linspace(neg0.to_value(f_unit), neg1.to_value(f_unit), n_neg),
                np.linspace(pos0.to_value(f_unit), pos1.to_value(f_unit), n_pos),
            ]
            << f_unit
        )


class TlalocOutputConfig(OutputConfig):
    """The tlaloc output config."""

    path: AbsDirectoryPath = Field(
        default=".",
        description="Tlaloc etc folder.",
    )
    n_chans_max: int = Field(
        default=1000,
        description="Maximum number of channels allowed.",
    )
    tone_phases: TonePhasesType = Field(
        default={
            "method": "random",
        },
        description="The tone phases config.",
    )
    placeholders: PlaceHolderTones = Field(
        default_factory=PlaceHolderTones,
        description="Config for generating placeholder tones.",
    )


@dataclass(kw_only=True)
class TlalocOutputData:
    """The data class for tlaloc output."""

    roach_tone_placeholders: QTable = ...
    roach_tones: QTable = ...


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
        cfg = context.config
        tlaloc_etc = TlalocEtcDataStore(cfg.path)
        logger.debug(f"{tlaloc_etc=}")
        # tlaloc_etc.write_tone_props_table(
        #     cls.make_roach_tone_props(
        #         data,
        #         tone_phases=cfg.tone_phases,
        #         placeholders=cfg.placeholders,
        #         n_chans_max=cfg.n_chans_max,
        #     ),
        # )
        return True

    @classmethod
    def make_roach_tone_props(
        cls,
        data: MultiSweep,
        tone_phases=None,
        placeholders=None,
        n_chans_max=TlalocOutputConfig.field_defaults["n_chans_max"],
    ):
        """Retrun roach tone props from kids find result."""
        ctx_kf = KidsFind.get_context(data)
        if not ctx_kf.completed:
            raise ValueError("kids find step has not run yet.")
        ctd_kf = ctx_kf.data
        swp = data

        roach = swp.meta["roach"]
        f_lo = swp.meta["f_lo_center"] << u.Hz
        mdl_grouped = ctd_kf.mdl_grouped
        n_tones = len(mdl_grouped)

        # information of previous placeholder tones
        n_chans0 = swp.n_chans
        if n_tones > n_chans0:
            raise ValueError(
                f"unalbe to allocate {n_tones=} in data with n_chans={n_chans0}.",
            )
        # make a sorted copy of the chan axis data
        # identify the placeholder idx for each half of the spectra.
        rtt0 = swp.meta["chan_axis_data"].copy()
        rtt0.sort("f_chan")
        mask_neg0 = rtt0["f_chan"] < f_lo
        mask_pos0 = rtt0["f_chan"] >= f_lo
        n_neg0 = mask_neg0.sum()
        n_pos0 = mask_pos0.sum()
        idx_placeholders_neg0 = np.nonzero((~rtt0["mask_tone"]) & mask_neg0)[0]
        idx_placeholders_pos0 = np.nonzero((~rtt0["mask_tone"]) & mask_pos0)[0]
        n_placeholders_neg0 = len(idx_placeholders_neg0)
        n_placeholders_pos0 = len(idx_placeholders_pos0)

        # #
        #
        # # now compute
        # mask_neg_out
        #
        #
        # np.nonzero(np.diff(rtt0["mask_tone"].astype(int))  == 1)
        # rtt = QTable(
        #     {
        #         "f_cha"
        #     }
        # )
        # tbl = QTable(
        #     {
        #         "f_tone": mdl_grouped["f"],
        #         "Qr": mdl_grouped["Qr"],
        #         "amp_tone": np.ones((n_tones, ), dtype=float),
        #         "phase_tone": tone_phases(n_tones),
        #     },
        #     meta={
        #         "roach": roach,
        #         "f_lo": f_lo,
        #     },
        # )
        return QTable()
