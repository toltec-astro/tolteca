from dataclasses import dataclass
from typing import Annotated, Literal

import astropy.units as u
import numpy as np
from astropy.table import QTable, vstack
from pydantic import BaseModel, ConfigDict, Field
from tollan.config.types import AbsDirectoryPath, AbsFilePath, FrequencyQuantityField
from tollan.utils.log import logger, timeit
from tollan.utils.table import TableValidator
from typing_extensions import assert_never

from tolteca_kidsproc.kidsdata.sweep import MultiSweep

from .kids_find import KidsFind
from .pipeline import Step, StepConfig, StepContext
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
        """Return random phase."""
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
        """Return phases from file."""
        tbl = QTable.read(self.path, format="ascii.no_header", names=["phase_tone"])
        phases = tbl["phase_tone"]
        if len(phases) < n_chans:
            raise ValueError("too few phases in file.")
        return phases


TonePhasesType = Annotated[
    TonePhasesRandom | TonePhasesFromFile,
    Field(discriminator="method"),
]


class PlaceHolderTones(BaseModel):
    """Helper to generate placeholder tones."""

    n: int = Field(
        default=0,
        description="number of placeholder tones.",
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
        description="Separation from the closest detector.",
    )
    f_step: FrequencyQuantityField = Field(
        default=50 << u.kHz,
        description="The step size of place holder tones.",
    )

    def __call__(self, f_dets, f_lo):
        """Return a list of place holder tone frequencies."""
        f_unit = u.Hz
        if self.n == 0:
            return np.array([]) << f_unit
        # this splits the spectrum to the below LO and above LO sections.
        f_det_neg = f_dets[f_dets < f_lo]
        f_det_pos = f_dets[f_dets >= f_lo]

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


ToneAmpsMethod = Literal["match", "interp", "ones"]


class TlalocOutputConfig(StepConfig):
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
    fix_n_chans: bool = Field(
        default=False,
        description="If True, the number of channels is the same as input.",
    )
    tone_amps_method: ToneAmpsMethod = Field(
        default="match",
        description="Method used to generate tone amplitudes.",
    )


@dataclass(kw_only=True)
class TlalocOutputData:
    """The data class for tlaloc output."""

    roach_tone_props: RoachToneProps = ...


class TlalocOutputContext(StepContext["TlalocOutput", TlalocOutputConfig]):
    """The context class for tlaloc output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: TlalocOutputData = Field(default_factory=TlalocOutputData)


class TlalocOutput(Step[TlalocOutputConfig, TlalocOutputContext]):
    """TlalocOutput.

    This step write files to tlaloc folder.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):
        """Run tlaloc output."""
        cfg = context.config
        ctd = context.data
        tlaloc_etc = TlalocEtcDataStore(cfg.path)
        logger.debug(f"{tlaloc_etc=}")
        rtp = ctd.roach_tone_props = cls.make_roach_tone_props(
            swp=data,
            tone_phases=cfg.tone_phases,
            placeholders=cfg.placeholders,
            n_chans_max=cfg.n_chans_max,
            fix_n_chans=cfg.fix_n_chans,
            tone_amps_method=cfg.tone_amps_method,
        )
        tlaloc_etc.write_tone_props(rtp, sort_fft=True)
        return True

    @classmethod
    def make_roach_tone_props(  # noqa: C901, PLR0912, PLR0915
        cls,
        swp: MultiSweep = None,
        detected: QTable = None,
        tone_phases=None,
        placeholders=None,
        n_chans_max=TlalocOutputConfig.field_defaults["n_chans_max"],
        fix_n_chans=TlalocOutputConfig.field_defaults["fix_n_chans"],
        tone_amps_method: ToneAmpsMethod = "match",
    ):
        """Retrun roach tone props from kids find result."""
        if sum([swp is None, detected is None]) != 1:
            raise ValueError("require one of swp or detected.")
        tblv = TableValidator()
        if swp is not None:
            ctx_kf = KidsFind.get_context(swp)
            if not ctx_kf.completed:
                raise ValueError("kids find step has not run yet.")
            detected = ctx_kf.data.detected
            tbl_chan_prop = ctx_kf.data.chan_matched
            roach = swp.meta["roach"]
            f_lo = swp.meta["f_lo_center"] << u.Hz
        elif detected is not None:
            if not tblv.has_any_col(detected, ["f", "fr"]):
                raise ValueError("no frequency column found in table.")
            roach = tblv.get_first_meta(detected.meta, ["roach", "nw"])
            if roach is None:
                raise ValueError("no roach id found in meta.")
            f_lo = tblv.get_first_meta(detected.meta, ["f_lo_center", "f_lo"]) << u.Hz
            if f_lo is None:
                raise ValueError("no f_lo found in meta.")
        else:
            assert_never()
        # make a copy so that we can add new columns to it
        tbl_dets = detected.copy()
        n_tones = len(tbl_dets)
        f_dets = tbl_dets["f_chan"] = tblv.get_first_col_data(tbl_dets, ["f", "fr"])
        tbl_dets["mask_tone"] = True
        # TODO: implement LUT interp
        if tone_amps_method == "match":
            amps = tblv.get_first_col_data(tbl_dets, ["amp_tone"])
            if amps is None:
                raise ValueError("no tone amps found in data.")
        elif tone_amps_method == "ones":
            amps = np.ones((n_tones,), dtype=float)
        elif tone_amps_method == "interp":
            raise NotImplementedError
        tbl_dets["amp_tone"] = amps

        # check if we have enough channel to hold the data
        def _trim_tones(tbl, n_keep, key_indices):
            n = len(tbl)
            keep_mask = np.zeros((n,), dtype=bool)
            keep_mask[key_indices[:n_keep]] = True
            tbl_keep = tbl[keep_mask]
            tbl_discard = tbl[~keep_mask]
            logger.debug(f"trimmed tones:\n{tbl_discard}")
            return tbl_keep

        n_chans0 = swp.n_chans
        if fix_n_chans and n_tones > n_chans0:
            # raise ValueError(
            #     f"unalbe to allocate {n_tones=} in data with n_chans={n_chans0}.",
            # )
            # trim the table to remove some tones that are blended.
            logger.info(
                f"unalbe to allocate {n_tones=} in data with n_chans={n_chans0}, "
                f"trim {n_tones - n_chans0} tones.",
            )
            tbl_dets = _trim_tones(
                tbl_dets,
                n_chans0,
                key_indices=np.abs(tbl_dets["d_phi"]).argsort(),
            )
        elif n_tones > n_chans_max:
            logger.info(
                f"number of tones {n_tones=} exceed {n_chans_max=}, "
                f"trim {n_tones - n_chans_max} tones.",
            )
            tbl_dets = _trim_tones(
                tbl_dets,
                n_chans_max,
                key_indices=np.flip(tbl_dets["Qr"].argsort()),
            )

        # update n_tones
        n_tones = len(tbl_dets)

        # call placeholder function
        if placeholders is not None:
            placeholders = placeholders(
                f_dets=f_dets,
                f_lo=f_lo,
            )
        if fix_n_chans and n_tones < n_chans0:
            n_chans = n_chans0
            # re-use existing placeholders
            rtt0 = swp.meta["chan_axis_data"]
            mask_neg0 = rtt0["f_chan"] < f_lo
            mask_pos0 = rtt0["f_chan"] >= f_lo
            # n_neg0 = mask_neg0.sum()
            # n_pos0 = mask_pos0.sum()
            idx_plh_neg0 = np.nonzero((~rtt0["mask_tone"]) & mask_neg0)[0]
            idx_plh_pos0 = np.nonzero((~rtt0["mask_tone"]) & mask_pos0)[0]
            n_plh_neg0 = len(idx_plh_neg0)
            n_plh_pos0 = len(idx_plh_pos0)
            n_plh0 = n_plh_neg0 + n_plh_pos0
            logger.debug(f"input n_chans={n_chans0} n_placeholders={n_plh0}")
            logger.debug(
                f"allocate {n_tones=} to {n_chans} "
                f"chans with {n_plh0} placeholders",
            )
            # get missed chans from chan matched
            n_missing = n_chans - n_tones
            missed_mask = np.zeros((n_chans,), dtype=bool)
            missed_mask[np.abs(tbl_chan_prop["d_phi"]).argsort()[-n_missing:]] = True
            tbl_chan_missing = rtt0[missed_mask][["f_chan", "mask_tone", "amp_tone"]]
            # these columns are neccessary to avoid missing value in the stacked table.
            tbl_chan_missing["f"] = tbl_chan_missing["f_chan"]
            tbl_chan_missing["Qr"] = 0.0
            tbl_chan_missing["bitmask"] = 0
            # tbl_chan_missing["mask_tone"] = False
            tbl_roach_tone = vstack([tbl_dets, tbl_chan_missing])
            assert n_chans == len(tbl_roach_tone)
        else:
            n_placeholders = len(placeholders)
            n_chans = n_tones + n_placeholders
            if n_chans > n_chans_max:
                n_chans = n_chans_max
                n_placeholders = n_chans - n_tones
            logger.debug(
                f"create {n_chans=} whth {n_tones=} {n_placeholders=}",
            )
            logger.debug(f"{placeholders=}")
            tbl_plhs = QTable(
                {
                    "f_chan": placeholders[:n_placeholders],
                    "mask_tone": np.zeros((n_placeholders,), dtype=bool),
                    "amp_tone": np.zeros((n_placeholders,), dtype=float),
                },
            )
            tbl_roach_tone = vstack([tbl_dets, tbl_plhs])
        tbl_roach_tone.meta.update(
            {
                "Header.Toltec.ObsNum": swp.meta["obsnum"],
                "Header.Toltec.SubObsNum": swp.meta["subobsnum"],
                "Header.Toltec.ScanNum": swp.meta["scannum"],
                "roach": roach,
                "f_lo_center": f_lo,
            },
        )
        tbl_roach_tone["f_tone"] = tbl_roach_tone["f_chan"] - f_lo
        tbl_roach_tone["phase_tone"] = tone_phases(n_chans)

        tbl_roach_tone.sort("f_chan")
        rtp = RoachToneProps(
            table=tbl_roach_tone,
            f_lo_key="f_lo_center",
        )
        logger.debug(f"roach tone props: {rtp}")
        return rtp
