from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Literal

import astropy.units as u
import dill
import numpy as np
import numpy.typing as npt
import plotly
import plotly.graph_objects as go
from astropy.table import QTable, hstack, unique, vstack
from pydantic import ConfigDict, Field
from scipy.ndimage import median_filter
from scipy.optimize import leastsq
from tollan.config.types import AbsAnyPath, FrequencyQuantityField, TimeQuantityField
from tollan.utils.fmt import pformat_mask
from tollan.utils.log import logger, timeit
from tollan.utils.np import attach_unit, make_complex, strip_unit
from typing_extensions import assert_never

from tolteca_kidsproc.kidsdata import MultiSweep

from .match1d import Match1D, Match1DResult
from .peaks1d import Peaks1D, Peaks1DResult
from .pipeline import Step, StepConfig, StepContext
from .plot import PlotConfig, PlotMixin
from .sweep_check import SweepCheck

__all__ = [
    "SegmentBitMask",
    "KidsFindConfig",
    "KidsFindData",
    "KidsFindContext",
    "KidsFind",
]


D21QuantityField = TimeQuantityField  # Hz^-1 = time


class SegmentBitMask(IntFlag):
    """A bit mask for segment."""

    doublet = auto()
    """The detection belongs to a group of 2."""

    triplet = auto()
    """The detection belongs to a group of 3."""

    manylet = auto()
    """The detection belongs to a group of 4 or beyond."""

    blended = doublet | triplet | manylet
    """The detection belongs to a group."""

    edge = auto()
    """The detection is on data edge."""

    # morphology
    peak_small = auto()
    """The detection peak is small."""

    snr_low = auto()
    """The detection peak has low SNR."""

    Qr_small = auto()
    """The detection Qr is small."""

    Qr_large = auto()
    """The detection Qr is large."""

    not_real = auto()
    """The detection is not real."""

    dark = auto()
    """The detection is a dark detector."""

    d21 = auto()
    """The detection is made in d21."""

    s21 = auto()
    """The detection is made in s21."""

    d21_collided = auto()
    """The d21 detection is merged with other d21."""

    s21_collided = auto()
    """The s21 detection is merged with two or more s21 detection."""


class KidsFindConfig(StepConfig):
    """The kids finding config."""

    model_config = ConfigDict(protected_namespaces=())

    ref_context_path: None | AbsAnyPath = Field(
        default=None,
        description="Reference kids find context to use as additional prior.",
    )

    Qr_min: float = Field(
        default=1000,
        description="Minimum Qr allowed for detection..",
    )

    Qr_dark_min: float = Field(
        default=20000,
        description="Minimum Qr for dark detectors.",
    )
    Qr_dark_max: float = Field(
        default=100000,
        description="Maximum Qr allowed for dark detection.",
    )

    d21_detect: Peaks1D = Field(
        default={
            "method": "peakdetect",
            "threshold": 0,
            "peakdetect_delta_threshold": 5,
        },
        description="Detection settings for D21.",
    )
    f_ref: FrequencyQuantityField = Field(
        default=450 << u.MHz,
        description="Reference frequency for frequency dependent thresholds.",
    )
    d21_peak_min: D21QuantityField = Field(
        default=0.1 << (u.Hz**-1),
        description="Minimum peak height for d21 detected kids at ref freq.",
    )
    d21_snr_min: float = Field(
        default=20.0,
        description="Minimum SNR for d21 detected kids at ref freq.",
    )
    d21_snr_dark_min: float = Field(
        default=100.0,
        description="Minimum SNR for dark kids at ref freq",
    )

    medfilt_size: int = Field(
        default=5,
        description=("Size of median filter used for S21 peak finding."),
    )

    detect: Peaks1D = Field(
        default={
            "method": "peakdetect",
            "threshold": 0,
            "peakdetect_delta_threshold": 5,
        },
        description="Detection settings for S21.",
    )
    peak_db_min: float = Field(
        default=0.2,
        description="Minimum peak height allowed for detection.",
    )
    snr_min: float = Field(
        default=10.0,
        description="Minimum SNR for detected kids.",
    )
    detect_sep_fwhm_min: float = Field(
        default=0.5,
        description=(
            "Segements with separation smaller than this are considered same detection."
        ),
    )
    model_sep_fwhm_min: float = Field(
        default=2,
        description=(
            "Segements with separation smaller than this are modeled as a group."
        ),
    )
    match: Match1D = Field(
        default={
            "method": "dtw_python",
        },
        description="Detection matching settings.",
    )
    match_ref: Literal["chan", "d21", "s21"] = Field(
        default="chan",
        description="The reference data to match.",
    )
    match_shift_max: FrequencyQuantityField = Field(
        default=10 << u.MHz,
        description="The maximum shift allowed in match.",
    )


@dataclass(kw_only=True)
class KidsFindData:
    """The data class for kids finding data."""

    refdata: "None | KidsFindData" = None
    bitmask: SegmentBitMask = ...
    bitmask_group: SegmentBitMask = ...

    d21_peaks: Peaks1DResult = ...
    d21_mask_not_real: npt.NDArray = ...
    d21_mask_dark: npt.NDArray = ...
    d21_mask_baseline: npt.NDArray = ...
    bitmask_d21: SegmentBitMask = ...

    mask_baseline: npt.NDArray = ...
    # chan_baseline_info: QTable = ...
    # s21_baseline: npt.NDArray = ...
    # as21_detrended: npt.NDArray = ...

    s21_peaks: Peaks1DResult = ...
    s21_mask_not_real: npt.NDArray = ...
    s21_mask_edge: npt.NDArray = ...
    bitmask_s21: SegmentBitMask = ...

    d21_detected: QTable = ...
    s21_detected: QTable = ...
    det_grouped: QTable = ...
    det_groups: QTable = ...
    bitmask_det: SegmentBitMask = ...

    mdl_grouped: QTable = ...
    mdl_groups: QTable = ...

    detected: QTable = ...
    matched: Match1DResult = ...
    detected_matched: QTable = ...
    chan_matched: QTable = ...

    matched_ref: Match1DResult = ...


class KidsFindContext(StepContext["KidsFind", KidsFindConfig]):
    """The context class for kids finding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: KidsFindData = Field(default_factory=KidsFindData)


class KidsFind(Step[KidsFindConfig, KidsFindContext]):
    """Kids finding step.

    This derives a list of data segments containing kids signature
    and the associated properties..
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):  # noqa: PLR0915, C901
        """Run kids find."""
        swp = data
        cfg = context.config
        ctd = context.data
        ctd_sc = SweepCheck.get_context(data).data

        # load ref context
        rpath = cfg.ref_context_path
        if rpath is not None:
            if rpath.is_dir():
                raise NotImplementedError
            ref_context = dill.load(cfg.ref_context_path)  # noqa: S301
        else:
            ref_context = None  # noqa: F841

        def _detect_postproc(r: Peaks1DResult):
            # this add Qr to the detected peak info table
            peaks = r.peaks
            peaks["Qr"] = (peaks["x"] / peaks["width"]).to_value(
                u.dimensionless_unscaled,
            )
            return r

        def _make_thresh_scale(f):
            return (f / cfg.f_ref).to_value(u.dimensionless_unscaled)

        # d21 detect

        d21_f = ctd_sc.d21_frequency
        d21_y = median_filter(
            ctd_sc.d21_detrended.to_value(u.Hz**-1),
            (cfg.medfilt_size,),
        ) << (u.Hz**-1)
        # d21_y = ctd_sc.d21_detrended
        d21_ey = ctd_sc.d21_baseline_rms
        # this helps eliminate negative peaks in d21 when the signal is small
        d21_y[d21_y < -d21_ey] = -d21_ey[d21_y < -d21_ey]
        d21_f_step = d21_f[1] - d21_f[0]

        def _calc_d21_fwhm():
            # use two tier Qr depends on the snr of data sample
            # this picks both dark and regular detectors
            thresh_scale = _make_thresh_scale(d21_f)
            snr = d21_y / d21_ey
            m_dark = snr > cfg.d21_snr_dark_min / thresh_scale
            Qr = np.full(d21_f.shape, cfg.Qr_dark_min)
            Qr[m_dark] = cfg.Qr_dark_max
            return d21_f / Qr / d21_f_step

        d21_peaks = ctd.d21_peaks = cfg.d21_detect(
            x=d21_f,
            y=d21_y,
            ey=d21_ey,
            fwhm=_calc_d21_fwhm(),
            postproc_hook=_detect_postproc,
        )
        d21_peak_info = d21_peaks.peaks
        if d21_peak_info is None:
            raise ValueError("no peaks found in d21 data.")

        # generate dark detector mask
        # this has to take into account the peak locations and scale
        # all the ref tagged thresholds from ref freq to the actual value
        d21_Qrs = d21_peak_info["Qr"]
        d21_snrs = d21_peak_info["snr"]
        d21_heights = d21_peak_info["height"]
        d21_thresh_scale = _make_thresh_scale(d21_peak_info["x"])

        d21_mask_peak_Qr_small = d21_peak_info["sbm_Qr_small"] = d21_Qrs < cfg.Qr_min
        d21_mask_peak_Qr_large = d21_peak_info["sbm_Qr_large"] = (
            d21_Qrs > cfg.Qr_dark_max
        )

        d21_mask_peak_peak_small = d21_peak_info["sbm_peak_small"] = (
            d21_heights < cfg.d21_peak_min / d21_thresh_scale
        )
        d21_mask_peak_snr_low = d21_peak_info["sbm_snr_low"] = (
            d21_snrs < cfg.d21_snr_min / d21_thresh_scale
        )
        d21_mask_peak_dark_snr_low = (d21_Qrs >= cfg.Qr_dark_min) & (
            d21_snrs < cfg.d21_snr_dark_min / d21_thresh_scale
        )
        d21_mask_peak_not_real = d21_peak_info["sbm_not_real"] = (
            d21_mask_peak_peak_small
            | d21_mask_peak_snr_low
            | d21_mask_peak_Qr_small
            | d21_mask_peak_Qr_large
            | d21_mask_peak_dark_snr_low
        )
        d21_mask_peak_dark = d21_peak_info["sbm_dark"] = (
            d21_Qrs >= cfg.Qr_dark_min
        ) & (~d21_mask_peak_not_real)
        bitmask_d21 = np.zeros((len(d21_peak_info),), dtype=int)
        bitmask_d21 |= (
            (d21_mask_peak_peak_small * SegmentBitMask.peak_small)
            | (d21_mask_peak_snr_low * SegmentBitMask.snr_low)
            | (d21_mask_peak_Qr_small * SegmentBitMask.Qr_small)
            | (d21_mask_peak_Qr_large * SegmentBitMask.Qr_large)
            | (d21_mask_peak_not_real * SegmentBitMask.not_real)
            | (d21_mask_peak_dark * SegmentBitMask.dark)
        )
        ctd.bitmask_d21 = bitmask_d21

        d21_mask_bad_n_fwhms = 3
        ctd.d21_mask_dark = d21_peaks.make_mask(
            d21_mask_peak_dark,
            n_fwhms=d21_mask_bad_n_fwhms,
        )
        ctd.d21_mask_not_real = d21_peaks.make_mask(
            d21_mask_peak_not_real,
            n_fwhms=d21_mask_bad_n_fwhms,
        )
        d21_mask_baseline = ctd.d21_mask_baseline = ~d21_peaks.make_mask(
            ~(d21_mask_peak_peak_small | d21_mask_peak_snr_low),
            n_fwhms=d21_mask_bad_n_fwhms,
        )
        d21_mask_peak_detected = (bitmask_d21 == 0) | (
            (bitmask_d21 & SegmentBitMask.dark) > 0
        )
        logger.debug(
            f"d21 detected peaks {pformat_mask(d21_mask_peak_detected)}",
        )
        d21_detected = ctd.d21_detected = d21_peak_info[d21_mask_peak_detected]

        # now work in s21
        # use d21 good peaks as prior
        d21_mask_peak_good = bitmask_d21 == 0
        logger.debug(
            f"with prior from unflagged d21 peaks {pformat_mask(d21_mask_peak_good)}",
        )
        d21_prior = d21_peak_info[d21_mask_peak_good]
        dp_f = d21_prior["x"]

        # some convinient varaibles
        n_chans = swp.n_chans
        # n_steps = swp.n_steps
        s21_f = swp.frequency
        s21_f_min = np.min(s21_f, axis=1)
        s21_f_max = np.max(s21_f, axis=1)
        # s21_x_center = np.mean(s21_x, axis=1)
        # s21_f_range = s21_f_max - s21_f_min
        s21_f_step = s21_f[0, 1] - s21_f[0, 0]

        map_chan_dp = (s21_f_min[:, np.newaxis] <= dp_f[np.newaxis, :]) & (
            dp_f[np.newaxis, :] <= s21_f_max[:, np.newaxis]
        )
        s21_data = swp.S21
        # s21_unc_data = swp.S21_unc

        ctd.mask_baseline = SweepCheck.make_data_mask_from_unified(
            s21_f,
            d21_f,
            d21_mask_baseline,
        )

        # do s21 detection

        # detrend and median filter
        def _cmedfilt(arr, shape):
            r = median_filter(arr.value.real, shape)
            i = median_filter(arr.value.imag, shape)
            return make_complex(r, i) << arr.unit

        as21_med = MultiSweep.calc_aS21(
            _cmedfilt(s21_data, (1, cfg.medfilt_size)),
        )

        as21_ymax = np.max(as21_med, axis=1)
        as21_y = as21_ymax[:, np.newaxis] - as21_med
        as21_ey = swp.aS21_unc

        def _calc_s21_fwhm():
            # generate Qr data for each channel
            Qr_default = np.quantile(d21_prior["Qr"], 0.9)
            logger.debug(f"{Qr_default=}")
            chan_dp_Qrs = np.tile(d21_prior["Qr"], (n_chans, 1))
            chan_dp_Qrs[~map_chan_dp] = -np.inf
            chan_dp_Qrs = np.max(chan_dp_Qrs, axis=1)
            chan_dp_Qrs[chan_dp_Qrs < 0] = Qr_default
            return s21_f / chan_dp_Qrs[:, np.newaxis] / s21_f_step

        as21_fwhm = _calc_s21_fwhm()

        def _s21_detect_postproc(r: Peaks1DResult):
            # this add Qr to the detected peak info table
            r = _detect_postproc(r)
            peaks = r.peaks
            # offset back to s21 and calculate s21 in db
            # idx_chunk is id_chan
            ci = peaks["idx_chan"] = peaks["idx_chunk"]
            y_orig = as21_ymax[ci] - peaks["y"]
            y_base_orig = as21_ymax[ci] - peaks["base"]
            y_db = peaks["y_db"] = MultiSweep.calc_db(y_orig)
            base_db = peaks["base_db"] = MultiSweep.calc_db(y_base_orig)
            peaks["height_db"] = base_db - y_db
            return r

        s21_peaks = ctd.s21_peaks = cfg.detect(
            x=s21_f.ravel(),
            y=as21_y.ravel(),
            ey=as21_ey.ravel(),
            fwhm=as21_fwhm.ravel(),
            chunks=np.arange(s21_f.size).reshape(s21_f.shape),
            postproc_hook=_s21_detect_postproc,
        )
        s21_peak_info = s21_peaks.peaks
        if s21_peak_info is None:
            raise ValueError("no peaks found in S21 data.")

        s21_Qrs = s21_peak_info["Qr"]
        s21_snrs = s21_peak_info["snr"]
        # s21_heights = s21_peak_info["height"]
        s21_heights_db = s21_peak_info["height_db"]
        # s21_thresh_scale = _make_thresh_scale(s21_peak_info["x"])
        # s21_thresh_db_offset = MultiSweep.calc_db(s21_thresh_scale)

        s21_mask_peak_Qr_small = s21_peak_info["sbm_Qr_small"] = s21_Qrs < cfg.Qr_min
        s21_mask_peak_Qr_large = s21_peak_info["sbm_Qr_large"] = (
            s21_Qrs > cfg.Qr_dark_max
        )
        s21_mask_peak_peak_small = s21_peak_info["sbm_peak_small"] = (
            s21_heights_db < cfg.peak_db_min
        )
        s21_mask_peak_snr_low = s21_peak_info["sbm_snr_low"] = s21_snrs < cfg.snr_min
        s21_mask_peak_not_real = s21_peak_info["sbm_not_real"] = (
            s21_mask_peak_peak_small
            | s21_mask_peak_snr_low
            | s21_mask_peak_Qr_small
            | s21_mask_peak_Qr_large
        )
        # handle edges
        # TODO: maybe make this configurable
        n_edge_fwhms = 0.5
        n_edges_min = 10
        n_edges = (as21_fwhm[:, [0, -1]] * n_edge_fwhms).astype(int)
        # enforce a minimum edge size
        n_edges[n_edges < n_edges_min] = n_edges_min
        s21_mask_peak_edge = s21_peak_info["sbm_edge"] = (
            s21_peak_info["idx_peak"] <= n_edges[s21_peak_info["idx_chunk"], 0]
        ) | (
            s21_peak_info["idx_peak"]
            >= (s21_peak_info["chunk_size"] - n_edges[s21_peak_info["idx_chunk"], 1])
        )
        logger.debug(f"edge peaks {pformat_mask(s21_mask_peak_edge)}")

        bitmask_s21 = np.zeros((len(s21_peak_info),), dtype=int)
        bitmask_s21 |= (
            (s21_mask_peak_peak_small * SegmentBitMask.peak_small)
            | (s21_mask_peak_snr_low * SegmentBitMask.snr_low)
            | (s21_mask_peak_Qr_small * SegmentBitMask.Qr_small)
            | (s21_mask_peak_Qr_large * SegmentBitMask.Qr_large)
            | (s21_mask_peak_not_real * SegmentBitMask.not_real)
            | (s21_mask_peak_edge * SegmentBitMask.edge)
        )
        ctd.bitmask_s21 = bitmask_s21

        s21_mask_bad_n_fwhms = 3
        ctd.s21_mask_edge = s21_peaks.make_mask(
            s21_mask_peak_edge,
            n_fwhms=s21_mask_bad_n_fwhms,
        ).reshape(s21_f.shape)
        ctd.s21_mask_not_real = s21_peaks.make_mask(
            s21_mask_peak_not_real,
            n_fwhms=s21_mask_bad_n_fwhms,
        )

        s21_mask_peak_detected = bitmask_s21 == 0
        logger.debug(
            f"s21 detected peaks {pformat_mask(s21_mask_peak_detected)}",
        )
        s21_detected = ctd.s21_detected = s21_peak_info[s21_mask_peak_detected]

        # merge detection list
        d21_detected["subdet"] = "d21"
        d21_detected["idx_subdet"] = range(len(d21_detected))
        d21_detected["bitmask"] = bitmask_d21[d21_mask_peak_detected]
        s21_detected["subdet"] = "s21"
        s21_detected["idx_subdet"] = range(len(s21_detected))
        s21_detected["bitmask"] = bitmask_s21[s21_mask_peak_detected]
        det_cols = [
            "x",
            "subdet",
            "idx_subdet",
            "idx",
            "idx_chunk",
            "idx_peak",
            "idx_chunk_offset",
            "chunk_size",
            "width",
            "Qr",
            "snr",
            "bitmask",
        ]
        det_info = vstack([d21_detected[det_cols], s21_detected[det_cols]])
        det_info["idx_det"] = range(len(det_info))
        logger.debug(f"merged detection info:\n{det_info}")

        def _agg_n_det(subdet, m):
            return np.ma.sum(m[subdet], axis=0)

        def _agg_mean_by_subdet(data_items, data_mask, subdets):
            ns = []
            vs = [[] for _ in range(len(data_items))]
            nms = []

            data_values, data_units = zip(*(map(strip_unit, data_items)), strict=True)
            for s in subdets:
                n = np.ma.sum(data_mask[s], axis=0)
                nm = n == 0
                for i, data in enumerate(data_values):
                    # mean in each subset
                    v = np.ma.mean(data[s], axis=0)
                    v[nm] = np.nan
                    vs[i].append(v)
                ns.append(n)
                nms.append(nm)
            # compute mean over all subsets
            nmm = np.array(nms)
            all_nan = np.all(nmm, axis=0)
            for i, vv in enumerate(vs):
                v = np.ma.array(vv, mask=nmm)
                v = np.ma.mean(v, axis=0)
                v[all_nan] = np.nan
                vs[i].append(v)
                for j, v in enumerate(vs[i]):
                    vs[i][j] = attach_unit(v, data_units[i])
            return ns, vs

        def _agg_func_det(m, x, d, make_masked, **_kw):
            (n_d21, n_s21), ((f_d21, f_s21, f), (Qr_d21, Qr_s21, Qr)) = (
                _agg_mean_by_subdet(
                    [x, make_masked(det_info["Qr"])],
                    m,
                    [m_subdet_d21, m_subdet_s21],
                )
            )
            return {
                "d_min": np.ma.min(d, axis=0),
                "d_max": np.ma.max(d, axis=0),
                "d_mean": np.ma.mean(d, axis=0),
                "bitmask": np.bitwise_or.reduce(
                    make_masked(det_info["bitmask"]),
                    axis=0,
                    where=m,
                ),
                "n_d21": n_d21,
                "n_s21": n_s21,
                "f_d21": f_d21,
                "f_s21": f_s21,
                "d_d21_s21": f_d21 - f_s21,
                "Qr_min": np.ma.min(Qr, axis=0),
                "Qr_max": np.ma.max(Qr, axis=0),
                "Qr_d21": Qr_d21,
                "Qr_s21": Qr_s21,
                "f": f,
                "Qr": Qr,
                "fwhm": f / Qr,
            }

        m_subdet_d21 = det_info["subdet"] == "d21"
        m_subdet_s21 = det_info["subdet"] == "s21"
        det_info["det_group_dist"] = det_info["width"] * cfg.detect_sep_fwhm_min
        # apply the thresh scale to d21 to increase tolerance for high freq side
        det_info["det_group_dist"][m_subdet_d21] *= _make_thresh_scale(
            det_info["x"][m_subdet_d21],
        )

        det_grouped, det_groups, det_group_mask = cls.make_groups1d(
            det_info["x"],
            det_info["det_group_dist"],
            agg_func=_agg_func_det,
        )
        ctd.det_grouped = det_grouped
        ctd.det_groups = det_groups
        # compose bitmask for each detection group.
        bitmask_det = ctd.bitmask_det = det_groups["bitmask"]
        bitmask_det_group_bits = (
            (det_groups["n_d21"] > 0) * SegmentBitMask.d21
            | (det_groups["n_s21"] > 0) * SegmentBitMask.s21
            | (det_groups["n_d21"] > 1) * SegmentBitMask.d21_collided
            | (det_groups["n_s21"] > 2) * SegmentBitMask.s21_collided  # noqa: PLR2004
        )
        bitmask_det |= bitmask_det_group_bits
        # map back group bits
        bitmask_det_det_group_bits = bitmask_det_group_bits[det_group_mask.nonzero()[1]]
        bitmask_d21[d21_mask_peak_detected] |= bitmask_det_det_group_bits[m_subdet_d21]
        bitmask_s21[s21_mask_peak_detected] |= bitmask_det_det_group_bits[m_subdet_s21]

        # run the model group
        def _agg_func_mdl(m, d, make_masked, **_kw):
            return {
                "d_min": np.ma.min(d, axis=0),
                "d_max": np.ma.max(d, axis=0),
                "d_mean": np.ma.mean(d, axis=0),
                "bitmask": np.bitwise_or.reduce(
                    make_masked(det_groups["bitmask"]),
                    axis=0,
                    where=m,
                ),
            }

        mdl_grouped, mdl_groups, mdl_group_mask = cls.make_groups1d(
            det_groups["f"],
            det_groups["fwhm"] * cfg.model_sep_fwhm_min,
            agg_func=_agg_func_mdl,
        )
        ctd.mdl_grouped = mdl_grouped
        ctd.mdl_groups = mdl_groups

        # model group bits
        bitmask_mdl_group_bits = (
            (mdl_grouped["groupsize"] == 2) * SegmentBitMask.doublet  # noqa: PLR2004
            | (mdl_grouped["groupsize"] == 3) * SegmentBitMask.triplet  # noqa: PLR2004
            | (mdl_grouped["groupsize"] > 3) * SegmentBitMask.manylet  # noqa: PLR2004
        )
        bitmask_det |= bitmask_mdl_group_bits
        # map back group bits
        bitmask_det_mdl_group_bits = bitmask_mdl_group_bits[det_group_mask.nonzero()[1]]
        bitmask_d21[d21_mask_peak_detected] |= bitmask_det_mdl_group_bits[m_subdet_d21]
        bitmask_s21[s21_mask_peak_detected] |= bitmask_det_mdl_group_bits[m_subdet_s21]

        # build the final detection table:
        detected = ctd.detected = hstack(
            [
                mdl_grouped,
                det_groups["f", "Qr", "fwhm"],
            ],
        )

        # do match to chan and ref
        tbl_chans = swp.meta["chan_axis_data"]
        mask_tone = tbl_chans["mask_tone"]
        tbl_chans = tbl_chans[mask_tone]

        def _match_postproc(r: Match1DResult):
            matched = r.matched
            iq = matched["idx_query"]
            Qr = det_groups["Qr"][iq]
            rr = 0.5 / Qr
            xx = matched["dist"] / matched["query"]
            matched["d_phi"] = np.rad2deg(np.arctan2(xx, rr)) << u.deg
            matched["bitmask_det"] = bitmask_det[iq]
            matched["idx_det"] = iq
            matched["f_det"] = det_groups["f"][iq]
            matched["Qr"] = Qr
            return r

        def _match_postproc_chan(r: Match1DResult):
            r = _match_postproc(r)
            matched = r.matched
            # add channel info
            ir = matched["idx_ref"]
            # note that chan_id is the id to the full chan list
            # idx_ref is only indexing to the chans with mask_tone=True.
            matched["idx_chan"] = tbl_chans["id"][ir]
            matched["f_chan"] = tbl_chans["f_chan"][ir]
            matched["amp_tone"] = tbl_chans["amp_tone"][ir]
            return r

        match_kw = {
            "shift_kw": {
                "shift_max": cfg.match_shift_max,
                "dx_resample": 1,
            },
        }
        matched = ctd.matched = cfg.match(
            query=detected["f"].to(u.MHz),
            ref=tbl_chans["f_chan"].to(u.MHz),
            postproc_hook=_match_postproc_chan,
            **match_kw,
        )
        # update back the matched tables
        detected_matched = ctd.detected_matched = matched.data["query_matched"]
        for colname in ["dist", "d_phi", "idx_chan", "f_chan", "amp_tone"]:
            detected[colname] = detected_matched[colname]

        ctd.chan_matched = matched.data["ref_matched"]

        if cfg.match_ref == "chan":
            ctd.matched_ref = matched
        else:
            if cfg.match_ref == "d21":
                f_ref = d21_detected["x"]
            elif cfg.match_ref == "s21":
                f_ref = s21_detected["x"]
            else:
                assert_never()
            ctd.matched_ref = cfg.match(
                query=detected["f"].to(u.MHz),
                ref=f_ref.to(u.MHz),
                postproc_hook=_match_postproc,
                **match_kw,
            )
        return True

    @staticmethod
    @timeit
    def make_groups1d(x, d, agg_func=None):
        """Group ``x`` by ``d``."""
        ii = np.argsort(x)
        xx = x[ii]
        dd = d[ii]
        dxx = np.diff(xx)
        assert np.all(dxx >= 0)
        br = []
        for i, dx in enumerate(dxx):
            if dx > dd[i] and dx > dd[i + 1]:
                br.append(i + 1)
        idx_groups = np.split(ii, br)
        n_groups = len(idx_groups)
        mask_groups = np.zeros((len(x), n_groups), dtype=bool)
        for i, igg in enumerate(idx_groups):
            mask_groups[igg, i] = True
        size_groups = np.sum(mask_groups, axis=0)
        unique, counts = np.unique(size_groups, return_counts=True)
        logger.debug(
            f"group {len(x)} items into {n_groups} groups with mean d={np.mean(d)}",
        )
        logger.debug(
            f"unique group sizes:\n{np.vstack([unique, counts])}",
        )
        groups = QTable(
            {
                "group": range(n_groups),
                "size": size_groups,
                "items": idx_groups,
            },
        )
        if agg_func is not None:
            # add stats per group
            def _make_masked(arr):
                return np.ma.array(np.tile(arr, (n_groups, 1)).T, mask=~mask_groups)

            xg = _make_masked(x)
            dg = _make_masked(d)
            for cname, value in agg_func(
                m=mask_groups,
                x=xg,
                d=dg,
                g=groups,
                make_masked=_make_masked,
            ).items():
                v = value.data if hasattr(value, "mask") else value
                groups[cname] = v
        logger.debug(f"groups:\n{groups}")
        group_idx = np.nonzero(mask_groups)[1]
        group_size = size_groups[group_idx]
        grouped = QTable(
            {
                "idx": range(len(x)),
                "group": group_idx,
                "groupsize": group_size,
            },
        )
        grouped["x"] = x
        grouped["d"] = d
        logger.debug(f"grouped:\n{grouped}")
        # grouped_sorted = grouped[ii]
        # logger.debug(f"grouped_sorted:\n{grouped_sorted}")
        return grouped, groups, mask_groups

    @staticmethod
    def fit_baseline_circle(swp: MultiSweep, mask_baseline):
        """Run circle fit to baseline data."""
        s21_value = swp.S21.value
        s21_unc_value = swp.S21_unc.value
        n_chans = swp.n_chans
        n_steps = swp.n_steps

        def _circle_objective_func(c, s21, _s21_unc):
            c = c[0] + 1.0j * c[1]
            r = np.abs(s21 - c)
            return r - np.mean(r)  # / np.abs(s21_unc)

        def _circle_jac_func(c, s21, _s21_unc):
            c = c[0] + 1.0j * c[1]
            r = np.abs(s21 - c)
            df_dc = np.empty((2, s21.size))
            df_dc[0] = (c.real - s21.real) / r  # dR/dxc
            df_dc[1] = (c.imag - s21.imag) / r  # dR/dyc
            return df_dc - df_dc.mean(axis=1, keepdims=True)

        chan_baseline_info = []
        baseline_fitsize_min = n_steps // 2

        for ci in range(n_chans):
            mm = mask_baseline[ci]
            fitsize = mm.sum()
            info = {
                "idx_chan": ci,
                "fitsize": fitsize,
                "center": 0.0 + 0.0j,
                "amp": 0.0,
                "amp_unc": np.nan,
                "converged": False,
            }
            if fitsize < baseline_fitsize_min:
                chan_baseline_info.append(info)
                continue
            # logger(f"fit baseline with {mm.sum()} points")
            y = s21_value[ci, mm]
            ey = s21_unc_value[ci, mm]
            y0, ier = leastsq(
                _circle_objective_func,
                (0.0, 0.0),
                args=(y, ey),
                Dfun=_circle_jac_func,
                col_deriv=True,
            )
            y0 = y0[0] + 1.0j * y0[1]
            ya = np.abs(y - y0)
            a = np.mean(ya)
            a_unc = np.std(ya)
            chan_baseline_info.append(
                info
                | {
                    "center": y0,
                    "amp": a,
                    "amp_unc": a_unc,
                    "converged": True,
                },
            )
        chan_baseline_info = QTable(chan_baseline_info)
        logger.debug(
            f"fitted baseline:\n{chan_baseline_info}"
            f"\nconverged: {pformat_mask(chan_baseline_info['converged'])}",
        )
        # interpolate for non-converged channels
        # def _cinterp(x, xp, yp):
        #     y0 = np.interp(x, xp, yp.real)
        #     y1 = np.interp(x, xp, yp.imag)
        #     return y0 + 1.j * y1
        #
        # cbi = chan_baseline_info
        # m = ~cbi["converged"]
        # if m.sum() > 0:
        #     chan_baseline_info["center"][m] = _cinterp(
        #         swp.f_chans[cbi["idx_chan"]][m],
        #         swp.f_chans[cbi["idx_chan"]][~m],
        #         cbi["center"][~m],
        #         )
        #     chan_baseline_info["amp"][m] = np.interp(
        #         swp.f_chans[cbi["idx_chan"]][m],
        #         swp.f_chans[cbi["idx_chan"]][~m],
        #         cbi["amp"][~m],
        #         )

        def _calc_baseline(y, y0, a):
            a = a[:, np.newaxis]
            y0 = y0[:, np.newaxis]
            ph = np.angle(y - y0)
            return a * np.exp(1.0j * ph) + y0, ph

        s21_baseline, ph = _calc_baseline(
            s21_value,
            chan_baseline_info["center"],
            chan_baseline_info["amp"],
        )
        chan_baseline_info["phi0"] = ph[:, 0]
        chan_baseline_info["phi1"] = ph[:, -1]
        chan_baseline_info["phi_center"] = np.mean(ph, axis=1)
        # when removing baseline, we shift it to be cutting with the maximum s21 value
        # so it can handle the not well-fitted case
        # as21_detrended = np.abs(s21_value / s21_baseline)
        # as21_detrended = np.abs(
        #     (s21_value - chan_baseline_info["center"][:, np.newaxis])
        # )
        # as21_detrended = ctd.as21_detrended = (
        #     np.max(as21_detrended, axis=1, keepdims=True) - as21_detrended
        # )
        # as21_y = median_filter(as21_detrended, (1, cfg.medfilt_size))

        return locals()


class KidsFindPlotConfig(PlotConfig):
    """The kids finding plot config."""


@dataclass(kw_only=True)
class KidsFindPlotData:
    """The data class for kids finding plot."""

    d21_peaks_summary: go.Figure = ...
    s21_peaks_summary: go.Figure = ...
    det_summary: go.Figure = ...
    # chan_baseline: go.Figure = ...
    peaks: go.Figure = ...
    matched: go.Figure = ...
    matched_ref: go.Figure = ...


class KidsFindPlotContext(StepContext["KidsFindPlot", KidsFindPlotConfig]):
    """The context class for kids finding plot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: KidsFindPlotData = Field(default_factory=KidsFindPlotData)


class KidsFindPlot(PlotMixin, Step[KidsFindPlotConfig, KidsFindPlotContext]):
    """Kids find plot.

    This step produces visualization for sweep finding step.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):  # noqa: PLR0915
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

        def _plot_peak_summary(
            bitmask,
            peaks: Peaks1DResult,
            x_unit,  # noqa: ARG001
            y_unit,
        ):
            peak_info = peaks.peaks
            height = peak_info["height"].to_value(y_unit)
            seg_data_items = [
                (
                    "Qr",
                    peak_info["Qr"],
                    {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min},
                ),
                (
                    "SNR",
                    peak_info["snr"],
                    {"zmin": 0, "zmax": np.quantile(peak_info["snr"], 0.9)},
                ),
                (
                    "FWHM",
                    peak_info["width"].to_value(u.Hz),
                    {"zmin": 1000, "zmax": 120000},
                ),
                (
                    "height",
                    height,
                    {
                        "zmin": 0,
                        "zmax": np.quantile(height, 0.9),
                    },
                ),
            ]
            if "height_db" in peak_info.colnames:
                height_db = peak_info["height_db"]
                seg_data_items.append(
                    (
                        "height_db",
                        height_db,
                        {
                            "zmin": 0,
                            "zmax": 1,
                        },
                    ),
                )
            return cls.make_summary_fig(bitmask, seg_data_items)

        ctd.d21_peaks_summary = _plot_peak_summary(
            ctd_kf.bitmask_d21,
            ctd_kf.d21_peaks,
            x_unit=u.MHz,
            y_unit=u.Hz**-1,
        )
        ctd.s21_peaks_summary = _plot_peak_summary(
            ctd_kf.bitmask_s21,
            ctd_kf.s21_peaks,
            x_unit=u.MHz,
            y_unit=u.dimensionless_unscaled,
        )
        bitmask_det = ctd_kf.bitmask_det
        det_groups = ctd_kf.det_groups
        ctd.det_summary = cls.make_summary_fig(
            bitmask_det,
            [
                (
                    "f",
                    det_groups["f"].to_value(u.MHz),
                    {},
                ),
                (
                    "Qr",
                    det_groups["Qr"],
                    {"zmin": cfg_kf.Qr_min, "zmax": cfg_kf.Qr_dark_min},
                ),
            ],
        )
        fig = ctd.peaks = cls.make_subplots(
            n_rows=3,
            n_cols=1,
            shared_xaxes="all",
            vertical_spacing=40 / 1000,
            fig_layout=cls.fig_layout_default
            | {
                "showlegend": True,
                "height": 1000,
            },
        )
        d21_panel_kw = {"row": 1, "col": 1}
        s21_panel_kw = {"row": 2, "col": 1}
        s21_data_panel_kw = {"row": 3, "col": 1}
        color_cycle = cls.color_palette.cycle()

        def _plot_peaks(  # noqa: PLR0913
            name,
            peaks: Peaks1DResult,
            x_unit,
            y_unit,
            panel_kw,
            overlay_masks,
        ):
            peak_info = peaks.peaks
            labels = peaks.labels
            x = peaks.x.to_value(x_unit)
            y = peaks.y.to_value(y_unit)
            height = peak_info["height"].to_value(y_unit)
            base = peak_info["base"].to_value(y_unit)
            # ey = peaks.ey.to_value(y_unit)
            for ll in np.unique(labels):
                color = next(color_cycle)
                showlegend = True
                m = labels == ll
                fig.add_scatter(
                    x=x[m],
                    y=y[m],
                    mode="lines",
                    line={
                        "color": color,
                    },
                    name=f"peak {ll}",
                    showlegend=showlegend,
                    **panel_kw,
                )
            customdata_info = [
                ("label", ".0f"),
                ("snr", ".3f"),
                ("Qr", ".3f"),
                ("height_db", ".3f"),
                ("halfmax_size", ".0f"),
                ("lookahead", ".0f"),
            ] + [(c, ".0f") for c in peak_info.colnames if c.startswith("sbm")]
            customdata_info = [c for c in customdata_info if c[0] in peak_info.colnames]
            fig.add_scatter(
                x=peak_info["x"].to_value(x_unit),
                y=height / 2 + base,
                error_x={
                    "type": "data",
                    "array": peak_info["width"].to_value(x_unit) / 2,
                    "width": 0,
                    "color": "green",
                },
                error_y={
                    "type": "data",
                    "array": height / 2,
                    "width": 0,
                    "color": "green",
                },
                customdata=np.stack(
                    [peak_info[ci[0]] for ci in customdata_info],
                ).T,
                hovertemplate=("f: %{x:.3f}<br>d21: %{y:.3f}")
                + "".join(
                    f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                    for i, c in enumerate(customdata_info)
                ),
                mode="markers",
                marker={
                    "color": "orange",
                    "size": 4,
                },
                name="peak info",
                **panel_kw,
            )
            # fig.add_scatter(
            #     x=x,
            #     y=peaks.delta,
            #     mode="lines",
            #     line={
            #         "color": "#aaaaaa",
            #     },
            #     name="delta",
            #     **panel_kw,
            # )
            for mask_name, mask, mask_color in overlay_masks:
                fig.add_scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="lines",
                    line={
                        "color": mask_color,
                    },
                    name=mask_name,
                    **panel_kw,
                )
            fig.update_yaxes(
                title={
                    "text": f"{name} ({y_unit})",
                },
                **panel_kw,
            )

        _plot_peaks(
            "|D21|",
            ctd_kf.d21_peaks,
            x_unit=u.MHz,
            y_unit=u.Hz**-1,
            panel_kw=d21_panel_kw,
            overlay_masks=[
                ("not real", ctd_kf.d21_mask_not_real, "gray"),
                ("dark", ctd_kf.d21_mask_dark, "red"),
                ("baseline", ctd_kf.d21_mask_baseline, "black"),
            ],
        )
        _plot_peaks(
            "|S21|",
            ctd_kf.s21_peaks,
            x_unit=u.MHz,
            y_unit=u.dimensionless_unscaled,
            panel_kw=s21_panel_kw,
            overlay_masks=[
                ("not real", ctd_kf.s21_mask_not_real, "gray"),
                ("edge", ctd_kf.s21_mask_edge.ravel(), "cyan"),
            ],
        )
        d21_dets = ctd_kf.d21_detected
        s21_y_max = ctd_kf.s21_peaks.y.max().value
        fig.add_scatter(
            x=d21_dets["x"].to_value(u.MHz),
            y=np.zeros((len(d21_dets),), dtype=float),
            error_y={
                "type": "constant",
                "value": s21_y_max,
                "valueminus": 0,
                "width": 0,
                "color": "cyan",
                "thickness": 0.5,
            },
            **s21_panel_kw,
        )
        # S21 trace
        fs = swp.frequency.to_value(u.MHz)
        as21_db = swp.aS21_db
        as21_unc_db = swp.aS21_unc_db
        mask_baseline = ctd_kf.mask_baseline
        color_cycle = cls.color_palette.cycle_alternated(1, 0.5)
        ds = slice(None, None, 4)
        for ci in range(fs.shape[0]):
            color = next(color_cycle)
            color2 = next(color_cycle)
            color_arr = [color if m else "black" for m in mask_baseline[ci, ds]]
            fig.add_scattergl(
                x=fs[ci, ds],
                y=as21_db[ci, ds],
                error_y={
                    "type": "data",
                    "array": as21_unc_db[ci, ds],
                    "width": 0,
                    "color": color2,
                },
                mode="markers",
                marker={
                    "size": 4,
                    "color": color_arr,
                },
                name=f"S21 {ci}",
                showlegend=False,
                **s21_data_panel_kw,
            )
        f_det = det_groups["f"].to_value(u.MHz)
        det_d_max = det_groups["d_max"].to_value(u.MHz)
        fwhm_det = det_groups["fwhm"].to_value(u.MHz)
        as21_min_idx = np.argmin(as21_db, axis=1, keepdims=True)
        as21_max_idx = np.argmax(as21_db, axis=1, keepdims=True)
        f_min = np.take_along_axis(fs, as21_min_idx, axis=1).ravel()
        f_max = np.take_along_axis(fs, as21_max_idx, axis=1).ravel()
        isort_min = np.argsort(f_min)
        isort_max = np.argsort(f_max)
        as21_det = np.interp(
            f_det,
            f_min[isort_min],
            np.take_along_axis(as21_db, as21_min_idx, axis=1).ravel()[isort_min],
        )
        as21_base = np.interp(
            f_det,
            f_max[isort_max],
            np.take_along_axis(as21_db, as21_max_idx, axis=1).ravel()[isort_max],
        )
        customdata_info = [
            ("group", ".0f"),
            ("size", ".0f"),
            ("d_min", ".3f"),
            ("d_max", ".3f"),
        ]
        fig.add_scatter(
            x=f_det,
            y=as21_det + 0.1,
            mode="markers",
            marker={
                "size": 4,
            },
            error_x={
                "type": "data",
                "array": det_d_max * 0.5,
                "width": 0,
                "color": "orange",
            },
            **s21_data_panel_kw,
        )

        fig.add_scatter(
            x=f_det,
            y=as21_det,
            mode="markers",
            marker={
                "size": 4,
            },
            error_x={
                "type": "data",
                "array": fwhm_det * 0.5,
                "width": 0,
                "color": "orange",
            },
            error_y={
                "type": "data",
                "array": (as21_base - as21_det) * 2,
                "arrayminus": np.zeros(f_det.shape),
                "width": 0,
                "color": "orange",
            },
            customdata=np.stack(
                [
                    det_groups[ci[0]]
                    for ci in customdata_info
                    if ci[0] in det_groups.colnames
                ],
            ).T,
            hovertemplate=("f: %{x:.3f}<br>s21: %{y:.3f}")
            + "".join(
                f"<br>{c[0]}: %{{customdata[{i}]:{c[1]}}}"
                for i, c in enumerate(customdata_info)
                if c[0] in det_groups.colnames
            ),
            **s21_data_panel_kw,
        )

        # s21_dets = ctd_kf.d21_detected
        fig.update_yaxes(
            title={
                "text": "|S21| (dB)",
            },
            **s21_data_panel_kw,
        )
        fig.update_xaxes(
            title={
                "text": "Frequency (MHz)",
            },
            **s21_data_panel_kw,
        )

        # matched
        ctd.matched = cls.make_matched_fig(ctd_kf.matched, "Chan")
        ctd.matched_ref = cls.make_matched_fig(
            ctd_kf.matched_ref,
            cfg_kf.match_ref.capitalize(),
        )
        cls.save_or_show(data, context)
        return True

    @classmethod
    def make_summary_fig(cls, bitmask_seg, seg_data_items):
        grid = cls.make_subplot_grid()
        grid.add_subplot(
            row=1,
            col=1,
            fig=cls.make_bitmask_seg_heatmap(
                bitmask_seg,
            ),
            row_height=1,
        )
        row0 = grid.shape[0] + 1
        for i, (name, value, trace_kw) in enumerate(seg_data_items):
            grid.add_subplot(
                row=row0 + i,
                col=1,
                fig=cls.make_seg_data_heatmap(
                    name,
                    value,
                    trace_kw,
                ),
                row_height=0.5 / len(seg_data_items),
            )
        fig = grid.make_figure(
            shared_xaxes="all",
            vertical_spacing=40 / 1200,
            fig_layout={
                "height": 1200,
            },
        )
        # add a range slider
        fig.update_xaxes(
            rangeslider={
                "autorange": True,
                "range": [0, bitmask_seg.shape[0]],
                "thickness": 0.05,
            },
            row=grid.shape[0],
            col=1,
        )
        return fig

    @classmethod
    def make_bitmask_seg_heatmap(cls, bitmask_seg, fig=None, panel_kw=None):
        names = []
        data = []
        for name, value in SegmentBitMask.__members__.items():
            names.append(name)
            data.append((bitmask_seg & value) > 0)
        data = np.vstack(data).astype(int)

        fig = fig or cls.make_subplots(1, 1)
        panel_kw = panel_kw or {}
        fig.add_heatmap(
            z=data,
            y=names,
            colorscale="rdylgn_r",
            zmin=0,
            zmax=1,
            **panel_kw,
        )
        fig.update_xaxes(
            title="Segment Id",
            **panel_kw,
        )
        fig.update_yaxes(
            **panel_kw,
        )
        fig.update_layout(
            title={
                "text": "Segment Bitmask",
            },
        )
        return fig

    @classmethod
    def make_seg_data_heatmap(  # noqa: PLR0913
        cls,
        name,
        data,
        trace_kw,
        fig=None,
        panel_kw=None,
    ):
        fig = fig or cls.make_subplots(1, 1)
        panel_kw = panel_kw or {}
        z = data[np.newaxis, :]
        y = [name]
        fig.add_heatmap(
            z=z,
            y=y,
            colorscale="rdylgn_r",
            **trace_kw,
            **panel_kw,
        )
        fig.update_xaxes(
            title="Segment Id",
            **panel_kw,
        )
        fig.update_layout(
            title={
                "text": name,
            },
        )
        return fig

    @classmethod
    def make_chan_baseline_info_fig(cls, swp, chan_baseline_info):
        # chan baseline info
        fig = cls.make_subplots(
            n_rows=2,
            n_cols=3,
            shared_xaxes=True,
            vertical_spacing=40 / 1000,
            fig_layout=cls.fig_layout_default
            | {
                "showlegend": True,
                "height": 1000,
            },
            specs=[
                [{"rowspan": 2}, {}, {"type": "polar"}],
                [None, {}, {"type": "polar"}],
            ],
        )
        cbi_amp_unc_cut = 1e3
        cbi = chan_baseline_info
        cbi = cbi[cbi["amp_unc"] < cbi_amp_unc_cut]
        fig.add_scatter(
            x=cbi["center"].real,
            y=cbi["center"].imag,
            mode="markers",
            marker={"size": 4},
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=swp.f_chans[cbi["idx_chan"]].to_value(u.MHz),
            y=np.abs(cbi["center"]),
            mode="markers+lines",
            marker={
                "size": 4,
            },
            row=1,
            col=2,
        )
        fig.add_scatter(
            x=swp.f_chans[cbi["idx_chan"]].to_value(u.MHz),
            y=cbi["amp"],
            error_y={
                "type": "data",
                "array": cbi["amp_unc"],
                "width": 0,
                "color": "gray",
            },
            mode="markers+lines",
            marker={
                "size": 4,
            },
            row=2,
            col=2,
        )
        fig.add_scatterpolar(
            theta=np.rad2deg(cbi["phi_center"]),
            r=np.abs(cbi["center"]),
            mode="markers",
            marker={
                "size": 4,
            },
            row=1,
            col=3,
        )
        fig.add_scatterpolar(
            theta=np.rad2deg(cbi["phi_center"]),
            r=cbi["amp"],
            mode="markers",
            marker={
                "size": 4,
            },
            row=2,
            col=3,
        )
        return fig

    @classmethod
    def make_matched_fig(cls, matched, ref_name):
        fig = cls.make_subplots(
            n_rows=3,
            n_cols=1,
            vertical_spacing=40 / 1200,
            fig_layout=cls.fig_layout_default
            | {
                "showlegend": False,
                "height": 1200,
            },
        )
        dist_panel_kw = {"row": 1, "col": 1}
        match_panel_kw = {"row": 2, "col": 1}
        density_panel_kw = {"row": 3, "col": 1}

        tbl_matched = matched.matched.copy()
        tbl_matched.sort("adist_shifted")
        tbl_matched = unique(tbl_matched, keys="idx_query")
        d_phi_good_max = 5 << u.deg
        d_phi_ok_max = d_phi_good_max * 3
        d_phi = tbl_matched["d_phi"]
        ad_phi = np.abs(d_phi)

        m_good = ad_phi < d_phi_good_max
        m_ok = (ad_phi >= d_phi_good_max) & (ad_phi < d_phi_ok_max)
        m_bad = ad_phi >= d_phi_ok_max
        m_dup = (tbl_matched["bitmask_det"] & SegmentBitMask.blended) > 0
        d_phi_good_max_value = d_phi_good_max.to_value(u.deg)
        bins = (
            np.arange(
                -90 - d_phi_good_max_value / 2,
                90 + d_phi_good_max_value * 1.1 / 2,
                d_phi_good_max_value,
            )
            << u.deg
        )
        x = (0.5 * (bins[1:] + bins[:-1])).to_value(u.deg)
        y_good_dup, _ = np.histogram(d_phi[m_good & m_dup], bins=bins)
        y_good, _ = np.histogram(d_phi[m_good & (~m_dup)], bins=bins)
        y_ok_dup, _ = np.histogram(d_phi[m_ok & m_dup], bins=bins)
        y_ok, _ = np.histogram(d_phi[m_ok & (~m_dup)], bins=bins)
        y_bad_dup, _ = np.histogram(d_phi[m_bad & m_dup], bins=bins)
        y_bad, _ = np.histogram(d_phi[m_bad & (~m_dup)], bins=bins)

        c00, c25, c75, c100 = plotly.colors.sample_colorscale(
            "rdylgn",
            samplepoints=[0, 0.25, 0.75, 1],
        )
        for y, name, color in [
            (y_bad, "bad", c75),
            (y_bad_dup, "bad_dup", c25),
            (y_ok, "ok", c75),
            (y_ok_dup, "ok_dup", c25),
            (y_good, "good", c100),
            (y_good_dup, "good_dup", c00),
        ]:
            fig.add_bar(
                x=x,
                y=y,
                marker={
                    "color": color,
                },
                name=name,
                **dist_panel_kw,
            )
        for x0, x1, opt in [
            (bins[0].to_value(u.deg), -d_phi_ok_max.to_value(u.deg), 0.3),
            (-d_phi_ok_max.to_value(u.deg), -d_phi_good_max.to_value(u.deg), 0.15),
            (-d_phi_good_max.to_value(u.deg), d_phi_good_max.to_value(u.deg), 0.0),
            (d_phi_good_max.to_value(u.deg), d_phi_ok_max.to_value(u.deg), 0.15),
            (d_phi_ok_max.to_value(u.deg), bins[-1].to_value(u.deg), 0.3),
        ]:
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="black", opacity=opt)
        fig.update_yaxes(
            title="Count",
            **dist_panel_kw,
        )
        fig.update_xaxes(
            title="phi (deg)",
            **dist_panel_kw,
        )

        matched.make_plotly_fig(
            type="match",
            fig=fig,
            panel_kw=match_panel_kw,
            label_value="Frequency (MHz)",
            label_ref=ref_name,
            label_query="Detect",
        )
        matched.make_plotly_fig(
            type="density",
            fig=fig,
            panel_kw=density_panel_kw,
            label_ref=f"Ref Id ({ref_name})",
            label_query="Detect Id",
        )
        fig.update_layout(barmode="stack")
        return fig
