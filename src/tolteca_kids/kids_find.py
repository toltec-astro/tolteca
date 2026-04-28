from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy.table import Column, QTable, hstack, unique, vstack
from pydantic import ConfigDict, Field
from scipy.ndimage import median_filter
from scipy.optimize import leastsq
from tollan.config.types import FrequencyQuantityField, TimeQuantityField
from tollan.utils.fmt import pformat_mask
from tollan.utils.log import logger, timeit
from tollan.utils.np import attach_unit, make_complex, strip_unit
from tollan.utils.table import TableValidator
from tollan.pipeline import Step, StepConfig, StepContext
from typing_extensions import assert_never

from tolteca_datamodels.toltec.kids import ReducedSweepView

from .match1d import Match1D, Match1DResult
from .peaks1d import Peaks1D, Peaks1DResult
from .sweep_check import SweepCheck, _extract_sweep_arrays

__all__ = [
    "KidsFind",
    "KidsFindConfig",
    "KidsFindContext",
    "KidsFindData",
    "SegmentBitMask",
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

    rejected = auto()
    """The detection does not pass select."""


class KidsFindConfig(StepConfig):
    """The kids finding config."""

    model_config = ConfigDict(protected_namespaces=(), validate_default=True)

    Qr_min: float = Field(
        default=1000,
        description="Minimum Qr allowed for detection..",
    )

    Qr_dark_min: float = Field(
        default=20000,
        description="Minimum Qr for dark detectors.",
    )
    Qr_dark_max: float = Field(
        default=120000,
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
    d21_peak_dark_min: D21QuantityField = Field(
        default=50 << (u.Hz**-1),
        description="Minimum peak height for dark kids at ref freq",
    )
    d21_snr_dark_min: float = Field(
        default=100.0,
        description="Minimum SNR for dark kids at ref freq",
    )
    d21_select: None | str = Field(
        default=None,
        description="Additional select clause to filter D21 peaks.",
    )
    medfilt_size: int = Field(
        default=5,
        description="Size of median filter used for S21 peak finding.",
    )
    detect: Peaks1D = Field(
        default={
            "method": "peakdetect",
            "threshold": 0,
            "peakdetect_delta_threshold": 5,
        },
        description="Detection settings for S21.",
    )
    detect_Qr_fallback: float = Field(
        default=5000,
        description="Qr to use for detection when no D21 prior is found.",
    )
    peak_db_min: float = Field(
        default=0.2,
        description="Minimum peak height allowed for detection.",
    )
    snr_min: float = Field(
        default=10.0,
        description="Minimum SNR for detected kids.",
    )
    select: None | str = Field(
        default=None,
        description="Additional select clause to filter peaks.",
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
    def run(cls, data: xr.DataTree, context):  # noqa: PLR0915, C901, PLR0912
        """Run kids find."""
        cfg = context.config
        ctd = context.data
        ctd_sc = SweepCheck.get_context(data).data

        # Load sweep data from xr.DataTree (replaces MultiSweep in v2)
        arrays = _extract_sweep_arrays(data)
        n_chans = arrays.n_chans
        s21_f = arrays.frequency  # Quantity [n_chans, n_steps] Hz
        s21_data = arrays.S21 << u.dimensionless_unscaled  # complex Quantity
        s21_f_min = np.min(s21_f.to_value(u.Hz), axis=1) << u.Hz
        s21_f_max = np.max(s21_f.to_value(u.Hz), axis=1) << u.Hz
        s21_f_step = s21_f[0, 1] - s21_f[0, 0]

        # Build channel axis table from ReducedSweepView f_lo
        # (replaces swp.meta["chan_axis_data"] in v2)
        view = ReducedSweepView(data)
        f_lo_da = view.f_lo
        if f_lo_da is not None:
            f_lo_hz = f_lo_da.values
        else:
            f_lo_hz = np.linspace(
                float(s21_f_min.to_value(u.Hz).mean()),
                float(s21_f_max.to_value(u.Hz).mean()),
                n_chans,
            )
        tbl_chans_all = QTable({
            "id": np.arange(n_chans),
            "f_chan": f_lo_hz << u.Hz,
            "mask_tone": np.ones(n_chans, dtype=bool),
            "amp_tone": np.ones(n_chans),
        })

        _tbl_validator = TableValidator()

        def _make_select_mask(tbl, expr):
            if expr is None:
                return np.ones((len(tbl),), dtype=bool)
            return _tbl_validator.eval(tbl, expr)

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
        d21_mask_peak_dark_peak_small = (d21_Qrs >= cfg.Qr_dark_min) & (
            d21_heights < cfg.d21_peak_dark_min / d21_thresh_scale
        )

        d21_mask_rejected = d21_peak_info["sbm_rejected"] = ~_make_select_mask(
            d21_peak_info,
            cfg.d21_select,
        )

        d21_mask_peak_not_real = d21_peak_info["sbm_not_real"] = (
            d21_mask_peak_peak_small
            | d21_mask_peak_snr_low
            | d21_mask_peak_Qr_small
            | d21_mask_peak_Qr_large
            | d21_mask_peak_dark_snr_low
            | d21_mask_peak_dark_peak_small
            | d21_mask_rejected
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
            | (d21_mask_rejected * SegmentBitMask.rejected)
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

        map_chan_dp = (
            s21_f_min.to_value(u.Hz)[:, np.newaxis] <= dp_f.to_value(u.Hz)[np.newaxis, :]
        ) & (
            dp_f.to_value(u.Hz)[np.newaxis, :] <= s21_f_max.to_value(u.Hz)[:, np.newaxis]
        )

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

        as21_med = np.abs(_cmedfilt(s21_data, (1, cfg.medfilt_size)).value)

        as21_ymax = np.max(as21_med, axis=1)
        as21_y = as21_ymax[:, np.newaxis] - as21_med
        as21_ey = arrays.aS21_unc

        def _calc_s21_fwhm():
            # generate Qr data for each channel
            n_d21_priors = len(d21_prior)
            n_d21_priors_min = 10
            if n_d21_priors >= n_d21_priors_min:
                Qr_default = np.quantile(d21_prior["Qr"], 0.9)
            elif n_d21_priors > 0:
                logger.debug("not enough prior found, use median value")
                Qr_default = np.quantile(d21_prior["Qr"], 0.5)
            else:
                Qr_default_no_prior = cfg.detect_Qr_fallback
                logger.debug(
                    f"no d21 prior found, use default value {Qr_default_no_prior=}",
                )
                Qr_default = Qr_default_no_prior

            logger.debug(f"{Qr_default=}")
            if n_d21_priors > 0:
                chan_dp_Qrs = np.tile(d21_prior["Qr"], (n_chans, 1))
                chan_dp_Qrs[~map_chan_dp] = -np.inf
                chan_dp_Qrs = np.max(chan_dp_Qrs, axis=1)
                chan_dp_Qrs[chan_dp_Qrs < 0] = Qr_default
            else:
                chan_dp_Qrs = np.full((n_chans,), Qr_default)
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
            y_db = peaks["y_db"] = 20.0 * np.log10(np.maximum(np.abs(y_orig), 1e-30))
            base_db = peaks["base_db"] = 20.0 * np.log10(np.maximum(np.abs(y_base_orig), 1e-30))
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
        s21_mask_rejected = s21_peak_info["sbm_rejected"] = ~_make_select_mask(
            s21_peak_info,
            cfg.select,
        )

        s21_mask_peak_not_real = s21_peak_info["sbm_not_real"] = (
            s21_mask_peak_peak_small
            | s21_mask_peak_snr_low
            | s21_mask_peak_Qr_small
            | s21_mask_peak_Qr_large
            | s21_mask_rejected
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
            | (s21_mask_rejected * SegmentBitMask.rejected)
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
        if len(d21_detected) > 0:
            d21_detected["subdet"] = "d21"
            d21_detected["idx_subdet"] = range(len(d21_detected))
            d21_detected["bitmask"] = bitmask_d21[d21_mask_peak_detected]
        else:
            d21_detected["subdet"] = Column(dtype=str)
            d21_detected["idx_subdet"] = Column(dtype=int)
            d21_detected["bitmask"] = Column(dtype=int)
        if len(s21_detected) > 0:
            s21_detected["subdet"] = "s21"
            s21_detected["idx_subdet"] = range(len(s21_detected))
            s21_detected["bitmask"] = bitmask_s21[s21_mask_peak_detected]
        else:
            s21_detected["subdet"] = Column(dtype=str)
            s21_detected["idx_subdet"] = Column(dtype=int)
            s21_detected["bitmask"] = Column(dtype=int)
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

        def _make_det_info_tbl(tbl, cols, cols_with_suffix, suffix):
            t = tbl[cols]
            for c in cols_with_suffix:
                t[f"{c}{suffix}"] = tbl[c]
            return t

        det_info = vstack(
            [
                _make_det_info_tbl(d21_detected, det_cols, ["height"], "_d21"),
                _make_det_info_tbl(s21_detected, det_cols, ["height_db"], "_s21"),
            ],
        )
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
                for i, d in enumerate(data_values):
                    # mean in each subset
                    v = np.ma.mean(d[s], axis=0)
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
                vv.append(v)
                for j, v in enumerate(vv):
                    vv[j] = attach_unit(v, data_units[i])
            return ns, vs

        def _agg_func_det(m, x, d, make_masked, **_kw):
            (n_d21, n_s21), (
                (f_d21, f_s21, f),
                (Qr_d21, Qr_s21, Qr),
                (snr_d21, snr_s21, snr),
                (_, height_db_s21, _),
                (height_d21, _, _),
            ) = _agg_mean_by_subdet(
                [
                    x,
                    make_masked(det_info["Qr"]),
                    make_masked(det_info["snr"]),
                    make_masked(det_info["height_db_s21"]),
                    make_masked(det_info["height_d21"]),
                ],
                m,
                [m_subdet_d21, m_subdet_s21],
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
                "height_d21": height_d21,
                "height_db_s21": height_db_s21,
                "snr_d21": snr_d21,
                "snr_s21": snr_s21,
                "snr": snr,
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
        _cols_extra = ["snr_d21", "snr_s21", "height_d21", "height_db_s21"]
        detected = ctd.detected = hstack(
            [
                mdl_grouped,
                det_groups[["f", "Qr", "fwhm"] + _cols_extra],
            ],
        )
        detected["bitmask"] = bitmask_det

        # do match to chan and ref
        tbl_chans = tbl_chans_all
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
            for c in _cols_extra:
                matched[c] = det_groups[c][iq]
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
                "dx": s21_f_step / 2,
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
    def fit_baseline_circle(s21_value, s21_unc_value, mask_baseline):
        """Run circle fit to baseline data.

        Parameters
        ----------
        s21_value : ndarray, complex [n_chans, n_steps]
        s21_unc_value : ndarray, complex [n_chans, n_steps]
        mask_baseline : ndarray, bool [n_chans, n_steps]
        """
        n_chans, n_steps = s21_value.shape

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
