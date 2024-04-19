from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.table import QTable
from pydantic import Field
from scipy.signal import peak_widths
from tollan.config.types import ImmutableBaseModel
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit
from tollan.utils.np import strip_unit
from typing_extensions import assert_never


def _peakdetect(y_axis, lookahead, delta):  # noqa: C901
    """Peak detect algorithm from ``avhn/peakdetect``."""
    # adapted from https://github.com/avhn/peakdetect/blob/master/peakdetect/peakdetect.py

    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    length = len(y_axis)

    # if np.isscalar(lookahead):
    #     lookahead = np.full((length,), lookahead, dtype=int)
    # if np.isscalar(delta):
    #     delta = np.full((length,), delta)

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.inf, -np.inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index in range(length):
        y = y_axis[index]
        x = index
        la = lookahead[x]
        if index + la > length:
            continue
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        # look for max
        if y < mx - delta[x] and mx != np.inf:  # noqa: SIM102
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + la].max() < mx:
                # max_peaks.append([mxpos, mx])
                max_peaks.append(mxpos)
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.inf
                mn = np.inf
                # if index + lookahead >= length:
                #     # end is within lookahead no more peaks can be found
                #     break

        # look for min
        if y > mn + delta[x] and mn != -np.inf:  # noqa: SIM102
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index : index + la].min() > mn:
                # min_peaks.append([mnpos, mn])
                min_peaks.append(mnpos)
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.inf
                mx = -np.inf
                # if index + lookahead >= length:
                #     # end is within lookahead no more peaks can be found
                #     break

    # Remove the false hit on the first value of the y_axis
    if dump:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
    else:
        # no peaks were found, should the function return empty lists?
        pass
    return np.array(max_peaks, dtype=int), np.array(min_peaks, dtype=int)


Peaks1DMethod = Literal["peakdetect",]


class Peaks1D(ImmutableBaseModel):
    """Find peaks in 1d data."""

    method: Peaks1DMethod = Field(
        default="peakdetect",
        description="The detection method.",
    )
    threshold: float = Field(
        default=2,
        description="The number of sigmas to use for accepting peaks.",
    )
    peakdetect_delta_threshold: float = Field(
        default=2,
        description="The number of sigmas to use for peakdetect delta parameter.",
    )

    def __call__(  # noqa: PLR0913
        self,
        x,
        y,
        ey=None,
        fwhm=None,
        chunks=None,
        postproc_hook=None,
    ):
        """Run the detection."""
        x, y, ey = self._check_xy(x, y, ey)
        logger.debug(
            f"peaks 1d on data shape {y.shape} "
            f"config:\n{pformat_yaml(self.model_dump())}",
        )
        method = self.method
        if method == "peakdetect":
            result = self._peakdetect(
                x,
                y,
                ey=ey,
                fwhm=fwhm,
                chunks=chunks,
            )
        else:
            assert_never()
        if result.peaks is None:
            logger.debug(f"{method} found no peaks")
        else:
            if postproc_hook is not None:
                result = postproc_hook(result)
            peaks = result.peaks
            logger.debug(f"{method} found {len(peaks)} peaks:\n{peaks}")
        return result

    @timeit
    def _peakdetect(  # noqa: PLR0915, C901, PLR0912, PLR0913
        self,
        x,
        y,
        ey=None,
        fwhm=None,
        chunks=None,
    ):
        ny = len(y)
        # use fwhm info to infer lookahead
        lookahead_min = min(ny // 10, 10)
        lookahead_max = ny // 2
        if fwhm is None:
            lookahead = min(ny // 10, 50)
            logger.debug(f"use default {lookahead=}")
        else:
            if isinstance(fwhm, u.Quantity):
                fwhm = fwhm.to_value(u.dimensionless_unscaled)
            lookahead = (fwhm * 0.5).astype(int)
            logger.debug(f"{lookahead=}")
            if np.any(lookahead < lookahead_min):
                logger.debug(f"some lookahead value clipped, use {lookahead_min=}")
                lookahead[lookahead < lookahead_min] = lookahead_min
            if np.any(lookahead > lookahead_max):
                logger.debug(f"some lookahead value clipped, use {lookahead_max=}")
                lookahead[lookahead > lookahead_max] = lookahead_max
        delta = 0.0 if ey is None else ey * self.peakdetect_delta_threshold
        delta_med = np.median(delta)
        if chunks is None:
            chunks = [slice(0, ny)]
        logger.debug(
            f"peakdetect with {lookahead=} {delta_med=} n_chunks={len(chunks)}",
        )

        x_value, x_unit = strip_unit(x)
        y_value, y_unit = strip_unit(y)
        if ey is not None:
            ey_value, _ = strip_unit(ey)
        else:
            ey_value = None
        delta_value, delta_unit = strip_unit(delta)

        labx = np.full(ny, 0, dtype=int)

        if np.isscalar(lookahead):
            lookahead = np.full((ny,), lookahead, dtype=int)
        if np.isscalar(delta_value):
            delta_value = np.full((ny,), delta_value)

        chunk_peaks = []
        count = 1  # label id
        for chunk in chunks:
            if isinstance(chunk, slice):
                c0 = chunk.start
                nc = chunk.stop - chunk.start
            else:
                c0 = chunk[0]
                nc = chunk[-1] + 1 - chunk[0]
            max_peaks, min_peaks = _peakdetect(
                y_axis=y_value[chunk],
                lookahead=lookahead[chunk],
                delta=delta_value[chunk],
            )
            chunk_peaks.append((c0, nc, np.r_[max_peaks]))
            # populate labels
            idx_valleys = np.r_[0, min_peaks, nc - 1]
            for i in range(idx_valleys.shape[0] - 1):
                if idx_valleys[i] != idx_valleys[i + 1]:
                    labx[c0 + idx_valleys[i] : c0 + idx_valleys[i + 1] + 1] = count
                    count += 1
            # logger.debug(f"found {len(max_peaks)} peaks in chunk {ci}")
        n_labels = count
        logger.debug(f"found total {n_labels=}")
        # calculate peak properties
        peak_info = []
        for ic, (ic_offset, chunk_size, idx_peaks) in enumerate(chunk_peaks):
            for ip_chunk in idx_peaks:
                ip = ip_chunk + ic_offset
                d = {
                    "idx": ip,
                    "idx_chunk": ic,
                    "idx_peak": ip_chunk,
                    "idx_chunk_offset": ic_offset,
                    "chunk_size": chunk_size,
                }
                ll = d["label"] = labx[ip]
                lm = labx == ll
                il = np.where(lm)[0]
                d["size"] = lm.sum()
                il0 = d["idx_label_left"] = il[0]
                il1 = d["idx_label_right"] = il[-1]
                d["x"] = x_value[ip]
                d["x_left"] = x_value[il0]
                d["x_right"] = x_value[il1]
                d["x_range"] = x_value[il1] - x_value[il0]
                yp = d["y"] = y_value[ip]
                yb = d["base"] = 0.5 * (y_value[il0] + y_value[il1])
                yh = d["height"] = yp - yb
                # number of samples above HM
                d["halfmax_size"] = (y_value[lm] >= yh * 0.5 + yb).sum()
                if ey_value is not None:
                    d["ey"] = np.mean(ey_value[lm])
                else:
                    d["ey"] = 0

                peak_info.append(d)

        if not peak_info:
            peak_info = None
        else:
            peak_info = QTable(rows=peak_info)
            # reject peaks with halfmax_size = 1
            m_hm_small = peak_info["halfmax_size"] < 2  # noqa: PLR2004
            n_hm_small = m_hm_small.sum()
            if n_hm_small > 0:
                logger.info(f"reject {n_hm_small} spikes.")
                # reset label
                for ll in peak_info["label"][m_hm_small]:
                    labx[labx == ll] = 0
                peak_info = peak_info[~m_hm_small]

            peak_width, peak_whs, left_ips, right_ips = peak_widths(
                y_value,
                peak_info["idx"],
                rel_height=0.5,
                prominence_data=(
                    peak_info["height"],
                    peak_info["idx_label_left"],
                    peak_info["idx_label_right"],
                ),
            )

            peak_info["lookahead"] = lookahead[peak_info["idx"]]
            peak_info["width_size"] = peak_width
            peak_info["width_height"] = peak_whs
            peak_info["idx_width_left"] = left_ips
            peak_info["idx_width_right"] = right_ips
            # idx_left = np.floor(left_ips).astype(int)
            # idx_right = np.ceil(right_ips).astype(int)
            peak_info["width_left"] = np.interp(left_ips, np.arange(ny), x_value)
            peak_info["width_right"] = np.interp(right_ips, np.arange(ny), x_value)
            peak_info["width"] = peak_info["width_right"] - peak_info["width_left"]
            peak_info["snr"] = peak_info["y"] / peak_info["ey"]
            peak_info["snr_height"] = peak_info["height"] / peak_info["ey"]
            peak_info["mask"] = peak_info["snr"] >= self.threshold
            # attach back units
            for c in [
                "x",
                "x_left",
                "x_right",
                "x_range",
                "width",
                "width_left",
                "width_right",
            ]:
                peak_info[c].unit = x_unit
            for c in ["height", "width_height", "ey", "y", "base"]:
                peak_info[c].unit = y_unit

        return Peaks1DResult(
            config=self,
            x=x,
            y=y,
            ey=ey,
            lookahead=lookahead,
            delta=delta,
            labels=labx,
            peaks=peak_info,
        )

    @staticmethod
    def _check_xy(x, y, ey):
        if not hasattr(x, "shape") or not hasattr(y, "shape"):
            raise ValueError("unknown data shape.")
        if len(x.shape) != len(y.shape):
            raise ValueError("mismatch data shape between x and y")
        if len(x.shape) != 1:
            raise ValueError("data has to be 1-d")
        if ey is not None:
            if not hasattr(ey, "shape"):
                raise ValueError("unknown error data shape.")
            if ey.shape != x.shape:
                raise ValueError("mismatch error shape.")
        return x, y, ey


@dataclass(kw_only=True)
class Peaks1DResult:
    """Result from Peaks1D."""

    config: Peaks1D = ...
    x: npt.NDArray = ...
    y: npt.NDArray = ...
    ey: None | npt.NDArray = None
    lookahead: None | npt.NDArray = None
    delta: None | npt.NDArray = None
    labels: npt.NDArray = ...
    peaks: QTable = ...

    def make_mask(self, peaks_select, n_fwhms):
        """Return a mask built from selected peaks."""
        ny = len(self.y)
        pks = self.peaks[peaks_select]
        mask = np.zeros((ny,), dtype=bool)
        w0 = pks["idx_width_left"]
        w = pks["width_size"]
        w1 = pks["idx_width_right"]
        m0 = np.floor(w0 - (n_fwhms - 1) * 0.5 * w).astype(int)
        m1 = np.ceil(w1 + (n_fwhms - 1) * 0.5 * w).astype(int)
        l0 = pks["idx_label_left"]
        l1 = pks["idx_label_right"]
        m0[m0 < l0] = l0[m0 < l0]
        m1[m1 > l1] = l1[m1 > l1]
        for i in range(len(pks)):
            mask[m0[i] : m1[i] + 1] = True
        return mask
