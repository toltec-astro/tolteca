"""Calibrated native-window and slow-trend inspection of complete integrations."""

from fractions import Fraction
from math import ceil

import astropy.units as u
import numpy as np
import xarray as xr
from pydantic import Field, model_validator
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly
from tollan.pipeline import Step, StepConfig, StepContext

from tolteca_kidsproc.analysis import PSDAnalysis

from .timestream import TimestreamView

_MIN_PSD_SAMPLES = 2


class TimestreamInspectConfig(StepConfig):
    """Sampling and spectral policy for a calibrated timestream block.

    Parameters
    ----------
    window_seconds, interval_seconds : float, optional
        Retain the first 10 seconds of every 300-second interval by default.
        Windows never cross a block boundary; a partial final window is retained.
    trend_rate_hz : float, optional
        Uniform filtered trend rate, default 1 Hz. Must be below the native rate.
    read_samples, channel_batch : int, optional
        Target input chunk and channel batch, defaults 32768 and 16. Rational
        resampling aligns chunks to its denominator and adds the FIR halo.
    native_psd, trend_psd : PSDAnalysis, optional
        Independent Welch settings for each native window and the continuous
        trend. Defaults use 512 and 256 samples respectively. Gapped native
        windows are never concatenated for a low-frequency spectrum.
    """

    window_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    interval_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    trend_rate_hz: float = Field(default=1, gt=0, allow_inf_nan=False)
    read_samples: int = Field(default=32768, ge=2)
    channel_batch: int = Field(default=16, ge=1)
    native_psd: PSDAnalysis = Field(default_factory=lambda: PSDAnalysis(nperseg=512))
    trend_psd: PSDAnalysis = Field(default_factory=lambda: PSDAnalysis(nperseg=256))

    @model_validator(mode="after")
    def _validate_windows(self):
        """Keep periodic inspection windows disjoint."""
        if self.window_seconds > self.interval_seconds:
            msg = "Inspection window must not exceed its interval"
            raise ValueError(msg)
        return self


def _resample(values, *, up, down):
    """Polyphase FIR with invalid-data support retained as NaNs."""
    finite = np.isfinite(values)
    result = resample_poly(np.where(finite, values, 0), up, down, axis=-1)
    radius = ceil(10 * max(up, down) / up)
    contaminated = maximum_filter1d(
        ~finite, size=2 * radius + 1, axis=-1, mode="constant", cval=0
    )
    positions = np.minimum(
        (np.arange(result.shape[-1]) * down // up), values.shape[-1] - 1
    )
    result[np.take(contaminated, positions, axis=-1)] = np.nan
    return result


def filtered_trend(signals, config, *, fs):
    """Return a lazy uniform trend with bounded, overlapping input reads.

    Parameters
    ----------
    signals : xarray.Dataset
        Channel/time arrays with units and elapsed-seconds coordinates.
    config : TimestreamInspectConfig
        Resampling and read-size settings.
    fs : float
        Recorded uniform input sampling frequency in Hz.

    Returns
    -------
    xarray.Dataset
        Zero-phase Kaiser-5 polyphase FIR result. Edge-support samples are NaN;
        invalid input contaminates its complete FIR support. Effective rate and
        filter identity are saved for reproducibility.
    """
    if config.trend_rate_hz >= fs:
        msg = "Trend rate must be lower than the native sample rate"
        raise ValueError(msg)
    ratio = (Fraction(str(config.trend_rate_hz)) / Fraction(str(fs))).limit_denominator(
        100000
    )
    up, down = ratio.numerator, ratio.denominator
    actual = fs * up / down
    if not np.isclose(actual, config.trend_rate_hz, rtol=1e-9):
        msg = (
            "Requested trend rate cannot be represented within 1e-9 relative tolerance"
        )
        raise ValueError(msg)
    n = signals.sizes["time"]
    radius = ceil(10 * down / up)
    halo = ceil(radius / down) * down
    width = max(down, config.read_samples // down * down)
    pieces = []
    for start in range(0, n, width):
        stop = min(start + width, n)
        left, right = max(0, start - halo), min(n, stop + halo)
        count = ceil((right - left) * up / down)
        source = signals.isel(time=slice(left, right)).drop_vars(
            "sample", errors="ignore"
        )
        arrays = {}
        for name, signal in source.data_vars.items():
            arrays[name] = xr.apply_ufunc(
                _resample,
                signal.chunk({"chan": config.channel_batch, "time": -1}),
                input_core_dims=[["time"]],
                output_core_dims=[["trend_time"]],
                kwargs={"up": up, "down": down},
                dask="parallelized",
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"trend_time": count}},
                keep_attrs=True,
            )
        offset = (start - left) * up // down
        size = ceil(stop * up / down) - start * up // down
        piece = xr.Dataset(arrays).isel(trend_time=slice(offset, offset + size))
        times = (np.arange(size) + start * up // down) / actual
        pieces.append(piece.assign_coords(trend_time=times))
    trend = xr.concat(pieces, dim="trend_time").rename(trend_time="time")
    support = (trend.time >= radius / fs) & (trend.time <= (n - 1 - radius) / fs)
    trend = trend.where(support)
    trend.time.attrs["units"] = "s"
    trend.attrs.update(
        f_smp=actual,
        filter="scipy.signal.resample_poly; Kaiser beta=5; zero padding",
        up=up,
        down=down,
        edge_seconds=radius / fs,
    )
    return trend


class TimestreamInspectContext(
    StepContext["TimestreamInspect", TimestreamInspectConfig]
):
    """Execution record for the configured inspection policy."""


class TimestreamInspect(Step[TimestreamInspectConfig, TimestreamInspectContext]):
    """Attach inspection products after calibrated solving, before publication.

    Native I/Q and calibrated r/x are required. The owning workflow prepares the
    exact calibration. This step does not discover, fit or choose a model, and
    the viewer never invokes it. Output remains lazy until DataProdOutput writes
    the tree while the acquisition is open.
    """

    @classmethod
    def result_path(cls, *, alias=None):
        """Return the block-relative inspection namespace."""
        return "timestream_inspect" if alias is None else f"timestream_inspect_{alias}"

    @classmethod
    def run(cls, data, context):
        """Retain periodic samples/PSDs and the full-duration filtered trend."""
        if "solved" not in data.children or not data["solved"].attrs.get("calibration"):
            msg = (
                "TimestreamInspect requires a solved "
                "block with its exact calibration binding"
            )
            raise ValueError(msg)
        cfg = context.config
        view = TimestreamView(data)
        signals = view.data_source
        if any(name not in signals for name in ("I", "Q", "r", "x")):
            msg = "TimestreamInspect requires calibrated r/x and raw I/Q"
            raise ValueError(msg)
        signals = signals[["I", "Q", "r", "x"]]
        fs = float(data.toltec_kids.block.meta.f_smp.to_value(u.Hz))
        n = signals.sizes["time"]
        out = xr.DataTree(
            xr.Dataset(
                attrs={
                    "schema": "timestream_inspect/1",
                    "config": cfg.model_dump(mode="json"),
                    "calibration": data["solved"].attrs["calibration"],
                    "native_rate_hz": fs,
                }
            )
        )
        _native_windows(out, signals, data["solved"].dataset, cfg, fs=fs)
        trend = filtered_trend(signals, cfg, fs=fs)
        out["trend"] = xr.DataTree(trend)
        # Exclude FIR edges; never concatenate valid islands across a data gap.
        edge = trend.attrs["edge_seconds"]
        interior = trend.sel(time=slice(edge, (n - 1) / fs - edge))
        if interior.sizes["time"] >= _MIN_PSD_SAMPLES:
            out["trend_psd"] = xr.DataTree(
                cfg.trend_psd.make_psd(
                    interior, f_smp=trend.attrs["f_smp"] * u.Hz, dim="time"
                )
            )
        for name in ("calibration_flags", "fit_flag"):
            if name in data["solved"].dataset:
                out[name] = data["solved"].dataset[name]
        data[cls.result_path(alias=cfg.alias)] = out


def _native_windows(out, signals, solved, cfg, *, fs):
    """Save disjoint native windows and retain their per-sample solver flags."""
    n = signals.sizes["time"]
    length = max(2, round(cfg.window_seconds * fs))
    spectra = []
    for i in range(ceil(n / (cfg.interval_seconds * fs))):
        start = round(i * cfg.interval_seconds * fs)
        if start >= n:
            break
        stop = min(n, start + length)
        window = signals.isel(time=slice(start, stop))
        out[f"windows/{i}"] = xr.DataTree(window)
        out[f"windows/{i}"].attrs.update(sample_start=start, sample_stop=stop)
        if "flags" in solved:
            flags = (
                solved["flags"].isel(sample=slice(start, stop)).rename(sample="time")
            )
            # Native files label samples; inspection labels elapsed seconds.
            flags = flags.assign_coords(time=window.time, chan=window.chan)
            out[f"windows/{i}"]["flags"] = flags
        if stop - start >= _MIN_PSD_SAMPLES:
            psd = cfg.native_psd.make_psd(
                window.drop_vars("sample", errors="ignore"),
                f_smp=fs * u.Hz,
                dim="time",
            )
            spectra.append(psd.expand_dims(window=[start / fs]))
    if spectra:
        out["window_psd"] = xr.DataTree(xr.concat(spectra, dim="window", join="outer"))
