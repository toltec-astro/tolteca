"""TolTEC KIDs end-to-end reduction pipeline for a single zarr interface store.

Runs :class:`~tolteca_kids.SweepCheck` followed by
:class:`~tolteca_kids.KidsFind` on a zarr store produced by
``tolteca_db.export.ZarrExporter`` and writes the results as new zarr
groups in the same store::

    sweep_check/
        bitmask_chan   int32  (chan,)  — SweepBitMask per channel
        mask_chan_bad  bool   (chan,)  — True = channel is bad
    kids_find/
        f_tone_hz      float64 (det,) — detected tone offsets from LO centre, Hz
        Qr             float64 (det,) — resonator Q_r
        bitmask_det    int32   (det,) — SegmentBitMask per detection
        .zattrs["meta"] — JSON string: n_detected, n_bad_chans,
                          tolteca_kids_version, config_hash

Usage::

    from tolteca_kids.pipeline import KidsPipeline

    result = KidsPipeline().run("/path/to/tcs-153522-0-1/nw0.zarr")
    print(result.n_detected, result.n_bad_chans, result.quality_score)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
import xarray as xr
import zarr as _zarr
from pydantic import Field
from tollan.config import FrozenBaseModel

from tolteca_datamodels.toltec.kids import ReducedSweepView
from tolteca_datamodels.toltec.kids.zarr import open_zarr_dataset

try:
    from tolteca._version import __version__ as _PKG_VERSION  # type: ignore[attr-defined]
except Exception:
    _PKG_VERSION = "unknown"

from .kids_find import KidsFind, KidsFindConfig
from .sweep_check import SweepCheck, SweepCheckConfig

__all__ = [
    "KidsPipeline",
    "KidsPipelineConfig",
    "KidsPipelineResult",
    "SWEEP_CHECK_GROUP",
    "KIDS_FIND_GROUP",
    "has_kids_reduction",
    "read_sweep_check",
    "read_kids_find",
]

logger = logging.getLogger(__name__)

SWEEP_CHECK_GROUP = "sweep_check"
KIDS_FIND_GROUP = "kids_find"

# ── Version helper ─────────────────────────────────────────────────────────────

try:
    _TOLTECA_KIDS_VERSION: str = _PKG_VERSION
except Exception:
    _TOLTECA_KIDS_VERSION = "unknown"


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class KidsPipelineResult:
    """Summary of one KidsPipeline.run() call.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr store that was processed.
    n_chans : int
        Total number of detector channels.
    n_bad_chans : int
        Channels flagged bad by SweepCheck.
    n_detected : int
        Resonators detected by KidsFind.
    quality_score : float
        Simple quality metric in [0, 1]: fraction of good channels with a
        detection.  Higher is better.
    """

    zarr_path: Path
    n_chans: int
    n_bad_chans: int
    n_detected: int
    quality_score: float
    skipped: bool = False
    skip_reason: str = ""

    @property
    def n_good_chans(self) -> int:
        """Channels not flagged bad."""
        return self.n_chans - self.n_bad_chans


# ── Zarr-reader helpers ────────────────────────────────────────────────────────


def has_kids_reduction(zarr_path: str | Path) -> bool:
    """Return True if the zarr store already has a ``kids_find`` group."""
    import zarr as _zarr

    try:
        root = _zarr.open_group(str(zarr_path), mode="r")
        return KIDS_FIND_GROUP in root
    except Exception:
        return False


@dataclass
class SweepCheckZarrData:
    """Data read from the ``sweep_check/`` zarr group."""

    bitmask_chan: np.ndarray  # int32, (chan,)
    mask_chan_bad: np.ndarray  # bool, (chan,)


@dataclass
class KidsFindZarrData:
    """Data read from the ``kids_find/`` zarr group."""

    f_tone_hz: np.ndarray  # float64, (det,) tone offsets from LO centre, Hz
    Qr: np.ndarray  # float64, (det,)
    bitmask_det: np.ndarray  # int32, (det,)
    meta: dict = field(default_factory=dict)


def read_sweep_check(zarr_path: str | Path) -> SweepCheckZarrData | None:
    """Read ``sweep_check/`` group from a kids-reduced zarr store.

    Returns ``None`` if the group is absent.
    """
    try:
        ds = xr.open_zarr(str(zarr_path), group=SWEEP_CHECK_GROUP)
        return SweepCheckZarrData(
            bitmask_chan=ds["bitmask_chan"].values.astype(np.int32),
            mask_chan_bad=ds["mask_chan_bad"].values.astype(bool),
        )
    except Exception:
        return None


def read_kids_find(zarr_path: str | Path) -> KidsFindZarrData | None:
    """Read ``kids_find/`` group from a kids-reduced zarr store.

    Returns ``None`` if the group is absent.
    """
    try:
        ds = xr.open_zarr(str(zarr_path), group=KIDS_FIND_GROUP)
        meta_str = ds.attrs.get("meta", "{}")
        try:
            meta = json.loads(meta_str)
        except Exception:
            meta = {}
        return KidsFindZarrData(
            f_tone_hz=ds["f_tone_hz"].values.astype(np.float64),
            Qr=ds["Qr"].values.astype(np.float64),
            bitmask_det=ds["bitmask_det"].values.astype(np.int32),
            meta=meta,
        )
    except Exception:
        return None


# ── DataTree builder from zarr format ─────────────────────────────────────────

_SWEEP_NS = "tolteca_datamodels.toltec.kids.sweep"


def _build_datatree_from_zarr(ds: xr.Dataset) -> xr.DataTree:
    """Build a DataTree compatible with ReducedSweepView from a zarr dataset.

    The zarr schema (``I``, ``Q``, ``tone_freq``, ``lo_freq``,
    ``lo_center_freq_hz``) is remapped to the namespaced reduced-sweep schema
    that :class:`~tolteca_datamodels.toltec.kids.ReducedSweepView` expects.

    Mapping:
    - ``ds.I(chan, sample)``          → ``{ns}.I(chan, sweep)``
    - ``ds.Q(chan, sample)``          → ``{ns}.Q(chan, sweep)``
    - ``ds.coords.lo_freq(sample)``   → coord ``sweep`` (offset from LO center)
    - ``ds.tone_freq(chan)``          → coord ``f_lo`` (abs tone freq per chan)
    - ``ds.attrs.lo_center_freq_hz`` → used for conversions (stored as attr)
    """
    lo_center: float = float(ds.attrs["lo_center_freq_hz"])
    tone_freq: np.ndarray = ds["tone_freq"].values  # (chan,), Hz offsets
    lo_freq: np.ndarray = ds.coords["lo_freq"].values  # (sample,), abs Hz

    n_chan = ds.sizes["chan"]
    n_sample = ds.sizes["sample"]

    # f_lo: absolute tone frequency per channel (centre LO + tone offset)
    f_lo = lo_center + tone_freq  # (chan,) Hz

    I_vals = ds["I"].values  # (chan, sample)
    Q_vals = ds["Q"].values  # (chan, sample)

    # Average repeated samples at the same LO frequency into one value per step.
    # The raw zarr may have multiple samples per LO step (e.g. 10 repeats);
    # the sweep algorithms require exactly one sample per unique LO step.
    unique_lo, inv_idx = np.unique(lo_freq, return_inverse=True)
    n_steps = len(unique_lo)
    if n_steps < 2:
        raise ValueError(
            f"degenerate sweep: only {n_steps} unique LO frequency step(s) "
            f"in {n_sample} samples — cannot run sweep analysis"
        )
    if n_steps < n_sample:
        logger.debug(
            "_build_datatree_from_zarr: averaging %d samples into %d steps",
            n_sample,
            n_steps,
        )
        I_reduced = np.zeros((n_chan, n_steps), dtype=I_vals.dtype)
        Q_reduced = np.zeros((n_chan, n_steps), dtype=Q_vals.dtype)
        counts = np.bincount(inv_idx)
        np.add.at(I_reduced, (slice(None), inv_idx), I_vals)
        np.add.at(Q_reduced, (slice(None), inv_idx), Q_vals)
        I_reduced /= counts[np.newaxis, :]
        Q_reduced /= counts[np.newaxis, :]
    else:
        I_reduced = I_vals
        Q_reduced = Q_vals
        unique_lo = lo_freq

    # sweep: LO step offset from LO centre per unique step
    sweep = unique_lo - lo_center  # (n_steps,) Hz

    # Build the reduced-sweep child dataset with namespaced variable names
    ds_reduced = xr.Dataset(
        {
            f"{_SWEEP_NS}.I": xr.DataArray(I_reduced, dims=["chan", "sweep"]),
            f"{_SWEEP_NS}.Q": xr.DataArray(Q_reduced, dims=["chan", "sweep"]),
        },
        coords={
            "chan": np.arange(n_chan),
            "sweep": sweep,  # Hz offsets from LO centre
            "f_lo": xr.DataArray(f_lo, dims=["chan"]),  # abs tone freq per chan
        },
        attrs={"lo_center_freq_hz": lo_center},
    )

    # Build the root dataset (pass-through of raw zarr data at root)
    ds_root = xr.Dataset(
        {
            "I": ds["I"].rename({"sample": "sample"}),
            "Q": ds["Q"].rename({"sample": "sample"}),
            "tone_freq": ds["tone_freq"],
        },
        coords={"lo_freq": ds.coords["lo_freq"]},
        attrs=ds.attrs,
    )
    _ = n_sample  # suppress unused variable warning

    dt = xr.DataTree.from_dict(
        {
            "/": ds_root,
            _SWEEP_NS: ds_reduced,
        }
    )
    return dt


# ── Config ─────────────────────────────────────────────────────────────────────


class KidsPipelineConfig(FrozenBaseModel):
    """Configuration for the KIDs reduction pipeline.

    Parameters
    ----------
    sweep_check : SweepCheckConfig
        Configuration for the SweepCheck step.
    kids_find : KidsFindConfig
        Configuration for the KidsFind step.
    skip_if_done : bool
        If True and the zarr already has a ``kids_find`` group, return a
        skipped result without re-running.  Default True.
    """

    sweep_check: SweepCheckConfig = Field(default_factory=SweepCheckConfig)
    kids_find: KidsFindConfig = Field(default_factory=KidsFindConfig)
    skip_if_done: bool = Field(default=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────


class KidsPipeline:
    """End-to-end KIDs reduction pipeline for one zarr interface store.

    Runs :class:`~tolteca_kids.SweepCheck` then
    :class:`~tolteca_kids.KidsFind` and writes results into the zarr store.

    Parameters
    ----------
    config : KidsPipelineConfig | None
        Pipeline configuration.  Uses defaults if None.

    Examples
    --------
    Basic use with default configuration:

    >>> pipeline = KidsPipeline()
    >>> result = pipeline.run("/path/to/tcs-153522-0-1/nw0.zarr")
    >>> print(result.n_detected, result.n_bad_chans)

    Custom configuration:

    >>> from tolteca_kids import SweepCheckConfig, KidsFindConfig
    >>> cfg = KidsPipelineConfig(
    ...     sweep_check=SweepCheckConfig(despike=True),
    ...     skip_if_done=False,
    ... )
    >>> pipeline = KidsPipeline(cfg)
    >>> result = pipeline.run("/path/to/nw0.zarr")
    """

    def __init__(self, config: KidsPipelineConfig | None = None) -> None:
        self._config = config or KidsPipelineConfig()

    @property
    def config(self) -> KidsPipelineConfig:
        """Pipeline configuration."""
        return self._config

    def _config_hash(self) -> str:
        """Short hash of the pipeline config for result metadata."""
        cfg_json = self._config.model_dump_json().encode()
        return hashlib.sha256(cfg_json).hexdigest()[:8]

    def run(self, zarr_path: str | Path) -> KidsPipelineResult:
        """Run SweepCheck + KidsFind on one zarr interface store.

        Parameters
        ----------
        zarr_path : str | Path
            Path to a ``dp_reduced_obs`` zarr store (one interface / one nw).
            Results are written in-place as ``sweep_check/`` and ``kids_find/``
            groups.

        Returns
        -------
        KidsPipelineResult
            Summary: n_chans, n_bad_chans, n_detected, quality_score.
        """
        zarr_path = Path(zarr_path)
        cfg = self._config

        if cfg.skip_if_done and has_kids_reduction(zarr_path):
            logger.debug("Kids reduction already present, skipping: %s", zarr_path)
            # Read existing results for the summary
            sc = read_sweep_check(zarr_path)
            kf = read_kids_find(zarr_path)
            n_chans = int(sc.bitmask_chan.shape[0]) if sc is not None else 0
            n_bad = int(sc.mask_chan_bad.sum()) if sc is not None else 0
            n_det = int(kf.f_tone_hz.shape[0]) if kf is not None else 0
            q = _quality_score(n_chans, n_bad, n_det)
            return KidsPipelineResult(
                zarr_path=zarr_path,
                n_chans=n_chans,
                n_bad_chans=n_bad,
                n_detected=n_det,
                quality_score=q,
                skipped=True,
                skip_reason="already_done",
            )

        logger.info("Running KidsPipeline on: %s", zarr_path)

        # ── 1. Load zarr ─────────────────────────────────────────────────────
        ds = open_zarr_dataset(zarr_path)
        lo_center: float = float(ds.attrs["lo_center_freq_hz"])

        # ── 2. Build DataTree compatible with ReducedSweepView ───────────────
        dt = _build_datatree_from_zarr(ds)

        # Verify the view can be constructed
        view = ReducedSweepView(dt)
        n_chans: int = dt[_SWEEP_NS].dataset.sizes["chan"]

        logger.debug(
            "Loaded %d channels, %d steps from %s",
            n_chans,
            dt[_SWEEP_NS].dataset.sizes["sweep"],
            zarr_path,
        )

        # ── 3. Run SweepCheck ────────────────────────────────────────────────
        sweep_check_step = SweepCheck(cfg.sweep_check)
        sweep_check_step(dt)
        sc_ctx = SweepCheck.get_context(dt)
        bitmask_chan: np.ndarray = sc_ctx.data.bitmask_chan.astype(np.int32)
        mask_chan_bad: np.ndarray = sc_ctx.data.mask_chan_bad.astype(bool)
        n_bad_chans: int = int(mask_chan_bad.sum())

        logger.debug(
            "SweepCheck: %d/%d bad channels", n_bad_chans, n_chans
        )

        # ── 4. Run KidsFind ──────────────────────────────────────────────────
        kids_find_step = KidsFind(cfg.kids_find)
        kids_find_step(dt)
        kf_ctx = KidsFind.get_context(dt)
        detected = kf_ctx.data.detected  # QTable

        n_detected: int = len(detected) if detected is not None else 0

        if n_detected > 0:
            f_abs_hz = detected["f"].to_value(u.Hz)  # absolute Hz
            f_tone_hz = f_abs_hz - lo_center  # offset from LO centre
            Qr_arr = detected["Qr"].astype(float)
            bitmask_det = np.asarray(detected["bitmask"], dtype=np.int32)
        else:
            f_tone_hz = np.empty(0, dtype=np.float64)
            Qr_arr = np.empty(0, dtype=np.float64)
            bitmask_det = np.empty(0, dtype=np.int32)

        logger.debug("KidsFind: %d detections", n_detected)

        # ── 5. Compute quality score ─────────────────────────────────────────
        quality_score = _quality_score(n_chans, n_bad_chans, n_detected)

        # ── 6. Write sweep_check group ───────────────────────────────────────
        ds_sc = xr.Dataset(
            {
                "bitmask_chan": xr.DataArray(bitmask_chan, dims=["chan"]),
                "mask_chan_bad": xr.DataArray(mask_chan_bad, dims=["chan"]),
            },
            coords={"chan": np.arange(n_chans)},
        )
        # Delete any partial/stale groups from a previous run before writing.
        # Use filesystem removal (shutil.rmtree) rather than the zarr API so
        # that old-schema directory trees are fully removed even when the zarr
        # group structure has changed between pipeline versions.
        for _grp in (SWEEP_CHECK_GROUP, KIDS_FIND_GROUP):
            _grp_path = Path(zarr_path) / _grp
            if _grp_path.exists():
                shutil.rmtree(_grp_path)
                logger.debug("Removed stale %s directory from %s", _grp, zarr_path)
        ds_sc.to_zarr(str(zarr_path), group=SWEEP_CHECK_GROUP, mode="a")
        logger.debug("Wrote %s group to %s", SWEEP_CHECK_GROUP, zarr_path)

        # ── 7. Write kids_find group ─────────────────────────────────────────
        n_det = len(f_tone_hz)
        det_dim: list[int] = list(range(n_det))
        ds_kf = xr.Dataset(
            {
                "f_tone_hz": xr.DataArray(f_tone_hz, dims=["det"]),
                "Qr": xr.DataArray(Qr_arr, dims=["det"]),
                "bitmask_det": xr.DataArray(bitmask_det, dims=["det"]),
            },
            coords={"det": det_dim},
            attrs={
                "meta": json.dumps(
                    {
                        "n_detected": n_detected,
                        "n_bad_chans": n_bad_chans,
                        "n_chans": n_chans,
                        "quality_score": quality_score,
                        "tolteca_kids_version": _TOLTECA_KIDS_VERSION,
                        "config_hash": self._config_hash(),
                    }
                )
            },
        )
        ds_kf.to_zarr(str(zarr_path), group=KIDS_FIND_GROUP, mode="a")
        logger.debug("Wrote %s group to %s", KIDS_FIND_GROUP, zarr_path)

        del view  # suppress "unused" warning; it validated the DataTree

        return KidsPipelineResult(
            zarr_path=zarr_path,
            n_chans=n_chans,
            n_bad_chans=n_bad_chans,
            n_detected=n_detected,
            quality_score=quality_score,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _quality_score(n_chans: int, n_bad: int, n_detected: int) -> float:
    """Fraction of good channels with a detection, clamped to [0, 1]."""
    n_good = max(n_chans - n_bad, 0)
    if n_good == 0:
        return 0.0
    return float(min(n_detected / n_good, 1.0))
