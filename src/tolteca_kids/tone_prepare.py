"""Prepare the v2 Tlaloc tone handoff from matched KIDs detections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.table import QTable, vstack
from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel
from tollan.config.types import AbsFilePath, FrequencyQuantityField
from tollan.pipeline import (
    ArtifactContext,
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
)

from .kids_find import KidsFind
from .sweep import ReducedSweepView
from .tlaloc_output import (
    TlalocOutputConfig,
    TlalocToneArtifact,
    TlalocTones,
    fill_tone_amplitudes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata


class TonePhasesRandom(FrozenBaseModel):
    """Random phases in radians; a seed makes the prepared result repeatable."""

    method: Literal["random"]
    seed: int | None = None

    def __call__(self, n_chans: int) -> np.ndarray:
        """Generate one random phase per requested tone.

        Parameters
        ----------
        n_chans : int
            Number of output tones.

        Returns
        -------
        numpy.ndarray
            Phases uniformly distributed over [0, 2 pi), in radians.
        """
        return np.random.default_rng(self.seed).random(n_chans) * (2 * np.pi)


class TonePhasesFromFile(FrozenBaseModel):
    """Phases in radians from a headerless single-column file."""

    method: Literal["file"]
    path: AbsFilePath

    def __call__(self, n_chans: int) -> np.ndarray:
        """Read the first n_chans phases from the configured file.

        Parameters
        ----------
        n_chans : int
            Number of output tones.

        Returns
        -------
        numpy.ndarray
            Loaded phases in radians, in file order.

        Raises
        ------
        ValueError
            The file contains fewer phases than requested.
        """
        phases = QTable.read(self.path, format="ascii.no_header")["col1"]
        if len(phases) < n_chans:
            msg = "Too few phases in file"
            raise ValueError(msg)
        return np.asarray(phases[:n_chans])


class PlaceHolderTones(FrozenBaseModel):
    """V2 placeholder placement around the detected frequency range."""

    n: int = Field(default=0, ge=0)
    loc: Literal["inner", "outer"] = "outer"
    sep_f_lo_min: FrequencyQuantityField = 5 * u.MHz
    sep_f_lo_max: FrequencyQuantityField = 500 * u.MHz
    sep_f_det: FrequencyQuantityField = 100 * u.kHz
    f_step: FrequencyQuantityField = 50 * u.kHz

    def __call__(self, f_dets, f_lo):
        """Place placeholder tones around detections using the v2 spacing.

        Parameters
        ----------
        f_dets : astropy.units.Quantity
            Absolute detection frequencies, with detections on both sides of
            the LO when placeholders are requested.
        f_lo : astropy.units.Quantity
            Center LO frequency.

        Returns
        -------
        astropy.units.Quantity
            Absolute placeholder frequencies in Hz; empty when n is zero.
        """
        if self.n == 0:
            return np.array([]) * u.Hz
        negative, positive = f_dets[f_dets < f_lo], f_dets[f_dets >= f_lo]
        n_neg = self.n // 2
        n_pos = self.n - n_neg
        if self.loc == "inner":
            neg0 = np.max(negative) + self.sep_f_det
            neg1 = min(neg0 + (n_neg - 1) * self.f_step, f_lo - self.sep_f_lo_min)
            pos1 = np.min(positive) - self.sep_f_det
            pos0 = max(pos1 - (n_pos - 1) * self.f_step, f_lo + self.sep_f_lo_min)
        else:
            neg1 = np.min(negative) - self.sep_f_det
            neg0 = max(neg1 - (n_neg - 1) * self.f_step, f_lo - self.sep_f_lo_max)
            pos0 = np.max(positive) + self.sep_f_det
            pos1 = min(pos0 + (n_pos - 1) * self.f_step, f_lo + self.sep_f_lo_max)
        return (
            np.r_[
                np.linspace(neg0.to_value(u.Hz), neg1.to_value(u.Hz), n_neg),
                np.linspace(pos0.to_value(u.Hz), pos1.to_value(u.Hz), n_pos),
            ]
            * u.Hz
        )


class TonePrepareArtifacts(ArtifactSpec[ArtifactContext]):
    """Prepared tone files retained for ordinary data-product publication."""

    tones: TlalocToneArtifact = Field(default_factory=TlalocToneArtifact)

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export the retained tone table to the supplied product directory.

        Parameters
        ----------
        data : tollan.pipeline.ArtifactContext
            Prepared tone artifacts with accessible table values.
        path : pathlib.Path
            Product export directory; existing tone files are rejected.

        Returns
        -------
        list of tollan.pipeline.ArtifactFile
            Frequency, amplitude, phase and mask file receipts.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(artifact.data, path=path)
        ]


class TonePrepareConfig(StepConfig):
    """Tone selection and phase settings; preparation does not install files.

    Parameters
    ----------
    enabled : bool, optional
        Enable preparation, default False. Requires a completed KidsFind step.
    n_chans_max : int, optional
        Maximum output tones, default 1000. Excess detections retain highest Qr.
    fix_n_chans : bool or None, optional
        Preserve the input count, trimming by absolute match phase distance or
        filling from missed probes. None uses v2 observing defaults: fixed for
        Tune, variable for standalone VNA and target sweeps.
    tone_phases : TonePhasesRandom or TonePhasesFromFile, optional
        Random phases by default. Phases are assigned before the common row sort.
    placeholders : PlaceHolderTones, optional
        Extra masked tones with zero amplitudes; none by default.
    artifacts : TonePrepareArtifacts, optional
        Frequency, amplitude, phase and mask export specifications.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    n_chans_max: int = Field(default=1000, ge=1, le=1000)
    fix_n_chans: bool | None = None
    tone_phases: Annotated[
        TonePhasesRandom | TonePhasesFromFile, Field(discriminator="method")
    ] = Field(default_factory=lambda: TonePhasesRandom(method="random"))
    placeholders: PlaceHolderTones = Field(default_factory=PlaceHolderTones)
    artifacts: TonePrepareArtifacts = Field(default_factory=TonePrepareArtifacts)


class TonePrepareContext(StepContext["TonePrepare", TonePrepareConfig]):
    """Prepared tones, effective output amplitude policy and artifacts."""

    data: TlalocTones | None = Field(default=None, exclude=True)
    output_config: TlalocOutputConfig = Field(
        default_factory=TlalocOutputConfig,
        description="Effective Tlaloc amplitude policy and LUT source for preparation.",
    )

    def make_step(self) -> TonePrepare:
        """Recreate preparation with its recorded output amplitude policy.

        Returns
        -------
        TonePrepare
            Step configured with the saved tone selection and amplitude policy.
        """
        return self.step_cls(self.config, tlaloc_config=self.output_config)


class TonePrepare(Step[TonePrepareConfig, TonePrepareContext]):
    """Prepare matched output tones for the next Tlaloc load.

    The tlaloc subtree retains the final common row order. Data-product output
    exports its four v2 files; TlalocOutput optionally installs them afterwards.

    Parameters
    ----------
    config : TonePrepareConfig or None, optional
        Tone selection and phase settings. None uses defaults.
    tlaloc_config : TlalocOutputConfig or None, optional
        Output amplitude policy and LUT source. None preserves v2 matching.
        The KIDs workflow supplies kids.tlaloc_output and records this policy
        in the preparation context even when installation is disabled.
    **kwargs
        TonePrepareConfig overrides, following the Step constructor.
    """

    def __init__(
        self,
        config: TonePrepareConfig | None = None,
        *,
        tlaloc_config: TlalocOutputConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            super().__init__(**kwargs)
        else:
            super().__init__(config, **kwargs)
        self._tlaloc_config = tlaloc_config or TlalocOutputConfig()

    def create_context(self, data: xr.DataTree) -> TonePrepareContext:
        """Attach a new invocation record with the configured output policy.

        Parameters
        ----------
        data : xarray.DataTree
            Sweep block receiving the preparation context.

        Returns
        -------
        TonePrepareContext
            Attached context retaining selection and amplitude settings.
        """
        context = super().create_context(data)
        context.output_config = self._tlaloc_config
        return context

    @classmethod
    def run(cls, data: xr.DataTree, context: TonePrepareContext) -> None:
        """Prepare one checked and found native or reopened sweep block.

        Parameters
        ----------
        data : xarray.DataTree
            Block with recorded probe metadata and a completed KidsFind context.
        context : TonePrepareContext
            Selection, phases and artifact settings, and prepared output.

        Returns
        -------
        None
            Installs a loaded tlaloc subtree and retains its typed artifact.

        Raises
        ------
        ValueError
            Finding is incomplete, metadata is missing, or no tones are prepared.
        """
        found = KidsFind.get_context(data)
        if not found.completed:
            msg = "Tone preparation requires completed KIDs finding"
            raise ValueError(msg)
        view = ReducedSweepView(data)
        if view.meta is None:
            msg = "Tone preparation requires acquisition metadata"
            raise ValueError(msg)
        table = prepare_tone_table(
            found.data.detected,
            found.data.chan_matched,
            view.chan_axis_table,
            view.meta,
            context.config,
            tlaloc_config=context.output_config,
            query_matches=found.data.detected_matched,
        )
        data["tlaloc"] = table
        context.data = TlalocTones(table)
        context.artifacts = ArtifactContext()
        context.artifacts.add(context.config.artifacts.tones, context.data)


def prepare_tone_table(  # noqa: PLR0913 -- science inputs and output policies
    detected: QTable,
    chan_matched: QTable,
    probes: QTable,
    meta: ToltecSweepMetadata,
    config: TonePrepareConfig,
    *,
    tlaloc_config: TlalocOutputConfig | None = None,
    query_matches: QTable | None = None,
) -> xr.Dataset:
    """Select and order tones using the v2 output policy.

    Parameters
    ----------
    detected, chan_matched : astropy.table.QTable
        Finding detections and input-probe matches, including d_phi and amp_tone.
    probes : astropy.table.QTable
        Recorded chan, f_chan, tone_amp and tone_mask columns in acquisition order.
    meta : ToltecSweepMetadata
        Acquisition identity, network, observation type and LO frequency in Hz.
    config : TonePrepareConfig
        Explicit tone selection and phase-generation settings.
    tlaloc_config : TlalocOutputConfig or None, optional
        Output amplitude policy and LUT source. None preserves v2 matching.
    query_matches : astropy.table.QTable or None, optional
        One row per detection, in detection order, with idx_chan and connected
        match-group size. Required for explicit fallback in match mode.

    Returns
    -------
    xarray.Dataset
        Loaded v2 report and companion columns in the same FFT order.
    """
    if meta.f_lo_center is None:
        msg = "Tone preparation requires a recorded LO frequency"
        raise ValueError(msg)
    f_lo = meta.f_lo_center * u.Hz
    tones = detected.copy()
    f_dets = tones["f_chan"] = tones["f"] if "f" in tones.colnames else tones["fr"]
    tones["mask_tone"] = True
    output = tlaloc_config or TlalocOutputConfig()
    unique_probe_ids = _amplitude_matches(tones, query_matches, output)
    fixed = config.fix_n_chans
    if fixed is None:
        if meta.obs_type not in {"tune", "vna", "targ"}:
            msg = "Automatic tone count requires a recorded sweep observation type"
            raise ValueError(msg)
        fixed = meta.obs_type == "tune"
    limit = len(probes) if fixed else config.n_chans_max
    if limit > config.n_chans_max:
        msg = "Fixed input tone count exceeds n_chans_max"
        raise ValueError(msg)
    if len(tones) > limit:
        ranking = (
            np.abs(tones["d_phi"]).argsort()
            if fixed
            else np.flip(tones["Qr"].argsort())
        )
        keep = np.zeros(len(tones), dtype=bool)
        keep[ranking[:limit]] = True
        tones = tones[keep]
    tones["amp_tone"], tones["amp_tone_filled"], amp_provenance = fill_tone_amplitudes(
        tones["amp_tone"],
        tones["f_chan"].to_value(u.Hz),
        probes,
        fill_mask=tones["amp_tone_ambiguous"],
        unique_probe_ids=unique_probe_ids,
        roach=meta.roach,
        config=output,
    )
    placeholders = config.placeholders(f_dets, f_lo)
    if fixed and len(tones) < len(probes):
        # Match indices refer to enabled probes; map their IDs back to the full
        # input table before selecting missed channels, including masked tones.
        distances = dict(
            zip(chan_matched["idx_chan"], np.abs(chan_matched["d_phi"]), strict=True)
        )
        scores = [
            distances.get(chan, np.inf * u.deg).to_value(u.deg)
            for chan in probes["chan"]
        ]
        missing = np.zeros(len(probes), dtype=bool)
        missing[np.argsort(scores)[-(len(probes) - len(tones)) :]] = True
        prior = probes[missing]
        fill = QTable(
            {
                "f_chan": prior["f_chan"],
                "mask_tone": prior["tone_mask"],
                "amp_tone": prior["tone_amp"],
                "amp_tone_matched": prior["tone_amp"],
                "amp_tone_filled": np.zeros(len(prior), dtype=bool),
                "amp_tone_ambiguous": np.zeros(len(prior), dtype=bool),
                "f": prior["f_chan"],
                "Qr": np.zeros(len(prior)),
                "bitmask": np.zeros(len(prior), dtype=int),
            }
        )
        tones = vstack([tones, fill])
    elif not fixed:
        count = min(len(placeholders), config.n_chans_max - len(tones))
        fill = QTable(
            {
                "f_chan": placeholders[:count],
                "mask_tone": np.zeros(count, dtype=bool),
                "amp_tone": np.zeros(count),
                "amp_tone_matched": np.zeros(count),
                "amp_tone_filled": np.zeros(count, dtype=bool),
                "amp_tone_ambiguous": np.zeros(count, dtype=bool),
            }
        )
        tones = vstack([tones, fill])
    if len(tones) == 0:
        msg = "No tones available for Tlaloc output"
        raise ValueError(msg)
    tones["f_tone"] = tones["f_chan"] - f_lo
    tones["phase_tone"] = config.tone_phases(len(tones))
    tones.sort("f_chan")
    f_tone = tones["f_tone"]
    # Preserve the production writer's FFT permutation for every column.
    offset = 3 * np.max(f_tone)
    order = sorted(
        range(len(tones)),
        key=lambda i: f_tone[i] + offset if f_tone[i] < 0 else f_tone[i],
    )
    tones = tones[order]
    return _finish_tone_dataset(tones, meta, output, amp_provenance)


def _amplitude_matches(tones, query_matches, output):
    """Mark ambiguity at the output boundary without changing the matcher."""
    if (
        output.tone_amps_method == "match"
        and output.tone_amps_fallback is not None
        and query_matches is None
    ):
        msg = "Amplitude fallback requires detection-to-probe match groups"
        raise ValueError(msg)
    unique_probe_ids = []
    ambiguous = np.zeros(len(tones), dtype=bool)
    if query_matches is not None:
        if len(query_matches) != len(tones):
            msg = "Amplitude match rows must align with detections"
            raise ValueError(msg)
        # Connected group size counts both probe and detection nodes. Only a
        # two-node group is unambiguous; no member of a larger group inherits.
        unique = np.asarray(query_matches["size"]) == 2  # noqa: PLR2004 -- one probe and one detection
        ambiguous = ~unique
        unique_probe_ids = query_matches["idx_chan"][unique]
    tones["amp_tone_matched"] = np.ma.asarray(tones["amp_tone"], dtype=float).filled(
        np.nan
    )
    tones["amp_tone_ambiguous"] = ambiguous
    return unique_probe_ids


def _finish_tone_dataset(tones, meta, output, amp_provenance):
    """Normalize coefficients jointly and retain fixed-total power diagnostics."""
    divisor = max(1.0, float(np.max(tones["amp_tone"])))
    if output.tone_amps_fallback is not None:
        tones["amp_tone"] /= divisor
    result = _tone_dataset(tones, meta)
    result.attrs.update(amp_provenance)
    if output.tone_amps_fallback is not None:
        # Use the same emitted-bin model as drive-fit commit. Import at the
        # computation boundary to avoid a configuration-module import cycle.
        from .drive_fit.power import (  # noqa: PLC0415
            POWER_MODEL,
            tone_power_fractions,
        )

        for name in ("amp_tone_matched", "amp_tone_filled", "amp_tone_ambiguous"):
            result[name] = ("tone", np.asarray(tones[name]))
        result["tone_power_fraction"] = (
            "tone",
            tone_power_fractions(result.amp_tone, result.f_centered, result.mask_tone),
        )
        result.attrs.update(
            tone_amps_normalization_divisor=divisor,
            tone_power_model=POWER_MODEL,
        )
    return result


def _tone_dataset(tones, meta):
    """Project the v2 model columns without changing their order or defaults."""

    def values(column):
        data = getattr(column, "value", column)
        return np.ma.asarray(data, dtype=float).filled(np.nan)

    columns = {
        "f_centered": tones["f_tone"].to_value(u.Hz),
        "f_out": tones["f_chan"].to_value(u.Hz),
        "f_in": tones["f_chan"].to_value(u.Hz),
        "flag": values(tones["flag"] if "flag" in tones.colnames else tones["bitmask"]),
    }
    aliases = {
        "fr": ("fr", "f"),
        "Qr": ("Qr",),
        "normI": ("normI", "g0"),
        "normQ": ("normQ", "g1"),
        "fp": ("fp", "f0"),
        "slopeI": ("slopeI", "k0"),
        "slopeQ": ("slopeQ", "k1"),
        "interceptI": ("interceptI", "m0"),
        "interceptQ": ("interceptQ", "m1"),
        "Qc": ("Qc",),
        "A": ("A",),
    }
    for name, names in aliases.items():
        source = next((col for col in names if col in tones.colnames), None)
        column = np.zeros(len(tones)) if source is None else values(tones[source])
        if name == "fr" and source is not None:
            columns["f_in"] = column.copy()
        columns[name] = np.where(tones["mask_tone"], column, np.nan)
    report_columns = list(columns)
    columns.update(
        {
            name: np.asarray(tones[name])
            for name in ("amp_tone", "phase_tone", "mask_tone")
        }
    )
    return xr.Dataset(
        {name: ("tone", column) for name, column in columns.items()},
        attrs={
            "kind": "tones",
            "roach": meta.roach,
            "f_lo_center": meta.f_lo_center,
            "report_columns": report_columns,
            "Header.Toltec.LoCenterFreq": meta.f_lo_center,
            "Header.Toltec.ObsNum": meta.obsnum,
            "Header.Toltec.SubObsNum": meta.subobsnum,
            "Header.Toltec.ScanNum": meta.scannum,
            "Header.Toltec.RoachIndex": meta.roach,
        },
    )
