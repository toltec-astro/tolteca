"""V2 Tlaloc tone files and commit settings shared by KIDs workflows."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr  # noqa: TC002 -- Pydantic resolves the dataclass field.
from astropy.table import QTable, Table
from filelock import FileLock
from pydantic import ConfigDict, Field, model_validator
from scipy.interpolate import CubicSpline
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field.
from tollan.pipeline import (
    ArtifactFile,
    ArtifactSpec,
    Step,
    StepConfig,
    StepContext,
    StepFailure,
    StepFailureKind,
)


@dataclass(frozen=True)
class TlalocSettings:
    """Validated amplitude and attenuation settings in recorded tone order.

    Parameters
    ----------
    roach : int
        Readout network index, from 0 through 12.
    amp_tone : numpy.ndarray
        One-dimensional relative tone amplitudes in [0, 1], at most 1000 entries.
    atten_drive : float
        Finite global drive attenuation in dB.
    f_chan : numpy.ndarray
        Recorded absolute probe frequencies in Hz, in amplitude row order.
    f_lo_center : float
        Recorded center LO frequency in Hz.
    """

    roach: int
    amp_tone: np.ndarray
    atten_drive: float
    f_chan: np.ndarray
    f_lo_center: float

    def __post_init__(self):
        if not 0 <= self.roach <= 12:  # noqa: PLR2004 -- instrument network range.
            msg = "Tlaloc settings require readout network 0 through 12"
            raise ValueError(msg)
        amps = np.asarray(self.amp_tone)
        if (
            amps.ndim != 1
            or not 0 < amps.size <= 1000  # noqa: PLR2004 -- firmware tone limit.
            or not np.isfinite(amps).all()
            or np.any((amps < 0) | (amps > 1))
            or not np.isfinite(self.atten_drive)
            or np.asarray(self.f_chan).shape != amps.shape
            or not np.isfinite(self.f_chan).all()
            or not np.isfinite(self.f_lo_center)
        ):
            msg = "Invalid Tlaloc tone amplitudes or drive attenuation"
            raise ValueError(msg)

    @classmethod
    def from_data(cls, data: xr.Dataset) -> TlalocSettings:
        """Read a prepared settings dataset.

        Parameters
        ----------
        data : xarray.Dataset
            amp_tone and f_chan variables with roach, atten_drive and f_lo_center
            attributes. Arrays must remain accessible until construction finishes.

        Returns
        -------
        TlalocSettings
            Loaded, validated settings independent of source storage.
        """
        return cls(
            int(data.attrs["roach"]),
            data.amp_tone.values.copy(),
            float(data.attrs["atten_drive"]),
            data.f_chan.values.copy(),
            float(data.attrs["f_lo_center"]),
        )

    def validate_destination(self, path: Path) -> None:
        """Check the prepared reload list before installing commit amplitudes.

        Parameters
        ----------
        path : pathlib.Path
            Network directory containing the existing v2 targ_freqs.dat.

        Returns
        -------
        None
            The LO and ordered DAC frequency bins match the acquired probes.

        Raises
        ------
        ValueError
            The LO, count or ordered frequencies differ from the commit target.
        FileNotFoundError
            The destination has no prepared frequency file.
        """
        table = Table.read(path / "targ_freqs.dat", format="ascii.ecsv")
        lo = table.meta.get("Header.Toltec.LoCenterFreq", table.meta.get("f_lo_center"))
        if lo != self.f_lo_center or "f_centered" not in table.colnames:
            msg = "Commit destination must retain the target LO and v2 tone list"
            raise ValueError(msg)

        def bins(values):
            # C++ round uses ties away from zero; NumPy rint uses ties to even.
            resolution = 488.28125  # 2 * 512 MHz / 2**21, Tlaloc DAC_FREQ_RES.
            return np.sign(values) * np.floor(np.abs(values) / resolution + 0.5)

        if not np.array_equal(
            bins(np.asarray(table["f_centered"])),
            bins(np.asarray(self.f_chan) - self.f_lo_center),
        ):
            msg = (
                "Commit destination tone count, frequencies or order differ from target"
            )
            raise ValueError(msg)


class TlalocSettingsArtifact(ArtifactSpec[TlalocSettings]):
    """Tlaloc-format settings files for a supplied network."""

    def write(self, data: TlalocSettings, *, path: Path) -> list[ArtifactFile]:
        """Write settings to a new network directory under the supplied root.

        Parameters
        ----------
        data : TlalocSettings
            Validated settings in recorded target-channel order.
        path : pathlib.Path
            Output root. Files go in toltecN/default_targ_amps.dat and
            toltecN/atten_drive.dat. Existing files raise FileExistsError.

        Returns
        -------
        list of ArtifactFile
            Headerless amplitudes and scalar attenuation in v2 file formats.
        """
        destination = path / f"toltec{data.roach}"
        destination.mkdir(parents=True, exist_ok=True)
        amps = destination / "default_targ_amps.dat"
        attenuation = destination / "atten_drive.dat"
        if amps.exists() or attenuation.exists():
            msg = f"Tlaloc settings already exist in {destination}"
            raise FileExistsError(msg)
        QTable({"amp_tone": data.amp_tone}).write(amps, format="ascii.no_header")
        attenuation.write_text(str(data.atten_drive))
        return [ArtifactFile(amps, "ascii"), ArtifactFile(attenuation, "text")]


@dataclass(frozen=True)
class TlalocTones:
    """Prepared v2 tone table in its final shared file order.

    Parameters
    ----------
    data : xarray.Dataset
        One-dimensional tone columns, including the v2 model report columns,
        amp_tone, phase_tone and mask_tone. Metadata identifies the acquisition,
        ROACH and LO frequency. Values must remain available until export.
    """

    data: xr.Dataset

    def __post_init__(self):
        required = {
            "f_centered",
            "f_out",
            "f_in",
            "flag",
            "amp_tone",
            "phase_tone",
            "mask_tone",
        }
        if not required.issubset(self.data.data_vars):
            msg = "Incomplete Tlaloc tone table"
            raise ValueError(msg)
        if (
            not 0 <= self.roach <= 12  # noqa: PLR2004 -- network range.
            or not 0 < self.data.sizes.get("tone", 0) <= 1000  # noqa: PLR2004 -- firmware limit.
        ):
            msg = "Invalid Tlaloc network or tone count"
            raise ValueError(msg)
        for name, column in self.data.data_vars.items():
            if column.dims != ("tone",):
                msg = f"Tlaloc column {name} must have only the tone dimension"
                raise ValueError(msg)
        for name in ("f_centered", "f_out", "amp_tone", "phase_tone"):
            if not np.isfinite(self.data[name]).all():
                msg = f"Nonfinite Tlaloc tone property: {name}"
                raise ValueError(msg)
        if np.any((self.data.amp_tone < 0) | (self.data.amp_tone > 1)):
            msg = "Tlaloc amplitudes must be in [0, 1]"
            raise ValueError(msg)

    @property
    def roach(self) -> int:
        """Readout network index."""
        return int(self.data.attrs["roach"])


class TlalocToneArtifact(ArtifactSpec[TlalocTones]):
    """The four v2 tone files, all sharing the prepared row order."""

    def write(self, data: TlalocTones, *, path: Path) -> list[ArtifactFile]:
        """Export frequencies, replacement amplitudes, phases and masks.

        Parameters
        ----------
        data : TlalocTones
            Prepared table. No independent sorting or amplitude scaling occurs.
        path : pathlib.Path
            New export root, containing toltecN. Existing files are rejected.

        Returns
        -------
        list of ArtifactFile
            ECSV frequency report and three headerless companion files.
        """
        ds = data.data
        destination = path / f"toltec{data.roach}"
        files = {
            "targ_freqs.dat": None,
            "default_targ_amps.dat": "amp_tone",
            "random_phases.dat": "phase_tone",
            "default_targ_masks.dat": "mask_tone",
        }
        if any((destination / name).exists() for name in files):
            msg = f"Tlaloc tone files already exist in {destination}"
            raise FileExistsError(msg)
        report = Table(
            {name: ds[name].values.astype(float) for name in ds.attrs["report_columns"]}
        )
        report.meta = {
            key: value
            for key, value in ds.attrs.items()
            if key.startswith("Header.Toltec.") or key in {"roach", "f_lo_center"}
        }
        destination.mkdir(parents=True, exist_ok=True)
        receipts = []
        for name, column in files.items():
            target = destination / name
            fmt = "ascii.ecsv" if column is None else "ascii.no_header"
            table = (
                report
                if column is None
                else Table(
                    {
                        column: ds[column].values.astype(int)
                        if column == "mask_tone"
                        else ds[column].values
                    }
                )
            )
            table.write(target, format=fmt)
            receipts.append(ArtifactFile(target, "ecsv" if column is None else "ascii"))
        return receipts


class TlalocOutputConfig(StepConfig):
    """Tone amplitudes and explicit installation in a Tlaloc etc directory.

    Parameters
    ----------
    enabled : bool, optional
        Enable file installation in a workflow, default False.
    path : pathlib.Path or None, optional
        Tlaloc etc root containing toltecN directories. Required when enabled.
        Ordinary finding replaces frequency, amplitude, phase and mask files.
        Commit requires the existing frequency list to match its saved target
        LO and ordered DAC bins; a mismatch raises before replacement. It
        replaces only amplitude and attenuation files. Writing files
        does not upload tones or apply hardware attenuation.
    tone_amps_method : {'match', 'ones'}, optional
        Inherit matched input amplitudes (default), or assign unity amplitudes
        to detected tones. Used during ordinary tone preparation only.
    tone_amps_fallback : {'median', 'lut'} or None, optional
        None (default) preserves v2 amplitude assignments, including ambiguous
        matches. An explicit policy replaces every ambiguous or unavailable
        match. ``median`` uses enabled input amplitudes. ``lut`` uses
        ``path/toltecN/amps_correction_lut.csv`` in absolute RF Hz, scaled by
        the median input/LUT amplitude ratio in dB at unambiguous probes.
        Missing LUTs or usable scale anchors fail preparation before output.
        A common coefficient normalization keeps the final amplitudes in [0, 1].
        Ordinary output leaves global drive attenuation unchanged. Commit and
        replay use saved amplitudes and do not evaluate this policy again.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    path: AbsAnyPath | None = None
    tone_amps_method: Literal["match", "ones"] = "match"
    tone_amps_fallback: Literal["median", "lut"] | None = None

    @model_validator(mode="after")
    def _require_path(self):
        if self.enabled and self.path is None:
            msg = "Enabled TlalocOutput requires an explicit path"
            raise ValueError(msg)
        return self


def fill_tone_amplitudes(  # noqa: PLR0913 -- output rows and input-probe anchors
    amplitudes,
    frequencies,
    probes,
    *,
    fill_mask,
    unique_probe_ids,
    roach,
    config,
):
    """Prepare relative amplitudes at fixed global attenuation.

    Parameters
    ----------
    amplitudes : array-like
        Matched voltage coefficients in output detection order.
    frequencies : array-like
        Absolute output RF frequencies in Hz, in the same order.
    probes : astropy.table.QTable
        Acquisition columns chan, f_chan (Hz), tone_amp and tone_mask.
    fill_mask : array-like of bool
        All detections in ambiguous match groups. Nonfinite or masked amplitude
        values also require filling when a fallback policy is enabled.
    unique_probe_ids : array-like
        Original channel IDs in one-probe/one-detection match groups. Positive,
        enabled input amplitudes at these IDs anchor the LUT scale.
    roach : int
        Network number selecting the LUT below the configured etc root.
    config : TlalocOutputConfig
        Amplitude policy and LUT source directory. Installation may be disabled.

    Returns
    -------
    amplitudes : numpy.ndarray
        Relative coefficients. LUT-filled coefficients can exceed one; the
        caller must normalize the entire selected comb by one common factor.
    filled : numpy.ndarray
        Rows assigned fallback values; false for v2 and unity policies.
    provenance : dict
        Applied policy and, when used, LUT identity, scale and anchor count.

    Raises
    ------
    ValueError
        Inputs are invalid, or the requested fallback has no usable population.
    FileNotFoundError
        A required LUT is missing. No alternative policy is silently substituted.
    """
    amps = np.ma.asarray(amplitudes, dtype=float).filled(np.nan).copy()
    frequency = np.asarray(frequencies, dtype=float)
    filled = np.asarray(fill_mask, dtype=bool).copy()
    if amps.ndim != 1 or frequency.shape != amps.shape or filled.shape != amps.shape:
        msg = "Tone amplitudes, frequencies and fill mask must have the same 1-D shape"
        raise ValueError(msg)
    provenance = {
        "tone_amps_method": config.tone_amps_method,
        "tone_amps_fallback": config.tone_amps_fallback,
    }
    filled |= ~np.isfinite(amps)
    if config.tone_amps_method == "ones":
        amps[:] = 1.0
        filled[:] = False
    elif config.tone_amps_fallback is None:
        filled[:] = False
    elif filled.any():
        prior = np.ma.asarray(probes["tone_amp"], dtype=float).filled(np.nan)
        enabled = np.asarray(probes["tone_mask"], dtype=bool)
        if config.tone_amps_fallback == "median":
            valid = enabled & np.isfinite(prior) & (prior >= 0)
            if not valid.any():
                msg = (
                    "Median amplitude fallback requires finite enabled input amplitudes"
                )
                raise ValueError(msg)
            amps[filled] = np.median(prior[valid])
        else:
            amps[filled], lut_provenance = _lut_fallback(
                frequency[filled], probes, unique_probe_ids, roach, config
            )
            provenance.update(lut_provenance)
    if not np.isfinite(amps).all() or np.any(amps < 0) or np.any(amps[~filled] > 1):
        msg = "Invalid matched or filled tone amplitudes"
        raise ValueError(msg)
    provenance["tone_amps_filled_count"] = int(filled.sum())
    return amps, filled, provenance


def _lut_fallback(frequency, probes, unique_probe_ids, roach, config):
    """Evaluate Tlaloc's spline in the input comb's relative amplitude scale."""
    prior = np.ma.asarray(probes["tone_amp"], dtype=float).filled(np.nan)
    enabled = np.asarray(probes["tone_mask"], dtype=bool)
    if config.path is None:
        msg = "LUT amplitude fallback requires tlaloc_output.path"
        raise ValueError(msg)
    path = config.path / f"toltec{roach}" / "amps_correction_lut.csv"
    content = path.read_bytes()
    lut = np.loadtxt(BytesIO(content), delimiter=",", ndmin=2)
    if (
        lut.shape[1] != 2  # noqa: PLR2004 -- RF Hz, amplitude
        or len(lut) < 3  # noqa: PLR2004 -- tk::spline minimum
        or not np.isfinite(lut).all()
        or not np.all(np.diff(lut[:, 0]) > 0)
        or np.any(lut[:, 1] < 0)
    ):
        msg = f"Invalid amplitude LUT: {path}"
        raise ValueError(msg)
    x, y = lut.T
    spline = CubicSpline(x, y, bc_type="natural")

    def evaluate(f):
        # Tlaloc's natural tk::spline uses linear extrapolation.
        values = spline(f)
        for edge, outside in [(0, f < x[0]), (-1, f > x[-1])]:
            values[outside] = y[edge] + spline(x[edge], 1) * (f[outside] - x[edge])
        if not np.isfinite(values).all() or np.any(values <= 0):
            msg = "LUT values must be positive at fallback and anchor frequencies"
            raise ValueError(msg)
        return values

    anchor = (
        enabled
        & np.isfinite(prior)
        & (prior > 0)
        & np.isin(probes["chan"], unique_probe_ids)
    )
    if not anchor.any():
        msg = "LUT amplitude scale requires positive unambiguous input probes"
        raise ValueError(msg)
    input_frequency = probes["f_chan"].to_value("Hz")
    ratio_db = 20 * np.log10(prior[anchor] / evaluate(input_frequency[anchor]))
    offset_db = float(np.median(ratio_db))
    scale = 10 ** (offset_db / 20)
    values = scale * evaluate(frequency)
    provenance = {
        "tone_amps_lut_path": str(path),
        "tone_amps_lut_sha256": sha256(content).hexdigest(),
        "tone_amps_lut_scale": scale,
        "tone_amps_lut_offset_db": offset_db,
        "tone_amps_lut_anchor_count": int(anchor.sum()),
    }
    return values, provenance


class TlalocOutputContext(StepContext["TlalocOutput", TlalocOutputConfig]):
    """Installed settings, written-file receipts and execution status."""

    data: TlalocSettings | TlalocTones | None = Field(default=None, exclude=True)
    files: list[ArtifactFile] = Field(default_factory=list)


class TlalocOutput(Step[TlalocOutputConfig, TlalocOutputContext]):
    """Install prepared v2 tone files or commit settings from a tlaloc subtree.

    This step can consume live or reopened ordinary tone and commit products.
    It validates the payload before replacing destination files under a
    per-network writer lock. Each file replacement is atomic; the set is not
    a filesystem transaction.
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: TlalocOutputContext) -> None:
        """Install prepared files without recomputing or reordering tones.

        Parameters
        ----------
        data : xarray.DataTree
            Interface tree with a tlaloc dataset containing prepared settings.
        context : TlalocOutputContext
            Destination settings and resulting payload/file receipts.

        Returns
        -------
        None
            Missing settings record a failure. Filesystem errors are raised and
            recorded by the step lifecycle.
        """
        if "tlaloc" not in data.children:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message="Tlaloc output requires prepared tone settings.",
                )
            )
            return
        if context.config.path is None:
            msg = "TlalocOutput requires an explicit path"
            raise ValueError(msg)
        from tempfile import TemporaryDirectory  # noqa: PLC0415 -- output boundary

        prepared = data["tlaloc"].to_dataset()
        if prepared.attrs.get("kind") == "tones":
            settings = TlalocTones(prepared.load())
        else:
            settings = TlalocSettings.from_data(prepared)
        root = context.config.path
        destination = root / f"toltec{settings.roach}"
        destination.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(destination / ".tlaloc-output.lock"),
            TemporaryDirectory(prefix=".tlaloc-output-", dir=root) as staging,
        ):
            if isinstance(settings, TlalocSettings):
                settings.validate_destination(destination)
                files = TlalocSettingsArtifact().write(settings, path=Path(staging))
            else:
                files = TlalocToneArtifact().write(settings, path=Path(staging))
            for receipt in files:
                target = destination / receipt.path.name
                receipt.path.replace(target)
                context.files.append(ArtifactFile(target, receipt.format))
        context.data = settings
