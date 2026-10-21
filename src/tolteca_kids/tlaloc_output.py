"""V2 Tlaloc tone files and commit settings shared by KIDs workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr  # noqa: TC002 -- Pydantic resolves the dataclass field.
from astropy.table import QTable, Table
from filelock import FileLock
from pydantic import ConfigDict, Field, model_validator
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
    """Explicit installation of prepared files in a Tlaloc etc directory.

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
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    path: AbsAnyPath | None = None

    @model_validator(mode="after")
    def _require_path(self):
        if self.enabled and self.path is None:
            msg = "Enabled TlalocOutput requires an explicit path"
            raise ValueError(msg)
        return self


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
