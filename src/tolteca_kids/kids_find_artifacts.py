"""Detection and input-channel tables published by sweep reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.table import QTable


FIND_COLUMNS = (
    "idx_det",
    "bitmask_det",
    "f_det",
    "Qr",
    "dist",
    "d_phi",
    "idx_chan",
    "f_chan",
    "amp_tone",
    "snr_d21",
    "snr_s21",
    "height_d21",
    "height_db_s21",
)


class KidsFindTableArtifact(ArtifactSpec["QTable"]):
    """One detection or input-probe table, with units and acquisition identity."""

    def write(self, data: QTable, *, path: Path) -> list[ArtifactFile]:
        """Write retained match tables without repeating detection or matching.

        Parameters
        ----------
        data : astropy.table.QTable
            A match table in its original row order.
        path : pathlib.Path
            Export directory. Existing files are rejected.

        Returns
        -------
        list of ArtifactFile
            The ECSV file named by this artifact's alias.
        """
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{self.alias or 'kids_find'}.ecsv"
        data.write(target, format="ascii.ecsv")
        return [ArtifactFile(target, "ecsv", name=self.alias or "kids_find")]


class KidsFindArtifacts(ArtifactSpec[ArtifactContext]):
    """The v2 detection and input-probe match tables, with units and identity."""

    table: KidsFindTableArtifact = Field(default_factory=KidsFindTableArtifact)

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """Export retained tables into a product artifact directory.

        Parameters
        ----------
        data : ArtifactContext
            Tables paired with their named export specifications.
        path : pathlib.Path
            Export directory; existing files are rejected.

        Returns
        -------
        list of ArtifactFile
            Detection and input-channel ECSV files.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(artifact.data, path=path)
        ]
