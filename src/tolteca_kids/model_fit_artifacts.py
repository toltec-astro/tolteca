"""Common model tables and backend diagnostics retained by ModelFit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from tollan.pipeline import ArtifactContext, ArtifactFile, ArtifactSpec

if TYPE_CHECKING:
    from pathlib import Path

KIDSCPP_FILES = {
    "blocks.ecsv": ("ecsv", "blocks"),
    "channels.ecsv": ("ecsv", "channels"),
    "command.json": ("json", "command"),
    "native.log": ("log", "native_log"),
    "processed.nc": ("nc", "processed"),
}


class ModelFitTable(ArtifactSpec[bytes]):
    """Calibration model table from either fitting backend."""

    def write(self, data: bytes, *, path: Path) -> list[ArtifactFile]:
        """Write the prepared table as ``kids_model.ecsv``.

        Parameters
        ----------
        data : bytes
            Validated ECSV contents, preserved without reserializing the table.
        path : pathlib.Path
            Block artifact destination supplied by product output.
            An existing model file is rejected.

        Returns
        -------
        list of ArtifactFile
            One ECSV file with the semantic name ``model``.
        """
        path.mkdir(parents=True, exist_ok=True)
        target = path / "kids_model.ecsv"
        with target.open("xb") as stream:
            stream.write(data)
        return [ArtifactFile(target, "ecsv", name="model")]


class KidscppFitFiles(ArtifactSpec[dict[str, bytes]]):
    """Native axis snapshots, command, log and optional processed sweep."""

    def write(self, data: dict[str, bytes], *, path: Path) -> list[ArtifactFile]:
        """
        Write retained native files without repeating the fit.

        Parameters
        ----------
        data : dict of str to bytes
            Native filenames and their contents, retained after computation.
        path : pathlib.Path
            ModelFit artifact destination supplied by product output.
            Files are written below ``kidscpp/``; existing files are rejected.

        Returns
        -------
        list of ArtifactFile
            Native diagnostic files with semantic names.
        """
        path = path / "kidscpp"
        path.mkdir(parents=True, exist_ok=True)
        result = []
        for filename, content in data.items():
            format_name, name = KIDSCPP_FILES[filename]
            target = path / filename
            with target.open("xb") as stream:
                stream.write(content)
            result.append(ArtifactFile(target, format_name, name=name))
        return result


class ModelFitArtifacts(ArtifactSpec[ArtifactContext]):
    """Common model and optional Kidscpp diagnostics for a fitted sweep block.

    The artifact context holds the model table under ``model``'s spec and,
    for native fits, diagnostic file contents under ``kidscpp``'s spec.
    """

    model: ModelFitTable = Field(default_factory=ModelFitTable)
    kidscpp: KidscppFitFiles = Field(default_factory=KidscppFitFiles)

    def write(self, data: ArtifactContext, *, path: Path) -> list[ArtifactFile]:
        """
        Materialize the block's model table and available backend diagnostics.

        Parameters
        ----------
        data : ArtifactContext
            Prepared model and optional diagnostics paired with their writers.
        path : pathlib.Path
            Block artifact destination supplied by product output.

        Returns
        -------
        list of ArtifactFile
            Model table and any native diagnostic files.
        """
        return [
            receipt
            for artifact in data.values()
            for receipt in artifact.spec.write(artifact.data, path=path)
        ]
