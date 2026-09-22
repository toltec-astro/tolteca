"""Resolve local file inputs and observation selectors into a catalog."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from tollan.utils.log import logger

from ..file import ToltecFileAccessor, guess_info_from_sources
from ..filestore import HeaderReadMode, identify_source
from .obsspec import ObsSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

__all__ = ["build_input_catalog", "resolve_sources"]


def build_input_catalog(
    sources: Iterable[str | Path],
    *,
    data_root: Path | None = None,
    header_mode: HeaderReadMode | str = HeaderReadMode.always,
) -> pd.DataFrame:
    """Discover files and identify metadata for configuration and selection.

    ``header_mode="never"`` uses filename guesses without opening files.
    ``"fallback"`` opens only unrecognized filenames; ``"always"`` (the default)
    reads headers for known telescope/KIDs and unrecognized netCDF candidates.
    Fallback does not fill missing fields in an otherwise recognized filename.

    Files are closed before returning or propagating a read error. KIDs reads
    scalar headers and dimension lengths into typed ``metadata``; telescope
    identification also reads two timestamp
    endpoints and the two target-coordinate pairs. I/Q and pointing streams are
    not loaded. Unsupported kinds retain filename guesses. Observation selectors
    filter filenames before this header-enrichment stage. Header fields such as
    ``atten_drive`` are available as query aliases and in flat source snapshots
    for conditional configuration. Actual tune block counts require a separate
    array read and are not included in header metadata.
    """
    header_mode = HeaderReadMode(header_mode)
    catalog = resolve_sources(sources, data_root=data_root)
    logger.debug("Discovered {} distinct input files", len(catalog))
    if header_mode is HeaderReadMode.never:
        logger.debug("Using filename metadata without opening source files")
        return catalog
    records = []
    for record in catalog.toltec_file.to_info_list():
        info = record
        supported = info.filepath.suffix == ".nc" and (
            info.interface in (None, "tel") or info.roach is not None
        )
        requested = header_mode is HeaderReadMode.always or info.filepath_info is None
        if supported and requested:
            logger.debug(
                "Identifying source {} (header_mode={})", info.source, header_mode
            )
            with xr.open_dataset(info.filepath) as data:
                info = identify_source(data, info)
        else:
            logger.debug("Keeping filename metadata for {}", info.source)
        logger.debug("Input metadata: {}", info)
        records.append(info)
    return ToltecFileAccessor.from_info_list(records)


def _resolve_obs_spec(spec: ObsSpec, root: Path) -> list[Path]:
    # Exact fields narrow discovery; sets/ranges are evaluated against metadata.
    master = "*" if spec.master is None else str(spec.master)
    interface = spec.interface or (
        f"toltec{spec.roach}" if isinstance(spec.roach, int) else "*"
    )
    obsnum = f"{spec.obsnum:06d}" if isinstance(spec.obsnum, int) else "*"
    if interface == "tel":
        # Telescope streams share an interface and live outside the TolTEC tree.
        # Their filenames put the writer date before the observation number.
        pattern = f"tel/tel_*_{obsnum}_*.nc"
    else:
        filename = (
            f"{interface}_*.nc" if spec.interface else f"{interface}_{obsnum}_*.nc"
        )
        pattern = f"toltec/{master}/{interface}/{filename}"
    candidates = sorted(root.glob(pattern))
    catalog = guess_info_from_sources(candidates)
    selected = spec.select(catalog)
    logger.debug(
        "ObsSpec {} under {} using {!r}: {} candidates, {} selected",
        spec,
        root,
        pattern,
        len(catalog),
        len(selected),
    )
    return [Path(source) for source in selected["source"]]


def resolve_sources(
    sources: Iterable[str | Path | ObsSpec], *, data_root: Path | None = None
) -> pd.DataFrame:
    """Resolve paths, filesystem globs, and shared observation selectors.

    Strings ending in a file extension, or starting with an explicit path
    prefix (/, ./, ../, ~/), are filesystem inputs. Other strings are ObsSpec
    expressions. Pass a Path object to disambiguate an extensionless filename.
    Selectors require data_root; paths and globs do not.

    An explicit ``tel`` interface searches ``data_root/tel``. Other selectors
    search the TolTEC master/interface tree. Filename discovery does not open
    headers; unknown constrained identifiers, including master, do not match.
    """
    paths = []
    for source in sources:
        if isinstance(source, ObsSpec):
            spec = source
        elif (
            isinstance(source, Path)
            or str(source).startswith(("/", "./", "../", "~/"))
            or Path(source).suffix
        ):
            path = Path(source).expanduser()
            # glob supports absolute patterns as well as relative ones.
            matches = sorted(Path(item).absolute() for item in glob.glob(str(path)))  # noqa: PTH207
            matches = [item for item in matches if item.is_file()]
            logger.debug("Filesystem input {!r}: {} files", str(source), len(matches))
            if not matches:
                msg = f"No input files matched {str(source)!r}"
                raise FileNotFoundError(msg)
            paths.extend(matches)
            continue
        else:
            spec = ObsSpec.parse(source)
        if data_root is None:
            msg = f"An observation selector requires data_root: {spec}"
            raise ValueError(msg)
        matches = _resolve_obs_spec(spec, data_root)
        if not matches:
            msg = f"No input files matched {str(spec)!r}"
            raise FileNotFoundError(msg)
        paths.extend(matches)
    # Selectors and explicit paths may reach the same file through a symlink.
    # Preserve the first spelling for metadata while comparing resolved paths.
    unique = {}
    for path in paths:
        unique.setdefault(path.resolve(), path)
    logger.debug(
        "Input discovery: {} matches, {} distinct files", len(paths), len(unique)
    )
    return guess_info_from_sources(unique.values())
