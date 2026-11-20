"""Preparation of sequence identities for fitting and product publication."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from tollan.utils.fileloc import fileloc

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tollan.utils.fileloc import FileLoc


def prepare_drive_fit_inputs(sources: Iterable[str | Path | FileLoc]) -> xr.DataTree:
    """Read input headers and prepare a working collection of interface trees.

    Parameters
    ----------
    sources : iterable of str, pathlib.Path or FileLoc
        Exact native sweep locations for one sequence. All inputs must have one
        master. Sources are closed before this function returns.

    Returns
    -------
    xarray.DataTree
        Interface scopes under ``interfaces/<interface>``, each with an inputs
        table. Pass an interface to DriveFitStep or the collection to publication.

    Raises
    ------
    ValueError
        Sources are empty, repeated, or span acquisition masters.
    """
    # Saved-result readers do not require native-file discovery. Keep that
    # dependency at the preparation boundary while its accessor API is ported.
    from tolteca_datamodels.toltec.file import guess_info_from_sources  # noqa: PLC0415

    urls = [fileloc(source).url for source in sources]
    if not urls:
        msg = "No sweeps supplied for drive fitting"
        raise ValueError(msg)
    if len(urls) != len(set(urls)):
        msg = "Duplicate drive-fit input locations"
        raise ValueError(msg)
    candidates = guess_info_from_sources(urls)
    candidates.toltec_file.read_metadata(on_error="raise")
    return _from_table(candidates)


def _from_table(table):
    if table["master"].nunique() != 1:
        msg = "A drive-fit sequence must have one acquisition master"
        raise ValueError(msg)
    working = xr.DataTree()
    for interface, rows in table.groupby("interface", sort=True):
        group = rows.sort_values(["obsnum", "subobsnum", "scannum"])
        fields = {
            name: ("input", group[name].to_numpy())
            for name in ("obsnum", "subobsnum", "scannum", "atten_drive")
        }
        fields.update(
            {
                "source": ("input", [fileloc(value).url for value in group.source]),
                "uid_raw_obs": ("input", group.uid_raw_obs.to_list()),
            }
        )
        attrs = {
            "master": str(group.master.iloc[0]),
            "interface": str(interface),
            "roach": int(group.roach.iloc[0]),
        }
        working[f"interfaces/{interface}/inputs"] = xr.Dataset(fields, attrs=attrs)
    return working
