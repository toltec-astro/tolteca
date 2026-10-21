"""Portable drive-fit results and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
import xarray as xr  # noqa: TC002 -- Pydantic inspects this result dataclass.
from astropy.table import QTable

__all__ = ["DriveFitResult"]


@dataclass
class DriveFitResult:
    """Drive-fit arrays and selections, independent of raw inputs and runtime handles.

    Parameters
    ----------
    data : xarray.Dataset
        Selected per-channel drive values and diagnostics indexed by ``power``
        and ``chan``. Power is negative drive attenuation in dB. Reopened arrays
        may be lazy; keep their product tree open until values are loaded.
    """

    data: xr.Dataset

    def __post_init__(self):
        if (
            self.data.attrs.get("schema") != "tolteca_kids.drive_fit"
            or self.data.attrs.get("schema_version") != 1
        ):
            msg = "Not a supported tolteca_kids.drive_fit result"
            raise ValueError(msg)

    @property
    def table(self) -> QTable:
        """Per-channel drive selection, with the production ECSV column names."""
        table = QTable({"idx_chan": self.data.chan.values.copy()})
        for name in (
            "adrv_best",
            "a_best",
            "p_best",
            "Qr_best",
            "p_flag",
            "f_chan",
            "amp_chan",
        ):
            table[name] = self.data[name].values.copy()
        table["f_chan"] = table["f_chan"] * u.Hz
        table.meta.update(self.data.attrs)
        table.meta["drive_fit_pows"] = self.data.power.values.tolist()
        return table

    @property
    def n_success(self) -> int:
        """Number of channels with an accepted nonlinearity crossing."""
        return int(np.sum(self.data.p_flag.values == 0))
