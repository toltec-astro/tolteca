"""Explicit application of partial acquisition files."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from typing import Literal

import xarray as xr
from pydantic import ConfigDict, Field

from ..core.types import FrozenDataModel
from .artifact import ArtifactBinding

MODEL_FIELDS = (
    "Header.Toltec.ModelParams",
    "Header.Toltec.ModelParamsHeader",
    "Header.Toltec.ModelParamsAccumLen",
)

TELESCOPE_COORD_FIELDS = tuple(
    f"Data.TelescopeBackend.{name}"
    for name in ("SourceRaAct", "SourceDecAct", "ActParAng")
)
TELESCOPE_MATCH_FIELDS = (
    "Header.TelescopeBackend.Master",
    "Header.Dcs.ObsNum",
    "Header.Dcs.SubObsNum",
    "Header.Dcs.ScanNum",
    "Data.TelescopeBackend.TelTime",
)


class StubApplication(FrozenDataModel):
    """One explicitly selected acquisition stub, in application order.

    Parameters
    ----------
    interface : str
        Target network or ancillary interface in the consuming acquisition.
    artifact : ArtifactBinding
        Exact stub artifact in a processed product owned by that acquisition.
    mode : {"override", "default"}
        Override replaces supplied variables; default fills absent variables.
        Neither mode treats zero, NaN or invalid values as absence.
    """

    model_config = ConfigDict(extra="forbid")
    interface: str = Field(min_length=1)
    artifact: ArtifactBinding
    mode: Literal["override", "default"]

    def resolve(self, index, *, raw_obs):
        """Resolve a stub and check its acquisition and interface ownership.

        Parameters
        ----------
        index : DataProdIndex
            Pinned processed-product realization containing the stub.
        raw_obs : RawObsRef
            Consuming acquisition, not the calibration sweep acquisition.

        Returns
        -------
        DataProdItem
            Matching NetCDF stub receipt. No payload is opened.

        Raises
        ------
        ValueError
            Ownership, interface, artifact kind or format differs.
        """
        item = self.artifact.resolve(index)
        parents = [
            a
            for a in index.product.assocs
            if a.is_primary and a.data_prod_assoc_type == "dpa_raw_obs_proc_raw_obs"
        ]
        if (
            index.product.meta.data_prod_type != "dp_raw_obs_proc"
            or len(parents) != 1
            or parents[0].uid != raw_obs.uid
            or parents[0].namespace_uid != raw_obs.namespace_uid
            or item.meta.get("interface") != self.interface
            or item.meta.get("kind") != "acquisition_stub"
            or item.meta.get("format") != "nc"
        ):
            msg = "Stub must belong to its consuming acquisition and interface"
            raise ValueError(msg)
        return item


@contextmanager
def open_acquisition(source, *, stubs=()):
    """Open native acquisition variables with explicitly ordered stub applications.

    Parameters
    ----------
    source : path-like
        Local raw NetCDF file, opened read-only with native numeric time values.
    stubs : sequence of (path-like, str), optional
        Explicit ordered pairs of stub path and mode (override or default).
        No files are discovered automatically. A later override wins; a default
        supplies only variables still absent. Model coefficients, column names,
        schema attributes and normalization are applied as one group.

    Yields
    ------
    xarray.Dataset
        Lazy effective acquisition. Load arrays needed beyond this scope before
        exit; all raw and stub resources close together. Raw I/Q is not copied.

    Raises
    ------
    ValueError
        A mode, target identity, dimension or coherent model group is invalid.
        Stub match variables must equal the raw file, including tone settings.
    """
    with ExitStack() as stack:
        raw = stack.enter_context(xr.open_dataset(source, decode_times=False))
        effective = raw.copy(deep=False)
        for path, mode in stubs:
            if mode not in {"override", "default"}:
                msg = f"Unknown stub mode: {mode}"
                raise ValueError(msg)
            stub = stack.enter_context(xr.open_dataset(path, decode_times=False))
            fields = _stub_fields(raw, stub)
            selected = _select_fields(effective, fields, mode)
            if set(MODEL_FIELDS) <= set(selected):
                # Model parameter dimensions belong to the replaced atomic group.
                # Acquisition channel/block dimensions remain on other variables.
                effective = effective.drop_vars(
                    [name for name in MODEL_FIELDS if name in effective]
                )
            for name in selected:
                array = stub[name]
                for dim, size in array.sizes.items():
                    if dim in effective.sizes and effective.sizes[dim] != size:
                        msg = f"Stub dimension differs from acquisition: {dim}"
                        raise ValueError(msg)
                effective[name] = array
        yield effective


def _stub_fields(raw, stub):
    fields = json.loads(stub.attrs["tolteca_stub_fields"])
    matches = json.loads(stub.attrs["tolteca_stub_match"])
    for names in (fields, matches):
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in stub for name in names)
            or len(set(names)) != len(names)
        ):
            msg = "Stub manifest must list distinct existing variables"
            raise ValueError(msg)
    if set(fields) & set(matches):
        msg = "Stub requires distinct payload and target match variables"
        raise ValueError(msg)
    if set(fields) & set(TELESCOPE_COORD_FIELDS):
        if not set(TELESCOPE_MATCH_FIELDS) <= set(matches):
            msg = "Telescope stub must match acquisition identity and TelTime"
            raise ValueError(msg)
        for name in set(fields) & set(TELESCOPE_COORD_FIELDS):
            if stub[name].dims != raw[TELESCOPE_MATCH_FIELDS[-1]].dims:
                msg = "Telescope coordinates must use the original TelTime dimensions"
                raise ValueError(msg)
    for name in matches:
        if name not in raw or not stub[name].identical(raw[name]):
            msg = f"Stub target differs from acquisition: {name}"
            raise ValueError(msg)
    return fields


def _select_fields(effective, fields, mode):
    selected = set(fields)
    if selected & set(MODEL_FIELDS):
        if not set(MODEL_FIELDS) <= selected:
            msg = "Model stub must supply the complete model group"
            raise ValueError(msg)
        present = [name in effective for name in MODEL_FIELDS]
        if mode == "default" and any(present):
            # Legacy complete coefficients may omit normalization. An orphan
            # normalization value must never be mixed with a new model.
            if not all(present[:2]):
                msg = "Cannot default a partially present model group"
                raise ValueError(msg)
            selected -= set(MODEL_FIELDS)
    if selected & set(TELESCOPE_COORD_FIELDS):
        if not set(TELESCOPE_COORD_FIELDS) <= selected:
            msg = "Telescope stub must supply the complete coordinate group"
            raise ValueError(msg)
        present = [name in effective for name in TELESCOPE_COORD_FIELDS]
        if mode == "default" and any(present):
            if not all(present):
                msg = "Cannot default a partially present telescope coordinate group"
                raise ValueError(msg)
            selected -= set(TELESCOPE_COORD_FIELDS)
    if mode == "default":
        selected = {name for name in selected if name not in effective}
    return [name for name in fields if name in selected]
