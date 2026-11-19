"""Metadata and sample access for synthetic counter datasets."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from operator import index
from typing import TYPE_CHECKING, cast

import astropy.units as u
import xarray as xr
from astropy.time import Time
from tollan.accessor import Mapping, NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.utils.typing import get_literal_value

from tolteca_datamodels.core.source import source_opener

from .metadata import DummyInstruInterfaceKindT, DummyInstruMetadata
from .types import DummyInstruDataModel, DummyInstruInfo

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.utils.fileloc import FileLoc

    class DummyInstruDataset(xr.Dataset):
        """Dataset with a typed dummy-instrument accessor."""

        dummy_instru: DummyInstruAccessor


__all__ = [
    "DummyInstruAccessor",
    "DummyInstruIOMapper",
    "DummyInstruIOSchema",
    "open_dummy_instru",
]


@dataclass
class DummyInstruIOSchema(Schema):
    """Attribute headers and counter samples in a dummy dataset."""

    instru: Mapping = NameMapping.attr("instru")
    interface: Mapping = NameMapping.attr("interface")
    master: Mapping = NameMapping.attr("master")
    obsnum: Mapping = NameMapping.attr("obsnum")
    subobsnum: Mapping = NameMapping.attr("subobsnum")
    scannum: Mapping = NameMapping.attr("scannum")
    filepath_orig: Mapping = NameMapping.attr("filepath_orig")
    t0: Mapping = NameMapping.attr("t0_unix")
    t1: Mapping = NameMapping.attr("t1_unix")
    t_int: Mapping = NameMapping.attr("t_int_s")
    counts: Mapping = NameMapping.var("counts")


class DummyInstruIOMapper(XarrayMapper[DummyInstruIOSchema]):
    """Map synthetic counter headers and samples."""

    def get_metadata(
        self, data_source: xr.Dataset | xr.DataArray
    ) -> DummyInstruMetadata:
        """Read observation headers without loading counter samples.

        Parameters
        ----------
        data_source : xarray.Dataset or xarray.DataArray
            Dummy data with observation attributes.

        Returns
        -------
        DummyInstruMetadata
            Validated observation metadata.
        """
        schema = self.schema
        interface = self.get_str(data_source, schema.interface)
        return DummyInstruMetadata(
            instru=cast(
                "DummyInstruInfo.InstruT", self.get_str(data_source, schema.instru)
            ),
            data_models=frozenset({DummyInstruDataModel.name}),
            interface=interface,
            interface_kind=get_literal_value(DummyInstruInterfaceKindT),
            channel=DummyInstruInfo.interface_channel[interface],
            master=self.get_str(data_source, schema.master),
            obsnum=index(self.get_scalar(data_source, schema.obsnum)),
            subobsnum=index(self.get_scalar(data_source, schema.subobsnum)),
            scannum=index(self.get_scalar(data_source, schema.scannum)),
            filepath_orig=self.get_str(data_source, schema.filepath_orig),
            t0=Time(self.get_scalar(data_source, schema.t0), format="unix"),
            t1=Time(self.get_scalar(data_source, schema.t1), format="unix"),
            t_int=self.get_scalar(data_source, schema.t_int) * u.s,
        )


@xr.register_dataset_accessor("dummy_instru")
class DummyInstruAccessor(XarrayAccessorBase[DummyInstruIOMapper]):
    """Access dummy metadata and counter samples.

    Parameters
    ----------
    data_source : xarray.Dataset
        Open dummy dataset. Keep it open while reading samples.
    mapper : DummyInstruIOMapper, optional
        Field mapper. If omitted, resolve fields against the dataset.
    """

    def _validate(self) -> None:
        if (
            not self.mapper.has_attr(self.data_source, self.mapper.schema.instru)
            or self.mapper.get_str(self.data_source, self.mapper.schema.instru)
            != DummyInstruInfo.instru
        ):
            msg = "Not a dummy instrument dataset"
            raise ValueError(msg)

    @cached_property
    def meta(self) -> DummyInstruMetadata:
        """Cached observation metadata."""
        return self.mapper.get_metadata(self.data_source)

    @property
    def counts(self) -> xr.DataArray:
        """Counter samples."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.counts)


def open_dummy_instru(source: str | Path | FileLoc, **kwargs) -> DummyInstruDataset:
    """Open a dummy-instrument netCDF dataset.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local netCDF file. Remote files must be staged before opening.
    **kwargs
        Additional arguments passed to `xarray.open_dataset`.

    Returns
    -------
    DummyInstruDataset
        Dataset with an initialized ``dummy_instru`` accessor. Headers and
        counter samples are read on demand. Close the dataset after use.

    Raises
    ------
    ValueError
        If the location is remote or the dataset is not recognized as
        dummy-instrument data. The dataset is closed if accessor initialization fails.
    """
    with ExitStack() as stack:
        ds = cast(
            "DummyInstruDataset",
            stack.enter_context(
                source_opener.open({"ext": "nc"}, source=source, **kwargs)
            ),
        )
        _ = ds.dummy_instru
        stack.pop_all()
    return ds
