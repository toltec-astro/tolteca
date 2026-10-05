"""Metadata and sample access for synthetic counter datasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter, index
from typing import TYPE_CHECKING, Literal, cast, overload

import astropy.units as u
import xarray as xr
from astropy.time import Time
from tollan.accessor import Mapping, NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper

from tolteca_datamodels.source import identify_reader, open_source

from .metadata import DummyInstruMetadata
from .types import DummyInstruInfo

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
    "identify_dummy_instru",
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
            interface=interface,
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

    def read_metadata(self) -> DummyInstruMetadata:
        """Read current dummy observation headers.

        Returns
        -------
        DummyInstruMetadata
            Fresh observation metadata. Previously cached ``meta`` is unchanged.
            Counter samples are not loaded.
        """
        mapper = type(self.mapper).from_data_source(self.data_source)
        return mapper.get_metadata(self.data_source)

    @property
    def counts(self) -> xr.DataArray:
        """Counter samples."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.counts)


@overload
def identify_dummy_instru(
    data_source: object, *, raise_on_error: Literal[True]
) -> DummyInstruAccessor: ...


@overload
def identify_dummy_instru(
    data_source: object, *, raise_on_error: Literal[False] = False
) -> DummyInstruAccessor | None: ...


@overload
def identify_dummy_instru(
    data_source: object, *, raise_on_error: bool
) -> DummyInstruAccessor | None: ...


def identify_dummy_instru(
    data_source: object, *, raise_on_error: bool = False
) -> DummyInstruAccessor | None:
    """Identify an opened dummy-instrument dataset.

    Parameters
    ----------
    data_source : object
        Opened data. Only Dataset objects are supported.
    raise_on_error : bool, optional
        Raise for unsupported types or accessor validation errors instead of
        returning None. Default is False. Other exceptions always propagate.

    Returns
    -------
    DummyInstruAccessor or None
        Registered accessor if the instrument attribute identifies dummy data,
        otherwise None. Repeated calls reuse the accessor. Read ``meta`` to
        validate and obtain the observation headers.
        Identification does not load counter samples or close the dataset.

    Raises
    ------
    TypeError
        If the payload type is unsupported and ``raise_on_error=True``.
    ValueError
        If the instrument is not recognized and ``raise_on_error=True``.
    """
    return identify_reader(
        data_source,
        source_types=(xr.Dataset,),
        reader_factory=attrgetter("dummy_instru"),
        raise_on_error=raise_on_error,
    )


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
        dummy-instrument data. The dataset is closed if identification fails.
    """
    dataset = open_source(
        source, opener=xr.open_dataset, identifier=identify_dummy_instru, **kwargs
    )
    return cast("DummyInstruDataset", dataset)
