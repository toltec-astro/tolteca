"""File tables for the dummy instrument and its telescope companions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd
import xarray as xr

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.obs_spec import obs_spec_context as lmt_obs_spec_context
from tolteca_datamodels.lmt.tel import identify_lmt_tel
from tolteca_datamodels.lmt_base._filepath import make_filepath_info_column_mapper
from tolteca_datamodels.lmt_base.funcs import make_uid_columns
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.model_table import (
    ModelColumnMapper,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)

from .data import identify_dummy_instru
from .filepath import DummyInstruFilepathInfo
from .metadata import DummyInstruMetadata
from .obs_spec import obs_spec_context

if TYPE_CHECKING:
    from tolteca_datamodels.file import _SourceInfoDiagnostics

__all__ = ["DummyInstruFileAccessor"]

_filepath_mapper = make_filepath_info_column_mapper("channel")
_metadata_mapper = ModelColumnMapper(
    name_rules=[(r".+", r"\g<0>")],
    type_rules=[
        QuantityColumnRule(physical_type="time", unit="s"),
        TimeColumnRule(format="mjd"),
        SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
        ScalarColumnRule(),
    ],
    include_computed=True,
    default=None,
)


@pd.api.extensions.register_dataframe_accessor("dummy_instru_file")
class DummyInstruFileAccessor(FileAccessor):
    """Build, select and open dummy-instrument file tables.

    Parameters
    ----------
    data : pandas.DataFrame
        File table containing source and source-location columns.

    Notes
    -----
    Filename guesses cover LMT and dummy-instrument files. Other inputs retain
    their rows with unidentified filename information. Local files are opened
    as xarray Datasets; supplied Dataset objects are borrowed. Opening requires
    recognized dummy-instrument or telescope data. Numeric interface selectors
    refer to dummy channels.
    """

    _io_obj_types: ClassVar = (xr.Dataset,)
    _source_opener: ClassVar = xr.open_dataset
    _source_readers: ClassVar = (identify_dummy_instru, identify_lmt_tel)
    _obs_spec_cls: ClassVar = ObsSpec
    _obs_spec_context: ClassVar = lmt_obs_spec_context | obs_spec_context
    _metadata_file_extensions: ClassVar = (".nc",)
    _filepath_info_types: ClassVar = (LmtFilepathInfo, DummyInstruFilepathInfo)
    _filepath_info_columns: ClassVar = {
        name: column
        for model in _filepath_info_types
        for name, column in _filepath_mapper(model).items()
    }
    _metadata_columns: ClassVar = {
        name: column
        for model in (LmtTelMetadata, DummyInstruMetadata)
        for name, column in _metadata_mapper(model).items()
    }

    def _guess_columns(self, *, diagnostics: _SourceInfoDiagnostics) -> set[int]:
        unidentified = super()._guess_columns(diagnostics=diagnostics)
        table = self._obj
        uids = make_uid_columns(
            table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
        )
        table[uids.columns] = uids
        return unidentified
