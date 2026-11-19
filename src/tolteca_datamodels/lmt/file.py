"""LMT telescope file tables."""

from typing import ClassVar

import pandas as pd
from tollan.accessor.pandas.model_table import Contains

from ..core.file import FileAccessorBase
from ..core.source import SourceReaderRules
from ..lmt_base.file import LmtBaseFileWorkflow, metadata_column_mapper
from . import filepath as _filepath  # noqa: F401 -- load this instrument's parsers.
from ._obs_spec import _obs_spec_context
from .metadata import LmtTelMetadata
from .tel import LmtTelAccessor, read_tel_interface_info
from .types import LmtDataModel

__all__ = ["LmtFileAccessor", "LmtFileWorkflow"]


class LmtFileWorkflow(LmtBaseFileWorkflow):
    """File-handling rules and column preparation for LmtFileAccessor.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table. Construction does not process rows.
    """

    _file_match_rules = ({"data_models": Contains(value=LmtDataModel.name)},)
    _source_reader_rules: ClassVar = SourceReaderRules(
        prepare_record=read_tel_interface_info,
        rules=(
            (
                {"data_models": Contains(value=LmtDataModel.name), "interface": "tel"},
                LmtTelAccessor,
            ),
        ),
    )
    _metadata_columns: ClassVar = {
        LmtTelMetadata: metadata_column_mapper(LmtTelMetadata),
    }
    _obs_spec_context: ClassVar = _obs_spec_context


@pd.api.extensions.register_dataframe_accessor("lmt_file")
class LmtFileAccessor(FileAccessorBase[LmtFileWorkflow]):
    """Collect filename information with LMT telescope observation columns.

    All supplied rows and available filename registrations are retained.
    Rows whose current ``data_models`` contain ``"lmt"`` receive UIDs.
    Other rows retain filename snapshots, marked metadata and an informational
    compatibility trace. `from_sources` guesses without opening resources.
    Use `open` to access their contents and `read_metadata` to
    replace guessed values with metadata.

    In a combined workflow, an instrument accessor may supersede this accessor
    on its matching telescope files. LMT still handles the other telescope rows.
    Observation selectors use the telescope interface labels.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access, including an empty DataFrame. Binding does not copy or
        modify it. Use `from_sources` to build a populated accessor.
    """
