"""File-table derived columns shared by LMT instruments."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from tollan.accessor.pandas.model_table import (
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)

from ..core.file import FileAccessorWorkflow
from .funcs import make_uid_columns
from .obs_spec import ObsSpec

if TYPE_CHECKING:
    from collections.abc import Hashable

    from tollan.accessor.pandas.model_table import RecordRuleMapper

__all__ = ["LmtBaseFileWorkflow", "metadata_column_mapper"]

metadata_column_mapper = ModelColumnMapper(
    name_rules=[(r".+", r"\g<0>")],
    type_rules=[
        ExplicitColumnRule(
            columns={
                "data_models": ModelColumn("data_models", dtype=object),
            }
        ),
        QuantityColumnRule.model_validate({"physical_type": "time", "unit": "s"}),
        QuantityColumnRule.model_validate({"physical_type": "frequency", "unit": "Hz"}),
        TimeColumnRule(format="mjd"),
        SkyCoordColumnRule.model_validate({"attrs": ["ra", "dec"], "unit": "deg"}),
        ScalarColumnRule(),
    ],
    include_computed=True,
    default=None,
)
"""Metadata columns in MJD, seconds, hertz and degrees, shared by LMT instruments."""


class LmtBaseFileWorkflow(FileAccessorWorkflow):
    """Derive observation columns for filenames accepted by the concrete accessor.

    Matching rows receive ``quartet``, ``uid_obs``, ``uid_raw_obs`` and
    ``uid_raw_obs_file``, derived from their current observation and interface
    columns. Filename snapshots and marked metadata remain available on every row.
    Identifiers are null when their required observation components are missing.

    Notes
    -----
    Extensions add derived values by overriding ``_prepare_derived_columns(positions)``.
    Return a mapping of column names to Series in the selected row order,
    without modifying the table. Destination columns are staged together before
    assignment, including columns returned by the extension.
    Observation selectors use the LMT field layout.
    """

    _obs_spec_cls: ClassVar = ObsSpec

    def _prepare_derived_columns(
        self, positions: list[int]
    ) -> dict[Hashable, pd.Series]:
        """Prepare UIDs from current values without modifying the table."""
        table = self.table
        uid_inputs = {}
        for name in ("master", "obsnum", "subobsnum", "scannum", "interface"):
            if name in table:
                uid_inputs[name] = table[name].take(positions)
            else:
                uid_inputs[name] = pd.Series(
                    None, index=table.index.take(positions), dtype=object
                )
        return dict(make_uid_columns(**uid_inputs).items())

    def _map_derived_column_from_current_values(
        self,
        positions: list[int],
        derived: dict[Hashable, pd.Series],
        mapper: RecordRuleMapper,
        *,
        dtype: str,
        fallback_columns: tuple[str, ...] = (),
    ) -> pd.Series:
        """Evaluate a rule column with prepared values overlaid on current rows."""
        rows = self.table.iloc[positions]
        for name, values in derived.items():
            rows[name] = values.array
        values = mapper.map_table(
            rows,
            column_names={
                name: f"file_{name}" for name in fallback_columns if name not in rows
            },
        )
        return pd.Series(values, index=rows.index, dtype=dtype)
