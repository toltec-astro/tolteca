"""Shared filename information for LMT instruments."""

from __future__ import annotations

import re
from functools import cached_property
from typing import Annotated, ClassVar

from pydantic import Field, computed_field
from tollan.accessor.pandas.model_table import (
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    ScalarColumnRule,
)

from .._filepath import FilepathInfo
from ..core import FrozenDataModel
from .metadata import InterfaceInfoMixin, LmtObsQuartetMixin
from .types import DatetimeT

__all__ = [
    "FilepathInfoMixin",
    "FilepathOrigMetadataMixin",
]


class _FileOnly:
    """Field projected only under its ``file_*`` name."""


class FilepathInfoMixin(LmtObsQuartetMixin, InterfaceInfoMixin):
    """Observation information encoded in an acquisition filepath."""

    _columns: ClassVar[dict[str, ModelColumn]]

    prefix: Annotated[str, _FileOnly]
    suffix: Annotated[str | None, _FileOnly]
    ext: Annotated[str, _FileOnly]
    ut: Annotated[DatetimeT, _FileOnly]

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()
        obs_fields = [
            name
            for name, field in cls.model_fields.items()
            if _FileOnly not in field.metadata
        ]
        cls._columns = ModelColumnMapper(
            name_rules=[
                (r".+", r"file_\g<0>"),
                ("|".join(map(re.escape, obs_fields)), r"\g<0>"),
            ],
            type_rules=[
                ExplicitColumnRule(
                    columns={
                        "data_models": ModelColumn(
                            "data_models", dtype=object, converter=list
                        ),
                    }
                ),
                ScalarColumnRule(),
            ],
            default=None,
        )(cls)


class FilepathOrigMetadataMixin(FrozenDataModel):
    """Mixin for the original acquisition filepath and its parsed information."""

    filepath_orig: str = Field(description="Original acquisition file path")

    @computed_field(repr=False)
    @cached_property
    def filepath_info_orig(self) -> FilepathInfo | None:
        """Information encoded in the original acquisition filepath."""
        return FilepathInfo.from_source(self.filepath_orig, raise_on_error=False)
