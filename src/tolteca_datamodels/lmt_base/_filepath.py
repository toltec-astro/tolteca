"""Shared filename information for LMT instruments."""

from __future__ import annotations

import re
from functools import cached_property
from typing import ClassVar

from pydantic import Field, computed_field
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from .._filepath import FilepathInfo
from ..core import FrozenDataModel
from ..model_table import ModelColumn, ModelColumnMapper, ScalarColumnRule
from .metadata import InterfaceInfoMixin, LmtObsQuartetMixin
from .types import DatetimeT

__all__ = [
    "FilepathInfoMixin",
    "FilepathOrigMetadataMixin",
]


class FilepathInfoMixin(LmtObsQuartetMixin, InterfaceInfoMixin):
    """Observation information encoded in an acquisition filepath."""

    _columns: ClassVar[dict[str, ModelColumn]]

    prefix: str
    suffix: str | None
    ext: str
    ut: DatetimeT

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()
        obs_fields = [
            name
            for name in cls.model_fields
            if name not in {"prefix", "suffix", "ext", "ut", "proc_suffix"}
        ]
        cls._columns = ModelColumnMapper(
            name_rules=[
                (r".+", r"file_\g<0>"),
                ("|".join(map(re.escape, obs_fields)), r"\g<0>"),
            ],
            type_rules=[ScalarColumnRule()],
            default=None,
        )(cls)


class FilepathOrigMetadataMixin[FilepathInfoT: FilepathInfo](FrozenDataModel):
    """Mixin for the original acquisition filepath and its parsed information."""

    # Runtime type: type[FilepathInfoT].
    _filepath_info_cls: ClassVar

    filepath_orig: str = Field(description="Original acquisition file path")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        bindings = resolve_concrete_generic_bindings(
            cls,
            FilepathOrigMetadataMixin,
            FilepathInfoT=GenericBindingRequirement("_filepath_info_cls", FilepathInfo),
        )
        if bindings is not None:
            cls._filepath_info_cls = bindings["FilepathInfoT"]

    @computed_field(repr=False)
    @cached_property
    def filepath_info_orig(self) -> FilepathInfoT | None:
        """Information encoded in the original acquisition filepath."""
        return self._filepath_info_cls.from_source(
            self.filepath_orig, raise_on_error=False
        )
