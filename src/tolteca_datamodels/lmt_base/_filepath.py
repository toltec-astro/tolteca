"""Shared filename information for LMT instruments."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from pydantic import BaseModel, Field, computed_field
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from .._filepath import FilepathInfo
from ..core import metadata_model_config
from .types import DatetimeT, LabelT, SeqNumT

__all__ = [
    "FilepathInfoMixin",
    "FilepathOrigMetadataMixin",
]


class FilepathInfoMixin(BaseModel):
    """Observation information encoded in an acquisition filepath."""

    prefix: str
    suffix: str | None = None
    ext: str
    interface: LabelT
    instru: LabelT
    instru_component: LabelT
    master: LabelT | None = None
    obsnum: SeqNumT
    subobsnum: SeqNumT
    scannum: SeqNumT
    ut: DatetimeT


class FilepathOrigMetadataMixin[FilepathInfoT: FilepathInfo](BaseModel):
    """Mixin for the original acquisition filepath and its parsed information."""

    model_config = metadata_model_config

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
