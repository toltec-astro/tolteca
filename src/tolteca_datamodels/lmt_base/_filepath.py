"""Shared filename information for LMT instruments."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field, computed_field
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from .._filepath import FilepathInfo
from ..core import metadata_model_config
from .types import DatetimeT, LabelT, SeqNumT

if TYPE_CHECKING:
    from ..model_table import ModelColumnMapper

__all__ = [
    "FilepathInfoMixin",
    "FilepathOrigMetadataMixin",
    "make_filepath_info_column_mapper",
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


def make_filepath_info_column_mapper(*extra_obs_fields: str) -> ModelColumnMapper:
    """Create a column mapper for acquisition filepath information.

    Parameters
    ----------
    *extra_obs_fields : str
        Additional field names to expose without the ``file_`` prefix, alongside
        interface, instru, instru_component, master, obsnum, subobsnum and scannum.
        Names are literal, not regular expressions.

    Returns
    -------
    ModelColumnMapper
        Mapper producing ``file_*`` columns for scalar filepath fields and
        unprefixed aliases for observation information. Missing attributes
        produce None. Call the mapper with a filepath model class to create
        column declarations.

    Examples
    --------
    >>> mapper = make_filepath_info_column_mapper()
    >>> columns = mapper(FilepathInfoMixin)
    >>> columns["obsnum"].attr, columns["file_obsnum"].attr
    ('obsnum', 'obsnum')
    >>> "prefix" in columns, "file_prefix" in columns
    (False, True)
    """
    # Filename models do not require table machinery until columns are requested.
    from ..model_table import ModelColumnMapper, ScalarColumnRule  # noqa: PLC0415

    obs_fields = [
        "interface",
        "instru",
        "instru_component",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        *extra_obs_fields,
    ]
    return ModelColumnMapper(
        name_rules=[
            (r".+", r"file_\g<0>"),
            ("|".join(map(re.escape, obs_fields)), r"\g<0>"),
        ],
        type_rules=[ScalarColumnRule()],
        default=None,
    )
