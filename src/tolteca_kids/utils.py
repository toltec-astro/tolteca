#!/usr/bin/env python
from pydantic import BaseModel
from functools import cached_property


class ImmutableBaseModel(BaseModel):
    """Base model class with mutation disabled."""

    class Config:
        allow_mutation = False
        keep_untouched = (cached_property,)
