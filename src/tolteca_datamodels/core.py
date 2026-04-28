"""Core metadata utilities."""

from __future__ import annotations

from pydantic import ConfigDict

__all__ = ["metadata_dataclass_config"]


metadata_dataclass_config = ConfigDict(strict=True)
"""Pydantic config for metadata dataclasses.

Use with @dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True).
"""
