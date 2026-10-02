"""Core metadata utilities."""

from __future__ import annotations

from pydantic import ConfigDict

__all__ = ["metadata_model_config"]


metadata_model_config = ConfigDict(strict=True, frozen=True)
"""Shared validation and immutability settings for observation metadata."""
