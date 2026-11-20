"""Columnar storage for completed finding results."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .kids_find_view import KidsFindMapper
from .match1d_result import _array, _make_match_result, _table_dataset

if TYPE_CHECKING:
    from .kids_find import KidsFindConfig, KidsFindData


def _make_find_result(
    values: KidsFindData, config: KidsFindConfig, identity: dict
) -> xr.DataTree:
    """Build the complete finding namespace before installing it in a block."""
    mapper = KidsFindMapper.from_defaults()
    schema = mapper.schema
    result = xr.DataTree()
    result.attrs[mapper.get_name(schema.config)] = config.model_dump_json()
    result.attrs[mapper.get_name(schema.identity)] = identity
    for name, masks in (
        ("d21", ("not_real", "dark", "baseline")),
        ("s21", ("not_real", "edge")),
    ):
        peaks = getattr(values, f"{name}_peaks")
        variables = {
            key: _array(getattr(peaks, key), "sample")
            for key in ("x", "y", "ey", "labels", "lookahead", "delta")
            if getattr(peaks, key) is not None
        }
        for mask in masks:
            variables[f"mask_{mask}"] = _array(
                np.asarray(getattr(values, f"{name}_mask_{mask}")).ravel(), "sample"
            )
        path = mapper.get_name(getattr(schema, name))
        result[f"{path}/samples"] = xr.DataTree(dataset=xr.Dataset(variables))
        ds = _table_dataset(peaks.peaks)
        ds["bitmask"] = _array(getattr(values, f"bitmask_{name}"), "row")
        result[f"{path}/peaks"] = xr.DataTree(dataset=ds)
    result[cast("str", mapper.get_name(schema.samples))] = xr.DataTree(
        dataset=xr.Dataset(
            {
                "mask_baseline": _array(values.mask_baseline, ("chan", "probe")),
            }
        )
    )
    for name in ("detected", "det_groups", "det_grouped", "mdl_groups", "mdl_grouped"):
        # 'group' in the corresponding *_grouped table is the numeric membership
        # relation; ragged 'items' arrays would duplicate it.
        result[cast("str", mapper.get_name(getattr(schema, name)))] = xr.DataTree(
            dataset=_table_dataset(getattr(values, name), omit=("items",))
        )
    result[cast("str", mapper.get_name(schema.matched))] = _make_match_result(
        values.matched
    )
    if config.match_ref != "chan":
        result[cast("str", mapper.get_name(schema.matched_ref))] = _make_match_result(
            values.matched_ref
        )
    return result
