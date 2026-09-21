"""Bounded storage/ownership experiment; not a scientific parity test.

Run with the workspace development Python. All products go to a temporary
directory. This is a design probe, not proposed production code.
"""

import json
from tempfile import TemporaryDirectory
from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


def main():
    checks = {}
    raw = xr.Dataset({"I": (("chan", "sample"), np.arange(12.0).reshape(3, 4))})
    selected = xr.Dataset({"I": (("chan", "sample"), np.arange(8.0).reshape(2, 4))})
    try:
        xr.DataTree.from_dict({"/": raw, "/selected": selected})
    except ValueError:
        checks["ancestor_channel_size_conflict_rejected"] = True
    else:
        raise AssertionError("Expected dimension conflict")

    original = raw["I"].values.copy()
    corrected = raw.copy(deep=False)
    corrected["I"] = raw["I"] + 1
    np.testing.assert_array_equal(raw["I"], original)
    assert not np.shares_memory(raw["I"].data, corrected["I"].data)
    checks["replacing_computed_variable_preserves_input"] = True
    shallow = raw.copy(deep=False)
    assert np.shares_memory(shallow["I"].data, raw["I"].data)
    checks["shallow_copy_still_shares_unmodified_array_buffers"] = True

    left = xr.DataArray([1, 2], dims="chan", coords={"chan": [0, 1]})
    right = xr.DataArray([3, 4], dims="chan", coords={"chan": [1, 2]})
    try:
        xr.align(left, right, join="exact", copy=False)
    except ValueError:
        checks["explicit_exact_alignment_rejects_mismatched_ids"] = True
    else:
        raise AssertionError("Expected alignment failure")

    lazy = da.from_array(np.arange(12.0).reshape(3, 4), chunks=(1, 4))
    tasks = []
    with TemporaryDirectory(prefix="toltec-kids-layout-") as tmp:
        root = Path(tmp)
        with Callback(posttask=lambda *args: tasks.append(args[0])):
            tree = xr.DataTree.from_dict({
                "/": xr.Dataset(attrs={"schema": "design-probe", "version": 1}),
                "/sweep": xr.Dataset({"I": (("chan", "sample"), lazy + 1)}),
                "/selected": selected,
                "/quality": xr.Dataset({"flags": ("chan", np.array([0, 1, 0], dtype="uint32"))}),
            })
            assert not tasks
            checks["empty_root_allows_independent_sibling_dimensions"] = True
            checks["building_result_tree_executes_no_dask_tasks"] = True
            tree.to_zarr(root / "arrays.zarr", zarr_format=3, consolidated=False)
            assert tasks
        reopened = xr.open_datatree(root / "arrays.zarr", engine="zarr", chunks=None, consolidated=False)
        try:
            np.testing.assert_array_equal(reopened["sweep/I"].values, original + 1)
            flags = reopened["quality/flags"]
            assert flags.dtype == np.dtype("uint32")
            np.testing.assert_array_equal(flags.values, [0, 1, 0])
            assert reopened["selected"].sizes["chan"] == 2
            checks["zarr_v3_roundtrip_preserves_zero_flags_and_sibling_shapes"] = True
        finally:
            reopened.close()

        table = pl.DataFrame({
            "detection_id": pl.Series([0, 1, 2], dtype=pl.UInt32),
            "frequency": pl.Series([1.0, None, float("nan")], dtype=pl.Float64),
        })
        arrow = table.to_arrow()
        fields = [arrow.schema.field(0), arrow.schema.field(1).with_metadata({b"unit": b"Hz"})]
        arrow = pa.Table.from_arrays(arrow.columns, schema=pa.schema(fields))
        pq.write_table(arrow, root / "detections.parquet")
        recovered = pq.read_table(root / "detections.parquet")
        assert recovered.schema.field("frequency").metadata == {b"unit": b"Hz"}
        frame = pl.read_parquet(root / "detections.parquet")
        assert frame["frequency"].null_count() == 1
        assert frame["frequency"].is_nan().sum() == 1
        assert frame.schema["detection_id"] == pl.UInt32
        checks["parquet_preserves_null_vs_nan_uint_ids_and_arrow_unit_metadata"] = True

    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
