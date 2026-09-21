"""Write and reopen self-contained reduced sweep products."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import xarray as xr

from .sweep_reduce import SweepReduce, SweepReduceView

__all__ = ["open_sweep_product", "write_sweep_product"]

_FORMAT = "tolteca.reduced-sweep.v1"


def write_sweep_product(
    data: xr.DataTree,
    path: Path,
    *,
    input_info: dict,
    output_config: dict | None = None,
) -> Path:
    """Write a new reduced product, marking it complete only after verification.

    Only the reducer result is exported, not the raw root or live resources.
    Existing destinations are never overwritten. A failed write may leave an
    incomplete directory for inspection; readers require its completion manifest.
    """
    view = SweepReduce.view(data)
    if not view.context.completed:
        msg = "Sweep reduction is not completed"
        raise ValueError(msg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    # Remove source-backend encodings; data values and unit attrs are retained.
    node = xr.DataTree(dataset=view.data_source.drop_encoding())
    product = xr.DataTree(children={SweepReduce.result_path(): node})
    # Format 2 preserves fixed-byte netCDF headers without relying on an
    # unstable Zarr 3 dtype extension or converting their values to text.
    product.to_zarr(path / "arrays.zarr", mode="w-", zarr_format=2, consolidated=False)
    with xr.open_datatree(
        path / "arrays.zarr", engine="zarr", consolidated=False
    ) as reopened:
        restored = SweepReduce.view(reopened)
        if dict(restored.data_source.sizes) != dict(view.data_source.sizes):
            msg = "Reduced array dimensions did not round-trip"
            raise ValueError(msg)
        if restored.context != view.context:
            msg = "Step context did not round-trip"
            raise ValueError(msg)
    manifest = {
        "format": _FORMAT,
        "input": input_info,
        "arrays": "arrays.zarr",
        "output_config": output_config,
        "software": {
            name: version(name)
            for name in ("tolteca", "tollan", "numpy", "xarray", "zarr")
        },
    }
    # The manifest is the completion marker, not a claim that partial arrays
    # are already available to catalog consumers.
    temporary = path / "manifest.pending"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.rename(path / "manifest.json")
    return path


def open_sweep_product(path: Path) -> xr.DataTree:
    """Open a completed local product; the caller owns the returned tree."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != _FORMAT or manifest.get("arrays") != "arrays.zarr":
        msg = "Unsupported reduced sweep product format"
        raise ValueError(msg)
    result = xr.open_datatree(path / "arrays.zarr", engine="zarr", consolidated=False)
    try:
        if not SweepReduceView(result).context.completed:
            msg = "Incomplete sweep reduction"
            raise ValueError(msg)  # noqa: TRY301 -- close the tree on validation failure
    except Exception:
        result.close()
        raise
    return result
