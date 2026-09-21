"""Write and reopen self-contained reduced sweep products."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr
from pydantic import AwareDatetime, ConfigDict
from tollan.accessor.xarray import ensure_dataset
from tollan.config import FrozenBaseModel

from tolteca_datamodels.toltec.file import SourceInfoModel

from .config import KidsConfig
from .sweep_reduce import SweepReduce, SweepReduceView

if TYPE_CHECKING:
    from .config import DataProdOutputConfig

__all__ = [
    "SweepProductManifest",
    "open_sweep_product",
    "read_sweep_manifest",
    "write_sweep_product",
]


class SweepProductManifest(FrozenBaseModel):
    """Completion metadata for a self-contained reduced sweep product.

    ``input`` preserves the source record, including its structured location.
    ``config`` contains the effective pipeline settings, after conditional
    configuration has been resolved. Either can be None for direct exports of
    in-memory calculations without that information. Arrays and execution
    contexts remain in the Zarr store.
    """

    model_config = ConfigDict(extra="forbid")
    format: Literal["tolteca.reduced-sweep.v2"]
    arrays: Literal["arrays.zarr"]
    time_created: AwareDatetime
    input: SourceInfoModel | None = None
    config: KidsConfig | None = None
    software: dict[str, str]


def read_sweep_manifest(path: Path) -> SweepProductManifest:
    """Read completion metadata without opening or computing numerical arrays.

    Raises FileNotFoundError if the completion marker is absent and a Pydantic
    validation error for invalid metadata or an unsupported product format.
    Use :func:`open_sweep_product` to also validate the saved step context.
    """
    return SweepProductManifest.model_validate_json(
        (Path(path) / "manifest.json").read_text()
    )


def write_sweep_product(
    data: xr.DataTree,
    path: Path,
    *,
    input_info: SourceInfoModel | None = None,
    output_config: DataProdOutputConfig | None = None,
) -> Path:
    """Write a new reduced product, marking it complete only after verification.

    Only the reducer result is exported, not the raw root or live resources.
    Existing destinations are never overwritten. A failed write may leave an
    incomplete directory for inspection; readers require its completion manifest.
    """
    view = SweepReduce.view(data)
    context = view.context
    if not context.completed:
        msg = "Sweep reduction is not completed"
        raise ValueError(msg)
    path = Path(path)
    manifest = SweepProductManifest(
        format="tolteca.reduced-sweep.v2",
        arrays="arrays.zarr",
        time_created=datetime.now(UTC),
        input=input_info,
        config=(
            KidsConfig(sweep_reduce=context.config, output=output_config)
            if output_config is not None
            else None
        ),
        software={
            name: version(name)
            for name in ("tolteca", "tollan", "numpy", "xarray", "zarr")
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    # Remove source-backend encodings; data values and unit attrs are retained.
    node = xr.DataTree(dataset=ensure_dataset(view.data_source).drop_encoding())
    product = xr.DataTree(children={SweepReduce.result_path(): node})
    # Format 2 preserves fixed-byte netCDF headers without relying on an
    # unstable Zarr 3 dtype extension or converting their values to text.
    product.to_zarr(
        path / manifest.arrays, mode="w-", zarr_format=2, consolidated=False
    )
    with xr.open_datatree(
        path / manifest.arrays, engine="zarr", consolidated=False
    ) as reopened:
        restored = SweepReduce.view(reopened)
        if dict(restored.data_source.sizes) != dict(view.data_source.sizes):
            msg = "Reduced array dimensions did not round-trip"
            raise ValueError(msg)
        if restored.context != context:
            msg = "Step context did not round-trip"
            raise ValueError(msg)
    # The manifest is the completion marker, not a claim that partial arrays
    # are already available to catalog consumers.
    temporary = path / "manifest.pending"
    temporary.write_text(manifest.model_dump_json(indent=2, round_trip=True) + "\n")
    temporary.rename(path / "manifest.json")
    return path


def open_sweep_product(path: Path) -> xr.DataTree:
    """Open a completed local product; the caller owns the returned tree."""
    path = Path(path)
    manifest = read_sweep_manifest(path)
    result = xr.open_datatree(path / manifest.arrays, engine="zarr", consolidated=False)
    try:
        context = SweepReduceView(result).context
        if not context.completed:
            msg = "Incomplete sweep reduction"
            raise ValueError(msg)  # noqa: TRY301 -- close the tree on validation failure
        if (
            manifest.config is not None
            and manifest.config.sweep_reduce != context.config
        ):
            msg = "Manifest and saved reduction configuration disagree"
            raise ValueError(msg)  # noqa: TRY301 -- close the tree on validation failure
    except Exception:
        result.close()
        raise
    return result
