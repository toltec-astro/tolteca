"""Compare local I/Q chunk layouts without modifying completed products.

Use temporary Zarr-v2 stores with the published signal codecs and values. Counts
are LocalStore operations, not HTTP requests. Timings have warm, uncontrolled OS
caches and exclude store opening. The output is diagnostic, not a storage policy.
"""

# ruff: noqa: INP001, T201 -- standalone diagnostic probe

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import xarray as xr
from tollan.accessor.xarray import ensure_dataset
from zarr.storage import LocalStore

from tolteca_kids.product import open_sweep_product
from tolteca_kids.sweep_reduce import SweepReduce


def read_selection(store_path, names, indexer):
    """Read I/Q and coordinates, counting only signal-chunk traffic."""
    events = []
    get = LocalStore.get

    async def record_get(store, key, *args, **kwargs):
        value = await get(store, key, *args, **kwargs)
        if Path(key).parent.name in names and not Path(key).name.startswith("."):
            events.append((key, 0 if value is None else len(value)))
        return value

    with (
        xr.open_dataset(store_path, engine="zarr", consolidated=True) as saved,
        patch.object(LocalStore, "get", record_get),
    ):
        start = perf_counter()
        actual = saved.isel(indexer).compute()
        seconds = perf_counter() - start
    return actual, {
        "signal_chunks": len(events),
        "compressed_bytes": sum(size for _, size in events),
        "seconds": seconds,
    }


def compare(path, channel_chunks):
    """Measure sparse, batch and full I/Q reads for one saved product."""
    with open_sweep_product(path) as tree:
        view = SweepReduce.view(tree)
        signal_I, signal_Q, f_chan = view.I, view.Q, view.f_chan
        assert signal_I is not None
        assert signal_Q is not None
        assert f_chan is not None
        names = [signal_I.name, signal_Q.name]
        chan_dim = f_chan.dims[-1]
        block_dim = view.mapper.get_name(view.mapper.schema.block)
        original = ensure_dataset(view.data_source)[names].load()
        encoding = {
            name: {
                key: value
                for key, value in original[name].encoding.items()
                if key in {"compressors", "filters", "dtype", "_FillValue", "chunks"}
            }
            for name in names
        }
        block_index = view.context.block_index
    n_channels = original.sizes[chan_dim]
    selections = {
        "adjacent_2": [0, 1],
        "separated_2": [0, n_channels - 1],
        "batch_16": list(range(min(16, n_channels))),
        "batch_64": list(range(min(64, n_channels))),
        "all": slice(None),
    }
    report = {
        "product": str(path),
        "shape": list(original[names[0]].shape),
        "layouts": [],
    }
    for channel_chunk in [None, *channel_chunks]:
        variant = original.drop_encoding()
        variant_encoding = {name: dict(values) for name, values in encoding.items()}
        if channel_chunk is not None:
            for name in names:
                variant_encoding[name]["chunks"] = tuple(
                    min(channel_chunk, size)
                    if dim == chan_dim
                    else encoding[name]["chunks"][index]
                    if dim == block_dim
                    else size
                    for index, (dim, size) in enumerate(
                        zip(variant[name].dims, variant[name].shape, strict=True)
                    )
                )
        with TemporaryDirectory(prefix="sweep-chunks-") as directory:
            store_path = Path(directory) / "signals.zarr"
            variant.to_zarr(
                store_path,
                zarr_format=2,
                consolidated=True,
                encoding=variant_encoding,
            )
            store_bytes = sum(
                p.stat().st_size for p in store_path.rglob("*") if p.is_file()
            )
            layout = {
                "channel_chunk": channel_chunk,
                "chunks": list(variant_encoding[names[0]]["chunks"]),
                "store_bytes": store_bytes,
                "reads": {},
            }
            for label, channels in selections.items():
                indexer = {chan_dim: channels}
                if block_dim in variant.dims:
                    indexer[block_dim] = block_index
                expected = original.isel(indexer)
                actual, metrics = read_selection(store_path, names, indexer)
                for name in names:
                    np.testing.assert_array_equal(actual[name], expected[name])
                    assert actual[name].attrs == expected[name].attrs
                layout["reads"][label] = metrics | {
                    "selected_bytes": sum(expected[name].nbytes for name in names),
                    "values_exact": True,
                }
            report["layouts"].append(layout)
    return report


def main():
    """Print a read-only-input chunk comparison as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("products", nargs="+", type=Path)
    parser.add_argument("--channel_chunks", nargs="+", type=int, default=[1, 16, 64])
    args = parser.parse_args()
    print(
        json.dumps(
            [compare(path, args.channel_chunks) for path in args.products], indent=2
        )
    )


if __name__ == "__main__":
    main()
