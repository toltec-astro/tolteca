"""Measure local store reads for opening and plotting saved sweep products.

The inputs are read-only. Consolidated-metadata comparisons use complete copies
in a temporary directory. Counts describe LocalStore operations, not HTTP requests
or remote latency; filesystem/OS caches are not controlled.
"""

# ruff: noqa: INP001, T201 -- standalone diagnostic probe

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import xarray as xr
import zarr
from tolteca_web.sweep.reduced import reduced_sweep_figure
from zarr.storage import LocalStore

from tolteca_kids.product import read_sweep_manifest
from tolteca_kids.sweep_reduce import SweepReduce


def measure(path, channels, *, consolidated):
    """Count reads separately during open, context validation and plotting."""
    manifest = read_sweep_manifest(path)
    events = []
    get = LocalStore.get
    get_partial = LocalStore.get_partial_values
    list_dir = LocalStore.list_dir

    async def record_get(store, key, *args, **kwargs):
        value = await get(store, key, *args, **kwargs)
        events.append(("get", key, None if value is None else len(value)))
        return value

    async def record_partial(store, prototype, key_ranges):
        key_ranges = list(key_ranges)
        values = await get_partial(store, prototype, key_ranges)
        for (key, _), value in zip(key_ranges, values, strict=True):
            events.append(("partial", key, None if value is None else len(value)))
        return values

    async def record_list(store, prefix):
        events.append(("list", prefix, None))
        async for key in list_dir(store, prefix):
            yield key

    def summarize(records, signal_names):
        gets = [(key, size) for method, key, size in records if method != "list"]
        metadata_names = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}
        chunks = [
            (key, size) for key, size in gets if Path(key).name not in metadata_names
        ]
        signal = [
            (key, size) for key, size in chunks if Path(key).parent.name in signal_names
        ]
        return {
            "get_calls": len(gets),
            "missing_gets": sum(size is None for _, size in gets),
            "list_calls": sum(method == "list" for method, _, _ in records),
            "returned_bytes": sum(size or 0 for _, size in gets),
            "chunk_gets": len(chunks),
            "signal_chunk_gets": len(signal),
            "signal_compressed_bytes": sum(size or 0 for _, size in signal),
            "chunk_variables": dict(
                Counter(str(Path(key).parent) for key, _ in chunks)
            ),
        }

    with (
        patch.object(LocalStore, "get", record_get),
        patch.object(LocalStore, "get_partial_values", record_partial),
        patch.object(LocalStore, "list_dir", record_list),
        xr.open_datatree(
            path / manifest.arrays, engine="zarr", consolidated=consolidated
        ) as tree,
    ):
        opened = len(events)
        view = SweepReduce.view(tree)
        assert view.context.completed
        validated = len(events)
        signal_names = {
            view.mapper.get_name(view.mapper.schema.I),
            view.mapper.get_name(view.mapper.schema.Q),
        }
        figure = reduced_sweep_figure(view, channels)
        assert len(figure.data) == 2 * len(channels)
    return {
        "open": summarize(events[:opened], signal_names),
        "context": summarize(events[opened:validated], signal_names),
        "plot": summarize(events[validated:], signal_names),
    }


def main():
    """Compare read counts without modifying supplied products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("products", nargs="+", type=Path)
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--compare_consolidated", action="store_true")
    args = parser.parse_args()
    reports = []
    for path in args.products:
        report = {"product": str(path), "channels": args.channels}
        report["unconsolidated"] = measure(path, args.channels, consolidated=False)
        if args.compare_consolidated:
            with TemporaryDirectory(prefix="sweep-consolidation-") as directory:
                copied = Path(directory) / path.name
                shutil.copytree(path, copied)
                manifest = read_sweep_manifest(copied)
                zarr.consolidate_metadata(copied / manifest.arrays)
                report["consolidated"] = measure(
                    copied, args.channels, consolidated=True
                )
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
