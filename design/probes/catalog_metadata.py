"""Inspect metadata available for catalog selection in each reader generation.

Run with the corresponding v2/v3 interpreter. Inputs are read-only; JSON is
written to stdout. Backend bytes are logical returned values, not disk traffic.
"""

# ruff: noqa: INP001, T201, PLC0415 -- standalone dual-environment diagnostic

import argparse
import json
from pathlib import Path

import reader_comparison as tracing
from loguru import logger


def inspect_metadata(version, path):
    """Read one source's metadata without reading I/Q data."""
    tracing.EVENTS.clear()
    tracing.TRACE = True
    reader = tracing.open_reader(version, "kids", path)
    try:
        meta = reader.meta if version == "v2" else reader.toltec_kids.meta
        result = tracing.summarize(meta)
    finally:
        reader.close()
        tracing.TRACE = False
    return {"metadata": result, "reads": list(tracing.EVENTS)}


def inspect_catalog(path):
    """Measure source identification independently of full reader metadata."""
    from tolteca_datamodels.toltec.filestore import build_input_catalog

    tracing.EVENTS.clear()
    tracing.TRACE = True
    try:
        catalog = build_input_catalog([path])
    finally:
        tracing.TRACE = False
    reads = list(tracing.EVENTS)
    try:
        selected = catalog.toltec_file.query("atten_drive >= 0")
        query = {"selected": len(selected)}
    except Exception as exc:  # noqa: BLE001 -- retain actual missing-field behavior
        query = {"error": f"{type(exc).__name__}: {exc}"}
    return {"columns": list(catalog), "query": query, "reads": reads}


def main():
    """Compare field availability and read costs on reference KIDs files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=["v2", "v3"])
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    logger.remove()
    tracing.install_trace(args.version)
    records = {}
    for path in sorted((args.data_root / "toltec/tcs/toltec0").glob("*.nc")):
        records[path.name] = inspect_metadata(args.version, path)
        if args.version == "v3":
            records[path.name]["catalog"] = inspect_catalog(path)
    print(json.dumps(records, indent=2, default=str))


if __name__ == "__main__":
    main()
