"""Inspect step-context and alias behavior on one read-only reference sweep."""

# ruff: noqa: INP001, T201 -- standalone diagnostic probe

import argparse
from pathlib import Path
from unittest.mock import patch

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.utils.log import logger

from tolteca_datamodels.toltec.kids import SweepReducer
from tolteca_kids.sweep_reduce import SweepReduce


def main():
    """Report the current lifecycle behavior without changing source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    logger.disable("tolteca_datamodels")
    logger.disable("tollan")
    with xr.open_dataset(args.source) as raw:
        working = ensure_datatree(raw)
        context = SweepReduce().create_context(working)
        print("Context attached by create_context:", SweepReduce.has_context(working))
        assert context.completed is False

        SweepReduce()(working)
        before = SweepReduce.get_context(working)
        invalid = SweepReduce(block=SweepReduce.view(working).n_blocks)
        try:
            invalid(working)
        except IndexError:
            pass
        else:
            msg = "Expected an out-of-range block failure"
            raise AssertionError(msg)
        after = SweepReduce.get_context(working)
        print("After failed rerun, context remains completed:", after.completed)
        print("After failed rerun, context equals previous success:", after == before)

        alias = SweepReduce.alias("review_probe")
        print(
            "Existing base context now recreates alias:",
            type(before.make_step()) is alias,
        )
        print(
            "Existing base context recreates base:",
            type(before.make_step()) is SweepReduce,
        )

        calls = []
        reduce = SweepReducer._reduce

        def count_reduction(reducer, data):
            calls.append(data)
            return reduce(reducer, data)

        with patch.object(SweepReducer, "_reduce", count_reduction):
            alias_working = ensure_datatree(raw)
            alias()(alias_working)
            alias()(alias_working)
        print("Numerical calls for two unforced alias-only invocations:", len(calls))


if __name__ == "__main__":
    main()
