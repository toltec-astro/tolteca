"""Extract unmodified v2 axis values with a separate interpreter."""

import sys

import numpy as np


def main():
    """Write reference axes to the requested temporary file."""
    from tolteca_datamodels.toltec.ncfile import NcFileIO  # noqa: PLC0415

    source, destination = sys.argv[1:]
    arrays = {}
    with NcFileIO(source) as reader:
        for block in range(reader.meta["n_blocks"]):
            chan = reader.get_chan_axis_data(block_index=block)
            for name in chan.colnames:
                arrays[f"{block}_chan_{name}"] = np.asarray(chan[name])
            if "Header.Toltec.ModelParams" in reader.nc_node.variables:
                params = reader.get_model_params_table(block_index=block)
                arrays[f"{block}_model_names"] = np.asarray(params.colnames)
                for name in params.colnames:
                    arrays[f"{block}_model_{name}"] = np.asarray(params[name])
            if reader.data_kind & type(reader.data_kind).RawSweep:
                sweep = reader.get_sweep_axis_data(block_index=block)
                for name in sweep.colnames:
                    arrays[f"{block}_sweep_{name}"] = np.asarray(sweep[name])
    np.savez(destination, **arrays)


if __name__ == "__main__":
    main()
