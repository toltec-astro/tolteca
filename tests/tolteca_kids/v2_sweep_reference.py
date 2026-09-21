"""Extract reader-level reference values using a separate v2 interpreter."""

import sys

import numpy as np


def main():
    """Read only; write comparison arrays to the supplied temporary NPZ file."""
    # This module is also discovered by v3's doctest collection. Import the
    # legacy-only reader only when executing with the requested v2 interpreter.
    from tolteca_datamodels.toltec.ncfile import NcFileIO  # noqa: PLC0415

    source, destination, block = sys.argv[1:]
    reader = NcFileIO(source)
    sweep = reader.read() if block == "-1" else reader.read(block=int(block))
    np.savez(
        destination,
        I=sweep.I.value,
        Q=sweep.Q.value,
        unc_I=sweep.I_unc.value,
        unc_Q=sweep.Q_unc.value,
        f_chan=sweep.f_chans.to_value("Hz"),
        sweep=sweep.f_sweep.to_value("Hz"),
        frequency=sweep.frequency.to_value("Hz"),
        units=[
            str(value.unit) for value in (sweep.I, sweep.Q, sweep.I_unc, sweep.Q_unc)
        ],
    )


if __name__ == "__main__":
    main()
