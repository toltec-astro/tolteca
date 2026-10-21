"""Extract production-v2 sweep-check results in a separate interpreter."""

import sys

import numpy as np
from astropy import units as u


def main():
    """Write all numerical core results without modifying the source file."""
    from tolteca_datamodels.toltec.ncfile import NcFileIO  # noqa: PLC0415 -- v2 only

    from tolteca_kids.sweep_check import SweepCheck  # noqa: PLC0415 -- v2 only

    source, destination, block, despike = sys.argv[1:]
    reader = NcFileIO(source)
    sweep = reader.read(block=int(block))
    SweepCheck(despike=despike == "true")(sweep)
    context = SweepCheck.get_context(sweep)
    assert context.completed
    values = {"S21_checked": sweep.S21.value, "aS21_unc_db": sweep.aS21_unc_db}
    for name, value in vars(context.data).items():
        if isinstance(value, u.Quantity):
            values[name] = value.value
            values[name + "_unit"] = str(value.unit)
        elif isinstance(value, (np.ndarray, float, int)):
            values[name] = value
        elif name.endswith("_slices"):
            values[name] = [(s.start, s.stop) for s in value]
    np.savez(destination, **values)


if __name__ == "__main__":
    main()
