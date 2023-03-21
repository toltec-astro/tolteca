#!/usr/bin/env python

from pathlib import Path
# from tollan.utils.log import init_log


if __name__ == "__main__":
    import sys
    obsnum = sys.argv[1]
    from tolteca.datamodels.toltec import BasicObsDataset

#    init_log(level='DEBUG')
    links = (
            list(Path("/data_lmt/toltec/tcs").glob(f'toltec*/toltec*{obsnum}_*[0-9].nc'))
            + list(Path("/data_lmt/toltec/ics").glob(f'toltec*/toltec*{obsnum}_*[0-9].nc')))
    f0 = links[0].resolve()
    stem = f0.stem.split("_", 1)[-1]
    print(f"apt_{stem}.ecsv")
