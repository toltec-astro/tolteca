#!/usr/bin/env python

from pathlib import Path
# from tollan.utils.log import init_log


if __name__ == "__main__":
    import sys
    obsnum = sys.argv[1]
    from tolteca.datamodels.toltec import BasicObsDataset

#    init_log(level='DEBUG')
    links = (
            list(Path("/data_lmt/toltec/tcs").glob(f'toltec*/toltec*{obsnum}_*.nc'))
            + list(Path("/data_lmt/toltec/ics").glob(f'toltec*/toltec*{obsnum}_*.nc')))
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    dataset.sort(['ut'])
    scannum = dataset[-1]['scannum']
    print(scannum)
