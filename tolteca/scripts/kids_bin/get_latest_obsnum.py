#!/usr/bin/env python

from pathlib import Path
# from tollan.utils.log import init_log


if __name__ == "__main__":
    from tolteca.datamodels.toltec import BasicObsDataset
#    init_log(level='DEBUG')
    links = (
            list(Path("/data/data_toltec").glob('toltec[0-9].nc'))
            + list(Path("/data/data_toltec").glob('toltec[0-9][0-9].nc')))
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    obsnum = max(dataset['obsnum'])
    print(obsnum)
