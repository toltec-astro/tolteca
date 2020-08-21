#!/usr/bin/env python

from pathlib import Path


if __name__ == "__main__":
    from tolteca.datamodels.toltec import BasicObsDataset

    links = (
            list(Path("/data/data_toltec").glob('toltec[0-9].nc'))
            + list(Path("/data/data_toltec").glob('toltec[0-9][0-9].nc')))
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    obsnum = max(dataset['obsnum'])
    print(obsnum)
