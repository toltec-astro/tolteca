#!/usr/bin/env python

if __name__ == "__main__":
    from tolteca.fs.toltec import ToltecDataFileStore, ToltecDataset

    datastore = ToltecDataFileStore("/data/data_toltec")
    links = datastore.runtime_datafile_links()
    
    ds = ToltecDataset.from_files(*links)
    obsid = max(ds['obsid'])
    print(obsid)
