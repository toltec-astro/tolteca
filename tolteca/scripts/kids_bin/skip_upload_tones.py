
import sys
from tolteca.datamodels.toltec import BasicObsData
from astropy.table import Table

if __name__ == '__main__':
    filepath, tbl_filepath = sys.argv[1:]

    bod = BasicObsData(filepath).open()

    tbl = Table.read(tbl_filepath, format='ascii.ecsv')
    print(tbl.meta)
    tbl.meta['Header.Toltec.ObsNum'] = bod.meta['obsnum']
    tbl.meta['Header.Toltec.SubObsNum'] = bod.meta['subobsnum']
    tbl.meta['Header.Toltec.ScanNum'] = bod.meta['scannum']
    print(tbl.meta)
    tbl.write(tbl_filepath, format='ascii.ecsv', overwrite=True)
