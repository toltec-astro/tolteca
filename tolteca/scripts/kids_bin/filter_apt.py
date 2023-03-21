#!/usr/bin/env python3
from astropy.table import Table


if __name__ == '__main__':
    import sys
    apt_filepath = sys.argv[1]
    apt = Table.read(apt_filepath, format='ascii.ecsv')
    m_good = apt['flag'] == 1
    print(f'select {m_good.sum()} of {len(apt)} detectors from {apt_filepath}')
    apt = apt[m_good]
    output_file = apt_filepath.replace(".ecsv", "_good.ecsv")
    print(f"filtered APT filepath: {output_file}")
    apt.write(output_file, format='ascii.ecsv', overwrite=True)
