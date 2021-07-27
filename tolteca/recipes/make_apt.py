#!/usr/bin/env python

"""Create array property table for a range of KIDs props."""


import itertools
from astropy.table import Table
import numpy as np
import astropy.units as u


def make_meshgrid_table(*args):
    """Return a table that is the direct product of a set of column values.

    Parameters
    ----------
    *args
        A collection of args each of which is a tuple of (colname, values).
    """

    colnames, values = zip(*args)
    records = list()
    for i, data in enumerate(itertools.product(*map(enumerate, values))):
        records.append(dict(meshgrid_index=i))
        for name, (idx, value) in zip(colnames, data):
            records[-1][name] = value
            records[-1][f'{name}_index'] = idx

    tbl = Table(records)
    tbl.meta['_meshgrid_table_args'] = {
            name: np.asarray(values).tolist()
            for name, values in args
            }
    return tbl


if __name__ == "__main__":

    apt = Table()

    Qrs = np.arange(8e3, 2.21e4, 1e3)[::2]
    readout_noises = np.arange(5, 51, 5)[::2]
    array_names = ['a1100', 'a1400', 'a2000']

    tbl = make_meshgrid_table(
            ['array_name', array_names],
            ['Qr', Qrs],
            ['sigma_readout', readout_noises],
            )
    # rename the index table for array_name
    tbl.rename_column('array_name_index', 'array')
    # make the uid the meshgrid index
    tbl.rename_column('meshgrid_index', 'uid')

    # make some other fiducial properties for the KIDs
    tbl['x'] = 0. << u.cm  # the physical position
    tbl['y'] = 0. << u.cm
    tbl['f'] = 500. << u.MHz  # the designed freq
    tbl['fr'] = tbl['f']  # the "measured" freq
    tbl['x_t'] = 0. << u.deg  # the projected position
    tbl['y_t'] = 0. << u.deg

    # make the nw match the TolTEC convention
    nw_map = {
            'a1100': 0,
            'a1400': 7,
            'a2000': 11
            }
    tbl['nw'] = [nw_map[array_name] for array_name in tbl['array_name']]
    # the simulator expects some the meta data
    tbl.meta['array_names'] = array_names
    array_meta = {
            'a1100': {
                'wl_center': 1.1 << u.mm,
                },
            'a1400': {
                'wl_center': 1.4 << u.mm,
                },
            'a2000': {
                'wl_center': 2.0 << u.mm,
                },
            }
    for array_name in array_names:
        tbl.meta.update(array_meta)
    print(tbl.meta)
    tbl.write("apt.ecsv", format='ascii.ecsv', overwrite=True)
