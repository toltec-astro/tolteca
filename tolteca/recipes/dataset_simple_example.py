#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/04/21 Zhiyuan Ma:
#       - Created.

"""
This recipe shows how to use the `tolteca.fs.toltec.ToltecDataset`
to open files, load data objects, save the data objects for repeated analysis,
and do some plots.

Example usage:

The below will load an dataset index table, and do some process,
and save a pickle file::

    # noqa: E501
    $ python dataset_simple_example.py --dataset my_local_data_dir/dataset_example.asc -s 'obsid == [10718, 10728]' --dump dataset_simple_example.pickle

Next time, one can load the pickle and do a plot

    $ python dataset_simple_example.py --dataset dataset_simple_example.pickle --plot
"""

import numpy as np
from tolteca.recipes import get_logger
from tolteca.fs.toltec import ToltecDataset
from kidsproc.kidsdata import Sweep
from astropy.table import Table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
            description='Build local dataset index using rsync as backend.')
    input_option_group = parser.add_mutually_exclusive_group()
    input_option_group.add_argument(
            "--dataset",
            help='The dataset to load. It can either'
                 ' be an index table, or a previously dumped dataset object.'
            )
    input_option_group.add_argument(
            "--files",
            nargs='+',
            help='Build dataset from this list of files.'
            )
    parser.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
            )
    parser.add_argument(
            "--dump",
            help='Save the dataset to disk for later to load.'
            )
    parser.add_argument(
            "--plot",
            action='store_true',
            help='Make some plots'
            )
    option = parser.parse_args()

    logger = get_logger()
    if option.dataset:
        try:
            # try load the dataset if it is a pickle
            dataset = ToltecDataset.load(option.dataset)
        except Exception:
            # now try load the dataset as an index table
            dataset = ToltecDataset(
                    Table.read(option.dataset, format='ascii'))
        except Exception:
            raise RuntimeError(f"cannot load dataset from {option.dataset}")
        logger.debug(f'loaded dataset: {dataset}')
    elif option.files:
        # or build the dataset from a list of file paths.
        dataset = ToltecDataset.from_files(*option.files)
    else:
        parser.error("specify the dataset via --dataset or --files.")
    if option.select:
        # we can use the .select function to get a subset of entries.
        # the syntax is the same as pandas.DataFrame.query.
        # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-query  # noqa: E501
        dataset = dataset.select(option.select)

    # the dataset have be used to open the load the data objects using
    # open_files and load_data these methods, along with .select, and some
    # others, all returns the dataset object so one can chain the operation in
    # one go
    if dataset.data_objs is None:
        dataset = dataset.open_files().load_data()
    else:
        # if the dataset were loaded from a previous pickle, the
        # data objects are available already amd we do needed
        # to load the data.
        # in fact, calling the open_file and load_data here will
        # overwrite the existing data objects.
        logger.debug("found data_objs from loaded dataset")
    # now, the dataset contains more columns under the hood, which are
    # accessible via .file_objs and .data_obj property
    logger.debug(f"file_objs shape: {dataset.file_objs.shape}")
    logger.debug(f"data_objs shape: {dataset.data_objs.shape}")

    # in the mean time, the index table now contains extra columns
    # of the meta data the data objects, such as fsmp, n_tones, etc.
    logger.debug(f'dataset: {dataset}')

    # we can check the type of data objects:
    logger.debug(
            f"data_objs types: {[d.__class__ for d in dataset.data_objs]}")

    # As an example, we can compute the D21 of a sweep,
    # This function is caches the computed result D21 spectrum
    # on the first call, so we just try to trigger the calculation here,
    # but will latter on access it via .d21() again
    # d21() accepts a number of arguments, here we use a common grid
    # for all of the data objects so make comparing things easier
    d21_kw = {
            'fstep': 1000,  # Hz, the step of the frequency grid.
            'flim': (4e8, 1.0e9)  # Hz, the range of the frequency grid.
            }
    for d in dataset.data_objs:
        if isinstance(d, Sweep):
            d.d21(**d21_kw)
    # one thing that one may want to do at this point is to save
    # the state of the data objects, since we've done some expensive
    # computation.
    # this can be done by save the dataset object as pickle:
    # the saved object can be just loaded via ToltecDataset.load
    if option.dump:
        dataset.dump(option.dump)

    if option.plot:
        # now let try do some plots with the data
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(dataset), 1)

        for ax, data_obj in zip(axes, dataset.data_objs):
            if isinstance(data_obj, Sweep):
                # this d21 call will just retrieve the already computed data
                d21fs, adiqs, adiqscov = data_obj.d21()
                ax.plot(d21fs, adiqs)
                # plot the individual tones on a twinx
                tax = ax.twinx()
                # sweeps have the following useful info tables
                # and two array data fs and iqs
                sweep_info = data_obj.meta['sweeps']
                tone_info = data_obj.meta['tones']
                logger.debug(f'sweep info: \n{sweep_info}')
                logger.debug(f'tone info: \n{tone_info}')
                aiqs = np.abs(data_obj.iqs)
                for ti in range(len(tone_info)):
                    tax.plot(data_obj.fs[ti], 20 * np.log10(aiqs[ti]))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("D21")
                tax.set_ylabel("|S21| (db)")
        plt.show()
