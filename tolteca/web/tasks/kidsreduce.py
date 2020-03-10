#! /usr/bin/env python

from dasha.web.extensions.ipc import ipc
from dasha.web.extensions.celery import get_celery_app, schedule_task
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from .toltecdb import get_toltec_file_info
from .shareddata import SharedToltecDataset
import subprocess


celery = get_celery_app()


class ReducedKidsData(object):

    @staticmethod
    def _make_datastore_key(info):
        return str(info)

    def __init__(self, info):
        self._info = info
        self._data = ipc.get_or_create(
                'redis', label=self._make_datastore_key(info))

    @classmethod
    def from_info(cls, info):
        return cls(info)


def _reduce_kidsdata(filepath):
    logger = get_logger()
    logger.debug(f"process file {filepath}")
    cmd = '/home/toltec/kids_bin/reduce.sh'
    cmd = [cmd, filepath, '-r']
    logger.debug(f"reduce cmd: {cmd}")
    try:
        result = subprocess.check_output(cmd)
    except Exception:
        logger.error(f"failed execute {cmd}", exc_info=True)
    else:
        logger.info(f"{result}")


if celery is not None:
    from celery_once import QueueOnce
    _dataset_label = 'kidsreduce'

    @celery.task(base=QueueOnce)
    def update_shared_toltec_dataset():
        logger = get_logger()
        dataset = SharedToltecDataset(_dataset_label)
        info = get_toltec_file_info(n_entries=50)
        # look in to datastore for files
        if info is None:
            return
        files = []
        reduced = []
        raw = []
        for i, entry in info.iterrows():
            logger.debug(f"get entry [{i}] info {entry}")
            f = dataset.files_from_info(entry)
            files.append(f)
            logger.debug(f"entry {pformat_yaml(entry)}")
            reduced.append(
                    list(filter(lambda n: 'reduced' in n, f)))
            raw.append(
                    list(filter(lambda n: 'reduced' not in n, f)))

        info['files'] = files
        info['reduced_files'] = reduced
        info['raw_files'] = raw
        logger.debug(f'info: {info}')
        dataset.set_index_table(info)

    @celery.task(base=QueueOnce)
    def reduce_kidsdata(*args, **kwargs):
        return _reduce_kidsdata(*args, **kwargs)

    @celery.task(base=QueueOnce)
    def reduce_kidsdata_on_db():
        dataset = SharedToltecDataset(_dataset_label)
        logger = get_logger()
        info = dataset.index_table
        if info is None:
            return
        logger.debug(f"reduce kidsdata on {len(info)} db entries")
        # make reduction file list
        files = []
        for i, entry in info.iterrows():
            if len(entry['reduced_files']) == 0:
                files.extend(entry['raw_files'])
        logger.debug(f"dispatch reduce files {files}")
        return reduce_kidsdata.map(files).delay()

    # run at 1s interval
    schedule_task(update_shared_toltec_dataset, schedule=1, args=tuple())
    schedule_task(reduce_kidsdata_on_db, schedule=1, args=tuple())
