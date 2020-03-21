#! /usr/bin/env python

from dasha.web.extensions.ipc import ipc
from dasha.web.extensions.celery import get_celery_app, schedule_task
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from .toltecdb import get_toltec_file_info
from .shareddata import SharedToltecDataset
from .. import tolteca_toltec_datastore 
from pathlib import Path
import subprocess
import shlex


def shlex_join(split_command):
    return ' '.join(shlex.quote(arg) for arg in split_command)


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


_reduce_state_store = ipc.get_or_create(
                'rejson', label='reduce_state')


def _make_reduce_state_key(filepath):
    info = tolteca_toltec_datastore.spec.info_from_filename(Path(filepath))
    return 'toltec{nwid}_{obsid}_{subobsid}_{scanid}'.format(**info)


def _reduce_kidsdata(filepath):
    logger = get_logger()
    logger.debug(f"process file {filepath}")
    cmd = '/home/toltec/kids_bin/reduce.sh'
    cmd = [cmd, filepath, '-r']
    logger.info(f"reduce cmd: {cmd}")

    state = {
            'cmd': shlex_join(cmd),
            'filepath': filepath,
            }
        
    def _decode(s):
        if isinstance(s, (bytes, bytearray)):
            return s.decode()
        return s

    try:
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f"failed execute {cmd} {e} {e.output}", exc_info=True)
        state['state'] = 'failed'
        state['returncode'] = e.returncode 
        state['stdout'] = _decode(e.stdout)
        state['stderr'] = _decode(e.stderr)
    else:
        logger.info(f"{r}")
        state['state'] = 'ok'
        state['returncode'] = r.returncode 
        state['stdout'] = _decode(r.stdout)
        state['stderr'] = _decode(r.stderr)
    _reduce_state_store.set(state, path=_make_reduce_state_key(filepath))


if celery is not None:
    from celery_once import QueueOnce
    _dataset_label = 'kidsreduce'

    @celery.task(base=QueueOnce)
    def update_shared_toltec_dataset():
        logger = get_logger()
        dataset = SharedToltecDataset(_dataset_label)
        info = get_toltec_file_info(n_entries=20)
        # look in to datastore for files
        if info is None:
            return
        files = []
        reduced = []
        raw = []
        for i, entry in info.iterrows():
            logger.debug(f"get entry [{i}] info {entry}")
            f = dataset.files_from_info(entry, master='repeat/**')
            fr = dataset.files_from_info(entry, master='reduced')
            f.extend(fr)
            logger.debug(f"found files {f}")
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
            if entry['ObsType'] == 'Nominal':
                logger.warn(f"skip files of obstype {entry['ObsType']} {entry}")
                continue
            if i == 0 and entry['Valid'] == 0:
                continue
            for filepath in entry['raw_files']:
                state = _reduce_state_store.get(_make_reduce_state_key(filepath))
                logger.debug(f"check file status {filepath} {state}")
                if state is None:
                    files.append(filepath)
        logger.info(f"dispatch reduce files {files}")
        return reduce_kidsdata.map(files).delay()

    # run at 1s interval
    schedule_task(update_shared_toltec_dataset, schedule=1, args=tuple())
    schedule_task(reduce_kidsdata_on_db, schedule=1, args=tuple())
