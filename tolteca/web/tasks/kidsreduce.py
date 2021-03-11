#! /usr/bin/env python

from dasha.web.extensions.db import dataframe_from_db
from dasha.web.extensions.ipc import ipc
from dasha.web.extensions.celery import celery_app, schedule_task, Q
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from .shareddata import SharedToltecDataset
from .. import toltec_datastore
from pathlib import Path
import cachetools.func
import subprocess
import shlex
from tollan.utils import odict_from_list
from ...datamodels.fs.toltec import meta_from_source


def shlex_join(split_command):
    return ' '.join(shlex.quote(arg) for arg in split_command)


_reduce_state_store = ipc.get_or_create(
                'rejson', label='reduce_state')
_reduce_state_store.ensure_obj(obj=dict())


def _make_reduce_state_key(filepath):
    info = meta_from_source(filepath)
    return 'toltec{roachid}_{obsnum}_{subobsnum}_{scannum}'.format(**info)


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
        r = subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def get_toltec_file_info(n_entries=20):
    query_template = {
            'query_base': 'select {use_cols} from {table} a'
            ' {join} {where} {group}'
            ' order by {order} limit {n_records}',
            'query_params': {'parse_dates': ["DateTime"]},
            'bind': 'toltecdb',
            'join': "",
            'where': '',
            'group': "",
            'order': 'a.id desc',
            'n_records': n_entries,
            'primary_key': 'id',
            }

    queries = odict_from_list([
            dict(query_template, **d)
            for d in [
                {
                    'title_text': "User Log",
                    'label': 'toltec_userlog',
                    'table': 'toltec.userlog',
                    'use_cols': ', '.join([
                        'a.id',
                        'TIMESTAMP(a.Date, a.Time) as DateTime',
                        'a.Obsnum',
                        'a.Entry', 'a.Keyword', ]),
                    },
                {
                    'title_text': "Files",
                    'label': 'toltec_files',
                    'table': 'toltec.toltec',
                    'use_cols': ', '.join([
                        'max(a.id) as id',
                        'a.Obsnum', 'a.SubObsNum', 'a.ScanNum',
                        'TIMESTAMP(a.Date, a.Time) as DateTime',
                        'GROUP_CONCAT('
                        'a.RoachIndex order by a.RoachIndex SEPARATOR ",")'
                        ' AS RoachIndex',
                        'CONCAT("clip", GROUP_CONCAT('
                        'distinct right(a.HostName, 1)'
                        ' order by a.RoachIndex SEPARATOR "/"))'
                        ' AS HostName',
                        'b.label as ObsType',
                        'c.label as Master',
                        'min(a.valid) as Valid',
                        ]),
                    'join': f"inner join toltec.obstype b on a.ObsType = b.id"
                            f" inner join toltec.master c on a.Master = c.id",
                    'group': 'group by a.ObsNum',
                    },
                ]
            ], key='label')
    q = queries['toltec_files']
    info = dataframe_from_db(
            q['query_base'].format(**q), bind='toltec',
            **q['query_params'])
    return info


_dataset_label = 'kidsreduce'


if celery_app is not None:

    QueueOnce = celery_app.QueueOnce

    @celery_app.task(base=QueueOnce, once={'timeout': 10}, time_limit=5)
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
            f = dataset.files_from_info(entry, master='ics/**')
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

    @celery_app.task(base=QueueOnce)
    def reduce_kidsdata(*args, **kwargs):
        return _reduce_kidsdata(*args, **kwargs)

    @celery_app.task(base=QueueOnce, once={'timeout': 10}, time_limit=5)
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
                logger.warn(
                        f"skip files of obstype {entry['ObsType']} {entry}")
                continue
            if entry['ObsType'] == 'Timestream':
                logger.warn(
                        f"skip files of obstype {entry['ObsType']} {entry}")
                continue
            if i == 0 and entry['Valid'] == 0:
                continue
            for filepath in entry['raw_files']:
                state = _reduce_state_store.get(
                        _make_reduce_state_key(filepath))
                logger.debug(f"check file status {filepath} {state}")
                if state is None:
                    files.append(filepath)
        logger.info(f"dispatch reduce files {files}")
        return reduce_kidsdata.map(files).delay()

    # run at 1s interval
    q = Q.normal_priority
    # lower number indicates higher priority, per
    # https://github.com/celery/celery/issues/4028#issuecomment-537587618
    schedule_task(update_shared_toltec_dataset, schedule=1, args=tuple(), options={'queue': q, 'priority': 0, 'expires': 1})
    schedule_task(reduce_kidsdata_on_db, schedule=1, args=tuple(), options={'queue': q, 'priority': 3, 'expires': 1})
