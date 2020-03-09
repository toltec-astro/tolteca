#! /usr/bin/env python
# from dasha.web.extensions.celery import get_celery_app
from dasha.web.extensions.ipc import ipc
import functools
import pandas as pd

# celery = get_celery_app()

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
        'n_records': 100,
        'primary_key': 'id',
        }

queries = [
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
                    ]),
                'join': f"inner join toltec.obstypes b on a.ObsType = b.id"
                        f" inner join toltec.masters c on a.Master = c.id",
                'group': 'group by a.ObsNum',
                },
            ]
        ]


def _get_datastore(label):
    return ipc.get_or_create('cache', label=label)


def _get_data(label):
    return _get_datastore(label).get()


tasks = [
        dict({
                'type': 'dasha.web.tasks.synceddatabase:SyncedDatabase',
                'update_interval': 1000 * 3,  # millisecond
                }, **{
                'label': d['label'],
                'bind': d['bind'],
                'query_init': d['query_base'].format(**d),
                'query_update': lambda x, d=d: d['query_base'].format(
                    **dict(d, where=f'where a.id > {x.iloc[0]["id"]}'),),
                'query_params': d['query_params'],
                'datastore': functools.partial(
                    _get_datastore, label=d['label']),
            })
        for d in queries
        ]

views = [
        dict(
            template='dasha.web.templates.dataframeview',
            title_text=d['title_text'],
            data=functools.partial(_get_data, label=d['label']),
            primary_key=d['primary_key'],
            update_interval=1000
            )
        for d in queries]


def get_toltec_file_info(n_entries=50):
    """This function joins the toltec user log with the toltec files."""
    userlog = _get_data('toltec_userlog')
    files = _get_data('toltec_files')

    if files is None:
        return None
    if userlog is None:
        result = files[:n_entries]
    else:
        result = pd.merge(
            files, userlog[['Obsnum', 'Entry']],
            on='Obsnum', how='left')[:n_entries]
    # split the roach index column
    # result['Interfaces'] = [
    #         ','.join(map(lambda i: f"toltec{i}", row.split(',')))
    #         for row in result['RoachIndex']
    #     ]
    # return result.drop(columns=['RoachIndex'])
    result.reset_index(inplace=True, drop=True)
    return result
