#! /usr/bin/env python
from tolteca.utils.log import get_logger
from tolteca.utils import odict_from_list


__all__ = [
        'label', 'template', 'title_text', 'title_icon',
        'update_interval',
        'n_records',
        'sources'
        ]

logger = get_logger()

# dasha props
label = "toltecdb"
template = 'live_db_view'
title_text = "TolTEC Database"
title_icon = "fas fa-table"

# source settings
update_interval = 4000  # ms
n_records = 50

source_common = {
            'query_init': f'select {{use_cols}} from {{table}} a'
                          f' {{join}} {{group}}'
                          f' order by {{order}} limit {{n_records}}',
            'query_update': f'select {{use_cols}} from {{table}} a'
                            f' {{join}}'
                            f' where a.id >= {{id_since}}'
                            f' {{group}}'
                            f' order by {{order}} limit {{n_records}}',
            'query_params': {'parse_dates': ["DateTime"]},
        }

sources = odict_from_list(map(lambda d: d.update(source_common) or d, [
        {
            'label': 'user_log',
            'title': 'User Log',
            'bind': 'lmt_toltec',
            'table': 'lmtmc_notes.userlog',
            'use_cols': ', '.join([
                'a.id',
                'TIMESTAMP(a.Date, a.Time) as DateTime',
                'a.Obsnum',
                'a.Entry', 'a.Keyword', ]),
            'join': "",
            'group': '',
            'order': 'a.id desc',
            },
        {
            'label': 'toltec_files',
            'title': 'TolTEC Files',
            'bind': 'lmt_toltec',
            'table': 'toltec.toltec',
            'use_cols': ', '.join([
                # 'GROUP_CONCAT(a.id SEPARATOR ",") AS id',
                'max(a.id) as id',
                'a.Obsnum', 'a.SubObsNum', 'a.ScanNum',
                'TIMESTAMP(a.Date, a.Time) as DateTime',
                'GROUP_CONCAT('
                'a.RoachIndex order by a.RoachIndex SEPARATOR ",")'
                ' AS RoachIndex',
                # 'a.RoachIndex',
                'CONCAT("clip", GROUP_CONCAT('
                'distinct right(a.HostName, 1)'
                ' order by a.RoachIndex SEPARATOR "/"))'
                ' AS HostName',
                # 'a.HostName',
                'b.label as ObsType',
                'c.label as Master',
                ]),
            'join': f"inner join toltec.obstypes b on a.ObsType = b.id"
                    f" inner join toltec.masters c on a.Master = c.id",
            # 'group': ''
            'group': 'group by a.ObsNum',
            'order': 'a.id desc'
            },
        ]), key=lambda v: v['label'])
