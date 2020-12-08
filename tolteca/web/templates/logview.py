#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
# import dash_core_components as dcc
from dasha.web.extensions.db import dataframe_from_db
from dasha.web.templates.common import LiveUpdateSection
from dash.dependencies import Input, Output
from dasha.web.templates.utils import partial_update_at
from dash_table import DataTable
import dash
import pandas as pd
from ..tasks.dbrt import dbrt
import cachetools.func
from sqlalchemy import select, and_
from sqlalchemy.sql import func as sqla_func
from datetime import datetime, timedelta
from pathlib import Path

from tollan.utils.log import get_logger


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_toltec_filelog(time_start, time_end, repeat=0):

    tname = 'toltec' if repeat == 0 else f'toltec_r{repeat}'

    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session
    df_file = dataframe_from_db(
            select(
                [
                    t[tname].c.id,
                    t[tname].c.ObsNum,
                    t[tname].c.SubObsNum,
                    t[tname].c.ScanNum,
                    sqla_func.timestamp(
                        t[tname].c.Date,
                        t[tname].c.Time).label('DateTime'),
                    t[tname].c.RoachIndex,
                    t['obstype'].c.label.label('ObsType'),
                    t[tname].c.FileName,
                    t[tname].c.Valid,
                    ]
                ).select_from(
                    t[tname]
                    .join(
                        t['obstype'],
                        onclause=(
                            t[tname].c.ObsType
                            == t['obstype'].c.id
                            )
                        )
                    ).where(
                    and_(
                        sqla_func.timestamp(
                            t[tname].c.Date,
                            t[tname].c.Time) >= time_start,
                        sqla_func.timestamp(
                            t[tname].c.Date,
                            t[tname].c.Time) <= time_end,
                    )), session=session)
    return df_file


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_toltec_userlog(time_start, time_end):

    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session
    df_userlog = dataframe_from_db(
            select(
                [
                    t['userlog'].c.id,
                    t['userlog'].c.ObsNum,
                    sqla_func.timestamp(
                        t['userlog'].c.Date,
                        t['userlog'].c.Time).label('DateTime'),
                    t['userlog'].c.Entry,
                    t['userlog'].c.User,
                    # t['userlog'].c.Keyword,
                    ]
                ).where(
                    and_(
                        sqla_func.timestamp(
                            t['userlog'].c.Date,
                            t['userlog'].c.Time) >= time_start,
                        sqla_func.timestamp(
                            t['userlog'].c.Date,
                            t['userlog'].c.Time) <= time_end,
                    )), session=session)
    return df_userlog


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_toltec_syslog(time_start, time_end):

    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session
    df_systemlog = dataframe_from_db(
            select(
                [
                    t['systemlog'].c.id,
                    t['systemlog'].c.Type,
                    sqla_func.timestamp(
                        t['systemlog'].c.Date,
                        t['systemlog'].c.Time).label('DateTime'),
                    t['systemlog'].c.Entry,
                    t['systemlog'].c.User,
                    ]
                ).where(
                    and_(
                        sqla_func.timestamp(
                            t['systemlog'].c.Date,
                            t['systemlog'].c.Time) >= time_start,
                        sqla_func.timestamp(
                            t['systemlog'].c.Date,
                            t['systemlog'].c.Time) <= time_end,
                    )), session=session)
    return df_systemlog


class LogView(ComponentTemplate):
    _component_cls = html.Div

    logger = get_logger()

    def setup_layout(self, app):
        container = self
        header_container, body = container.grid(2, 1)
        header = header_container.child(
                LiveUpdateSection(
                    title_component=html.H3("Log View"),
                    interval_options=[2000, 5000, 10000],
                    interval_option_value=2000
                    ))

        controls_container, log_container = body.grid(2, 1)

        controls_form = controls_container.child(dbc.Form, inline=True)

        source_select_drp = controls_form.child(
            dbc.Checklist,
            options=[
                {
                    'label': c,
                    'value': c,
                    }
                for c in ['user', 'sys', 'file', 'file_replica']
                ],
            value=['user', 'sys', 'file', 'file_replica'],
            inline=True, switch=True,
            )

        log_dt = log_container.child(
                DataTable,
                style_cell={
                    'padding': '0.5em',
                    'width': '0px',
                    },
                css=[
                    {
                        'selector': (
                            '.dash-spreadsheet-container '
                            '.dash-spreadsheet-inner *, '
                            '.dash-spreadsheet-container '
                            '.dash-spreadsheet-inner *:after, '
                            '.dash-spreadsheet-container '
                            '.dash-spreadsheet-inner *:before'),
                        'rule': 'box-sizing: inherit; width: 100%;'
                    }
                ],
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Entry'},
                        'textAlign': 'left',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        },
                    ],
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{source} = "user"'
                        },
                        'backgroundColor': '#ffeeff',
                    },
                    {
                        'if': {
                            'filter_query': '{source} = "file"'
                        },
                        'backgroundColor': '#eeffaa',
                    },
                    {
                        'if': {
                            'filter_query': '{source} = "replica"'
                        },
                        'backgroundColor': '#aaffaa',
                    },
                    {
                        'if': {
                            'filter_query': '{Type} = "ERROR"'
                        },
                        'backgroundColor': '#ffaaaa',
                    },
                    {
                        'if': {
                            'filter_query': '{Entry} contains "(in progress)"'
                        },
                        'backgroundColor': '#ffffaa',
                    },
                ]
                )

        super().setup_layout(app)

        @app.callback(
            [
                Output(log_dt.id, 'columns'),
                Output(log_dt.id, 'data'),
                Output(header.loading.id, 'children'),
                Output(header.banner.id, 'children'),
                ],
            header.timer.inputs + [
                Input(source_select_drp.id, 'value'),
                ],
                )
        def update(n_calls, source_select_value):
            if len(source_select_value) == 0:
                raise dash.exceptions.PreventUpdate
            time_end = datetime.now()
            time_start = datetime.now() - timedelta(hours=24)
            dfs = dict()

            def make_filelog_entry(r):
                if int(r.Valid) > 0:
                    suffix = ''
                else:
                    suffix = ' (in progress)'
                return f'{Path(r.FileName).name}{suffix}'

            def prepare_filelog_df(df):
                df['Entry'] = [
                        make_filelog_entry(r) for r in df.itertuples()
                        ]
                df['Type'] = df['ObsType']
                df = df.reindex(columns=[
                    'source', 'id', 'ObsNum', 'DateTime', 'Entry', 'Type'
                    ])
                return df

            try:
                if 'user' in source_select_value:
                    df_userlog = query_toltec_userlog(time_start, time_end)
                    df_userlog['source'] = 'user'
                    df_userlog['Type'] = None
                    dfs['user'] = df_userlog
                if 'sys' in source_select_value:
                    df_syslog = query_toltec_syslog(time_start, time_end)
                    df_syslog['source'] = 'sys'
                    dfs['sys'] = df_syslog
                if 'file' in source_select_value:
                    df_filelog = query_toltec_filelog(time_start, time_end)
                    df_filelog['source'] = 'file'
                    dfs['file'] = prepare_filelog_df(df_filelog)
                if 'file_replica' in source_select_value:
                    df_replog = query_toltec_filelog(time_start, time_end, repeat=1)
                    df_replog['source'] = 'replica'
                    dfs['file_replica'] = prepare_filelog_df(df_replog)
            except Exception as e:
                self.logger.debug(f'Error query db: {e}', exc_info=True)
                return partial_update_at(
                        -1, dbc.Alert(
                            f'Error query db: {e}', color='danger'))
            df = pd.concat(dfs.values(), join='outer', axis=0)
            df = df.sort_values(by='DateTime', ascending=False)
            if len(df) > 100:
                df = df.head(100)
            # use_cols = [
            #         'source', 'id', 'ObsNum',
            #         'Entry', 'Type'
            #         ]
            use_cols = list(df.columns)
            use_cols.remove('source')
            use_cols.insert(0, 'source')
            df = df.reindex(columns=use_cols)
            data = df.to_dict('record')
            columns = [
                    {
                        'label': c,
                        'id': c
                        }
                    for c in df.columns
                    ]
            return columns, data, '', dash.no_update
