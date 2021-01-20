#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dasha.web.extensions.db import dataframe_from_db
from dasha.web.templates.common import LiveUpdateSection
from dash.dependencies import Input, Output, State
from dasha.web.templates.utils import partial_update_at
from dash_table import DataTable
import dash
from ..tasks.dbrt import dbrt
import cachetools.func
import sqlalchemy.sql.expression as se
from sqlalchemy.sql import func as sqla_func
from datetime import datetime, timedelta, timezone
import astropy.units as u
from tollan.utils.db.conventions import utcnow

from tollan.utils.log import get_logger


def _get_toltec_userlog_id_latest():
    logger = get_logger()

    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables['userlog']
    session = dbrt['toltec'].session
    session.commit()

    stmt = se.select([t.c.id]).order_by(se.desc(t.c.id)).limit(1)
    id_latest = session.execute(stmt).scalar()

    logger.debug(f"latest id: {id_latest}")
    return id_latest


def _get_toltecdb_obsnum_latest():
    logger = get_logger()

    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables['toltec']
    session = dbrt['toltec'].session
    session.commit()
    stmt = se.select([t.c.ObsNum]).order_by(se.desc(t.c.ObsNum)).limit(1)
    obsnum_latest = session.execute(stmt).scalar()

    logger.debug(f"latest obsnum: {obsnum_latest}")
    return obsnum_latest


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_toltec_userlog(time_start=None, time_end=None, n_entries=None):
    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session
    session.commit()

    conditions = []
    if time_start is not None:
        conditions.append(
            sqla_func.timestamp(
                t['userlog'].c.Date,
                t['userlog'].c.Time) >= time_start,
                )
    if time_end is not None:
        conditions.append(
            sqla_func.timestamp(
                t['userlog'].c.Date,
                t['userlog'].c.Time) <= time_end,
                )
    if n_entries is not None:
        id_latest = _get_toltec_userlog_id_latest()
        id_since = id_latest - n_entries + 1
        conditions.extend(
                [
                    t['userlog'].c.id <= id_latest,
                    t['userlog'].c.id >= id_since
                    ])
    df_userlog = dataframe_from_db(
            se.select(
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
                    se.and_(*conditions)), session=session)
    return df_userlog


def insert_to_toltec_userlog(user, obsnum, entry):
    logger = get_logger()
    logger.debug(f"insert to userlog obsnum={obsnum} entry={entry}")

    dbrt.ensure_connection('toltec_userlog_tool')
    t = dbrt['toltec_userlog_tool'].tables
    session = dbrt['toltec'].session
    session.commit()
    stmt = (
        se.insert(t['userlog']).
        values(
            {
                'User': user,
                'ObsNum': obsnum,
                'Entry': entry,
                'Date': utcnow(),
                'Time': utcnow(),
                })
        )
    logger.debug(f"insert stmt: {stmt}")
    session.execute(stmt)
    session.commit()
    return


def make_labeled_drp(form, label, **kwargs):
    igrp = form.child(dbc.InputGroup, size='sm', className='pr-2')
    igrp.child(dbc.InputGroupAddon(label, addon_type='prepend'))
    return igrp.child(dbc.Select, **kwargs)


def make_labeled_input(
        form, label, input_cls=dbc.Input, make_extra_container=False,
        **kwargs):
    igrp = form.child(dbc.InputGroup, size='sm', row=True)
    width = kwargs.pop('width', 10)
    label_width = 12 - width
    lbl = igrp.child(dbc.Label, label, width=label_width)
    if make_extra_container:
        extra_container = igrp.child(dbc.Col, className='d-flex')
        inp_container = extra_container
    else:
        inp_container = igrp.child(dbc.Col)
        extra_container = None
    inp = inp_container.child(input_cls, width=width, **kwargs)
    lbl.html_for = inp.id
    if make_extra_container:
        return inp, extra_container
    return inp


class UserLogTool(ComponentTemplate):
    _component_cls = html.Div

    logger = get_logger()

    def setup_layout(self, app):
        container = self
        header_container, body = container.grid(2, 1)
        header = header_container.child(
                LiveUpdateSection(
                    title_component=html.H3("User Log Tool"),
                    interval_options=[2000, 5000, 10000],
                    interval_option_value=2000
                    ))

        inputs_container, controls_container, view_container = body.grid(3, 1)

        inputs_container.className = 'mt-4 mb-4'

        inputs_form = inputs_container.child(
                dbc.Form,
                style={'width': '50vw'}
                )

        input_user = make_labeled_input(
                inputs_form, "User",
                input_cls=dbc.Input,
                debounce=True,
                type='text',
                bs_size="sm",
                style={
                    'width': '15em',
                    }
                )

        input_obsnum, input_obsnum_extra_container = make_labeled_input(
                inputs_form, "ObsNum",
                input_cls=dbc.Input,
                make_extra_container=True,
                debounce=True,
                type='number',
                min=0, step=1,
                bs_size="sm",
                style={
                    'width': '10em',
                    }
                )

        latest_obsnum_btn = input_obsnum_extra_container.child(
                dbc.Button, 'Fill latest ObsNum',
                size='sm', color='link',
                className='ml-2 mb-2',
                )

        input_entry = make_labeled_input(
                inputs_form, "Entry",
                input_cls=dbc.Textarea,
                # debounce=True,
                bs_size="sm",
                className='mb-2',
                required=True,
                minLength=1,
                )

        # here we wrap the submit btn in a loading state to debounce
        submit_btn_loading, submit_btn_extra_container = make_labeled_input(
                inputs_form, "",
                input_cls=dcc.Loading,
                make_extra_container=True,
                # className='mb-2',
                type='dot',
                parent_style={
                    'height': '38px'  # this matches the button
                    }
                )
        submit_btn = submit_btn_loading.child(
                dbc.Button,
                # size="sm",
                color='primary',
                children='Submit',
                className='mb-2 mr-2'
                )
        # the button itself can't be the trigger so we need another
        # dummy div in the dcc.Loading
        on_submit_trigger = submit_btn_loading.child(html.Div)

        response_container = submit_btn_extra_container.child(html.Div)

        controls_container.className = 'mt-2'
        controls_form = controls_container.child(dbc.Form, inline=True)

        view_latest_since_drp = make_labeled_drp(
                controls_form, 'Show entries of last',
                options=[
                    {
                        'label': f'{n}',
                        'value': n,
                        }
                    for n in ['1 d', '7 d', '30 d']],
                value='1 d',
                )

        view_n_entries_max_drp = make_labeled_drp(
                controls_form, 'Show maximum',
                options=[
                    {'label': f'{n} entries', 'value': n}
                    for n in [50, 200, 1000]
                    ],
                value=50,
                )

        view_container.className = 'mt-2'
        log_dt = view_container.child(
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
                # style_data_conditional=[
                # ]
                )

        super().setup_layout(app)

        @app.callback(
                Output(input_obsnum.id, 'value'),
                [
                    Input(latest_obsnum_btn.id, 'n_clicks')
                    ]
                )
        def fill_latest_obsnum(n_clicks):
            latest_obsnum = _get_toltecdb_obsnum_latest()
            return int(latest_obsnum)

        @app.callback(
                [
                    Output(response_container.id, 'children'),
                    Output(on_submit_trigger.id, 'children'),
                    ],
                [
                    Input(submit_btn.id, 'n_clicks'),
                    ],
                [
                    State(input_user.id, 'value'),
                    State(input_obsnum.id, 'value'),
                    State(input_entry.id, 'value'),
                    ],
                prevent_initial_call=True
                )
        def on_submit(n_clicks, user, obsnum, entry):

            def make_output(color, message):
                return [
                    dbc.Alert(
                        message, color=color,
                        duration=3000,
                        fade=True,
                        className='mx-0 my-0',
                        style={
                            # these matches the button
                            'height': '38px',
                            'padding-top': '0.375em',
                            'padding-bottom': '0.375em',
                            }
                        ),
                    ""]

            if obsnum is None or obsnum < 0 or entry in [None, '']:
                return make_output(
                        'danger',
                        'Error: incomplete form data.')
            # create entry and push to db
            try:
                insert_to_toltec_userlog(
                        user=user, obsnum=obsnum, entry=entry)
            except Exception:
                self.logger.error("failed create record in db", exc_info=True)
                return make_output(
                        'danger',
                        'Error: unable to update database.'
                        )
            return make_output(
                    'success',
                    'Success.')

        @app.callback(
            [
                Output(log_dt.id, 'columns'),
                Output(log_dt.id, 'data'),
                Output(header.loading.id, 'children'),
                Output(header.banner.id, 'children'),
                ],
            header.timer.inputs + [
                Input(view_latest_since_drp.id, 'value'),
                Input(view_n_entries_max_drp.id, 'value'),
                # this is here to trigger update on submit
                Input(on_submit_trigger.id, 'children'),
                ]
            )
        def update_view(
                n_calls, view_latest_since_value, view_n_entries_max_value,
                submit_btn_loading_state):
            latest_since_value, latest_since_unit = \
                    view_latest_since_value.split()
            latest_since = latest_since_value << u.Unit(latest_since_unit)
            view_n_entries_max = int(view_n_entries_max_value)

            time_end = datetime.now(timezone.utc)
            time_start = time_end - timedelta(
                    hours=latest_since.to_value('hr'))

            try:
                df_userlog = query_toltec_userlog(
                        time_start=time_start,
                        time_end=time_end,
                        n_entries=view_n_entries_max)
            except Exception as e:
                self.logger.debug(f'Error query db: {e}', exc_info=True)
                return partial_update_at(
                        -1, dbc.Alert(
                            f'Error query db: {e}', color='danger'))
            df = df_userlog
            df = df.sort_values(by='DateTime', ascending=False)
            data = df.to_dict('record')
            columns = [
                    {
                        'label': c,
                        'id': c
                        }
                    for c in df.columns
                    ]
            return columns, data, '', dash.no_update
