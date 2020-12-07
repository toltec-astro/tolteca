#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
from dasha.web.templates.common import (
        LabeledDropdown, LabeledChecklist,
        LabeledInput,
        CollapseContent)
from dasha.web.extensions.db import dataframe_from_db
from dasha.web.templates.utils import partial_update_at

from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils import fileloc, odict_from_list

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
from dash.dependencies import Output, Input, State

from astropy.table import Table

import json
from schema import Schema, Optional, Or, Use

from pathlib import Path
import cachetools.func
import sqlalchemy.sql.expression as se
from sqlalchemy.sql import func as sqla_func

from ...tasks.dbrt import dbrt
from ....datamodels.toltec.basic_obs_data import BasicObsDataset
from ... import env_registry, env_prefix
from ....utils import get_user_data_dir


PROCESSED_KIDSDATA_SEARCH_PATH_ENV = (
        f"{env_prefix}_CUSTOM_PROCESSED_KIDSDATA_PATH")

env_registry.register(
        PROCESSED_KIDSDATA_SEARCH_PATH_ENV,
        "The path to locate processed KIDs data.",
        get_user_data_dir())

_processed_kidsdata_search_paths = list(map(
            Path,
            env_registry.get(PROCESSED_KIDSDATA_SEARCH_PATH_ENV).split(':')))


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


def _get_bods_index_from_toltecdb(
        obs_type='VNA', n_obs=500, obsnum_latest=None):
    logger = get_logger()

    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session

    if obsnum_latest is None:
        obsnum_latest = _get_toltecdb_obsnum_latest()
    obsnum_since = obsnum_latest - n_obs + 1

    logger.debug(
            f"query toltecdb for obsnum [{obsnum_since}:{obsnum_latest}]")
    stmt = se.select(
            [
                sqla_func.timestamp(
                    t['toltec'].c.Date,
                    t['toltec'].c.Time).label('time_obs'),
                t['toltec'].c.ObsNum.label('obsnum'),
                t['toltec'].c.SubObsNum.label('subobsnum'),
                t['toltec'].c.ScanNum.label('scannum'),
                t['toltec'].c.RoachIndex.label('roachid'),
                t['toltec'].c.RepeatLevel.label('repeat'),
                t['toltec'].c.TargSweepObsNum.label('cal_obsnum'),
                t['toltec'].c.TargSweepSubObsNum.label('cal_subobsnum'),
                t['toltec'].c.TargSweepScanNum.label('cal_scannum'),
                t['obstype'].c.label.label('raw_obs_type'),
                t['master'].c.label.label('master'),
                t['toltec'].c.FileName.label('source_orig'),
                ]).select_from(
                    t['toltec']
                    .join(
                        t['obstype'],
                        onclause=(
                            t['toltec'].c.ObsType
                            == t['obstype'].c.id
                            )
                    ).join(
                        t['master'],
                        onclause=(
                            t['toltec'].c.Master
                            == t['master'].c.id
                            )
                    )
                ).where(
                    se.and_(
                        t['toltec'].c.ObsNum <= obsnum_latest,
                        t['toltec'].c.ObsNum >= obsnum_since,
                        t['obstype'].c.label == obs_type,
                        t['master'].c.label == 'ICS'
                        ))
    tbl_raw_obs = Table.from_pandas(
            dataframe_from_db(stmt, session=session))
    # logger.debug(f"tbl_raw_obs: {tbl_raw_obs}")

    # make the required columns for
    # tolteca.datamodels.toltec.BasicObsDataset
    tbl_raw_obs['interface'] = [
            f'toltec{i}' for i in tbl_raw_obs['roachid']]

    # this need to handle various cases for remote file search
    tbl_raw_obs['source'] = [
            f'{s}' for s in tbl_raw_obs['source_orig']]

    return tbl_raw_obs


@cachetools.func.ttl_cache(maxsize=None, ttl=1)
def get_processed_file(raw_file_url):
    logger = get_logger()

    raw_filepath = fileloc(raw_file_url).path
    processed_filename = f'{raw_filepath.name[:-3]}_processed.nc'

    logger.debug(
            f"search processed file in "
            f"{list(map(str, _processed_kidsdata_search_paths))}")
    for p in _processed_kidsdata_search_paths:
        p = p.joinpath(processed_filename)
        logger.debug(f"check {p}")
        if p.exists():
            return fileloc(p).uri
    logger.debug(f"unable to find processed file for {raw_filepath}")
    return None


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_basic_obs_data(**kwargs):

    logger = get_logger()

    logger.debug(f'query basic obs data kwargs={kwargs}')

    tbl_bods = _get_bods_index_from_toltecdb(**kwargs)

    logger.debug(
            f'collect {len(tbl_bods)} entries from toltec files db'
            f' obsnum=[{tbl_bods["obsnum"][0]}:'
            f' {tbl_bods["obsnum"][-1]}]')

    # now collate by obs
    group_keys = ['obsnum', 'subobsnum', 'scannum', 'master', 'repeat']
    grouped = tbl_bods.group_by(group_keys)
    result = []  # this holds all the per obs info as a table
    raw_obs_sources = []
    for key, tbl in zip(grouped.groups.keys, grouped.groups):
        # for each group, we collate all per-interface entry to
        # a single raw obs
        tbl.sort(['roachid'])
        result.append(tbl[group_keys][0])
        ds = BasicObsDataset(index_table=tbl, open_=False)
        # common meta data for this data product
        raw_obs_source = {c: tbl[0][c] for c in tbl.colnames}
        # collect data items
        raw_obs_source['data_items'] = [
                {
                    'url': d.meta['file_loc'].uri,
                    'meta': {
                        k: d.meta[k]
                        for k in [
                            'interface',
                            'roachid',
                            ]
                        }
                    }
                for d in ds.bod_list
                ]
        raw_obs_sources.append(raw_obs_source)
    result = Table(rows=result, names=group_keys)
    result['source'] = raw_obs_sources
    result.sort('obsnum', reverse=True)
    return result


class KidsDataSelect(ComponentTemplate):

    _component_cls = dbc.Form
    _component_schema = Schema({
        Optional('multi', default=lambda: ['nwid', ]): Or(
            [str], Use(lambda v: list() if v is None else v)),
        })

    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # obsnum_multi = 'obsnum' in self.multi
        nwid_multi = self._nwid_multi = 'nwid' in self.multi

        container = self
        obsnum_input_container = container.child(
            dbc.Form, inline=True)
        self._obsnum_select = obsnum_input_container.child(
                LabeledDropdown(
                    label_text='Select ObsNum',
                    className='mt-3 w-auto mr-3',
                    size='sm',
                    )).dropdown
        self._obsnum_input = obsnum_input_container.child(
                LabeledInput(
                    label_text='Set Query Range',
                    className='mt-3 w-auto',
                    size='sm',
                    input_props={
                        # 'bs_size': 'sm'
                        'type': 'number',
                        'placeholder': 'e.g., 10886',
                        'min': 0
                        },
                    )).input
        self._nwid_select = container.child(
                LabeledChecklist(
                    label_text='Select Network',
                    checklist_props={
                        'options': make_network_options(),
                    },
                    multi=nwid_multi
                    )).checklist
        self._filepaths_store = container.child(dcc.Store, data=None)

    def setup_layout(self, app):

        details_container = self.child(
                CollapseContent(button_text='Details ...')).content

        super().setup_layout(app)

        @app.callback(
                [
                    Output(self.network_select.id, 'options'),
                    Output(self.network_select.id, 'value'),
                    ],
                [
                    Input(self.obsnum_select.id, 'value'),
                    ],
                [
                    State(self.network_select.id, 'value'),
                    ],
                )
        def update_network_options(obsnum_value, network_value):
            if obsnum_value is None:
                options = make_network_options(enabled=set())
                return options, dash.no_update
            obsnum_value = json.loads(obsnum_value)
            if network_value is None:
                network_value = []
            elif not self._nwid_multi:
                network_value = [network_value]
            network_value = set(network_value)
            # check processed state

            def has_processed(v):
                return get_processed_file(v['url']) is not None

            enabled = set(
                    v['meta']['roachid']
                    for v in obsnum_value if has_processed(v))
            network_value = network_value.intersection(enabled)
            if self._nwid_multi:
                network_value = list(network_value)
            elif len(network_value) > 0:
                network_value = next(iter(network_value))
            else:
                network_value = next(iter(enabled))
            options = make_network_options(
                    enabled=enabled)
            return options, network_value

        @app.callback(
                [
                    Output(self._filepaths_store.id, 'data'),
                    Output(details_container.id, 'children'),
                    ],
                [
                    Input(self.obsnum_select.id, 'value'),
                    Input(self.network_select.id, 'value'),
                    ]
                )
        @timeit
        def update_filepaths(obsnum_value, network_value):
            if obsnum_value is None or network_value is None:
                raise dash.exceptions.PreventUpdate
            obsnum_value = json.loads(obsnum_value)
            if not isinstance(network_value, list):
                network_value = [network_value]
            d = odict_from_list(
                    obsnum_value, key=lambda v: v['meta']['roachid'])

            def make_filepaths(nw):
                return {
                        'raw_obs': d[nw]['url'],
                        'raw_obs_processed': get_processed_file(d[nw]['url'])
                        }
            filepaths = {
                    nw: make_filepaths(nw)
                    for nw in network_value
                    }
            if not self._nwid_multi:
                # just return the single item
                filepaths = next(iter(filepaths.values()))

            details = html.Pre(
                    pformat_yaml({
                        'obsnum_value': obsnum_value,
                        'network_value': network_value,
                        'basic_obs_select_value': filepaths
                        }))

            return filepaths, details

    def setup_live_update_section(self, app, section, **kwargs):
        """Setup live update with section template.

        Parameters
        ----------
        section : `~dasha.web.templates.common.LiveUpdateSection`
            The live update section template instance to setup.

        **kwargs :
            Keyword arguments passed to :meth:`setup_live_update`.
        """
        self.setup_live_update(
                app,
                section.timer.inputs[0],
                loading_output=Output(section.loading.id, 'children'),
                error_output=Output(section.banner.id, 'children'),
                **kwargs
                )

    def setup_live_update(
            self, app,
            timer_input,
            loading_output=None,
            error_output=None,
            query_kwargs=None):
        """Setup live update.

        Parameters
        ----------
        timer_input : `~dash.dependencies.Input`
            The inputs of the live update callback.

        loading_output : `~dash.dependencies.Input`, optional
            The output of the live update indicator.

        error_outputs : `~dash.dependencies.Input`, optional
            The outputs for the error message.

        query_kwargs :
            Keyword arguments passed to the query function.
        """
        outputs = [
                Output(self.obsnum_select.id, 'options')
                ]
        if loading_output is not None:
            outputs.append(loading_output)
        if error_output is not None:
            outputs.append(error_output)

        @app.callback(
                outputs,
                [
                    timer_input,
                    ],
                # prevent_initial_call=True
                [
                    State(self._obsnum_input.id, 'value')
                    ]
                )
        @timeit
        def update_obsnum_select(n_calls, obsnum_latest):
            self.logger.debug(f"update obsnum select with obsnum_latest={obsnum_latest}")
            error_content = dbc.Alert(
                    'Unable to get data file list', color='danger')
            try:
                tbl_raw_obs = query_basic_obs_data(
                        obsnum_latest=obsnum_latest, **query_kwargs)
            except Exception as e:
                self.logger.debug(
                        f"error getting obs list: {e}", exc_info=True)
                if error_output is not None:
                    return partial_update_at(
                            slice(-2, None), ["", error_content])

            def make_option_label(s):
                return f'{s["obsnum"]}-{s["subobsnum"]}-{s["scannum"]}'

            def make_option_value(s):
                return json.dumps(s['data_items'])

            options = []
            for source in tbl_raw_obs['source']:
                option = {
                        'label': make_option_label(source),
                        'value': make_option_value(source)
                        }
                # for d in source['data_items']:
                #     if get_processed_file(d['url']) is None:
                #         option['disabled'] = True
                options.append(option)
            return options, "", ""

    @property
    def obsnum_select(self):
        return self._obsnum_select

    @property
    def network_select(self):
        return self._nwid_select

    @property
    def inputs(self):
        return [Input(self._filepaths_store.id, 'data')]


def make_network_options(enabled=None, disabled=None):
    """Return the options dict for select TolTEC detector networks."""
    if enabled is None:
        enabled = set(range(13))
    if disabled is None:
        disabled = set()
    if len(enabled.intersection(disabled)) > 0:
        raise ValueError('conflict in enabled and disabled kwargs.')
    result = list()
    for i in range(13):
        d = {
                'label': i,
                'value': i,
                'disabled': (i not in enabled) or (i in disabled)
                }
        result.append(d)
    return result
