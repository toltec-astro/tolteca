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
from sqlalchemy.sql import alias
from sqlalchemy.sql import func as sqla_func

from ...tasks.dbrt import dbrt
from ....datamodels.toltec.basic_obs_data import BasicObsDataset, BasicObsData
from ... import env_registry, env_prefix
from ....utils import get_user_data_dir


PROCESSED_KIDSDATA_SEARCH_PATH_ENV = (
        f"{env_prefix}_CUSTOM_PROCESSED_KIDSDATA_PATH")

env_registry.register(
        PROCESSED_KIDSDATA_SEARCH_PATH_ENV,
        "The path to locate processed KIDs data.",
        get_user_data_dir().as_posix())

_processed_kidsdata_search_paths = list(map(
            Path,
            env_registry.get(PROCESSED_KIDSDATA_SEARCH_PATH_ENV).split(':')))


def _get_toltecdb_obsnum_latest():
    logger = get_logger()

    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables['toltec']
    stmt = se.select([t.c.ObsNum]).order_by(se.desc(t.c.id)).limit(1)

    with dbrt['toltec'].session_context as session:
        obsnum_latest = session.execute(stmt).scalar()
    logger.debug(f"latest obsnum: {obsnum_latest}")
    return obsnum_latest


def _get_bods_index_from_toltecdb(
        obs_type='VNA', n_obs=1, obsnum_latest=None):
    logger = get_logger()

    tname = 'toltec_r1'

    dbrt.ensure_connection('toltec')
    t = dbrt['toltec'].tables

    if obsnum_latest is None:
        obsnum_latest = _get_toltecdb_obsnum_latest()

    logger.debug(f"latest obsnum: {obsnum_latest}")
    obsnum_since = obsnum_latest - n_obs + 1
    logger.debug(
            f"query toltecdb for obsnum [{obsnum_since}:{obsnum_latest}] to find id range")

    # run a query to figure out actual id for obsnum_since to obsnum_latest
    stmt = se.select(
            [
                sqla_func.min(t[tname].c.id).label('id_min'),
                sqla_func.max(t[tname].c.id).label('id_max'),
                sqla_func.max(t[tname].c.ObsNum).label('ObsNum'),
                ]).select_from(
                    t[tname]
                    .join(
                        t['obstype'],
                        onclause=(
                            t[tname].c.ObsType
                            == t['obstype'].c.id
                            )
                    ).join(
                        t['master'],
                        onclause=(
                            t[tname].c.Master
                            == t['master'].c.id
                            )
                    )
                ).where(
                    se.and_(
                        t[tname].c.ObsNum <= obsnum_latest,
                        t[tname].c.ObsNum >= obsnum_since,
                        t['obstype'].c.label == obs_type,
                        # t['master'].c.label == 'ICS'
                        )
                ).group_by(
                    t[tname].c.ObsNum.label('obsnum'),
                    t[tname].c.SubObsNum.label('subobsnum'),
                    t[tname].c.ScanNum.label('scannum'),
                    t[tname].c.RepeatLevel.label('repeat'),
                    t[tname].c.Master.label('master_id'),
                ).order_by(
                        se.desc(t[tname].c.id)
                ).limit(n_obs)
    df_group_ids = dataframe_from_db(stmt, session=dbrt['toltec'].session)
    if len(df_group_ids) == 0:
        # no recent timestream obs found
        return
    id_min = df_group_ids['id_min'].min()
    id_max = df_group_ids['id_max'].max()
    logger.debug(
            f"id range of n_obs={n_obs} is [{id_min}, {id_max}] "
            f"obsnum range [{df_group_ids['ObsNum'].min()}, {df_group_ids['ObsNum'].max()}]")

    t_cal = alias(t[tname])
    stmt = se.select(
            [
                sqla_func.timestamp(
                    t[tname].c.Date,
                    t[tname].c.Time).label('time_obs'),
                t[tname].c.ObsNum.label('obsnum'),
                t[tname].c.SubObsNum.label('subobsnum'),
                t[tname].c.ScanNum.label('scannum'),
                t[tname].c.RoachIndex.label('roachid'),
                t[tname].c.RepeatLevel.label('repeat'),
                t[tname].c.TargSweepObsNum.label('cal_obsnum'),
                t[tname].c.TargSweepSubObsNum.label('cal_subobsnum'),
                t[tname].c.TargSweepScanNum.label('cal_scannum'),
                t_cal.c.FileName.label('cal_source_orig'),
                t['obstype'].c.label.label('raw_obs_type'),
                t['master'].c.label.label('master'),
                t[tname].c.FileName.label('source_orig'),
                ]).select_from(
                    t[tname]
                    .join(
                        t['obstype'],
                        onclause=(
                            t[tname].c.ObsType
                            == t['obstype'].c.id
                            )
                    ).join(
                        t['master'],
                        onclause=(
                            t[tname].c.Master
                            == t['master'].c.id
                            )
                    ).join(
                        t_cal,
                        onclause=(
                            se.and_(
                                t[tname].c.TargSweepObsNum == t_cal.c.ObsNum,
                                t[tname].c.TargSweepSubObsNum == t_cal.c.SubObsNum,
                                t[tname].c.TargSweepScanNum == t_cal.c.ScanNum,
                                t[tname].c.RoachIndex == t_cal.c.RoachIndex,
                                t[tname].c.Master == t_cal.c.Master,
                                )),
                        isouter=True,
                    ),
                ).where(
                    se.and_(
                        t[tname].c.id <= id_max,
                        t[tname].c.id >= id_min,
                        t['obstype'].c.label == obs_type,
                        # t['master'].c.label == 'ICS'
                        ))

    session = dbrt['toltec'].session
    # re_cal
    tbl_raw_obs = Table.from_pandas(
            dataframe_from_db(stmt, session=session))
    logger.debug(f"tbl_raw_obs: {tbl_raw_obs}")

    # make the required columns for
    # tolteca.datamodels.toltec.BasicObsDataset
    tbl_raw_obs['interface'] = [
            f'toltec{i}' for i in tbl_raw_obs['roachid']]

    # this need to handle various cases for remote file search
    # TODO fix this handling of path in db
    def fix_raw_filepath(p):
        p = str(p)
        if p.startswith('/data/'):
            return p
        return f'/data/{p}'
    tbl_raw_obs['source'] = [
            fix_raw_filepath(s) for s in tbl_raw_obs['source_orig']]
    tbl_raw_obs['cal_source'] = [
            fix_raw_filepath(s) for s in tbl_raw_obs['cal_source_orig']]
    # tbl_raw_obs['obsnum', 'subobsnum', 'scannum', 'roachid'].pprint_all()

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
            logger.debug(f'use processed file {p}')
            return fileloc(p).uri
    logger.debug(f"unable to find processed file for {raw_filepath}")
    return None


@cachetools.func.ttl_cache(maxsize=1, ttl=2)
def query_basic_obs_data(**kwargs):

    logger = get_logger()

    logger.debug(f'query basic obs data kwargs={kwargs}')

    with timeit('query toltecdb'):
        obs_type = kwargs.pop('obs_type')
        if obs_type == 'Timestream':
            tbl_bods = [
                    _get_bods_index_from_toltecdb(obs_type='Timestream', **kwargs),
                    _get_bods_index_from_toltecdb(obs_type='Nominal', **kwargs)
                    ]
            tbl_bods = vstack([t for t in tbl_bods if t is not None])
        else:
            tbl_bods = _get_bods_index_from_toltecdb(obs_type=obs_type, **kwargs)
        if tbl_bods is None:
            return

    logger.debug(
            f'collect {len(tbl_bods)} entries from toltec files db'
            f' obsnum=[{tbl_bods["obsnum"][0]}:'
            f' {tbl_bods["obsnum"][-1]}]')

    # now collate by obs
    group_keys = ['obsnum', 'subobsnum', 'scannum', 'master', 'repeat']
    with timeit('group baods by roach id'):
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
                    'url_cal': fileloc(cal_source).uri,
                    'meta': {
                        k: d.meta[k]
                        for k in [
                            'interface',
                            'roachid',
                            ]
                        }
                    }
                for d, cal_source in zip(ds.bod_list, ds['cal_source'])
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
        if nwid_multi:
            array_multi = self._array_multi = 'array' in self.multi
            self._array_select = container.child(
                    LabeledChecklist(
                        label_text='Select by Array',
                        checklist_props={
                            'options': make_array_options(),
                        },
                        multi=array_multi
                        )).checklist
        else:
            self._array_select = None
        self._filepaths_store = container.child(dcc.Store, data=None)

    def setup_layout(self, app):

        details_container = self.child(
                CollapseContent(button_text='Details ...')).content

        super().setup_layout(app)

        @app.callback(
                Output(self.network_select.id, 'options'),
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
                return options
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
            options = make_network_options(enabled=enabled)
            return options

        def update_network_value_for_options(network_options, network_value):
            enabled = set(o['value'] for o in network_options if not o['disabled'])
            if network_value is None:
                network_value = []
            if not self._nwid_multi:
                if not isinstance(network_value, list):  # this happends somehow
                    # make list of values
                    network_value = [network_value]
            network_value = list(set(network_value).intersection(enabled))
            if self._nwid_multi:
                pass
            elif len(network_value) > 0:
                network_value = network_value[0]
            elif len(enabled) > 0:
                network_value = next(iter(enabled))
            else:
                network_value = None
            return network_value


        if self.array_select is not None:
            @app.callback(
                    [
                        Output(self.array_select.id, 'options'),
                        Output(self.array_select.id, 'value'),
                        ],
                    [
                        Input(self.network_select.id, 'options'),
                        ],
                    )
            def update_array_options(network_options):
                options = make_array_options(
                        network_options=network_options)
                return options, ''

            @app.callback(
                    Output(self.network_select.id, 'value'),
                    [
                        Input(self.network_select.id, 'options'),
                        Input(self.array_select.id, 'value'),
                        ],
                    [
                        State(self.network_select.id, 'value'),
                        ]
                    )
            def update_network_select_value_with_array(
                    network_select_options, array_select_value,
                    network_select_value,
                    ):
                value = get_networks_for_array(array_select_value)
                return update_network_value_for_options(network_select_options, value)
        else:
            @app.callback(
                    Output(self.network_select.id, 'value'),
                    [
                        Input(self.network_select.id, 'options'),
                        ],
                    [
                        State(self.network_select.id, 'value'),
                        ],
                    )
            def update_network_select_value_without_array(
                    options, network_value):
                return update_network_value_for_options(options, network_value)

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
                if nw not in d:
                    return None
                return {
                        'raw_obs': d[nw]['url'],
                        'raw_obs_processed': get_processed_file(d[nw]['url']),
                        'cal_obs': d[nw]['url_cal'],
                        'cal_obs_processed': get_processed_file(d[nw]['url_cal']),
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
            self.logger.debug(
                    f"update obsnum select with obsnum_latest={obsnum_latest}")
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
    def array_select(self):
        return self._array_select

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


_array_option_specs = {
        'a1100': {
            'label': '1.1mm',
            'nwids': [0, 1, 2, 3, 4, 5, 6],
            },
        'a1400': {
            'label': '1.4mm',
            'nwids': [7, 8, 9, 10],
            },
        'a2000': {
            'label': '2.0mm',
            'nwids': [11, 12],
            },
        'ALL': {
            'label': 'All',
            'nwids': list(range(13))
            }
        }


def make_array_options(enabled=None, disabled=None, network_options=None):
    """Return the options dict for select TolTEC arrays."""
    if enabled is None:
        enabled = set(_array_option_specs.keys())
    if disabled is None:
        disabled = set()
    if network_options is not None:
        nw_disabled = set(
                int(n['value']) for n in network_options if n['disabled'])
        for k, a in _array_option_specs.items():
            if set(a['nwids']).issubset(nw_disabled):
                disabled.add(k)
                enabled.discard(k)
    if len(enabled.intersection(disabled)) > 0:
        raise ValueError('conflict in enabled and disabled kwargs.')

    result = list()
    for k, a in _array_option_specs.items():
        d = {
                'label': a['label'],
                'value': k,
                'disabled': (k not in enabled) and (k in disabled)
                }
        result.append(d)
    return result


def get_networks_for_array(array_select_values):
    if array_select_values is None:
        return None
    checked = set()
    for k in array_select_values:
        checked = checked.union(set(_array_option_specs[k]['nwids']))
    return list(checked)
