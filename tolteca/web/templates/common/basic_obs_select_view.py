#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
from dasha.web.templates.collapsecontent import CollapseContent
import dash_html_components as html
from ...tasks.dbrt import dbrt
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Output, Input
from dasha.web.extensions.db import dataframe_from_db
from dasha.web.templates.utils import partial_update_at, fa
from tollan.utils.log import get_logger, timeit
from dash_table import DataTable
import dash
from sqlalchemy import select, and_
from sqlalchemy.sql import func as sqla_func
import networkx as nx
import json
import dash_cytoscape as cyto
import cachetools.func
import functools
from tolteca.datamodels.toltec import BasicObsData, BasicObsDataset
from types import SimpleNamespace
from tollan.utils import odict_from_list, fileloc


cyto.load_extra_layouts()


@functools.lru_cache(maxsize=None)
def get_bod(filepath):
    return BasicObsData(filepath, )


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_raw_obs():

    logger = get_logger()
    t = dbrt['tolteca'].tables
    session = dbrt['tolteca'].session
    # need this alias to join self
    reduced_data_prod = t['data_prod'].alias('reduced_data_prod')
    df_raw_obs = dataframe_from_db(
            select(
                [
                    t['dp_raw_obs'].c.pk,
                    t['data_prod_type'].c.label.label('dp_type'),
                    t['dp_raw_obs_type'].c.label.label('raw_obs_type'),
                    t['dp_raw_obs_master'].c.label.label('raw_obs_master')
                    ]
                + [
                    c for c in t['dp_raw_obs'].columns
                    if not c.name.endswith('pk')]
                + [
                    c for c in t['data_prod'].columns
                    if not c.name.endswith('pk')]
                + [
                    c for c in t['dpa_raw_obs_sweep_obs'].columns
                    if c.name not in ['pk', 'dp_raw_obs_pk']]
                + [
                    c for c in t['dpa_basic_reduced_obs_raw_obs'].columns
                    if c.name not in ['pk', 'dp_raw_obs_pk']]
                + [
                    reduced_data_prod.c.source_url.label(
                        'basic_reduced_obs_source_url'),
                    reduced_data_prod.c.source.label(
                        'basic_reduced_obs_source')
                    ]
                ).select_from(
                    t['dp_raw_obs']
                    .join(
                        t['dpa_raw_obs_sweep_obs'],
                        isouter=True,
                        onclause=(
                            t['dpa_raw_obs_sweep_obs'].c.dp_raw_obs_pk
                            == t['dp_raw_obs'].c.pk
                            )
                        )
                    .join(
                        t['dpa_basic_reduced_obs_raw_obs'],
                        isouter=True,
                        )
                    .join(t['data_prod'])
                    .join(t['data_prod_type'])
                    .join(t['dp_raw_obs_type'])
                    .join(t['dp_raw_obs_master'])
                    .join(
                        reduced_data_prod,
                        isouter=True,
                        onclause=(
                            t['dpa_basic_reduced_obs_raw_obs'].c
                            .dp_basic_reduced_obs_pk
                            == reduced_data_prod.c.pk
                            )
                        )
                ),
            session=session)
    # the pk columns needs to be in int, because when there is null
    # the automatic type is float
    for col in df_raw_obs.columns:
        if col.endswith('_pk') or col == 'pk':
            df_raw_obs[col] = df_raw_obs[col].fillna(-1.).astype(int)

    def get_source_keys(s):
        if s is None:
            return None
        return [ss['key'] for ss in s['sources']]

    df_raw_obs['_source_keys'] = df_raw_obs['source'].apply(
            lambda s: get_source_keys(s))
    df_raw_obs['source_keys'] = df_raw_obs['_source_keys'].apply(
            lambda v: None if v is None else ','.join(v))
    df_raw_obs['_source_keys_reduced'] = \
        df_raw_obs['basic_reduced_obs_source'] \
        .apply(lambda s: get_source_keys(s))
    df_raw_obs['source_keys_reduced'] = df_raw_obs[
            '_source_keys_reduced'].apply(
                lambda v: None if v is None else ','.join(v))
    df_raw_obs.set_index('pk', drop=False, inplace=True)
    logger.debug(f"get {len(df_raw_obs)} entries from dp_raw_obs")
    logger.debug(f"dtypes: {df_raw_obs.dtypes}")
    return df_raw_obs


@cachetools.func.ttl_cache(maxsize=1, ttl=1)
def query_toltec_userlog(obsnum_start, obsnum_stop):

    t = dbrt['toltec'].tables
    session = dbrt['toltec'].session
    df_userlog = dataframe_from_db(
            select(
                [
                    t['userlog'].c.ObsNum,
                    sqla_func.timestamp(
                        t['userlog'].c.Date,
                        t['userlog'].c.Time).label('DateTime'),
                    t['userlog'].c.User,
                    t['userlog'].c.Entry,
                    t['userlog'].c.Keyword,
                    ]
                ).where(
                    and_(
                        t['userlog'].c.ObsNum >= obsnum_start,
                        t['userlog'].c.ObsNum < obsnum_stop,
                    )), session=session)
    return df_userlog


def get_display_columns(df):
    return [
            c for c in df.columns
            if c not in [
                'source', 'source_url',
                'basic_reduced_obs_source',
                'basic_reduced_obs_source_url'
                ] and not c.startswith('_')
            ]


def get_calgroups(df_raw_obs):

    logger = get_logger()
    # create a list of connected components
    # each is a calgroup
    has_cal = df_raw_obs['dp_sweep_obs_pk'] > 0
    dg = nx.DiGraph()
    dg.add_edges_from(zip(
            df_raw_obs['dp_sweep_obs_pk'][has_cal], df_raw_obs['pk'][has_cal]))
    cc = list(nx.weakly_connected_components(dg))
    logger.debug(f'found {len(cc)} calgroups: {cc}')
    return cc


class BasicObsSelectView(ComponentTemplate):
    """This is a view that allow one to browse basic obs data.

    """
    _component_cls = html.Div

    logger = get_logger()

    def __init__(self, *args, file_search_paths=None, **kwargs):
        super().__init__(*args, **kwargs)
        # these paths are used to construct data file store objects
        # to locate the files.
        self._file_search_paths = file_search_paths

    def setup_layout(self, app):

        container = self
        # control_section, control_graph_section = container.grid(2, 1)
        # control_section.width = 3
        # control_graph_section.width = 9
        # ctx = {
        #         'control_graph_section': control_graph_section
        #         }

        control_section = container
        calgroup_selection_container, dataitem_selection_container = \
            control_section.grid(2, 1)

        ctx = self._setup_calgroup_selection(
                app, calgroup_selection_container, {})
        ctx = self._setup_dataitem_selection(
                app, dataitem_selection_container, ctx)

        super().setup_layout(app)
        self._ctx = ctx

    @property
    def ctx(self):
        """A dict that contains various related objects.
        """
        return getattr(self, '_ctx', None)

    @property
    def select_inputs(self):
        ctx = self.ctx
        if ctx is None:
            raise ValueError(
                    "setup_layout has to be called first")
        return [
                    Input(ctx['dataitem_select_drp'].id, 'value'),
                    Input(ctx['network_select_drp'].id, 'value'),
                    ]

    @staticmethod
    def bods_from_select_inputs(dataitem_value, network_value):
        """Return basic obs dataset from the selected items.

        This is to be used in user defined callbacks.
        """
        logger = get_logger()

        if dataitem_value is None or network_value is None:
            logger.debug("no update")
            raise dash.exceptions.PreventUpdate
        logger.debug(
                f"update bod with {dataitem_value} {network_value}")

        _df_raw_obs = query_raw_obs()
        raw_obs_pks = dataitem_value
        df_raw_obs = _df_raw_obs.loc[raw_obs_pks]

        nw = network_value
        bods = [get_bod(r.source['sources'][nw]['url'])
                for r in df_raw_obs.itertuples()]
        bods = BasicObsDataset(bod_list=bods)

        # get assoc objs
        def get_bod_from_source(source):
            if source is None:
                return None
            s = odict_from_list(source['sources'], key='key')
            key = f'toltec{nw}'
            if key in s:
                return get_bod(s[key]['url'])
            return None

        def get_sweep_bod(r):
            if r is None or r.dp_sweep_obs_pk <= 0:
                return None
            return get_bod_from_source(
                    _df_raw_obs.loc[r.dp_sweep_obs_pk]['source'])

        def get_reduced_bod(r):
            if r is None or r.basic_reduced_obs_source is None:
                return None
            return get_bod_from_source(
                    r.basic_reduced_obs_source)

        # add reduced bod to the bods table
        bods['reduced_bod'] = [
                get_reduced_bod(r)
                for r in df_raw_obs.itertuples()
                ]
        bods['sweep_bod'] = [
                get_sweep_bod(r)
                for r in df_raw_obs.itertuples()
                ]
        bods['reduced_sweep_bod'] = [
                get_reduced_bod(
                    None if r.dp_sweep_obs_pk <= 0 else
                    SimpleNamespace(
                        **_df_raw_obs.loc[r.dp_sweep_obs_pk].to_dict())
                    )
                for r in df_raw_obs.itertuples()
                ]
        return bods

    def _setup_calgroup_selection(self, app, container, ctx):

        select_container, error_container = container.grid(2, 1)

        select_container_form = select_container.child(dbc.Form, inline=True)
        calgroup_select_drp = select_container_form.child(
                dcc.Dropdown,
                placeholder="Select cal group",
                style={
                    'min-width': '240px'
                    },
                className='mr-2'
                )
        calgroup_refresh_btn = select_container_form.child(
                    dbc.Button, fa('fas fa-sync'),
                    color='link', className='my-2',
                    size='sm'
                    )
        details_container = select_container_form.child(
                        CollapseContent(button_text='Details ...')).content
        details_container.parent = container
        df_raw_obs_dt = details_container.child(
                DataTable,
                # style_table={'overflowX': 'scroll'},
                page_action='native',
                page_current=0,
                page_size=10,
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',
                    # 'overflowY': 'auto',
                    # 'height': '25vh',
                    },
                )

        @app.callback(
                [
                    Output(calgroup_select_drp.id, 'options'),
                    Output(df_raw_obs_dt.id, 'columns'),
                    Output(df_raw_obs_dt.id, 'data'),
                    Output(error_container.id, 'children'),
                    ],
                [
                    Input(calgroup_refresh_btn.id, 'n_clicks'),
                    ],
                )
        @timeit
        def update_calgroup(n_clicks):
            self.logger.debug("update calgroup")
            try:
                df_raw_obs = query_raw_obs()
            except Exception as e:
                self.logger.debug(f"error query db: {e}", exc_info=True)
                error_notify = dbc.Alert(
                        f'Query failed: {e.__class__.__name__}')
                return partial_update_at(-1, error_notify)
            use_cols = get_display_columns(df_raw_obs)
            df = df_raw_obs.reindex(columns=use_cols)
            cols = [{'label': c, 'id': c} for c in df.columns]
            data = df.to_dict('record')
            # make cal group options

            def make_option(g):
                g = list(g)
                n = len(g)
                c = df_raw_obs[df_raw_obs.index.isin(g)]
                c = c.sort_values(
                        by=['obsnum', 'subobsnum', 'scannum'])
                r = c.iloc[0]
                r1 = c.iloc[-1]
                label = f'{r["obsnum"]} - {r1["obsnum"]} ({n})'
                value = json.dumps(g)

                # check source exists
                def has_data(t):
                    for s in t['source']:
                        if s is None:
                            continue
                        for ss in s['sources']:
                            if fileloc(ss['url']).exists():
                                return True
                    return False

                return {
                        'label': label,
                        'value': value,
                        'disabled': not has_data(c)
                        }

            calgroups = get_calgroups(df_raw_obs)
            calgroups = sorted(calgroups, key=lambda g: max(g), reverse=True)
            options = list(map(make_option, calgroups))
            return options, cols, data, ""

        ctx['calgroup_select_drp'] = calgroup_select_drp
        return ctx

    def _setup_dataitem_selection(self, app, container, ctx):

        control_container, assoc_view_graph_container = container.grid(1, 2)
        control_container.width = 4
        assoc_view_graph_container.width = 8
        dataitem_select_container, network_select_container, error_container =\
            control_container.grid(3, 1)
        dataitem_select_container_form = dataitem_select_container.child(
                dbc.Form, inline=True)
        dataitem_select_drp = dataitem_select_container_form.child(
            dcc.Dropdown,
            placeholder="Select obs",
            multi=True,
            style={
                'min-width': '240px'
                },
            )
        dataitem_details_container = dataitem_select_container_form.child(
                        CollapseContent(button_text='Details ...')).content
        dataitem_userlog_container = dataitem_select_container_form.child(
                        CollapseContent(button_text='User logs ...')).content
        dataitem_details_container.parent = container
        dataitem_userlog_container.parent = container
        dataitem_details_container.className = 'mb-4'  # fix the bottom margin
        df_raw_obs_dt = dataitem_details_container.child(
                DataTable,
                # style_table={'overflowX': 'scroll'},
                page_action='native',
                page_current=0,
                page_size=20,
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',
                    # 'overflowY': 'auto',
                    # 'height': '25vh',
                    },
                )

        # assoc_view_graph_collapse = assoc_view_graph_container.child(
        #                 CollapseContent(button_text='Select on graph ...')
        #                 ).content
        assoc_view_graph_collapse = assoc_view_graph_container
        assoc_view_ctx = self._setup_assoc_view(
                app, assoc_view_graph_collapse)
        assoc_view_graph = assoc_view_ctx['assoc_view_graph']
        assoc_view_graph_legend = assoc_view_ctx['assoc_view_graph_legend']

        # network_options_store = container.child(dcc.Store, data=None)
        # network_select_ctx = self._setup_network_selection(
        #         app, container.child(dbc.Row).child(dbc.Col),
        #         {
        #             'network_options_store': network_options_store
        #             }
        #         )
        network_select_container_form = network_select_container.child(
                dbc.Form, inline=True)
        network_select_container.className = 'mt-1'  # text-monospace'
        network_select_ctx = self._setup_network_selection_simple(
                app, network_select_container_form, {})
        network_select_drp = network_select_ctx['network_select_drp']
        network_details_container = network_select_container_form.child(
                        CollapseContent(button_text='Details ...')).content
        network_details_container.parent = container
        network_details_container.className = 'mb-4'  # fix the bottom margin
        df_bod_dt = network_details_container.child(
                DataTable,
                # style_table={'overflowX': 'scroll'},
                page_action='native',
                page_current=0,
                page_size=20,
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',
                    # 'overflowY': 'auto',
                    # 'height': '25vh',
                    },
                )

        @app.callback(
                Output(dataitem_select_drp.id, 'value'),
                [
                    Input(assoc_view_graph.id, 'selectedNodeData'),
                    ]
                )
        def update_from_assoc_graph_view(data):
            if not data:
                raise dash.exceptions.PreventUpdate
            return [int(d['id']) for d in data]

        assoc_view_graph.stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                    },
                },
            {
                'selector': ':selected',
                "style": {
                    "border-width": 2,
                    "border-color": "#222222",
                    "border-opacity": 1,
                    }
                }
            ]
        dispatch_type_color = {
                'TUNE': '#ccaaff',
                'Targ': '#aaaaff',
                'Timestream': '#99ccff'
                }
        type_stylesheet = [
            {
                'selector': f'[type = "{t}"]',
                'style': {
                    'background-color': c,
                    },
                }
            for t, c in dispatch_type_color.items()
            ]
        assoc_view_graph.stylesheet.extend(type_stylesheet)
        assoc_view_graph_legend.stylesheet = [
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)'
                        },
                    },
                ] + type_stylesheet

        @app.callback(
                Output(df_raw_obs_dt.id, 'style_data_conditional'),
                [
                    Input(dataitem_select_drp.id, 'value'),
                    ]
                )
        def update_selected_data_items(dataitem_value):
            if dataitem_value is None:
                return None
            dt_style = [
                    {
                        'if': {
                            'filter_query': '{{{}}} = {}'.format(
                                'pk', pk),
                        },
                        'backgroundColor': '#eeeeee',
                        }
                    for pk in dataitem_value
                    ]
            return dt_style

        @app.callback(
                [
                    Output(dataitem_select_drp.id, 'options'),
                    Output(network_select_drp.id, 'options'),
                    Output(df_raw_obs_dt.id, 'columns'),
                    Output(df_raw_obs_dt.id, 'data'),
                    Output(dataitem_userlog_container.id, 'children'),
                    Output(assoc_view_graph.id, 'elements'),
                    Output(assoc_view_graph_legend.id, 'elements'),
                    Output(error_container.id, 'children'),
                    ],
                [
                    Input(ctx['calgroup_select_drp'].id, 'value'),
                    ],
                )
        def update_calgroup(calgroup_value):
            if calgroup_value is None:
                raise dash.exceptions.PreventUpdate

            self.logger.debug("update dataitem")
            try:
                df_raw_obs = query_raw_obs()
                raw_obs_pks = json.loads(calgroup_value)
                df_raw_obs = df_raw_obs.loc[raw_obs_pks]
            except Exception as e:
                self.logger.debug(f"error query db: {e}", exc_info=True)
                error_notify = dbc.Alert(
                        f'Query failed: {e.__class__.__name__}')
                return partial_update_at(-1, error_notify)
            use_cols = get_display_columns(df_raw_obs)
            df = df_raw_obs.reindex(columns=use_cols)
            cols = [{'label': c, 'id': c} for c in df.columns]
            data = df.to_dict('record')

            # build the DAG to show the assocs
            nodes = [
                {
                    'data': {
                        'id': str(r.pk),
                        'label': f"{r.obsnum}-{r.subobsnum}-{r.scannum}",
                        'type': r.raw_obs_type,
                        'reduced': r.dp_basic_reduced_obs_pk > 0
                        },
                    }
                for r in df.itertuples()
                ]
            edges = []
            for r in df.itertuples():
                if r.dp_sweep_obs_pk < 0:
                    continue
                edges.append(
                    {
                        'data': {
                            'source': str(r.dp_sweep_obs_pk),
                            'target': str(r.pk),
                            'lable': ''
                            },
                        'selectable': False
                    })
            self.logger.debug(
                    f"collected {len(nodes)} nodes and {len(edges)} edges")
            elems = nodes + edges
            # legend
            elems_legend = [
                    {
                        'data': {
                            'id': t,
                            'label': t,
                            'type': t,
                            },
                        }
                    for t in set(df['raw_obs_type'])
                    ]
            # make cal group options

            def make_dataitem_option(r):
                label = (
                    f'{r.obsnum}-{r.subobsnum}-{r.scannum}-{r.raw_obs_type}')
                value = r.pk
                return {'label': label, 'value': value}

            options = list(map(make_dataitem_option, df_raw_obs.itertuples()))
            # get all keys
            source_keys = set()
            for k in df_raw_obs['_source_keys']:
                if k is None:
                    continue
                source_keys = source_keys.union(set(k))

            def make_network_options(source_key):
                value = int(source_key.replace('toltec', ''))
                for r in df_raw_obs.itertuples():
                    if r.source is None:
                        return {
                                'label': source_key,
                                'value': value
                                }
                    s = odict_from_list(r.source['sources'], key='key')
                    if source_key in s:
                        if 'meta' not in s[source_key]:
                            return {
                                'label': source_key,
                                'value': value
                                }
                        m = s[source_key]['meta']
                        return {
                                'label': (
                                    f'{source_key} '
                                    f'({m["n_tones"]}/{m["n_tones_design"]})'),
                                'value': value,
                                }

            network_options = [
                    make_network_options(source_key)
                    for source_key in source_keys
                    ]

            df_toltec_userlog = query_toltec_userlog(
                    min(df_raw_obs['obsnum']),
                    max(df_raw_obs['obsnum']),
                    )

            df_toltec_userlog_dt = DataTable(
                style_cell={'padding': '0.5em'},
                style_table={
                    # 'overflowX': 'auto',
                    'width': '100%',
                    },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Entry'},
                        'textAlign': 'left',
                        },
                    ],
                data=df_toltec_userlog.to_dict('record'),
                columns=[
                    {
                        'label': c,
                        'id': c
                        }
                    for c in df_toltec_userlog.columns
                    ]
                )
            return (
                    options, network_options, cols, data,
                    df_toltec_userlog_dt,
                    elems, elems_legend,
                    "")

        _outputs = [
                    Output(df_bod_dt.id, 'columns'),
                    Output(df_bod_dt.id, 'data'),
                    ]

        @app.callback(
                _outputs,
                [
                    Input(dataitem_select_drp.id, 'value'),
                    Input(network_select_drp.id, 'value'),
                    ],
                )
        def update_bod_dt(dataitem_value, network_value):
            bods = self.bods_from_select_inputs(dataitem_value, network_value)
            tbl = bods.index_table
            # get all displayable columns
            use_cols = [
                    c for c in tbl.colnames
                    if not (
                        tbl.dtype[c].hasobject
                        or c in ['source', 'filename_orig']
                        or tbl.dtype[c].ndim > 0
                        )]
            df = tbl[use_cols].to_pandas()
            cols = [{'label': c, 'id': c} for c in df.columns]
            data = df.to_dict('record')
            return cols, data
        ctx.update({
                'control_container': control_container,
                'dataitem_select_drp': dataitem_select_drp,
                'network_select_drp': network_select_drp,
                })
        return ctx

    def _setup_assoc_view(self, app, container):

        graph_container, graph_controls_container = container.grid(2, 1)
        graph_controls_container.className = 'px-0'
        graph_controls_form = graph_controls_container.child(
                dbc.Form, inline=True)
        graph_reset_btn = graph_controls_form.child(
                dbc.Button,
                [
                    fa('fas fa-undo pr-2'),
                    'Reset pan/zoom'
                    ],
                size='sm', color='link',
                className='pl-0 pr-4'
                )
        graph_controls_form.child(
                dbc.Button(
                    [
                        fa('fas fa-info pr-2'),
                        'Shift + Drag to select multiple nodes'
                        ],
                    size='sm',
                    color='link',
                    style={
                        'color': '#aaaaaa'
                        },
                    className='pl-0 pr-4'
                    )
                )

        # graph_layout_select_group = graph_controls_container.child(
        #         dbc.FormGroup)
        # graph_layout_select_group.child(
        #         dbc.Label,
        #         'Layout:',
        #         className='mr-3'
        #         )
        # cyto_layouts = [
        #         'random', 'grid', 'circle', 'concentric',
        #         'breadthfirst', 'cose', 'cose-bilkent',
        #         'cola', 'euler', 'spread', 'dagre',
        #         'klay',
        #         ]

        # graph_layout_select = graph_layout_select_group.child(
        #         dbc.Select,
        #         options=[
        #             {'label': l, 'value': l}
        #             for l in cyto_layouts
        #             ],
        #         value='dagre',
        #         )

        def _get_layout(value):
            return {
                    'name': value,
                    'animate': True,
                    'nodeDimensionsIncludeLabels': True,
                    'rankDir': 'LR',
                    }

        height = '250px'
        graph_container_row = graph_container.child(dbc.Row)
        graph = graph_container_row.child(
                dbc.Col, width=10,
                className='border',
                style={'border-color': '#aaaaaa'}
                ).child(
                cyto.Cytoscape,
                # layout_=_get_layout(graph_layout_select.value),
                layout_=_get_layout('dagre'),
                elements=[],
                style={
                    'min-height': height,
                    },
                minZoom=0.6,
                # zoomingEnabled=True,
                # userZoomingEnabled=True,
                # userPanningEnabled=False,
                boxSelectionEnabled=True,
                autoungrabify=True,
                )
        graph_legend = graph_container_row.child(dbc.Col, width=2).child(
                cyto.Cytoscape,
                layout_=_get_layout('grid'),
                elements=[],
                style={
                    'min-height': height,
                    },
                # minZoom=0.8,
                userZoomingEnabled=True,
                userPanningEnabled=False,
                boxSelectionEnabled=False,
                autoungrabify=True,
                )

        @app.callback(
                Output(graph.id, 'layout'),
                [
                    # Input(graph_layout_select.id, 'value')
                    Input(graph_reset_btn.id, 'n_clicks')
                    ]
                )
        def update_graph_layout(value):
            if value is None:
                return dash.no_update
            layout = _get_layout('dagre')
            layout['n_clicks'] = value
            return layout

        return {
                'assoc_view_graph': graph,
                'assoc_view_graph_legend': graph_legend,
                }

    def _setup_network_selection(self, app, container, ctx):
        # set up a network selection section
        network_select_section = container.child(dbc.Row).child(dbc.Col)
        network_select_section.child(dbc.Label, 'Select network(s):')
        network_select_row = network_select_section.child(
                dbc.Row, className='mx-0')
        preset_container = network_select_row.child(dbc.Col)
        preset = preset_container.child(
                dbc.Checklist, persistence=False,
                labelClassName='pr-1', inline=True)
        preset.options = [
                {'label': 'All', 'value': 'all'},
                {'label': '1.1mm', 'value': '1.1 mm Array'},
                {'label': '1.4mm', 'value': '1.4 mm Array'},
                {'label': '2.0mm', 'value': '2.0 mm Array'},
                ]
        preset.value = []
        network_container = network_select_row.child(dbc.Col)
        # make three button groups
        network_select = network_container.child(
                dbc.Checklist, persistence=False,
                labelClassName='pr-1', inline=True)

        network_options = [
                {'label': f'N{i}', 'value': i}
                for i in range(13)
                ]
        array_names = ['1.1 mm Array', '1.4 mm Array', '2.0 mm Array']
        preset_networks_map = dict()
        preset_networks_map['1.1 mm Array'] = set(
                o['value'] for o in network_options[0:7])
        preset_networks_map['1.4 mm Array'] = set(
                o['value'] for o in network_options[7:10])
        preset_networks_map['2.0 mm Array'] = set(
                o['value'] for o in network_options[10:13])
        preset_networks_map['all'] = functools.reduce(
                set.union, (preset_networks_map[k] for k in array_names))

        # this shall be a dcc.store that provide mapping of
        # value to enabled state
        network_options_store = ctx['network_options_store']

        # a callback to update the check state
        @app.callback(
                [
                    Output(network_select.id, "options"),
                    Output(network_select.id, "value"),
                    ],
                [
                    Input(preset.id, "value"),
                    Input(network_options_store.id, 'data')
                ]
            )
        def on_preset_change(preset_values, network_options_store_data):
            # this is all the nws
            nw_values = set()
            for pv in preset_values:
                nw_values = nw_values.union(preset_networks_map[pv])
            options = [
                    dict(**o) for o in network_options
                    if o['value'] in nw_values]
            values = list(nw_values)

            for option in options:
                v = option['value']
                # this is to update the option with the store data
                # json makes all the dict keys str
                option.update(
                        network_options_store_data.get(str(v), dict()))
                if option['disabled']:
                    values.remove(v)
            return options, values

        return {'network_select_chklst': network_select}

    def _setup_network_selection_simple(self, app, container, ctx):
        # set up a network selection section
        network_select_section = container
        # make three button groups
        network_select_drp = network_select_section.child(
                dcc.Dropdown,
                placeholder="Select network",
                style={
                    'min-width': '240px'
                    },
                )
        return {'network_select_drp': network_select_drp}
