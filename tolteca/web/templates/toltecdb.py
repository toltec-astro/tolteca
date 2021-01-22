#! /usr/bin/env python

from dasha.web.templates import ComponentTemplate
from dash_table import DataTable
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from tollan.utils.log import get_logger, timeit
from dash.dependencies import Input, Output, State
import dash_html_components as html
# from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.extensions.db import db, dataframe_from_db
from dasha.web.extensions.dasha import resolve_url
import cachetools.func
from sqlalchemy import select
# from flask_sqlalchemy import BaseQuery
import pandas as pd
import dash_defer_js_import as dji
import json
import dash_cytoscape as cyto
import dash
from tollan.utils.db import SqlaDB
import base64
from astropy.table import Table
from tolteca.datamodels.toltec import BasicObsDataset
from tolteca.datamodels.db.toltec import data_prod


cyto.load_extra_layouts()


class ToltecDB(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_live_update_header(self, app, container, title, interval):
        header = container.child(dbc.Row, className='mb-2').child(
                dbc.Col, className='d-flex align-items-center')
        header.child(html.H5, title, className='mr-4 my-0')
        timer = header.child(dcc.Interval, interval=interval)
        loading = header.child(dcc.Loading)
        error = container.child(dbc.Row).child(dbc.Col)
        return timer, loading, error

    def _setup_databases(self):
        _db = dict()
        for bind in ('tolteca', 'toltec'):
            try:
                _db[bind] = SqlaDB.from_flask_sqla(db, bind=bind)
            except Exception as e:
                self.logger.warning(
                        f"unable to connect to db bind={bind}: {e}",
                        exc_info=True)
        # tolteca conn is required
        if 'tolteca' not in _db:
            raise RuntimeError("unable to connect to required db tolteca.")
        # load tolteca tables
        data_prod.init_db(_db['tolteca'])

        # reflect toltec tables if it exists
        try:
            _db['toltec'].reflect_tables()
        except Exception as e:
            self.logger.warning(
                    f"unable to connect to toltec db: {e}",
                    exc_info=True)

        self._db = _db
        return self._db

    def setup_layout(self, app):
        self._setup_databases()

        container = self
        body = container.child(dbc.Row).child(dbc.Col)

        tolteca_db_view_container = body.child(
                dbc.Row).child(dbc.Col, width=12)
        graph_view_container = body.child(dbc.Row).child(dbc.Col, width=12)
        dataset_upload_container = body.child(dbc.Row).child(
                dbc.Col, width=12)

        # tolteca db tables
        exclude_choices = []
        for bindkey, table_name, section_name in [
                ('toltec', 'raw_obs_table_info', 'Raw Obs Tables'),
                ('tolteca', 'data_prod_type', 'Data Prod Tables'),
                ('tolteca', 'data_prod_assoc_type', 'Data Prod Assoc Tables'),
                # this one is to capture all other tables
                # has to be the last in the list
                ('tolteca', '_table_info', 'Info Tables'),
                ]:
            section = tolteca_db_view_container.child(
                    dbc.Row).child(dbc.Col, width=12)
            section_header = section.child(dbc.Row).child(
                    dbc.Col, width=12)
            section_header.child(html.Hr())
            section_body = section.child(dbc.Row)
            select_container = section_body.child(
                    dbc.Col, className='mb-4',
                    xl=4, lg=6, xs=12)
            view_container = section_body.child(
                    dbc.Col,
                    xl=8, lg=6, xs=12)

            # get choices
            if table_name == 'raw_obs_table_info':
                # data file tables
                choices, info = self._get_raw_obs_table_info()
            else:
                # data prod tables
                choices, info = self._query_choices_table(table_name)
                # filter choices so they don't duplicate
                if table_name == '_table_info':
                    _choices = [c for c in choices if c not in exclude_choices]
                    mask = choices.isin(_choices)
                    choices = choices[mask]
                    info = info[mask]
                else:
                    exclude_choices.extend(choices.tolist())
            radio_items = self._setup_db_select(
                    app,
                    select_container,
                    section_name, choices, info, bindkey)

            self._setup_db_view(
                    app,
                    view_container,
                    radio_items, bindkey)

        update_listener = self._setup_graph_view(app, graph_view_container)
        self._setup_dataset_upload(
                app, dataset_upload_container, update_listener)
        super().setup_layout(app)

    def _get_raw_obs_table_info(self):
        data = [
                ('master', 'The master.'),
                ('obstype', 'The obs type.'),
                ('userlog', 'The user log.'),
                ('toltec', 'The TolTEC data files.'),
                ('toltec_r1',
                    'The locally repeated data files (repeat level 1)'),
                ]
        colnames = ['name', 'desc']
        info = pd.DataFrame(data={
            c: v for c, v in zip(colnames, zip(*data))
            })
        return info['name'], info

    @cachetools.func.ttl_cache(maxsize=1, ttl=60 * 60)
    def _query_choices_table(self, table_name, exclude_parent=False):
        session = self._db['tolteca'].session
        tbl = self._db['tolteca'].tables[table_name]

        stmt = select([tbl])

        info = dataframe_from_db(
                stmt, session=session)

        if 'name' in info.columns:
            key = 'name'
        else:
            key = 'label'
        if exclude_parent:
            info = info[info[key] != table_name.rsplit('_type')[0]]
        return info[key], info

    def _setup_db_select(self, app, container, title, choices, info, bindkey):
        timer, loading, error = self._setup_live_update_header(
                app, container, title, 1000)

        select_container = container.child(dbc.Row).child(dbc.Col)
        _db = self._db[bindkey]

        _t = _db.tables

        @timeit
        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def query_table_sizes(*table_names):
            result = []
            with _db.session_context as session:
                for n in table_names:
                    if n not in _t:
                        result.append(None)
                    else:
                        result.append(
                            session.execute(_t[n].count()).scalar())
            return result

        radio_items = select_container.child(
            dbc.RadioItems,
            persistence=True,
            labelClassName='pr-3',
            inline=True,
            )
        radio_items.options = [
                {
                    'label': c,
                    'value': c,
                    'label_id': f'{radio_items.id}-{i}',
                    }
                for i, c in enumerate(choices)
                ]
        radio_items.value = radio_items.options[0]['value']
        label_ids = [o['label_id'] for o in radio_items.options]
        extra_label_items = [
                select_container.child(
                    dbc.Badge, children='',
                    pill=True, color='light',
                    className='ml-2 font-weight-normal text-primary',
                    style={
                        'width': '30px',
                        })
                for _ in range(len(radio_items.options))
                ]

        # attach the descriptions as tooltips
        for i, desc in enumerate(info['desc']):
            select_container.child(
                    dbc.Tooltip, desc,
                    target=f'{radio_items.id} > '
                           f'.custom-radio:nth-of-type({i + 1}) label',
                    placement='right',
                    )

        # this is to update the label with a badge to indicate table size
        @app.callback(
                [
                    Output(extra_label_item.id, 'children')
                    for extra_label_item in extra_label_items
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_db_table_sizes(n_intervals):
            return query_table_sizes(*choices)

        # this is to move the input item badges
        js_hack_url = resolve_url(f'/js/hack_input_items_{radio_items.id}.js')
        container.child(dji.Import, src=js_hack_url)

        @app.server.route(js_hack_url, endpoint=f'{radio_items.id}')
        def js_hack():
            return ''.join(["""
document.getElementById('{dest}').appendChild(
    document.getElementById('{src}')
);""".format(dest=label_id, src=extra_label_item.id)
                for label_id, extra_label_item in zip(
                    label_ids, extra_label_items)
                ])

        return radio_items

    def _setup_db_view(self, app, container, radio_items, bindkey):

        timer = container.child(dcc.Interval, interval=1000)

        _db = self._db[bindkey]
        _t = _db.tables

        # from sqlalchemy.ext.automap import automap_base
        # base = automap_base(metadata=meta)
        # base.prepare()

        tbl = container.child(
                DataTable,
                page_current=0,
                page_size=10,
                page_action='custom',
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',
                    'overflowY': 'auto',
                    'height': '25vh',
                },
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '60px',
                    'maxWidth': '500px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                css=[
                    # this is needed to make the markdown <p> vertically
                    # centered.
                    {
                        'selector': '.cell-markdown p, pre',
                        'rule': '''
                            margin: 0.5rem 0
                        '''
                        },
                    {
                        'selector': '.cell-markdown pre',
                        'rule': '''
                            background-color: #fafafa;
                            font-size: 12px;
                        '''
                        },
                    ],
                style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'maxWidth': '200px',
                            }
                        for c in ['context', 'source']
                        ] + [{
                            'if': {'column_id': 'source_url'},
                            'maxWidth': '100px',
                            'wordWrap': 'break-word'
                            }
                    ],
                )

        @app.callback(
            Output(tbl.id, 'page_current'),
            [
                Input(radio_items.id, 'value'),
                ]
            )
        def reset_table_page(table_name):
            return 0

        @app.callback(
            [
                Output(tbl.id, 'columns'),
                Output(tbl.id, 'data'),
                Output(tbl.id, 'page_count')
                ],
            [
                Input(radio_items.id, 'value'),
                Input(tbl.id, "page_current"),
                Input(tbl.id, "page_size"),
                Input(timer.id, 'n_intervals')
                ]
            )
        def update_table(table_name, page_current, page_size, n_intervals):
            if table_name is None:
                return None, None, 1
            session = _db.session
            if table_name not in _t:
                raise dash.exceptions.PreventUpdate(
                        f'table {table_name} not defined in db')
            db_table = _t[table_name]
            stmt = select([db_table]).limit(page_size).offset(
                    page_current * page_size)
            info = dataframe_from_db(
                    stmt, session=session)
            # we need to handle json types separately
            json_cols = {'context', 'source'}.intersection(info.columns)
            for col in json_cols:
                info[col] = info[col].apply(
                        lambda x:
                        '' if x is None else
                        f"```json\n{json.dumps(x, indent=2)}\n```")
            columns = [
                    {
                        "name": i,
                        "id": i,
                        "presentation":
                            "markdown" if i in json_cols else "input"
                        } for i in info.columns
                    ]
            data = info.to_dict('records')
            with _db.session_context as session:
                size = session.execute(db_table.count()).scalar()
            if size == 0:
                n_pages = 1
            else:
                n_pages = size // page_size + (
                        1 if size % page_size > 0 else 0)
            return columns, data, n_pages

    def _setup_graph_view(self, app, container):
        container.child(html.Hr())
        timer, loading, error = self._setup_live_update_header(
                app, container, 'Dependency Graph', 10000)

        graph_container = container.child(dbc.Row).child(dbc.Col)

        cyto_layouts = [
                'random', 'grid', 'circle', 'concentric',
                'breadthfirst', 'cose', 'cose-bilkent',
                'cola', 'euler', 'spread', 'dagre',
                'klay',
                ]

        graph_controls_container = container.child(
                dbc.Form, inline=True)
        graph_layout_select_group = graph_controls_container.child(
                dbc.FormGroup)
        graph_layout_select_group.child(
                dbc.Label,
                'Layout:',
                className='mr-3'
                )
        graph_layout_select = graph_layout_select_group.child(
                dbc.Select,
                options=[
                    {'label': layout, 'value': layout}
                    for layout in cyto_layouts
                    ],
                value='cola',
                )

        def _get_layout(value):
            return {
                    'name': value,
                    'animate': True,
                    'nodeDimensionsIncludeLabels': True,
                    }

        graph = graph_container.child(
                cyto.Cytoscape,
                layout_=_get_layout(graph_layout_select.value),
                style={'width': '100%', 'height': '1200px'},
                elements=[],
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'label': 'data(label)',
                            'background-color': '#66ccff',
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'curve-style': 'bezier',
                            'target-arrow-shape': 'triangle',
                            },
                        },
                    ],
                userZoomingEnabled=False
                )

        _t = self._db['tolteca'].tables

        dpa_types, dpa_info = self._query_choices_table(
                'data_prod_assoc_type', exclude_parent=True)
        dispatch_idcols = {
                dpa_type: {
                    'src': list(_t[dpa_type].c)[1].name,
                    'dst': list(_t[dpa_type].c)[2].name
                    }
                for dpa_type in dpa_types
                }

        @app.callback(
                Output(graph.id, 'layout'),
                [
                    Input(graph_layout_select.id, 'value')
                    ]
                )
        def update_graph_layout(value):
            if value is None:
                return dash.no_update
            return _get_layout(value)

        update_listener = container.child(html.Div)

        @app.callback(
                [
                    Output(graph.id, 'elements'),
                    Output(loading.id, 'children')
                    ],
                [
                    Input(timer.id, 'n_intervals'),
                    Input(update_listener.id, 'children')
                    ]
                )
        def update_graph_elements(n_intervals, trigger_update):
            session = self._db['tolteca'].session
            dataprods = dataframe_from_db(
                    select([
                        _t['data_prod'],
                        _t['data_prod_type'].c['label'].label(
                            'data_prod_type')
                        ]).select_from(
                            _t['data_prod'].join(
                                _t['data_prod_type'])
                        ).limit(100),
                    session=session)
            assocs = dict()
            for dpa_type in dpa_types:
                assoc = dataframe_from_db(
                    select([
                        _t[dpa_type],
                        _t['data_prod_assoc'].c['data_prod_assoc_info_pk'],
                        ] + [
                            c for c in _t['data_prod_assoc_info'].c
                            if c.name != 'pk'
                            ]).select_from(
                        _t[dpa_type].outerjoin(
                            _t['data_prod_assoc']
                            ).outerjoin(
                                _t['data_prod_assoc_info'])), session=session)
                if not assoc.empty:
                    assocs[dpa_type] = assoc
            nodes = [
                    {
                        'data': {
                            'id': r['pk'],
                            'label': f"{r['data_prod_type']}-{r['pk']}"
                            }
                        }
                    for _, r in dataprods.iterrows()
                    ]
            edges = []
            for dpa_type, a in assocs.items():
                idcols = dispatch_idcols[dpa_type]
                for _, r in a.iterrows():
                    edges.append(
                        {
                            'data': {
                                'source': r[idcols['src']],
                                'target': r[idcols['dst']],
                                'label':
                                    f"assoc-{r['data_prod_assoc_info_pk']}"
                                }
                        })
            return nodes + edges, ""

        return update_listener

    def _setup_dataset_upload(self, app, container, update_listener):
        message_box = container.child(
                dbc.Row, className='mt-4 mb-2').child(dbc.Col)
        upload_area = container.child(dbc.Row).child(dbc.Col).child(
            dcc.Upload,
            style={
                'width': '100%',
                'height': '200px',
                'lineHeight': '200px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
                },
            # Allow multiple files to be uploaded
            multiple=True
            )
        upload_area.child(
            html.Div,
            [
                'Add Data Product(s) by Drag and Drop or ',
                html.A('Select Files')
                ])

        @app.callback(
                [
                    Output(message_box.id, 'children'),
                    Output(update_listener.id, 'children')
                    ],
                [
                    Input(upload_area.id, 'contents')
                    ],
                [
                    State(upload_area.id, 'filename'),
                    State(upload_area.id, 'last_modified')
                    ])
        def update_output(list_of_contents, list_of_names, list_of_dates):
            if list_of_contents is None:
                return dash.no_update, dash.no_update
            items = []
            for c, n, d in zip(
                    list_of_contents, list_of_names, list_of_dates):
                content_type, content_string = c.split(',')
                decoded = base64.b64decode(content_string)
                content = decoded.decode('utf-8')
                items.append({
                    'content': content,
                    'filename': n,
                    'last_modified': d
                    })
            from tolteca.recipes.collect_data_prods import (
                    collect_data_prods)

            def dataset_from_string(content, name):
                dataset = BasicObsDataset(
                        index_table=Table.read(content, format='ascii'))
                dataset.source = name
                return dataset
            datasets = [
                    dataset_from_string(item['content'], item['filename'])
                    for item in items
                    ]
            n_items = collect_data_prods(self._db['tolteca'], datasets)
            return dbc.Alert(
                    f'Added {n_items} items.',
                    dismissable=True, duration=4000), ""
